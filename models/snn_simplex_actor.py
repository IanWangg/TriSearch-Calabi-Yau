from __future__ import annotations

import torch
import torch.nn as nn

from core.snn_simplex_topology import build_top_degree_down_laplacian

from .act_resolver import activation_resolver


class ChebyshevSimplicialConvolution(nn.Module):
    """Torch-native Chebyshev convolution on one simplex degree."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        chebyshev_order: int = 3,
        enable_bias: bool = True,
    ):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if int(out_channels) <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if int(chebyshev_order) <= 0:
            raise ValueError(f"chebyshev_order must be positive, got {chebyshev_order}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.chebyshev_order = int(chebyshev_order)
        self.theta = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.chebyshev_order)
        )
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.theta)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, laplacian: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be 2D with shape [num_simplices, channels], got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"x has {x.size(1)} channels, expected {self.in_channels}.")
        if laplacian.dim() != 2 or laplacian.size(0) != laplacian.size(1):
            raise ValueError("laplacian must be square.")
        if laplacian.size(0) != x.size(0):
            raise ValueError(
                "laplacian size must match number of simplex features: "
                f"{laplacian.size(0)} vs {x.size(0)}."
            )

        terms = [x]
        if self.chebyshev_order > 1:
            terms.append(torch.sparse.mm(laplacian, terms[0]))
        for order in range(2, self.chebyshev_order):
            terms.append(2.0 * torch.sparse.mm(laplacian, terms[order - 1]) - terms[order - 2])

        features = torch.stack(terms, dim=-1)
        output = torch.einsum("mik,oik->mo", features, self.theta)
        if self.bias is not None:
            output = output + self.bias
        return output


class SNNSimplexActor(nn.Module):
    """Top-simplex SNN actor compatible with PyG-batched local vertex ids."""

    _CACHED_TOPOLOGY_ATTRS = (
        "snn_laplacian_row",
        "snn_laplacian_col",
        "snn_laplacian_value",
        "num_snn_laplacian_entries",
        "snn_candidate",
        "snn_simplex",
        "num_snn_candidate_simplex_memberships",
    )

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int,
        num_layers: int,
        chebyshev_order: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        layer_dims = [int(channels)]
        if int(num_layers) == 1:
            layer_dims.append(int(channels))
        else:
            layer_dims.extend([int(hidden_channels)] * (int(num_layers) - 1))
            layer_dims.append(int(channels))

        self.layers = nn.ModuleList(
            ChebyshevSimplicialConvolution(
                layer_dims[layer_idx],
                layer_dims[layer_idx + 1],
                chebyshev_order=chebyshev_order,
            )
            for layer_idx in range(len(layer_dims) - 1)
        )
        self.act = activation_resolver(act)

    @staticmethod
    def _extract_topology(batch, device, *, validate_counts: bool = True):
        if not hasattr(batch, "simplex_vertices"):
            raise AttributeError(
                "Batched graph is missing `simplex_vertices`; build data with "
                "include_simplex_topology=True for subcomplex_actor_type='snn_simplex'."
            )
        if not hasattr(batch, "num_top_simplices"):
            raise AttributeError(
                "Batched graph is missing `num_top_simplices`; build data with "
                "include_simplex_topology=True for subcomplex_actor_type='snn_simplex'."
            )

        simplex_vertices = batch.simplex_vertices
        if not isinstance(simplex_vertices, torch.Tensor):
            raise TypeError("`simplex_vertices` must be a torch.Tensor.")
        if simplex_vertices.dim() == 1:
            simplex_vertices = simplex_vertices.view(1, -1)
        elif simplex_vertices.dim() != 2:
            raise ValueError(
                "`simplex_vertices` must be 1D or 2D, got shape "
                f"{tuple(simplex_vertices.shape)}."
            )
        simplex_vertices = simplex_vertices.to(device=device, dtype=torch.long)

        num_top_simplices = batch.num_top_simplices
        if not isinstance(num_top_simplices, torch.Tensor):
            num_top_simplices = torch.tensor(num_top_simplices, device=device, dtype=torch.long)
        else:
            num_top_simplices = num_top_simplices.to(device=device, dtype=torch.long).view(-1)
        if validate_counts and int(num_top_simplices.sum().item()) != int(simplex_vertices.size(0)):
            raise ValueError(
                "Mismatch between batched `simplex_vertices` rows and `num_top_simplices`."
            )
        return simplex_vertices, num_top_simplices

    @classmethod
    def _has_cached_topology(cls, batch) -> bool:
        return all(hasattr(batch, attr_name) for attr_name in cls._CACHED_TOPOLOGY_ATTRS)

    @staticmethod
    def _as_long_vector(value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.long).view(-1)
        return torch.tensor(value, device=device, dtype=torch.long).view(-1)

    @staticmethod
    def _as_float_vector(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype).view(-1)
        return torch.tensor(value, device=device, dtype=dtype).view(-1)

    @staticmethod
    def _pool_simplex_vertex_embeddings(
        node_embeddings: torch.Tensor,
        simplex_vertices: torch.Tensor,
        *,
        node_offset: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if simplex_vertices.numel() == 0:
            return node_embeddings.new_empty((0, node_embeddings.size(-1)))
        if int(simplex_vertices.min().item()) < 0:
            raise ValueError("`simplex_vertices` must not contain padding.")
        if int(simplex_vertices.max().item()) >= int(num_nodes):
            raise IndexError("Simplex vertex index out of range for graph.")
        global_simplex_vertices = simplex_vertices + node_offset
        gathered = node_embeddings[global_simplex_vertices]
        return gathered.amax(dim=1)

    @staticmethod
    def _pool_candidate_simplex_embeddings(
        simplex_embeddings: torch.Tensor,
        simplex_vertices: torch.Tensor,
        candidate_vertices: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_vertices.size(0) == 0:
            return simplex_embeddings.new_empty((0, simplex_embeddings.size(-1)))
        if simplex_embeddings.size(0) == 0:
            raise ValueError("Cannot score candidates without current top simplices.")

        contains = (
            simplex_vertices[:, None, :, None]
            == candidate_vertices[None, :, None, :]
        ).any(dim=-1).all(dim=-1).transpose(0, 1)
        if bool((~contains.any(dim=1)).any().item()):
            raise ValueError(
                "At least one candidate subcomplex contains no current top-dimensional simplex."
            )

        pooled_inputs = simplex_embeddings.unsqueeze(0).expand(candidate_vertices.size(0), -1, -1)
        pooled_inputs = pooled_inputs.masked_fill(~contains.unsqueeze(-1), float("-inf"))
        return pooled_inputs.amax(dim=1)

    def _apply_snn_layers(self, laplacian: torch.Tensor, simplex_embeddings: torch.Tensor) -> torch.Tensor:
        z = simplex_embeddings
        for layer_idx, layer in enumerate(self.layers):
            z = layer(laplacian, z)
            if layer_idx != len(self.layers) - 1:
                z = self.act(z)
        return z

    def _forward_cached(
        self,
        *,
        node_embeddings: torch.Tensor,
        subcomplex_vertices: torch.Tensor,
        num_available_subcomplexes: torch.Tensor,
        node_ptr: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = node_embeddings.device
        dtype = node_embeddings.dtype

        simplex_vertices, num_top_simplices = self._extract_topology(
            batch=batch,
            device=device,
            validate_counts=False,
        )
        batch_size = int(num_available_subcomplexes.numel())
        if int(node_ptr.numel()) != batch_size + 1:
            raise ValueError("node_ptr must have one more element than the number of graphs.")
        if subcomplex_vertices.size(0) == 0:
            raise ValueError("Each graph must provide at least one candidate subcomplex.")
        if simplex_vertices.size(0) == 0:
            raise ValueError("Cannot score candidates without current top simplices.")

        graph_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        simplex_start = torch.cumsum(num_top_simplices, dim=0) - num_top_simplices
        candidate_start = torch.cumsum(num_available_subcomplexes, dim=0) - num_available_subcomplexes

        simplex_graph = torch.repeat_interleave(graph_ids, num_top_simplices)
        node_offsets = node_ptr[:-1].to(device=device, dtype=torch.long)
        global_simplex_vertices = simplex_vertices + node_offsets[simplex_graph].unsqueeze(1)
        simplex_embeddings = node_embeddings[global_simplex_vertices].amax(dim=1)

        num_snn_laplacian_entries = self._as_long_vector(
            batch.num_snn_laplacian_entries,
            device,
        )
        laplacian_entry_graph = torch.repeat_interleave(graph_ids, num_snn_laplacian_entries)
        snn_laplacian_row = self._as_long_vector(batch.snn_laplacian_row, device)
        snn_laplacian_col = self._as_long_vector(batch.snn_laplacian_col, device)
        snn_laplacian_value = self._as_float_vector(
            batch.snn_laplacian_value,
            device=device,
            dtype=dtype,
        )
        laplacian_row = snn_laplacian_row + simplex_start[laplacian_entry_graph]
        laplacian_col = snn_laplacian_col + simplex_start[laplacian_entry_graph]
        laplacian = torch.sparse_coo_tensor(
            torch.stack([laplacian_row, laplacian_col], dim=0),
            snn_laplacian_value,
            size=(simplex_vertices.size(0), simplex_vertices.size(0)),
            device=device,
            dtype=dtype,
        ).coalesce()
        simplex_embeddings = self._apply_snn_layers(laplacian, simplex_embeddings)

        num_memberships = self._as_long_vector(
            batch.num_snn_candidate_simplex_memberships,
            device,
        )
        membership_graph = torch.repeat_interleave(graph_ids, num_memberships)
        snn_candidate = self._as_long_vector(batch.snn_candidate, device)
        snn_simplex = self._as_long_vector(batch.snn_simplex, device)
        global_candidate = snn_candidate + candidate_start[membership_graph]
        global_simplex = snn_simplex + simplex_start[membership_graph]

        candidate_features = node_embeddings.new_full(
            (subcomplex_vertices.size(0), node_embeddings.size(-1)),
            float("-inf"),
        )
        candidate_features.scatter_reduce_(
            0,
            global_candidate.unsqueeze(1).expand(-1, node_embeddings.size(-1)),
            simplex_embeddings[global_simplex],
            reduce="amax",
            include_self=True,
        )
        candidate_graph_indices = torch.repeat_interleave(graph_ids, num_available_subcomplexes)
        return candidate_features, candidate_graph_indices

    def _forward_slow(
        self,
        *,
        node_embeddings: torch.Tensor,
        subcomplex_vertices: torch.Tensor,
        num_available_subcomplexes: torch.Tensor,
        node_ptr: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        simplex_vertices, num_top_simplices = self._extract_topology(
            batch=batch,
            device=node_embeddings.device,
        )
        batch_size = int(num_available_subcomplexes.numel())
        if int(node_ptr.numel()) != batch_size + 1:
            raise ValueError("node_ptr must have one more element than the number of graphs.")

        candidate_start = torch.cumsum(num_available_subcomplexes, dim=0) - num_available_subcomplexes
        simplex_start = torch.cumsum(num_top_simplices, dim=0) - num_top_simplices
        candidate_features = []
        candidate_graph_indices = []

        for graph_idx in range(batch_size):
            num_candidates = int(num_available_subcomplexes[graph_idx].item())
            if num_candidates == 0:
                continue

            graph_candidate_vertices = subcomplex_vertices[
                candidate_start[graph_idx] : candidate_start[graph_idx] + num_candidates
            ]
            graph_candidate_vertices = graph_candidate_vertices.clamp_min(-1)
            candidate_valid_mask = graph_candidate_vertices >= 0
            if bool((candidate_valid_mask.sum(dim=1) <= 0).any().item()):
                raise ValueError("Encountered an empty subcomplex candidate.")
            max_candidate_vertex = graph_candidate_vertices.masked_fill(~candidate_valid_mask, 0).amax()
            num_nodes = int((node_ptr[graph_idx + 1] - node_ptr[graph_idx]).item())
            if int(max_candidate_vertex.item()) >= num_nodes:
                raise IndexError("Subcomplex vertex index out of range for graph.")
            graph_simplex_vertices = simplex_vertices[
                simplex_start[graph_idx] : simplex_start[graph_idx] + num_top_simplices[graph_idx]
            ]

            node_offset = node_ptr[graph_idx].to(device=node_embeddings.device, dtype=torch.long)
            simplex_embeddings = self._pool_simplex_vertex_embeddings(
                node_embeddings,
                graph_simplex_vertices,
                node_offset=node_offset,
                num_nodes=num_nodes,
            )
            laplacian = build_top_degree_down_laplacian(
                graph_simplex_vertices,
                dtype=node_embeddings.dtype,
            )
            simplex_embeddings = self._apply_snn_layers(laplacian, simplex_embeddings)
            candidate_features.append(
                self._pool_candidate_simplex_embeddings(
                    simplex_embeddings,
                    graph_simplex_vertices,
                    graph_candidate_vertices,
                )
            )
            candidate_graph_indices.append(
                torch.full(
                    (num_candidates,),
                    graph_idx,
                    device=node_embeddings.device,
                    dtype=torch.long,
                )
            )

        if not candidate_features:
            raise ValueError("Each graph must provide at least one candidate subcomplex.")
        return torch.cat(candidate_features, dim=0), torch.cat(candidate_graph_indices, dim=0)

    def forward(
        self,
        *,
        node_embeddings: torch.Tensor,
        subcomplex_vertices: torch.Tensor,
        num_available_subcomplexes: torch.Tensor,
        node_ptr: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._has_cached_topology(batch):
            return self._forward_cached(
                node_embeddings=node_embeddings,
                subcomplex_vertices=subcomplex_vertices,
                num_available_subcomplexes=num_available_subcomplexes,
                node_ptr=node_ptr,
                batch=batch,
            )
        return self._forward_slow(
            node_embeddings=node_embeddings,
            subcomplex_vertices=subcomplex_vertices,
            num_available_subcomplexes=num_available_subcomplexes,
            node_ptr=node_ptr,
            batch=batch,
        )
