import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from .act_resolver import activation_resolver
from .snn_simplex_actor import SNNSimplexActor


class GCNSubcomplexAgent(nn.Module):
    SUPPORTED_SUBCOMPLEX_ACTOR_TYPES = ("mlp", "gnn", "circuit_pool", "snn_simplex", "default")
    LEGACY_SUBCOMPLEX_ACTOR_ALIASES = {"default": "gnn"}

    """
    Subcomplex PPO policy with a GCN encoder.

    This is the non-EGNN architecture ablation for subcomplex policies.  It
    preserves the EGNNSubcomplexAgent public interface used by training and
    evaluation, but never updates coordinates and never instantiates EGNN
    modules.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64,
        num_layers=3,
        share_encoder=True,
        mlp_hidden_channel_list=[64],
        use_projection=True,
        act="silu",
        subcomplex_actor_type="gnn",
        device="cpu",
    ):
        super().__init__()
        del share_encoder

        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.use_projection = bool(use_projection)
        self.device = device
        self.act = activation_resolver(act)
        self.subcomplex_actor_type = self._normalize_subcomplex_actor_type(
            subcomplex_actor_type
        )

        layer_dims = [int(in_channels)]
        if int(num_layers) == 1:
            layer_dims.append(int(out_channels))
        else:
            layer_dims.extend([int(hidden_channels)] * (int(num_layers) - 1))
            layer_dims.append(int(out_channels))

        self.encoder_layers = nn.ModuleList(
            gnn.GCNConv(layer_dims[idx], layer_dims[idx + 1])
            for idx in range(len(layer_dims) - 1)
        )
        if self.subcomplex_actor_type == "gnn":
            decoder_layer_dims = [int(out_channels)]
            if int(num_layers) == 1:
                decoder_layer_dims.append(int(out_channels))
            else:
                decoder_layer_dims.extend([int(hidden_channels)] * (int(num_layers) - 1))
                decoder_layer_dims.append(int(out_channels))
            self.subcomplex_decoder_layers = nn.ModuleList(
                gnn.GCNConv(decoder_layer_dims[idx], decoder_layer_dims[idx + 1])
                for idx in range(len(decoder_layer_dims) - 1)
            )
            self.subcomplex_decoder_head = nn.Linear(out_channels, 1)
        elif self.subcomplex_actor_type == "snn_simplex":
            self.snn_simplex_actor = SNNSimplexActor(
                channels=out_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                act=act,
            )
            self.subcomplex_decoder_head = nn.Linear(out_channels, 1)

        if self.use_projection:
            mlp_full_layer_list = [out_channels] + list(mlp_hidden_channel_list) + [out_channels]
            self.projection = gnn.models.MLP(mlp_full_layer_list, act=act)
        else:
            self.projection = nn.Identity()

        self.value_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            activation_resolver(act),
            nn.Linear(out_channels, 1),
        )

        actor_input_channels = out_channels
        if self.subcomplex_actor_type == "mlp":
            actor_input_channels = out_channels * 2
        self.subcomplex_head = nn.Sequential(
            nn.Linear(actor_input_channels, out_channels),
            activation_resolver(act),
            nn.Linear(out_channels, 1),
        )

        self.to(device)

    @classmethod
    def _normalize_subcomplex_actor_type(cls, subcomplex_actor_type):
        resolved_actor_type = str(subcomplex_actor_type).strip().lower()
        resolved_actor_type = cls.LEGACY_SUBCOMPLEX_ACTOR_ALIASES.get(
            resolved_actor_type,
            resolved_actor_type,
        )
        if resolved_actor_type not in cls.SUPPORTED_SUBCOMPLEX_ACTOR_TYPES:
            raise ValueError(
                f"Unsupported subcomplex_actor_type '{subcomplex_actor_type}'. "
                f"Expected one of: {', '.join(cls.SUPPORTED_SUBCOMPLEX_ACTOR_TYPES)}."
            )
        return resolved_actor_type

    def encode(self, h, edge_index):
        z = h
        for layer_idx, conv in enumerate(self.encoder_layers):
            z = conv(z, edge_index)
            if layer_idx != len(self.encoder_layers) - 1:
                z = self.act(z)
        return z

    def _extract_batched_subcomplex_data(self, batch, device):
        if not hasattr(batch, "subcomplex_vertices"):
            raise AttributeError("Batched graph is missing `subcomplex_vertices`.")
        if not hasattr(batch, "num_available_subcomplexes"):
            raise AttributeError("Batched graph is missing `num_available_subcomplexes`.")

        subcomplex_vertices = batch.subcomplex_vertices
        if not isinstance(subcomplex_vertices, torch.Tensor):
            raise TypeError("`subcomplex_vertices` must be a torch.Tensor.")
        if subcomplex_vertices.dim() == 1:
            subcomplex_vertices = subcomplex_vertices.view(1, -1)
        elif subcomplex_vertices.dim() != 2:
            raise ValueError(
                "`subcomplex_vertices` must be 1D or 2D, got shape "
                f"{tuple(subcomplex_vertices.shape)}."
            )
        subcomplex_vertices = subcomplex_vertices.to(device=device, dtype=torch.long)

        num_available = batch.num_available_subcomplexes
        if not isinstance(num_available, torch.Tensor):
            num_available = torch.tensor(num_available, device=device, dtype=torch.long)
        else:
            num_available = num_available.to(device=device, dtype=torch.long).view(-1)
        if int(num_available.sum().item()) != int(subcomplex_vertices.size(0)):
            raise ValueError(
                "Mismatch between batched `subcomplex_vertices` rows and "
                "`num_available_subcomplexes`."
            )
        return subcomplex_vertices, num_available

    def _pool_batched_subcomplex_embeddings(
        self,
        node_embeddings,
        subcomplex_vertices,
        num_available_subcomplexes,
        node_ptr,
    ):
        if subcomplex_vertices.size(0) == 0:
            raise ValueError("Each graph must provide at least one candidate subcomplex.")

        candidate_graph_index = torch.repeat_interleave(
            torch.arange(
                num_available_subcomplexes.numel(),
                device=node_embeddings.device,
                dtype=torch.long,
            ),
            num_available_subcomplexes,
        )
        if int(candidate_graph_index.numel()) != int(subcomplex_vertices.size(0)):
            raise ValueError("Candidate graph assignment does not match subcomplex rows.")

        valid_mask = subcomplex_vertices >= 0
        if not bool(valid_mask.any().item()):
            raise ValueError("Encountered a batch with only empty subcomplex candidates.")

        node_offsets = node_ptr[:-1].to(device=node_embeddings.device, dtype=torch.long)
        global_vertices = subcomplex_vertices.clamp_min(0) + node_offsets[candidate_graph_index].unsqueeze(1)

        node_counts = (node_ptr[1:] - node_ptr[:-1]).to(device=node_embeddings.device, dtype=torch.long)
        max_local_index = subcomplex_vertices.masked_fill(~valid_mask, 0).amax(dim=1)
        if bool((max_local_index >= node_counts[candidate_graph_index]).any().item()):
            raise IndexError("Subcomplex vertex index out of range for at least one graph in the batch.")

        gathered = node_embeddings[global_vertices]
        gathered = gathered.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        return gathered.amax(dim=1), candidate_graph_index

    @staticmethod
    def _build_complete_subcomplex_edges(num_nodes, device):
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        node_ids = torch.arange(num_nodes, device=device, dtype=torch.long)
        row = node_ids.repeat_interleave(num_nodes)
        col = node_ids.repeat(num_nodes)
        mask = row != col
        return torch.stack([row[mask], col[mask]], dim=0)

    def _decode_subcomplex(self, sub_h, sub_edges):
        for layer_idx, conv in enumerate(self.subcomplex_decoder_layers):
            sub_h = conv(sub_h, sub_edges)
            if layer_idx != len(self.subcomplex_decoder_layers) - 1:
                sub_h = self.act(sub_h)
        return sub_h

    @staticmethod
    def _build_batched_complete_subcomplex_edges(valid_counts, max_subcomplex_width):
        device = valid_counts.device
        if int(max_subcomplex_width) <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        local_ids = torch.arange(max_subcomplex_width, device=device, dtype=torch.long)
        local_src = local_ids.repeat_interleave(max_subcomplex_width)
        local_dst = local_ids.repeat(max_subcomplex_width)
        non_self_mask = local_src != local_dst

        valid_edge_mask = (
            non_self_mask.unsqueeze(0)
            & (local_src.unsqueeze(0) < valid_counts.unsqueeze(1))
            & (local_dst.unsqueeze(0) < valid_counts.unsqueeze(1))
        )
        node_starts = torch.cumsum(valid_counts, dim=0) - valid_counts
        src = (node_starts.unsqueeze(1) + local_src.unsqueeze(0)).masked_select(valid_edge_mask)
        dst = (node_starts.unsqueeze(1) + local_dst.unsqueeze(0)).masked_select(valid_edge_mask)
        return torch.stack([src, dst], dim=0)

    def _build_batched_subcomplex_graph_inputs(
        self,
        subcomplex_vertices,
        num_available_subcomplexes,
        node_ptr,
        device,
    ):
        if subcomplex_vertices.size(0) == 0:
            raise ValueError("Each graph must provide at least one candidate subcomplex.")

        candidate_graph_index = torch.repeat_interleave(
            torch.arange(
                num_available_subcomplexes.numel(),
                device=device,
                dtype=torch.long,
            ),
            num_available_subcomplexes,
        )
        if int(candidate_graph_index.numel()) != int(subcomplex_vertices.size(0)):
            raise ValueError("Candidate graph assignment does not match subcomplex rows.")

        valid_mask = subcomplex_vertices >= 0
        valid_counts = valid_mask.sum(dim=1)
        if bool((valid_counts <= 0).any().item()):
            raise ValueError("Encountered an empty subcomplex candidate.")

        node_offsets = node_ptr[:-1].to(device=device, dtype=torch.long)
        node_counts = (node_ptr[1:] - node_ptr[:-1]).to(device=device, dtype=torch.long)
        max_local_index = subcomplex_vertices.masked_fill(~valid_mask, 0).amax(dim=1)
        if bool((max_local_index >= node_counts[candidate_graph_index]).any().item()):
            raise IndexError("Subcomplex vertex index out of range for at least one graph in the batch.")

        global_vertices = subcomplex_vertices.clamp_min(0) + node_offsets[candidate_graph_index].unsqueeze(1)
        flat_global_vertices = global_vertices[valid_mask]
        candidate_node_batch = torch.repeat_interleave(
            torch.arange(
                subcomplex_vertices.size(0),
                device=device,
                dtype=torch.long,
            ),
            valid_counts,
        )
        subcomplex_edges = self._build_batched_complete_subcomplex_edges(
            valid_counts=valid_counts,
            max_subcomplex_width=subcomplex_vertices.size(1),
        )
        return flat_global_vertices, candidate_node_batch, subcomplex_edges, candidate_graph_index

    def _decode_and_pool_batched_subcomplex_embeddings(
        self,
        node_embeddings,
        subcomplex_vertices,
        num_available_subcomplexes,
        node_ptr,
    ):
        flat_global_vertices, pooled_batch, subcomplex_edges, candidate_graph_index = (
            self._build_batched_subcomplex_graph_inputs(
                subcomplex_vertices=subcomplex_vertices,
                num_available_subcomplexes=num_available_subcomplexes,
                node_ptr=node_ptr,
                device=node_embeddings.device,
            )
        )
        pooled_inputs = self._decode_subcomplex(
            node_embeddings[flat_global_vertices],
            subcomplex_edges,
        )
        return (
            gnn.pool.global_max_pool(
                pooled_inputs,
                pooled_batch,
                size=subcomplex_vertices.size(0),
            ),
            candidate_graph_index,
        )

    def _build_padded_logits(self, logits_flat, num_available_subcomplexes):
        batch_size = int(num_available_subcomplexes.numel())
        max_candidates = int(num_available_subcomplexes.max().item())
        padded_logits = torch.full(
            (batch_size, max_candidates),
            float("-inf"),
            device=logits_flat.device,
            dtype=logits_flat.dtype,
        )
        candidate_graph_index = torch.repeat_interleave(
            torch.arange(batch_size, device=logits_flat.device, dtype=torch.long),
            num_available_subcomplexes,
        )
        candidate_start = torch.cumsum(num_available_subcomplexes, dim=0) - num_available_subcomplexes
        candidate_position = (
            torch.arange(logits_flat.size(0), device=logits_flat.device, dtype=torch.long)
            - torch.repeat_interleave(candidate_start, num_available_subcomplexes)
        )
        padded_logits[candidate_graph_index, candidate_position] = logits_flat
        return padded_logits

    def get_value_and_logits(self, batch):
        node_feature = batch.x
        edge_index = batch.edge_index

        z_before_proj = self.encode(node_feature, edge_index)
        global_feature = gnn.pool.global_max_pool(z_before_proj, batch.batch)
        value = self.value_head(global_feature)

        if self.subcomplex_actor_type in ("gnn", "circuit_pool", "snn_simplex"):
            policy_node_embeddings = z_before_proj
        else:
            policy_node_embeddings = self.projection(z_before_proj)
        subcomplex_vertices, num_available_subcomplexes = self._extract_batched_subcomplex_data(
            batch=batch,
            device=policy_node_embeddings.device,
        )
        if self.subcomplex_actor_type == "gnn":
            subcomplex_features, _candidate_graph_index = self._decode_and_pool_batched_subcomplex_embeddings(
                node_embeddings=policy_node_embeddings,
                subcomplex_vertices=subcomplex_vertices,
                num_available_subcomplexes=num_available_subcomplexes,
                node_ptr=batch.ptr,
            )
            logits_flat = self.subcomplex_decoder_head(subcomplex_features).squeeze(-1)
        elif self.subcomplex_actor_type == "snn_simplex":
            subcomplex_features, _candidate_graph_index = self.snn_simplex_actor(
                node_embeddings=policy_node_embeddings,
                subcomplex_vertices=subcomplex_vertices,
                num_available_subcomplexes=num_available_subcomplexes,
                node_ptr=batch.ptr,
                batch=batch,
            )
            logits_flat = self.subcomplex_decoder_head(subcomplex_features).squeeze(-1)
        else:
            subcomplex_features, candidate_graph_index = self._pool_batched_subcomplex_embeddings(
                node_embeddings=policy_node_embeddings,
                subcomplex_vertices=subcomplex_vertices,
                num_available_subcomplexes=num_available_subcomplexes,
                node_ptr=batch.ptr,
            )
            if self.subcomplex_actor_type == "circuit_pool":
                policy_features = subcomplex_features
            else:
                policy_features = torch.cat(
                    [subcomplex_features, global_feature[candidate_graph_index]],
                    dim=-1,
                )
            logits_flat = self.subcomplex_head(policy_features).squeeze(-1)
        logits_padded = self._build_padded_logits(logits_flat, num_available_subcomplexes)
        return value.squeeze(-1), logits_padded

    def forward(self, batch, deterministic=False):
        value, logits_padded = self.get_value_and_logits(batch)

        probs_padded = F.softmax(logits_padded, dim=1)
        dist = torch.distributions.Categorical(probs=probs_padded)

        if deterministic:
            action_indices = torch.argmax(probs_padded, dim=1)
        else:
            action_indices = dist.sample()

        log_probs = dist.log_prob(action_indices)
        entropy = dist.entropy()

        subcomplex_vertices, num_available_subcomplexes = self._extract_batched_subcomplex_data(
            batch=batch,
            device=logits_padded.device,
        )
        candidate_start = torch.cumsum(num_available_subcomplexes, dim=0) - num_available_subcomplexes
        selected_actions_padded = subcomplex_vertices[candidate_start + action_indices]

        return selected_actions_padded, action_indices, value.squeeze(-1), log_probs, entropy

    def get_value(self, batch):
        z_before_proj = self.encode(batch.x, batch.edge_index)
        global_feature = gnn.pool.global_max_pool(z_before_proj, batch.batch)
        value = self.value_head(global_feature)
        return value.squeeze(-1)

    def get_log_prob(self, logits, action_indices):
        probs_padded = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs_padded)
        action_indices = action_indices.to(device=logits.device, dtype=torch.long).view(-1)
        return dist.log_prob(action_indices), dist.entropy()
