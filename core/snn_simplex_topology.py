from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class SNNSimplexTopologyTensors:
    snn_laplacian_row: torch.Tensor
    snn_laplacian_col: torch.Tensor
    snn_laplacian_value: torch.Tensor
    num_snn_laplacian_entries: int
    snn_candidate: torch.Tensor
    snn_simplex: torch.Tensor
    num_snn_candidate_simplex_memberships: int


def simplex_vertices_tensor_from_simplices(
    simplices: Iterable[Iterable[int]],
) -> torch.Tensor:
    canonical_simplices = tuple(
        sorted(tuple(sorted(int(vertex) for vertex in simplex)) for simplex in simplices)
    )
    if not canonical_simplices:
        return torch.empty((0, 0), dtype=torch.long, device="cpu")

    simplex_width = len(canonical_simplices[0])
    if any(len(simplex) != simplex_width for simplex in canonical_simplices):
        raise ValueError("All top-dimensional simplices must have the same width.")
    return torch.tensor(canonical_simplices, dtype=torch.long, device="cpu")


def _normalize_sparse_laplacian(laplacian: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    laplacian = laplacian.coalesce()
    if laplacian._nnz() == 0:
        return laplacian

    indices = laplacian.indices()
    values = laplacian.values()
    row_abs_sums = torch.zeros(
        laplacian.size(0),
        device=values.device,
        dtype=values.dtype,
    )
    row_abs_sums.scatter_add_(0, indices[0], values.abs())
    scale = row_abs_sums.max().clamp_min(eps)
    return torch.sparse_coo_tensor(
        indices,
        values / scale,
        size=laplacian.shape,
        device=values.device,
        dtype=values.dtype,
    ).coalesce()


def build_top_degree_down_laplacian(
    simplex_vertices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build normalized L_d = B_d^T B_d for top-dimensional simplices."""

    if simplex_vertices.dim() != 2:
        raise ValueError(
            "`simplex_vertices` must be 2D with shape [num_simplices, simplex_width], "
            f"got {tuple(simplex_vertices.shape)}."
        )
    device = simplex_vertices.device
    num_simplices = int(simplex_vertices.size(0))
    simplex_width = int(simplex_vertices.size(1))
    if num_simplices == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            size=(0, 0),
            device=device,
            dtype=dtype,
        ).coalesce()

    if simplex_width <= 1:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            size=(num_simplices, num_simplices),
            device=device,
            dtype=dtype,
        ).coalesce()

    facets = []
    signs = []
    for omitted_vertex_position in range(simplex_width):
        keep_mask = torch.ones(simplex_width, dtype=torch.bool, device=device)
        keep_mask[omitted_vertex_position] = False
        facets.append(simplex_vertices[:, keep_mask])
        sign = 1.0 if omitted_vertex_position % 2 == 0 else -1.0
        signs.append(torch.full((num_simplices,), sign, dtype=dtype, device=device))

    facet_tensor = torch.cat(facets, dim=0)
    unique_facets, inverse = torch.unique(facet_tensor, dim=0, return_inverse=True)
    simplex_ids = torch.arange(num_simplices, dtype=torch.long, device=device).repeat(simplex_width)
    boundary_indices = torch.stack([inverse.to(dtype=torch.long), simplex_ids], dim=0)
    boundary_values = torch.cat(signs, dim=0)
    boundary = torch.sparse_coo_tensor(
        boundary_indices,
        boundary_values,
        size=(int(unique_facets.size(0)), num_simplices),
        device=device,
        dtype=dtype,
    ).coalesce()

    laplacian = torch.sparse.mm(boundary.transpose(0, 1), boundary).coalesce()
    return _normalize_sparse_laplacian(laplacian)


def _validate_simplex_vertices(simplex_vertices: torch.Tensor) -> torch.Tensor:
    if not isinstance(simplex_vertices, torch.Tensor):
        raise TypeError("`simplex_vertices` must be a torch.Tensor.")
    if simplex_vertices.dim() != 2:
        raise ValueError(
            "`simplex_vertices` must be 2D with shape [num_simplices, simplex_width], "
            f"got {tuple(simplex_vertices.shape)}."
        )
    simplex_vertices = simplex_vertices.detach().to(device="cpu", dtype=torch.long)
    if simplex_vertices.numel() > 0 and bool((simplex_vertices < 0).any().item()):
        raise ValueError("`simplex_vertices` must not contain padding.")
    return simplex_vertices.contiguous()


def _validate_subcomplex_vertices(subcomplex_vertices: torch.Tensor) -> torch.Tensor:
    if not isinstance(subcomplex_vertices, torch.Tensor):
        raise TypeError("`subcomplex_vertices` must be a torch.Tensor.")
    if subcomplex_vertices.dim() == 1:
        subcomplex_vertices = subcomplex_vertices.view(1, -1)
    elif subcomplex_vertices.dim() != 2:
        raise ValueError(
            "`subcomplex_vertices` must be 1D or 2D, got shape "
            f"{tuple(subcomplex_vertices.shape)}."
        )
    subcomplex_vertices = subcomplex_vertices.detach().to(device="cpu", dtype=torch.long)
    valid_counts = (subcomplex_vertices >= 0).sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Encountered an empty subcomplex candidate.")
    return subcomplex_vertices.contiguous()


def _build_candidate_simplex_memberships(
    *,
    simplex_vertices: torch.Tensor,
    subcomplex_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if subcomplex_vertices.size(0) == 0:
        return (
            torch.empty((0,), dtype=torch.long, device="cpu"),
            torch.empty((0,), dtype=torch.long, device="cpu"),
        )
    if simplex_vertices.size(0) == 0:
        raise ValueError("Cannot score candidates without current top simplices.")

    candidate_ids = []
    simplex_ids = []
    for candidate_id, candidate_vertices in enumerate(subcomplex_vertices.tolist()):
        candidate_vertex_set = {int(vertex) for vertex in candidate_vertices if int(vertex) >= 0}
        for simplex_id, simplex in enumerate(simplex_vertices.tolist()):
            if all(int(vertex) in candidate_vertex_set for vertex in simplex):
                candidate_ids.append(candidate_id)
                simplex_ids.append(simplex_id)
        if not candidate_ids or candidate_ids[-1] != candidate_id:
            raise ValueError(
                "At least one candidate subcomplex contains no current top-dimensional simplex."
            )

    return (
        torch.tensor(candidate_ids, dtype=torch.long, device="cpu"),
        torch.tensor(simplex_ids, dtype=torch.long, device="cpu"),
    )


def build_snn_simplex_topology_tensors(
    *,
    simplex_vertices: torch.Tensor,
    subcomplex_vertices: torch.Tensor,
) -> SNNSimplexTopologyTensors:
    simplex_vertices = _validate_simplex_vertices(simplex_vertices)
    subcomplex_vertices = _validate_subcomplex_vertices(subcomplex_vertices)

    laplacian = build_top_degree_down_laplacian(
        simplex_vertices,
        dtype=torch.float32,
    ).coalesce()
    laplacian_indices = laplacian.indices()
    snn_candidate, snn_simplex = _build_candidate_simplex_memberships(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )
    return SNNSimplexTopologyTensors(
        snn_laplacian_row=laplacian_indices[0].to(device="cpu", dtype=torch.long).contiguous(),
        snn_laplacian_col=laplacian_indices[1].to(device="cpu", dtype=torch.long).contiguous(),
        snn_laplacian_value=laplacian.values().to(device="cpu", dtype=torch.float32).contiguous(),
        num_snn_laplacian_entries=int(laplacian._nnz()),
        snn_candidate=snn_candidate.contiguous(),
        snn_simplex=snn_simplex.contiguous(),
        num_snn_candidate_simplex_memberships=int(snn_candidate.numel()),
    )


def add_snn_simplex_topology_to_data(
    data,
    topology: SNNSimplexTopologyTensors,
) -> None:
    data.snn_laplacian_row = topology.snn_laplacian_row
    data.snn_laplacian_col = topology.snn_laplacian_col
    data.snn_laplacian_value = topology.snn_laplacian_value
    data.num_snn_laplacian_entries = topology.num_snn_laplacian_entries
    data.snn_candidate = topology.snn_candidate
    data.snn_simplex = topology.snn_simplex
    data.num_snn_candidate_simplex_memberships = topology.num_snn_candidate_simplex_memberships
