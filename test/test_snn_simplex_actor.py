from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from core.snn_simplex_topology import (
    add_snn_simplex_topology_to_data,
    build_snn_simplex_topology_tensors,
    build_top_degree_down_laplacian,
)
from models.snn_simplex_actor import SNNSimplexActor


def _ring_edges(num_nodes: int) -> torch.Tensor:
    src, dst = [], []
    for vertex in range(num_nodes):
        next_vertex = (vertex + 1) % num_nodes
        src.extend([vertex, next_vertex])
        dst.extend([next_vertex, vertex])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_graph(
    *,
    num_nodes: int,
    subcomplex_vertices: list[list[int]],
    simplex_vertices: list[list[int]],
    include_cached_topology: bool,
) -> Data:
    subcomplex_tensor = torch.tensor(subcomplex_vertices, dtype=torch.long)
    simplex_tensor = torch.tensor(simplex_vertices, dtype=torch.long)
    data = Data(
        x=torch.zeros((num_nodes, 3), dtype=torch.float32),
        edge_index=_ring_edges(num_nodes),
        subcomplex_vertices=subcomplex_tensor,
        num_available_subcomplexes=subcomplex_tensor.size(0),
        simplex_vertices=simplex_tensor,
        num_top_simplices=simplex_tensor.size(0),
    )
    if include_cached_topology:
        add_snn_simplex_topology_to_data(
            data,
            build_snn_simplex_topology_tensors(
                simplex_vertices=simplex_tensor,
                subcomplex_vertices=subcomplex_tensor,
            ),
        )
    return data


def _build_actor_batch(*, include_cached_topology: bool) -> Batch:
    graph_1 = _build_graph(
        num_nodes=4,
        subcomplex_vertices=[
            [0, 1, 2, -1],
            [0, 2, 3, -1],
            [0, 1, 2, 3],
        ],
        simplex_vertices=[
            [0, 1, 2],
            [0, 2, 3],
        ],
        include_cached_topology=include_cached_topology,
    )
    graph_2 = _build_graph(
        num_nodes=5,
        subcomplex_vertices=[
            [0, 1, 4, -1],
            [1, 2, 4, -1],
            [0, 1, 2, 4],
        ],
        simplex_vertices=[
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
        ],
        include_cached_topology=include_cached_topology,
    )
    return Batch.from_data_list([graph_1, graph_2])


def _cached_laplacian_dense(
    *,
    simplex_vertices: torch.Tensor,
    subcomplex_vertices: torch.Tensor,
) -> torch.Tensor:
    topology = build_snn_simplex_topology_tensors(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )
    return torch.sparse_coo_tensor(
        torch.stack([topology.snn_laplacian_row, topology.snn_laplacian_col], dim=0),
        topology.snn_laplacian_value,
        size=(simplex_vertices.size(0), simplex_vertices.size(0)),
    ).to_dense()


def test_cached_laplacian_matches_reference_builder():
    simplex_vertices = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 3],
        ],
        dtype=torch.long,
    )
    subcomplex_vertices = torch.tensor(
        [
            [0, 1, 2, -1],
            [0, 2, 3, -1],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )

    cached = _cached_laplacian_dense(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )
    reference = build_top_degree_down_laplacian(
        simplex_vertices,
        dtype=torch.float32,
    ).to_dense()

    assert torch.allclose(cached, reference, atol=0.0, rtol=0.0)


def test_cached_candidate_memberships_match_dense_containment():
    simplex_vertices = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [1, 2, 4],
        ],
        dtype=torch.long,
    )
    subcomplex_vertices = torch.tensor(
        [
            [0, 1, 2, -1],
            [0, 2, 3, -1],
            [0, 1, 2, 3],
            [1, 2, 4, -1],
        ],
        dtype=torch.long,
    )

    topology = build_snn_simplex_topology_tensors(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )
    contains = (
        simplex_vertices[:, None, :, None] == subcomplex_vertices[None, :, None, :]
    ).any(dim=-1).all(dim=-1).transpose(0, 1)
    expected_candidate, expected_simplex = contains.nonzero(as_tuple=True)

    assert torch.equal(topology.snn_candidate, expected_candidate)
    assert torch.equal(topology.snn_simplex, expected_simplex)


def test_lower_dimensional_circuit_memberships_use_top_simplex_cofaces():
    simplex_vertices = torch.tensor(
        [
            [0, 1, 2, 3, 5],
            [0, 1, 3, 4, 6],
            [0, 1, 2, 5, 6],
            [0, 2, 3, 4, 6],
        ],
        dtype=torch.long,
    )
    subcomplex_vertices = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    topology = build_snn_simplex_topology_tensors(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )

    assert torch.equal(topology.snn_candidate, torch.tensor([0, 0, 0]))
    assert torch.equal(topology.snn_simplex, torch.tensor([0, 1, 3]))


def test_pyg_batch_preserves_cached_local_topology_without_incrementing():
    graph_1 = _build_graph(
        num_nodes=4,
        subcomplex_vertices=[[0, 1, 2, -1], [0, 2, 3, -1]],
        simplex_vertices=[[0, 1, 2], [0, 2, 3]],
        include_cached_topology=True,
    )
    graph_2 = _build_graph(
        num_nodes=4,
        subcomplex_vertices=[[0, 1, 2, -1], [0, 2, 3, -1]],
        simplex_vertices=[[0, 1, 2], [0, 2, 3]],
        include_cached_topology=True,
    )

    batch = Batch.from_data_list([graph_1, graph_2])
    first_laplacian_entries = int(graph_1.num_snn_laplacian_entries)
    second_laplacian_entries = int(graph_2.num_snn_laplacian_entries)
    first_memberships = int(graph_1.num_snn_candidate_simplex_memberships)
    second_memberships = int(graph_2.num_snn_candidate_simplex_memberships)

    assert torch.equal(
        batch.snn_laplacian_row[:first_laplacian_entries],
        graph_1.snn_laplacian_row,
    )
    assert torch.equal(
        batch.snn_laplacian_row[first_laplacian_entries:first_laplacian_entries + second_laplacian_entries],
        graph_2.snn_laplacian_row,
    )
    assert torch.equal(
        batch.snn_candidate[:first_memberships],
        graph_1.snn_candidate,
    )
    assert torch.equal(
        batch.snn_candidate[first_memberships:first_memberships + second_memberships],
        graph_2.snn_candidate,
    )
    assert batch.snn_laplacian_row.max().item() < graph_1.num_top_simplices
    assert batch.snn_candidate.max().item() < graph_1.num_available_subcomplexes


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_cached_batched_actor_matches_slow_reference(device_name: str):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = torch.device(device_name)
    cached_batch = _build_actor_batch(include_cached_topology=True).to(device)
    slow_batch = _build_actor_batch(include_cached_topology=False).to(device)

    torch.manual_seed(123)
    actor = SNNSimplexActor(
        channels=4,
        hidden_channels=6,
        num_layers=2,
    ).to(device)
    actor.eval()
    node_embeddings = torch.randn(
        cached_batch.num_nodes,
        4,
        device=device,
        dtype=torch.float32,
    )
    num_available = cached_batch.num_available_subcomplexes.to(device=device, dtype=torch.long).view(-1)

    with torch.no_grad():
        cached_features, cached_graph = actor(
            node_embeddings=node_embeddings,
            subcomplex_vertices=cached_batch.subcomplex_vertices,
            num_available_subcomplexes=num_available,
            node_ptr=cached_batch.ptr,
            batch=cached_batch,
        )
        slow_features, slow_graph = actor._forward_slow(
            node_embeddings=node_embeddings,
            subcomplex_vertices=slow_batch.subcomplex_vertices,
            num_available_subcomplexes=num_available,
            node_ptr=slow_batch.ptr,
            batch=slow_batch,
        )

    assert torch.equal(cached_graph, slow_graph)
    assert torch.allclose(cached_features, slow_features, atol=1e-5, rtol=1e-5)
