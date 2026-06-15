import pytest

pytest.importorskip("cytools.polytope")

from core.cy_data_utils import (
    create_cy_batch_from_states_with_subcomplex,
    create_data_from_cy_state_with_subcomplex,
    load_cy_3d_dataset,
    load_cy_3d_states,
    load_k3_records,
)


def _points_to_tuple_set(points):
    return {tuple(int(coord) for coord in point) for point in points}


def test_load_k3_records_reads_first_record():
    records = load_k3_records("cy_data/k3.txt", max_polytopes=1)
    assert len(records) == 1

    record = records[0]
    assert record.ambient_dim == 3
    assert record.num_matrix_columns == 4
    assert len(record.m_vertices) == 4
    assert all(len(vertex) == 3 for vertex in record.m_vertices)


def test_load_cy_3d_dataset_uses_n_lattice_triangulation():
    entries = load_cy_3d_dataset(
        "cy_data/k3.txt",
        max_polytopes=1,
        include_points_interior_to_facets=False,
        precompute_actions=False,
    )
    assert len(entries) == 1

    entry = entries[0]
    assert entry["lattice_space"] == "N"

    tri_points = _points_to_tuple_set(entry["triangulation"].points())
    n_points = _points_to_tuple_set(entry["n_polytope"].points())
    assert tri_points.issubset(n_points)


def test_cy_state_subcomplex_encoding_from_k3():
    states = load_cy_3d_states(
        "cy_data/k3.txt",
        max_polytopes=1,
        include_points_interior_to_facets=False,
        precompute_actions=True,
    )
    assert len(states) == 1

    state = states[0]
    data = create_data_from_cy_state_with_subcomplex(state, ensure_actions_ready=False)
    assert data.x.size(1) == 3
    assert data.subcomplex_vertices.dim() == 2
    assert int(data.num_available_subcomplexes) == data.subcomplex_vertices.size(0)
    assert not hasattr(data, "simplex_vertices")
    assert not hasattr(data, "snn_laplacian_row")

    topology_data = create_data_from_cy_state_with_subcomplex(
        state,
        ensure_actions_ready=False,
        include_simplex_topology=True,
    )
    assert topology_data.simplex_vertices.dim() == 2
    assert int(topology_data.num_top_simplices) == topology_data.simplex_vertices.size(0)
    assert topology_data.snn_laplacian_row.dim() == 1
    assert topology_data.snn_candidate.dim() == 1

    batch = create_cy_batch_from_states_with_subcomplex(states, ensure_actions_ready=False)
    assert int(batch.num_graphs) == 1

    topology_batch = create_cy_batch_from_states_with_subcomplex(
        states,
        ensure_actions_ready=False,
        include_simplex_topology=True,
    )
    assert hasattr(topology_batch, "simplex_vertices")
    assert hasattr(topology_batch, "snn_laplacian_row")
