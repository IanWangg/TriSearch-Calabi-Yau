"""Two-face restrictions, backend action semantics, and packed PyG indices."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from core.cy_data_utils import configure_cy_data_tensor_caches, get_cy_data_tensor_cache_stats, prune_cy_data_tensor_caches
from core.cy_two_face_data import create_two_face_data_from_state, build_two_face_data, TwoFaceData
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices, two_face_restrictions


def square_state(*, dimension=3, one_face=False):
    points = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 1), (0, 0, 1))
    if dimension == 4:
        points = tuple(point + (0,) for point in points)
    faces = ((10, 20, 30, 40), (10, 20, 50, 60))[:1 if one_face else 2]
    triangles = ((10, 20, 30), (10, 30, 40), (10, 20, 50), (10, 50, 60))[:2 if one_face else 4]
    configuration = CYPointConfiguration(7, points, points, (10, 20, 30, 40, 50, 60), False, faces)
    return CyStateRecord(configuration, frozenset(triangles), "two_neighbors", True, True)


def square_actions(state):
    return list(state.configuration.two_face_labels)


@pytest.fixture(autouse=True)
def reset_tensor_caches():
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()
    yield
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()


def assert_same_data(first, second):
    assert set(first.keys()) == set(second.keys())
    for key in first.keys():
        if isinstance(first[key], torch.Tensor):
            assert torch.equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


def test_noncontinuous_labels_completion_invariance_and_canonical_restrictions():
    first = square_state()
    second = replace(first, simplices=first.simplices | {(90, 91, 92, 93, 94)})
    assert first.key != second.key and first.two_face_key == second.two_face_key
    data = create_two_face_data_from_state(first, square_actions(first))
    assert_same_data(data, create_two_face_data_from_state(second, square_actions(first)))
    assert data.x.shape == (8, 3) and data.num_faces == 2
    assert data.subcomplex_vertices.tolist() == [list(action) for action in square_actions(first)]
    assert data.removed_triangle_ids.tolist() == [[0, 1], [2, 3]]
    assert data.added_triangle_vertices.tolist() == [[[0, 1, 3], [1, 2, 3]], [[4, 5, 7], [5, 6, 7]]]
    assert torch.equal(data.node_face[data.edge_index[0]], data.node_face[data.edge_index[1]])
    assert torch.equal(data.triangle_face[data.triangle_edge_index[0]], data.triangle_face[data.triangle_edge_index[1]])
    assert not {"simplex_vertices", "num_top_simplices", "key"}.intersection(data.keys())
    assert first._edges is None  # Observation never reads full-FRST edges.
    shuffled = replace(first.configuration, labels=tuple(reversed(first.configuration.labels)),
                       points=tuple(reversed(first.configuration.points)),
                       two_face_labels=tuple(tuple(reversed(face)) for face in reversed(first.configuration.two_face_labels)))
    assert_same_data(data, create_two_face_data_from_state(replace(first, configuration=shuffled), square_actions(first)))


def test_pyg_offsets_round_trip_and_action_order():
    first, second = square_state(), square_state(one_face=True)
    data = create_two_face_data_from_state(first, list(reversed(square_actions(first))))
    empty = create_two_face_data_from_state(second)
    last = create_two_face_data_from_state(second, square_actions(second))
    batch = Batch.from_data_list([data, empty, last])
    assert batch.num_faces.tolist() == [2, 1, 1]
    assert batch.subcomplex_vertices.tolist() == [list(action) for action in list(reversed(square_actions(first))) + square_actions(second)]
    assert batch.action_face.tolist() == [1, 0, 3]
    assert batch.removed_triangle_ids[-1].tolist() == [6, 7]
    assert batch.added_triangle_vertices[-1].tolist() == [[12, 13, 15], [13, 14, 15]]
    for original, recovered in zip((data, empty, last), batch.to_data_list()):
        assert isinstance(recovered, TwoFaceData)
        assert_same_data(original, recovered)


@pytest.mark.parametrize("actions", [[(10, 20, 30)], [(10, 20, 30, 30)], [(10, 20, 30, 50)]])
def test_invalid_circuit_is_an_error(actions):
    with pytest.raises(ValueError, match="four distinct|exactly one"):
        create_two_face_data_from_state(square_state(), actions)


def test_metadata_and_invalid_removed_triangles_fail_early():
    state = square_state()
    with pytest.raises(ValueError, match="ambient 2-face"):
        create_two_face_data_from_state(replace(state, configuration=replace(state.configuration, two_face_labels=())))
    with pytest.raises(ValueError, match="exactly two"):
        create_two_face_data_from_state(replace(state, simplices=state.simplices - {(10, 30, 40)}), square_actions(state))


def test_cache_budget_order_configuration_eviction_and_full_key_pruning():
    state = square_state()
    configure_cy_data_tensor_caches(max_bytes=200_000, max_entries=2)
    original = create_two_face_data_from_state(state, square_actions(state))
    equivalent = replace(state, simplices=state.simplices | {(90, 91, 92, 93)})
    alias = create_two_face_data_from_state(equivalent, square_actions(state))
    assert original.x.data_ptr() == alias.x.data_ptr()
    assert original.removed_triangle_ids.data_ptr() == alias.removed_triangle_ids.data_ptr()
    reverse = create_two_face_data_from_state(state, list(reversed(square_actions(state))))
    assert reverse.action_face.tolist() == [1, 0]
    prune_cy_data_tensor_caches(keep_keys=[state.key], max_entries=2)
    assert get_cy_data_tensor_cache_stats()["two_face_topology"]["entries"] == 1
    for index in range(8):
        configuration = replace(state.configuration, points=tuple(tuple(c + index for c in point) for point in state.configuration.points))
        create_two_face_data_from_state(replace(state, configuration=configuration), square_actions(state))
        stats = get_cy_data_tensor_cache_stats()
        assert sum(cache["bytes"] for cache in stats.values()) <= 200_000
        assert all(cache["entries"] <= 2 for cache in stats.values())
    configure_cy_data_tensor_caches(max_bytes=0)
    assert_same_data(original, create_two_face_data_from_state(state, square_actions(state)))
    assert all(cache["entries"] == 0 for cache in get_cy_data_tensor_cache_stats().values())


@pytest.fixture(scope="module")
def real_representatives():
    from core.cytools_config import configure_cytools
    configure_cytools(worker=True)
    from mdp import cy_geometry_worker as worker

    path = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["frst_list"] = row["frst_list"][:1]
    collection = worker.execute_geometry_request(dict(operation="build_collection", row=row,
        neighbor_mode="two_neighbors", include_points_interior_to_facets=False, include_two_face_metadata=True))
    first = collection["base_states"][0]
    polytope = worker._get_polytope(first.configuration)
    tri = worker._triangulate(polytope, first.configuration, canonical_simplices(first.simplices))
    equivalent = next(neighbor for neighbor in tri.neighbor_triangulations(only_fine=True, only_regular=True, only_star=True)
                      if two_face_restrictions(first.configuration, neighbor.simplices()) == two_face_restrictions(first.configuration, first.simplices))
    second = replace(first, simplices=frozenset(canonical_simplices(equivalent.simplices())))
    return row, first, second


def _expansion_signature(state):
    from mdp.cy_geometry_worker import execute_geometry_request
    expansion = execute_geometry_request(dict(operation="expand", configuration=state.configuration,
        state=state.to_payload(), objective_mode=True, action_order="canonical"))
    before = dict(two_face_restrictions(state.configuration, state.simplices))
    data = create_two_face_data_from_state(state, expansion.candidate_actions)
    signatures = set()
    for index, (action, transition) in enumerate(expansion.transitions):
        destination = transition.simplices_from(state.simplices)
        after = dict(two_face_restrictions(state.configuration, destination))
        changed = [face for face in before if before[face] != after[face]]
        assert len(changed) == 1
        face = changed[0]
        assert int(data.action_face[index]) == list(before).index(face)
        removed, added = set(before[face]) - set(after[face]), set(after[face]) - set(before[face])
        assert len(removed) == len(added) == 2
        label_map = [label for labels in before for label in labels]
        actual_removed = {tuple(sorted(label_map[node] for node in triangle))
                          for triangle in data.triangle_vertices[data.removed_triangle_ids[index]].tolist()}
        actual_added = {tuple(sorted(label_map[node] for node in triangle))
                        for triangle in data.added_triangle_vertices[index].tolist()}
        assert actual_removed == removed and actual_added == added
        signatures.add((face, tuple(sorted(removed)), tuple(sorted(added)), tuple(after.items())))
    assert not expansion.ambiguous_actions
    return expansion, signatures


def test_real_quotient_neighborhood_tracked_fallback_and_rebuild(real_representatives, monkeypatch):
    from cytools.triangulation import Triangulation
    from mdp import cy_geometry_worker as worker

    _, first, second = real_representatives
    assert first.key != second.key and first.two_face_key == second.two_face_key
    tracked = [_expansion_signature(state) for state in (first, second)]
    assert tracked[0][1] == tracked[1][1] and tracked[0][1]
    for state in (first, second):
        tri = worker._triangulate(worker._get_polytope(state.configuration), state.configuration, state.simplices)
        assert dict(two_face_restrictions(state.configuration, state.simplices)) == {
            face: canonical_simplices(triangles) for face, triangles in worker._face_restrictions(tri).items()}
    original = Triangulation.neighbor_triangulations
    def without_track(self, *, two_neighbors=False, **kwargs):
        return original(self, two_neighbors=two_neighbors, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(Triangulation, "neighbor_triangulations", without_track)
        assert [_expansion_signature(state)[1] for state in (first, second)] == [item[1] for item in tracked]
    worker.clear_geometry_caches()
    assert [_expansion_signature(state)[1] for state in (first, second)] == [item[1] for item in tracked]
    assert_same_data(create_two_face_data_from_state(first, tracked[0][0].candidate_actions),
                     create_two_face_data_from_state(second, tracked[0][0].candidate_actions))


def test_real_three_dimensional_face_restrictions():
    from core.cytools_config import configure_cytools
    configure_cytools(worker=True)
    from cytools import Polytope
    from mdp.cy_geometry_worker import _face_restrictions

    vertices = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1))
    polytope = Polytope(vertices)
    tri = polytope.triangulate(include_points_interior_to_facets=False)
    configuration = CYPointConfiguration(11, vertices, tuple(tuple(map(int, point)) for point in polytope.points()),
        tuple(map(int, polytope.labels)), False, tuple(tuple(map(int, face.labels)) for face in polytope.faces(2)))
    restrictions = two_face_restrictions(configuration, tri.simplices())
    assert dict(restrictions) == {face: canonical_simplices(triangles) for face, triangles in _face_restrictions(tri).items()}
    data = build_two_face_data(configuration, restrictions)
    assert data.x.shape[1] == 3 and data.num_faces == 4
