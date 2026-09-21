from __future__ import annotations

import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mdp.cy_geometry_worker import execute_geometry_request
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices


def _square_row():
    return {
        "polytope_index": 7101,
        "vertices": [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
        "frst_list": [{
            "simplices": [[0, 1, 2], [0, 2, 4], [0, 3, 4], [0, 1, 3]],
            "triangulation_list": [{"simplices": [[1, 2, 4], [1, 3, 4]]}],
        }],
    }


def test_geometry_protocol_imports_no_geometry_or_training_libraries():
    code = (
        "import sys; import mdp.cy_state_record; import mdp.cy_geometry_worker; "
        "assert not any(k.split('.')[0] in {'cytools','sage','torch','numpy'} for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_records_share_coordinates_compute_edges_lazily_and_exclude_providers_from_pickle():
    configuration = CYPointConfiguration(7, ((0, 0), (1, 0), (0, 1)),
                                        ((0, 0), (1, 0), (0, 1)), (0, 1, 2), True)
    state = CyStateRecord(configuration, frozenset({(2, 0, 1)}))
    assert state.key == "7:((0, 1, 2),)"
    assert state.vertices is configuration.input_vertices
    assert state._edges is None
    state.bind_objective_provider(lambda record, name: 42.0)
    assert state.objective_value("max_cy_volume") == 42.0
    restored = pickle.loads(pickle.dumps(state))
    assert restored.key == state.key
    assert restored._objective_provider is None
    assert state.edges == frozenset({(0, 1), (0, 2), (1, 2)})


def test_real_regular_expansion_matches_legacy_and_returns_deltas():
    pytest.importorskip("cytools")
    row = _square_row()
    result = execute_geometry_request({"operation": "build_collection", "row": row})
    configuration = result["configuration"]
    initial = {state.key: state for state in result["base_states"]}[result["initial_keys"][0]]
    expansion = execute_geometry_request({"operation": "expand", "configuration": configuration,
                                          "state": initial.to_payload(), "objective_mode": True})
    from mdp.cy_geometry_worker import _get_polytope, _triangulate
    from mdp.cy_rollout import _expand_cy_state_worker
    from mdp.cy_triangulation_state import CYTriangulationState

    triangulation = _triangulate(_get_polytope(configuration), configuration, initial.simplices)
    rich = CYTriangulationState(vertices=row["vertices"], point_config_index=configuration.index,
                               simplices=initial.simplices, cy_triangulation=triangulation)
    legacy = _expand_cy_state_worker((rich, True))
    assert expansion.candidate_actions == legacy.candidate_actions
    assert expansion.ambiguous_actions == legacy.ambiguous_actions
    assert len(expansion.transitions) > 0
    for (action, actual), (expected_action, expected) in zip(expansion.transitions, legacy.transitions):
        assert action == expected_action
        assert actual.next_key == expected.next_key
        assert actual.next_is_target == expected.next_is_target
        assert actual.next_simplices == ()
        assert actual.simplices_from(initial.simplices) == expected.next_simplices
    rich.release_geometry_cache()
    assert rich.neighbours is None
    assert rich._subcomplex_neighbour_cache == {}


def test_worker_cache_eviction_preserves_exact_geometry_and_objectives():
    pytest.importorskip("cytools")
    from mdp.cy_geometry_worker import _POLYTOPES

    result = execute_geometry_request({"operation": "build_collection", "row": _square_row()})
    state = result["base_states"][-1]
    request = {"operation": "expand", "state": state.to_payload(),
               "configuration": result["configuration"], "objective_mode": True}
    first = execute_geometry_request(request)
    execute_geometry_request({"operation": "trim"})
    assert len(_POLYTOPES) == 0
    assert execute_geometry_request(request) == first
    objective = execute_geometry_request({**request, "operation": "objective", "reward_name": "min_tri"})
    assert objective == float(len(state.simplices))


def test_reconstruction_preserves_backend_action_order_for_original_constructor_inputs():
    pytest.importorskip("cytools")
    dataset = (Path(__file__).resolve().parents[1]
               / "data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl")
    with dataset.open(encoding="utf-8") as handle:
        row = next(row for row in map(json.loads, handle) if row["polytope_index"] == 11)
    result = execute_geometry_request({"operation": "build_collection", "row": row})
    by_key = {state.key: state for state in result["base_states"]}
    requests = [{"operation": "expand", "state": by_key[key].to_payload(),
                 "configuration": result["configuration"], "objective_mode": True}
                for key in result["initial_keys"]]
    original = [execute_geometry_request(request) for request in requests]
    execute_geometry_request({"operation": "trim"})
    reconstructed = [execute_geometry_request(request) for request in requests]
    assert reconstructed == original


def test_canonical_expansion_order_matches_across_worker_counts_and_cache_eviction():
    pytest.importorskip("cytools")
    from mdp.cy_rollout import build_cy_rollout_collection, create_transition_pool

    dataset = (Path(__file__).resolve().parents[1]
               / "data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl")
    with dataset.open(encoding="utf-8") as handle:
        row = next(row for row in map(json.loads, handle) if row["polytope_index"] == 11)
    expected = None
    for workers in (1, 2):
        with create_transition_pool(num_workers=workers) as pool:
            collection = build_cy_rollout_collection(
                [row], include_points_interior_to_facets=True, transition_pool=pool,
            )
            requests = [{"operation": "expand", "state": state.to_payload(),
                         "configuration": state.configuration, "objective_mode": True,
                         "action_order": "canonical"} for state in collection.initial_states]
            assert len(requests) == 5
            actual = list(pool.imap(execute_geometry_request, requests))
            if expected is None:
                expected = actual
            assert actual == expected
            list(pool.imap(execute_geometry_request, [{"operation": "trim"}] * workers))
            assert list(pool.imap(execute_geometry_request, requests)) == expected


def test_zero_worker_cache_budget_bypasses_geometry_admission():
    pytest.importorskip("cytools")
    from mdp.cy_geometry_worker import configure_geometry_worker
    from mdp.cy_rollout import create_transition_pool

    with create_transition_pool(num_workers=1, initializer=configure_geometry_worker, initargs=(0,)) as pool:
        result = next(pool.imap(execute_geometry_request, [{"operation": "build_collection", "row": _square_row()}]))
        state = result["base_states"][-1]
        request = {"operation": "expand", "configuration": result["configuration"],
                   "state": state.to_payload(), "objective_mode": True, "action_order": "canonical"}
        expansions = list(pool.imap(execute_geometry_request, [request] * 12))
        assert all(expansion == expansions[0] for expansion in expansions)
        stats = next(pool.imap(execute_geometry_request, [{"operation": "cache_stats"}]))
        assert all(cache["bytes"] == 0 and cache["entries"] == 0 for cache in stats.values())
        assert stats["polytopes"]["bypasses"] >= 12


def test_objective_cache_distinguishes_reused_indices_after_configuration_eviction(monkeypatch):
    from mdp import cy_geometry_worker as worker
    from mdp.cy_state_record import BoundedCache
    import reward_functions

    # Keep only one immutable registration while scalar results remain cached.
    monkeypatch.setattr(worker, "_CONFIGURATIONS", BoundedCache(max_entries=1))
    monkeypatch.setattr(worker, "_POLYTOPES", BoundedCache(max_entries=2))
    monkeypatch.setattr(worker, "_OBJECTIVES", BoundedCache(max_entries=10))

    class FakePolytope:
        def __init__(self, points):
            self._points = points
            self.labels = tuple(range(len(points)))

        def points(self):
            return self._points

        def triangulate(self, **kwargs):
            return SimpleNamespace(metric=float(self._points[-1][-1]))

    monkeypatch.setattr(worker, "_polytope_class", lambda: FakePolytope)
    monkeypatch.setattr(reward_functions, "get_objective", lambda *args, **kwargs: lambda state: state.cy_triangulation.metric)
    first_points = ((0, 0), (1, 0), (0, 1))
    second_points = ((0, 0), (1, 0), (0, 2))
    first = CYPointConfiguration(7, first_points, first_points, (0, 1, 2), True)
    second = CYPointConfiguration(7, second_points, second_points, (0, 1, 2), True)
    unrelated = CYPointConfiguration(8, first_points, first_points, (0, 1, 2), True)
    payload = CyStateRecord(first, frozenset({(0, 1, 2)})).to_payload()
    request = {"operation": "objective", "state": payload, "reward_name": "max_cy_volume"}
    assert execute_geometry_request({**request, "configuration": first}) == 1.0
    execute_geometry_request({"operation": "register", "configurations": [unrelated]})
    assert 7 not in worker._CONFIGURATIONS
    assert len(worker._OBJECTIVES) == 1
    assert execute_geometry_request({**request, "configuration": second}) == 2.0


def test_regular_action_order_ambiguity_and_duplicate_filtering():
    from mdp.cy_geometry_worker import _expand

    source = frozenset({(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)})
    vertices = tuple((i, 0) for i in range(8))
    configuration = CYPointConfiguration(7311, vertices, vertices, tuple(range(8)), True)
    record = CyStateRecord(configuration, source)
    flips = [
        ({(0, 1, 2), (0, 2, 3)}, {(0, 1, 3)}),
        ({(0, 4, 5)}, {(0, 4, 6), (0, 5, 6)}),
        ({(0, 3, 4), (0, 4, 5)}, {(0, 3, 5)}),
        ({(0, 1, 2)}, {(0, 1, 7), (0, 2, 7)}),
        ({(0, 4, 5)}, {(0, 4, 6), (4, 5, 6)}),
        ({(0, 1, 2), (0, 2, 3)}, {(0, 1, 3)}),
    ]

    class FakeTriangulation:
        def __init__(self, simplices):
            self._simplices = simplices

        def simplices(self):
            return self._simplices

        def is_fine(self):
            return False

        def neighbor_triangulations(self, *, only_regular=True):
            return [FakeTriangulation((source - removed) | added) for removed, added in flips]

    polytope = SimpleNamespace(triangulate=lambda **kwargs: FakeTriangulation(source))
    expansion = _expand({"objective_mode": True}, configuration, record.to_payload(), polytope)
    assert expansion.candidate_actions == ((0, 1, 2, 7), (0, 1, 2, 3), (0, 3, 4, 5))
    assert expansion.ambiguous_actions == frozenset({(0, 4, 5, 6)})
    assert len(expansion.transitions) == 3


def test_worker_geometry_objectives_match_existing_reward_classes():
    pytest.importorskip("cytools")
    from mdp.cy_geometry_worker import _get_polytope, _triangulate
    from reward_functions import get_objective

    dataset = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    with dataset.open(encoding="utf-8") as handle:
        row = json.loads(next(handle))
    row = {**row, "frst_list": row["frst_list"][:1]}
    result = execute_geometry_request({"operation": "build_collection", "row": row,
                                      "neighbor_mode": "two_neighbors",
                                      "include_points_interior_to_facets": False})
    record = result["base_states"][0]
    configuration = result["configuration"]
    triangulation = _triangulate(_get_polytope(configuration), configuration, record.simplices)
    state = SimpleNamespace(key=record.key, simplices=record.simplices, cy_triangulation=triangulation)
    for name in ("max_cy_volume", "max_kcup", "max_toric_cy_volume"):
        expected = get_objective(name)(state)
        actual = execute_geometry_request({"operation": "objective", "state": record.to_payload(),
                                           "configuration": configuration, "reward_name": name})
        assert actual == pytest.approx(expected, rel=1e-10)


def test_two_neighbors_tracked_and_fallback_paths_match(monkeypatch):
    pytest.importorskip("cytools")
    from mdp import cy_geometry_worker as worker
    from cytools.triangulation import Triangulation

    dataset = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    with dataset.open(encoding="utf-8") as handle:
        row = json.loads(next(handle))
    row = {**row, "frst_list": row["frst_list"][:1]}
    result = execute_geometry_request({"operation": "build_collection", "row": row,
                                      "neighbor_mode": "two_neighbors",
                                      "include_points_interior_to_facets": False})
    source = result["base_states"][0]
    request = {"operation": "expand", "state": source.to_payload(),
               "configuration": result["configuration"], "objective_mode": True}
    tracked = execute_geometry_request(request)
    original = Triangulation.neighbor_triangulations

    def without_track(self, *, two_neighbors=False, **kwargs):
        return original(self, two_neighbors=two_neighbors, **kwargs)

    monkeypatch.setattr(Triangulation, "neighbor_triangulations", without_track)
    counts = {"source": 0}
    original_restrictions = worker._face_restrictions

    def counted_restrictions(triangulation):
        if canonical_simplices(triangulation.simplices()) == canonical_simplices(source.simplices):
            counts["source"] += 1
        return original_restrictions(triangulation)

    monkeypatch.setattr(worker, "_face_restrictions", counted_restrictions)
    fallback = execute_geometry_request(request)
    assert tracked == fallback
    assert tracked.candidate_actions
    assert all(len(action) == 4 for action in tracked.candidate_actions)
    assert counts["source"] == 1
    assert all(transition.next_is_target and transition.next_is_frst for _, transition in tracked.transitions)
