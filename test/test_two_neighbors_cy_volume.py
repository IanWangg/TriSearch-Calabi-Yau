from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.evaluate_rl_cy import parse_args as parse_eval_args
from core.cy_data_utils import create_data_from_cy_state_with_subcomplex
from core.train_cy import (
    parse_args as parse_train_args,
    validate_cy_volume_reward_transform_args,
    validate_neighbor_mode_args,
)
from mdp import cy_rollout
from mdp.cy_triangulation_state import CYTriangulationState
from reward_functions import get_objective, get_reward, infer_goal


class _FakeTriangulation:
    def __init__(self, simplices):
        self._simplices = tuple(
            tuple(int(vertex) for vertex in simplex) for simplex in simplices
        )

    def simplices(self):
        return [list(simplex) for simplex in self._simplices]

    def is_fine(self):
        return True

    def is_star(self):
        return True

    def is_regular(self, **_kwargs):
        return True


class _FakePolytope:
    def __init__(self, vertices):
        self.vertices = vertices

    def triangulate(self, *, simplices, **_kwargs):
        return _FakeTriangulation(simplices)


def _frst_row():
    return {
        "polytope_index": 7,
        "vertices": [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ],
        "frst_list": [
            {
                "frst_index": 0,
                "simplices": [[0, 1, 2, 3, 4]],
                "triangulation_list": [
                    {"simplices": [[0, 1, 2, 3, 5]]}
                ],
            }
        ],
        "non_fine_triangulation_list": [
            {"simplices": [[0, 1, 2, 4, 5]]}
        ],
    }


def test_neighbor_mode_cli_defaults_and_validation():
    assert parse_train_args([]).neighbor_mode == "regular"
    assert parse_eval_args([]).neighbor_mode == "regular"
    args = parse_train_args(
        [
            "--neighbor_mode",
            "two_neighbors",
            "--no-include_points_interior_to_facets",
        ]
    )
    assert args.neighbor_mode == "two_neighbors"
    validate_neighbor_mode_args(args)

    invalid_args = parse_train_args(["--neighbor_mode", "two_neighbors"])
    with pytest.raises(
        ValueError,
        match="--no-include_points_interior_to_facets",
    ):
        validate_neighbor_mode_args(invalid_args)


def test_two_neighbors_collection_uses_only_validated_frst_entries(monkeypatch):
    monkeypatch.setattr(cy_rollout, "Polytope", _FakePolytope)

    collection = cy_rollout.build_cy_rollout_collection(
        [_frst_row()],
        include_points_interior_to_facets=False,
        neighbor_mode="two_neighbors",
    )

    assert len(collection.base_states) == 1
    assert len(collection.initial_states) == 1
    assert collection.initial_states[0].is_frst is True
    assert collection.initial_states[0].key.startswith("two_neighbors|")


def test_regular_collection_default_behavior_is_unchanged(monkeypatch):
    monkeypatch.setattr(cy_rollout, "Polytope", _FakePolytope)

    collection = cy_rollout.build_cy_rollout_collection(
        [_frst_row()],
        include_points_interior_to_facets=True,
    )

    assert len(collection.base_states) == 3
    assert len(collection.initial_states) == 2
    assert all(not state.key.startswith("two_neighbors|") for state in collection.base_states.values())


def test_neighbor_mode_separates_state_keys_and_action_caches():
    simplices = {(0, 1, 2), (0, 2, 3)}
    vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
    regular_state = CYTriangulationState(
        vertices=vertices,
        point_config_index=9,
        simplices=simplices,
        is_frst=False,
    )
    cached_action = (0, 1, 2, 3)
    cache_value = (
        (cached_action,),
        {cached_action: (frozenset({(0, 1, 2)}), frozenset({(0, 2, 3)}))},
        frozenset(),
    )
    CYTriangulationState._SHARED_SUBCOMPLEX_CACHE[regular_state.key] = cache_value
    try:
        regular_copy = CYTriangulationState(
            vertices=vertices,
            point_config_index=9,
            simplices=simplices,
            is_frst=False,
        )
        two_neighbor_state = CYTriangulationState(
            vertices=vertices,
            point_config_index=9,
            simplices=simplices,
            is_frst=False,
            neighbor_mode="two_neighbors",
        )

        assert regular_copy.subcomplex_actions_ready is True
        assert regular_copy.available_subcomplex_actions == (cached_action,)
        assert regular_copy.key != two_neighbor_state.key
        assert two_neighbor_state.subcomplex_actions_ready is False
        assert two_neighbor_state.available_subcomplex_actions == tuple()
    finally:
        CYTriangulationState._SHARED_SUBCOMPLEX_CACHE.pop(regular_state.key, None)


class _FakeCY:
    def __init__(self, volume):
        self.volume = float(volume)
        self.volume_calls = 0

    def dimension(self):
        return 3

    def mori_cone_cap(self, *, in_basis):
        assert in_basis is True
        return self

    def dual(self):
        return self

    def tip_of_stretched_cone(self, *, c):
        assert c == 1
        return [1.0]

    def compute_cy_volume(self, tip):
        assert tip == [1.0]
        self.volume_calls += 1
        return self.volume


class _FakeCYTriangulation:
    def __init__(self, cy):
        self.cy = cy

    def get_cy(self):
        return self.cy


def test_max_cy_volume_registration_direction_caching_and_metric_reuse():
    source_cy = _FakeCY(4.0)
    destination_cy = _FakeCY(9.5)
    source = SimpleNamespace(
        key="source",
        cy_triangulation=_FakeCYTriangulation(source_cy),
    )
    destination = SimpleNamespace(
        key="destination",
        cy_triangulation=_FakeCYTriangulation(destination_cy),
    )
    reward = get_reward("max_cy_volume")
    objective = get_objective("max_cy_volume", reward=reward)

    assert infer_goal("max_cy_volume") == "max"
    assert objective(source) == 4.0
    assert reward(source, destination) == 5.5
    assert objective(destination) == 9.5
    assert source_cy.volume_calls == 1
    assert destination_cy.volume_calls == 1


def test_max_cy_volume_log_reward_is_exact_and_keeps_raw_metric():
    source_cy = _FakeCY(4.0)
    destination_cy = _FakeCY(9.5)
    source = SimpleNamespace(
        key="source",
        cy_triangulation=_FakeCYTriangulation(source_cy),
    )
    destination = SimpleNamespace(
        key="destination",
        cy_triangulation=_FakeCYTriangulation(destination_cy),
    )
    reward = get_reward(
        "max_cy_volume",
        cy_volume_reward_transform="log",
    )
    objective = get_objective("max_cy_volume", reward=reward)

    assert reward(source, destination) == pytest.approx(math.log(9.5 / 4.0))
    assert objective(source) == 4.0
    assert objective(destination) == 9.5
    assert source_cy.volume_calls == 1
    assert destination_cy.volume_calls == 1


@pytest.mark.parametrize("current_volume,next_volume", [(0.0, 2.0), (2.0, 0.0), (-1.0, 2.0)])
def test_max_cy_volume_log_reward_rejects_nonpositive_volumes(
    current_volume,
    next_volume,
):
    source_cy = _FakeCY(current_volume)
    destination_cy = _FakeCY(next_volume)
    source = SimpleNamespace(
        key="source",
        cy_triangulation=_FakeCYTriangulation(source_cy),
    )
    destination = SimpleNamespace(
        key="destination",
        cy_triangulation=_FakeCYTriangulation(destination_cy),
    )
    reward = get_reward(
        "max_cy_volume",
        cy_volume_reward_transform="log",
    )

    with pytest.raises(ValueError, match="strictly positive volumes"):
        reward(source, destination)
    with pytest.raises(ValueError, match="strictly positive volumes"):
        reward(source, destination)
    assert source_cy.volume_calls == 1
    assert destination_cy.volume_calls == 1


def test_cy_volume_reward_transform_cli_and_validation():
    assert parse_train_args([]).cy_volume_reward_transform == "raw"
    assert parse_eval_args([]).cy_volume_reward_transform == "raw"

    train_args = parse_train_args(
        [
            "--reward",
            "max_cy_volume",
            "--cy_volume_reward_transform",
            "log",
        ]
    )
    eval_args = parse_eval_args(
        [
            "--reward",
            "max_cy_volume",
            "--cy_volume_reward_transform",
            "log",
        ]
    )
    validate_cy_volume_reward_transform_args(train_args)
    validate_cy_volume_reward_transform_args(eval_args)

    invalid_args = parse_train_args(
        ["--reward", "min_tri", "--cy_volume_reward_transform", "log"]
    )
    with pytest.raises(ValueError, match="requires --reward max_cy_volume"):
        validate_cy_volume_reward_transform_args(invalid_args)
    with pytest.raises(ValueError, match="only valid with the max_cy_volume"):
        get_reward("min_tri", cy_volume_reward_transform="log")


def test_cytools_two_neighbors_actions_and_kcup_representative_invariance():
    cytools = pytest.importorskip("cytools")
    from core.cytools_config import configure_cytools

    configure_cytools()
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data/cy/two_neighbors_h11_12.samples.jsonl"
    )
    row = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    polytope = cytools.Polytope(row["vertices"])
    source_tri = polytope.triangulate(
        simplices=row["frst_list"][0]["simplices"],
        include_points_interior_to_facets=False,
        check_input_simplices=False,
    )
    source_state = CYTriangulationState(
        vertices=row["vertices"],
        point_config_index=row["polytope_index"],
        simplices=source_tri.simplices(),
        cy_triangulation=source_tri,
        neighbor_mode="two_neighbors",
    )

    source_state.find_available_actions()
    topology_data = create_data_from_cy_state_with_subcomplex(
        source_state,
        ensure_actions_ready=False,
        include_simplex_topology=True,
    )
    membership_counts = topology_data.snn_candidate.bincount(
        minlength=int(topology_data.num_available_subcomplexes)
    )
    assert bool((membership_counts > 0).all().item())
    actions = source_state.get_available_subcomplex_actions()
    assert actions
    assert all(len(action) == 4 for action in actions)

    action = actions[0]
    destination_simplices, _destination_edges, destination_key = (
        source_state.get_transition_output_from_subcomplex_action(action)
    )
    destination_tri = source_state.get_next_cy_triangulation_from_subcomplex_action(
        action
    )
    full_changed_vertices = set().union(
        *(
            set(simplex)
            for simplex in (
                (source_state.simplices - destination_simplices)
                | (destination_simplices - source_state.simplices)
            )
        )
    )
    assert len(full_changed_vertices) > 4
    assert destination_simplices == frozenset(
        tuple(sorted(int(vertex) for vertex in simplex))
        for simplex in destination_tri.simplices()
    )
    assert destination_key.startswith("two_neighbors|")
    assert destination_tri.is_fine()
    assert destination_tri.is_star()
    assert destination_tri.is_regular()

    def restriction_signature(triangulation):
        return tuple(
            (
                tuple(int(label) for label in face.labels),
                tuple(
                    sorted(
                        tuple(sorted(int(vertex) for vertex in simplex))
                        for simplex in face.simplices()
                    )
                ),
            )
            for face in triangulation.restrict(as_poly=True)
        )

    source_restriction = restriction_signature(source_tri)
    equivalent_tri = next(
        neighbour
        for neighbour in source_tri.neighbor_triangulations(
            only_fine=True,
            only_regular=True,
            only_star=True,
        )
        if restriction_signature(neighbour) == source_restriction
    )
    equivalent_state = CYTriangulationState(
        vertices=row["vertices"],
        point_config_index=row["polytope_index"],
        simplices=equivalent_tri.simplices(),
        cy_triangulation=equivalent_tri,
        neighbor_mode="two_neighbors",
    )
    volume_metric = get_objective("max_cy_volume")
    assert volume_metric(source_state) == pytest.approx(
        volume_metric(equivalent_state),
        rel=1e-10,
        abs=1e-10,
    )
