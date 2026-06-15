from __future__ import annotations

import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import core.evaluate_rl_cy as evaluate_rl_cy
from mdp import cy_rollout


def _state(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, visitation=0)


class _FakeTriangulation:
    def __init__(self, simplices):
        self._simplices = tuple(tuple(int(vertex) for vertex in simplex) for simplex in simplices)

    def simplices(self):
        return [list(simplex) for simplex in self._simplices]

    def is_fine(self):
        return False

    def is_star(self):
        return False

    def is_regular(self):
        return True


class _FakePolytope:
    def __init__(self, vertices):
        self.vertices = vertices

    def triangulate(self, *, simplices, **kwargs):
        return _FakeTriangulation(simplices)


def test_resolve_dataset_split_prefers_explicit_polytope_indices():
    rows = [
        {"polytope_index": 0, "vertices": [[0], [1], [2]]},
        {"polytope_index": 0, "vertices": [[0], [1], [2]]},
        {"polytope_index": 1, "vertices": [[0], [1], [2], [3], [4]]},
        {"polytope_index": 2, "vertices": [[0], [1], [2], [3]]},
    ]

    split = evaluate_rl_cy.resolve_dataset_split(
        rows,
        num_eval_polytopes=1,
        polytope_indices=[2, 0, 2],
    )

    assert split.eval_polytope_indices == [2, 0]
    assert split.train_polytope_indices == [1]
    assert [int(row["polytope_index"]) for row in split.eval_rows] == [0, 0, 2]
    assert [int(row["polytope_index"]) for row in split.train_rows] == [1]


def test_normalize_polytope_indices_preserves_order_and_deduplicates():
    assert evaluate_rl_cy.normalize_polytope_indices([5, 2, 5, 3, 2]) == [5, 2, 3]


def test_resolve_policy_in_channels_infers_dataset_dimension():
    rows = [
        {"polytope_index": 0, "vertices": [[0, 0, 0, 0], [1, 0, 0, 0]]},
        {"polytope_index": 1, "vertices": [[0, 0, 0, 0], [0, 1, 0, 0]]},
    ]

    assert evaluate_rl_cy.resolve_policy_in_channels(rows, None) == 4

    with pytest.raises(ValueError, match="--in_channels must match"):
        evaluate_rl_cy.resolve_policy_in_channels(rows, 3)


def test_build_cy_rollout_collection_supports_4d_random_flip_rows(monkeypatch):
    monkeypatch.setattr(cy_rollout, "Polytope", _FakePolytope)
    row = {
        "polytope_index": 7,
        "vertices": [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        "frst_list": [
            {
                "frst_index": 0,
                "simplices": [[0, 1, 2, 3, 4]],
                "triangulation_list": [{"simplices": [[0, 1, 2, 3, 5]]}],
            }
        ],
    }

    collection = cy_rollout.build_cy_rollout_collection(
        [row],
        include_points_interior_to_facets=True,
    )

    assert len(collection.initial_states) == 1
    assert len(collection.base_states) == 2
    assert any(state.is_frst for state in collection.base_states.values())
    assert len(collection.initial_states[0].vertices[0]) == 4
    assert len(next(iter(collection.initial_states[0].simplices))) == 5


@pytest.mark.parametrize(
    ("row", "coordinate_dim", "simplex_width", "expected_initial_states"),
    [
        (
            {
                "polytope_index": 2000,
                "vertices": [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                ],
                "non_fine_triangulation_list": [
                    {"signature": [[0, 1, 2, 3]]},
                    {"signature": [[0, 1, 2, 4]]},
                ],
            },
            3,
            4,
            2,
        ),
        (
            {
                "polytope_index": 3000,
                "vertices": [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1],
                ],
                "non_fine_triangulation_list": [
                    {"signature": [[0, 1, 2, 3, 4]]},
                    {"simplices": [[0, 1, 2, 3, 5]]},
                ],
            },
            4,
            5,
            2,
        ),
    ],
)
def test_build_cy_rollout_collection_supports_random_heights_rows(
    monkeypatch,
    row,
    coordinate_dim,
    simplex_width,
    expected_initial_states,
):
    monkeypatch.setattr(cy_rollout, "Polytope", _FakePolytope)

    collection = cy_rollout.build_cy_rollout_collection(
        [row],
        include_points_interior_to_facets=True,
    )

    assert len(collection.initial_states) == expected_initial_states
    assert len(collection.base_states) == expected_initial_states
    assert all(len(state.vertices[0]) == coordinate_dim for state in collection.initial_states)
    assert all(len(next(iter(state.simplices))) == simplex_width for state in collection.initial_states)


def test_collect_policy_rollout_over_initial_states_uses_each_initial_state_once(monkeypatch):
    initial_states = [_state("s0"), _state("s1"), _state("s2")]
    call_state_keys: list[list[str]] = []
    captured_preprocessing = []
    step_results = [
        SimpleNamespace(
            rewards=[1.0, 0.0, -1.0],
            dones=[True, False, True],
            terminal_reasons=["frt_or_frst", "continue", "single_simplex"],
            transitioned_states=[_state("s0_terminal"), _state("s1_mid"), _state("s2_terminal")],
            next_states=[_state("reset0"), _state("s1_next"), _state("reset2")],
            frt_hits=1,
            collapsed_hits=1,
            dead_end_hits=0,
            reset_count=2,
            expanded_states=5,
            discovered_states=7,
            used_multiprocessing=False,
            action_candidates=[((1,),), ((2,),), ((3,),)],
            valid_action_mask=torch.tensor([True, True, True]),
            candidate_expand_sec=0.1,
            policy_data_build_sec=0.2,
            policy_batch_transfer_sec=0.3,
            policy_value_inference_sec=0.4,
            policy_action_inference_sec=0.5,
            transition_apply_sec=0.6,
        ),
        SimpleNamespace(
            rewards=[1.0],
            dones=[True],
            terminal_reasons=["frt_or_frst"],
            transitioned_states=[_state("s1_terminal")],
            next_states=[_state("reset1")],
            frt_hits=1,
            collapsed_hits=0,
            dead_end_hits=0,
            reset_count=1,
            expanded_states=11,
            discovered_states=13,
            used_multiprocessing=True,
            action_candidates=[((4,),)],
            valid_action_mask=torch.tensor([True]),
            candidate_expand_sec=1.0,
            policy_data_build_sec=1.1,
            policy_batch_transfer_sec=1.2,
            policy_value_inference_sec=1.3,
            policy_action_inference_sec=1.4,
            transition_apply_sec=1.5,
        ),
    ]

    def fake_rollout_step_with_policy(engine, states, policy, **kwargs):
        call_state_keys.append([state.key for state in states])
        captured_preprocessing.append(kwargs.get("vertex_preprocessor"))
        return step_results.pop(0)

    monkeypatch.setattr(evaluate_rl_cy, "rollout_step_with_policy", fake_rollout_step_with_policy)

    summary = evaluate_rl_cy.collect_policy_rollout_over_initial_states(
        engine=object(),
        policy=object(),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        initial_states=initial_states,
        rollout_length=3,
        gamma=0.5,
        deterministic=True,
        use_multiprocessing=False,
        transition_pool=None,
        transition_mp_chunksize=1,
        transition_mp_min_batch=1,
        report_every=0,
        label="eval",
        vertex_preprocessor="prep",
    )

    assert call_state_keys == [["s0", "s1", "s2"], ["s1_next"]]
    assert captured_preprocessing == ["prep", "prep"]
    assert summary.success_rate == 2.0 / 3.0
    assert math.isclose(summary.discounted_reward, 1.0 / 6.0)
    assert summary.finished_fraction == 1.0
    assert summary.finished_count == 3
    assert summary.frt_hits == 2
    assert summary.collapsed_hits == 1
    assert summary.dead_end_hits == 0
    assert summary.all_step_reset_count == 0
    assert summary.all_step_frt_hits == 2
    assert summary.all_step_collapsed_hits == 1
    assert summary.all_step_dead_end_hits == 0
    assert summary.expanded_states == 16
    assert summary.discovered_states == 20
    assert summary.multiprocessing_steps == 1
    assert summary.total_candidates == 4
    assert summary.total_valid_actions == 4
    assert summary.rollout_lengths == [1, 2, 1]
    assert math.isclose(summary.rollout_length_mean, 4.0 / 3.0)
    assert summary.rollout_length_min == 1
    assert summary.rollout_length_max == 2


def test_collect_random_rollout_over_initial_states_uses_uniform_random_step():
    initial_states = [_state("r0"), _state("r1"), _state("r2")]

    class DummyEngine:
        def __init__(self):
            self.nodes_by_key = {
                "r0": SimpleNamespace(candidate_actions=((1,),)),
                "r1": SimpleNamespace(candidate_actions=((2,), (3,))),
                "r2": SimpleNamespace(candidate_actions=tuple()),
                "r1_next": SimpleNamespace(candidate_actions=((4,),)),
            }
            self.calls = 0

        def rollout_step(self, states, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    input_states=list(states),
                    transitioned_states=[_state("r0_term"), _state("r1_mid"), _state("r2_term")],
                    next_states=[_state("reset0"), _state("r1_next"), _state("reset2")],
                    rewards=[1.0, 0.0, 0.0],
                    dones=[True, False, True],
                    terminal_reasons=["frt_or_frst", "continue", "dead_end_current"],
                    reset_count=2,
                    frt_hits=1,
                    collapsed_hits=0,
                    dead_end_hits=1,
                    expanded_states=5,
                    discovered_states=7,
                    used_multiprocessing=False,
                )
            return SimpleNamespace(
                input_states=list(states),
                transitioned_states=[_state("r1_term")],
                next_states=[_state("reset1")],
                rewards=[-1.0],
                dones=[True],
                terminal_reasons=["single_simplex"],
                reset_count=1,
                frt_hits=0,
                collapsed_hits=1,
                dead_end_hits=0,
                expanded_states=11,
                discovered_states=13,
                used_multiprocessing=True,
            )

    summary = evaluate_rl_cy.collect_random_rollout_over_initial_states(
        engine=DummyEngine(),
        rng=np.random.default_rng(0),
        initial_states=initial_states,
        rollout_length=3,
        gamma=0.5,
        use_multiprocessing=False,
        transition_pool=None,
        transition_mp_chunksize=1,
        transition_mp_min_batch=1,
        report_every=0,
        label="eval",
    )

    assert summary.success_rate == 1.0 / 3.0
    assert math.isclose(summary.discounted_reward, 1.0 / 6.0)
    assert summary.finished_fraction == 1.0
    assert summary.finished_count == 3
    assert summary.frt_hits == 1
    assert summary.collapsed_hits == 1
    assert summary.dead_end_hits == 1
    assert summary.all_step_reset_count == 0
    assert summary.all_step_frt_hits == 1
    assert summary.all_step_collapsed_hits == 1
    assert summary.all_step_dead_end_hits == 1
    assert summary.expanded_states == 16
    assert summary.discovered_states == 20
    assert summary.multiprocessing_steps == 1
    assert summary.total_candidates == 4
    assert summary.total_valid_actions == 3
    assert summary.policy_action_inference_sec == 0.0
    assert summary.rollout_lengths == [1, 2, 1]
    assert math.isclose(summary.rollout_length_mean, 4.0 / 3.0)
    assert summary.rollout_length_min == 1
    assert summary.rollout_length_max == 2


def test_build_summary_payload_includes_rollout_length_record():
    summary = evaluate_rl_cy.attach_rollout_length_record(
        evaluate_rl_cy.PolicyRolloutSummary(
            final_states=[],
            rollout_buffer=None,
            success_rate=0.5,
            discounted_reward=1.25,
            finished_fraction=0.75,
            finished_count=3,
            frt_hits=2,
            collapsed_hits=1,
            dead_end_hits=0,
            all_step_reset_count=0,
            all_step_frt_hits=2,
            all_step_collapsed_hits=1,
            all_step_dead_end_hits=0,
            expanded_states=8,
            discovered_states=13,
            multiprocessing_steps=0,
            total_candidates=21,
            total_valid_actions=17,
            candidate_expand_sec=0.1,
            policy_data_build_sec=0.2,
            policy_batch_transfer_sec=0.3,
            policy_value_inference_sec=0.4,
            policy_action_inference_sec=0.5,
            transition_apply_sec=0.6,
        ),
        rollout_lengths=[2, 4, 4],
    )

    payload = evaluate_rl_cy.build_summary_payload(
        checkpoint_path=None,
        policy_mode="random",
        preprocessing="rms_radius",
        device=torch.device("cpu"),
        eval_initial_states=[object(), object(), object()],
        eval_polytope_indices=[3, 5],
        eval_summary=summary,
        eval_steps=6,
        eval_sec=1.5,
        eval_mean_vertices=7.0,
        graph_node_count=10,
        graph_edge_count=12,
        cached_states=9,
        hot_cache_size=4,
        shared_cache_sizes={
            "subcomplex": 1,
            "neighbour_flip": 2,
            "subcomplex_transition": 3,
            "subcomplex_neighbour": 4,
        },
    )

    assert payload["preprocessing"] == "rms_radius"
    assert payload["rollout_lengths"] == [2, 4, 4]
    assert math.isclose(payload["rollout_length_mean"], 10.0 / 3.0)
    assert payload["rollout_length_min"] == 2
    assert payload["rollout_length_max"] == 4


def test_resolve_eval_vertex_preprocessor_rejects_random_rollout_preprocessing():
    with pytest.raises(ValueError, match="requires policy evaluation"):
        evaluate_rl_cy.resolve_eval_vertex_preprocessor(
            random_policy=True,
            preprocessing="whitening",
        )


def test_import_evaluate_rl_cy_does_not_emit_swig_deprecation_warnings():
    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", "import core.evaluate_rl_cy"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
