from types import SimpleNamespace

import numpy as np
import pytest
import torch

import core.cy_policy_rollout_utils as policy_rollout
from core.evaluate_rl_cy import build_summary_payload, parse_args as parse_eval_args
from core.train_cy import parse_args as parse_train_args
from core.training_types import PolicyRolloutSummary
from mdp.cy_rollout import CYRandomRolloutEngine
from reward_functions import get_objective, get_reward, infer_goal


class FakeState:
    def __init__(
        self,
        *,
        key,
        simplices,
        transitions=None,
        is_target=False,
        point_config_index=0,
    ):
        self.key = str(key)
        self.simplices = frozenset(tuple(simplex) for simplex in simplices)
        self.point_config_index = int(point_config_index)
        self.is_target = bool(is_target)
        self.actions_ready = False
        self.available_subcomplex_actions = tuple((transitions or {}).keys())
        self.ambiguous_subcomplex_actions = frozenset()
        self.transitions = dict(transitions or {})

    def find_available_actions(self):
        self.actions_ready = True

    def get_available_subcomplex_actions(self):
        return self.available_subcomplex_actions

    def get_transition_output_from_subcomplex_action(self, action):
        return self.transitions[tuple(action)]


def _state_factory(states_by_simplices):
    def factory(point_config_index, simplices):
        canonical = frozenset(tuple(simplex) for simplex in simplices)
        return states_by_simplices[(int(point_config_index), canonical)]

    return factory


def _four_to_one_states(*, target_destination=False):
    destination_simplices = frozenset({(0, 1, 2, 3)})
    source_simplices = frozenset(
        {
            (0, 1, 2, 4),
            (0, 1, 3, 4),
            (0, 2, 3, 4),
            (1, 2, 3, 4),
        }
    )
    action = (0, 1, 2, 3, 4)
    destination = FakeState(
        key="destination",
        simplices=destination_simplices,
        is_target=target_destination,
    )
    source = FakeState(
        key="source",
        simplices=source_simplices,
        transitions={
            action: (destination_simplices, frozenset(), destination.key),
        },
    )
    states_by_simplices = {
        (0, source_simplices): source,
        (0, destination_simplices): destination,
    }
    return source, destination, action, states_by_simplices


def _objective_engine(source, states_by_simplices):
    return CYRandomRolloutEngine(
        base_states={source.key: source},
        initial_states=[source],
        state_factory=_state_factory(states_by_simplices),
        reward_function=get_reward("min_tri"),
    )


def test_min_tri_registry_contract():
    state_four, state_one, _, _ = _four_to_one_states()

    reward = get_reward("min_tri")
    objective = get_objective("min_tri")

    assert reward(state_four, state_one) == 3.0
    assert reward(state_one, state_four) == -3.0
    assert objective(state_four) == 4.0
    assert infer_goal("min_tri") == "min"
    with pytest.raises(ValueError, match="Unknown reward"):
        get_reward("unknown")


def test_reward_cli_aliases_are_opt_in():
    assert parse_train_args([]).reward_function is None
    assert parse_train_args(["--reward", "min_tri"]).reward_function == "min_tri"
    assert parse_train_args(["--reward_function", "min_tri"]).reward_function == "min_tri"
    assert parse_eval_args([]).reward_function is None
    assert parse_eval_args(["--reward", "min_tri"]).reward_function == "min_tri"


def test_objective_mode_expands_target_state_while_sampler_mode_stops():
    action = (0, 1, 2, 3)
    next_simplices = frozenset({(0, 1, 2)})
    target = FakeState(
        key="target",
        simplices={(0, 1, 3), (1, 2, 3)},
        transitions={action: (next_simplices, frozenset(), "next")},
        is_target=True,
    )

    sampler = CYRandomRolloutEngine(base_states={target.key: target}, initial_states=[target])
    sampler_actions, _ = sampler.candidate_actions_for_states([target])
    assert sampler_actions == [tuple()]

    objective_target = FakeState(
        key="target",
        simplices=target.simplices,
        transitions={action: (next_simplices, frozenset(), "next")},
        is_target=True,
    )
    objective = CYRandomRolloutEngine(
        base_states={objective_target.key: objective_target},
        initial_states=[objective_target],
        reward_function=get_reward("min_tri"),
    )
    objective_actions, _ = objective.candidate_actions_for_states([objective_target])
    assert objective_actions == [(action,)]


def test_random_objective_step_scores_destination_before_dead_end_reset():
    source, destination, _, states_by_simplices = _four_to_one_states(
        target_destination=True
    )
    engine = _objective_engine(source, states_by_simplices)

    result = engine.rollout_step(
        [source],
        rng=np.random.default_rng(0),
        initial_state_pool=[source],
    )

    assert result.rewards == [3.0]
    assert result.dones == [True]
    assert result.terminal_reasons == ["dead_end_next"]
    assert result.transitioned_states == [destination]
    assert result.next_states == [source]
    assert result.frt_hits == 0
    assert result.collapsed_hits == 0


def test_policy_objective_step_scores_destination_before_dead_end_reset(monkeypatch):
    source, destination, action, states_by_simplices = _four_to_one_states(
        target_destination=True
    )
    engine = _objective_engine(source, states_by_simplices)

    monkeypatch.setattr(
        policy_rollout,
        "batched_policy_action_selection",
        lambda *args, **kwargs: SimpleNamespace(
            action_lists=[(action,)],
            action_index_tensor=torch.tensor([0]),
            actions_tensor=torch.tensor([action]),
            log_prob_tensor=torch.tensor([0.0]),
            entropy_tensor=torch.tensor([0.0]),
            value_tensor=torch.tensor([0.0]),
            valid_action_mask=torch.tensor([True]),
            data_build_sec=0.0,
            batch_transfer_sec=0.0,
            value_inference_sec=0.0,
            policy_inference_sec=0.0,
            data_list=[],
        ),
    )

    result = policy_rollout.rollout_step_with_policy(
        engine,
        [source],
        object(),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        initial_state_pool=[source],
    )

    assert result.rewards == [3.0]
    assert result.terminal_reasons == ["dead_end_next"]
    assert result.transitioned_states == [destination]
    assert result.next_states == [source]
    assert result.collapsed_hits == 0


def test_objective_summary_reports_goal_aware_improvement():
    summary = PolicyRolloutSummary(
        final_states=[],
        rollout_buffer=None,
        success_rate=0.0,
        discounted_reward=1.0,
        finished_fraction=0.0,
        finished_count=0,
        frt_hits=0,
        collapsed_hits=0,
        dead_end_hits=0,
        all_step_reset_count=0,
        all_step_frt_hits=0,
        all_step_collapsed_hits=0,
        all_step_dead_end_hits=0,
        expanded_states=1,
        discovered_states=1,
        multiprocessing_steps=0,
        total_candidates=1,
        total_valid_actions=1,
        candidate_expand_sec=0.0,
        policy_data_build_sec=0.0,
        policy_batch_transfer_sec=0.0,
        policy_value_inference_sec=0.0,
        policy_action_inference_sec=0.0,
        transition_apply_sec=0.0,
        objective_name="min_tri",
        objective_goal="min",
        objective_initial_values=[4.0, 5.0],
        objective_final_values=[1.0, 6.0],
        objective_best_values=[1.0, 5.0],
    )
    payload = build_summary_payload(
        checkpoint_path="checkpoint.pth",
        policy_mode="policy",
        preprocessing="none",
        device=torch.device("cpu"),
        eval_initial_states=[object(), object()],
        eval_polytope_indices=[1],
        eval_summary=summary,
        eval_steps=2,
        eval_sec=0.1,
        eval_mean_vertices=5.0,
        graph_node_count=2,
        graph_edge_count=1,
        cached_states=2,
        hot_cache_size=1,
        shared_cache_sizes={
            "subcomplex": 0,
            "neighbour_flip": 0,
            "subcomplex_transition": 0,
            "subcomplex_neighbour": 0,
        },
    )

    assert payload["objective"]["initial_mean"] == 4.5
    assert payload["objective"]["best_mean"] == 3.0
    assert payload["objective"]["mean_improvement"] == 1.5
    assert payload["objective"]["improved_fraction"] == 0.5
