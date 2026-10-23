"""Finite-episode objective, contextual inference, and terminal masking contracts."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.cy_episode_context import BEST_SO_FAR_LOG_VOLUME, EpisodeContext, best_so_far_reward
from core.cy_policy_inference import (
    batched_policy_action_selection, build_cy_data_list, evaluate_policy_actions_from_data_list,
    evaluate_policy_scores, configure_policy_execution, query_objective_values,
)
from core.cy_policy_rollout import collect_policy_rollout
from core.cy_ppo import PPORolloutBuffer, train_policy_from_rollout
from core.cy_training_config import parse_args
from eval.algorithm import RLStochasticPolicy, RLValueBestFirst, RLValueBeamSearch
from eval.batched_rollout import run_batched_rollouts
from eval.policy import EvaluationPolicy
from mdp.cy_rollout import CYRandomRolloutEngine
from models.subcomplex_policy_factory import build_subcomplex_agent
from reward_functions import get_reward
from test_cy_bounded_ppo import _step
from test_cy_two_face_data import square_state, square_actions
from test_eval_rl import GraphPolicy
from test_eval_rollout import GraphPool

DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_execution():
    torch.set_default_device("cpu")
    configure_policy_execution()
    yield
    configure_policy_execution()


def context_policy(encoder="deep_sets"):
    torch.manual_seed(17)
    return build_subcomplex_agent(
        model_type="egnn", subcomplex_actor_type="two_face_deep_sets", in_channels=4,
        hidden_channels=8, out_channels=8, num_layers=2,
        training_reward_mode=BEST_SO_FAR_LOG_VOLUME, episode_horizon=128,
        two_face_encoder=encoder,
    ).eval()


def test_best_reward_only_rewards_new_records_and_telescopes():
    best, rewards = 2., []
    for value in [4., 1., 4., 4., 8., 2.]:
        best, reward = best_so_far_reward(best, value)
        rewards.append(reward)
    assert rewards == pytest.approx([math.log(2), 0, 0, 0, math.log(2), 0])
    assert sum(rewards) == pytest.approx(math.log(best / 2))
    assert EpisodeContext(2, 8, 0).features() == pytest.approx([math.log(2), math.log(4), 0])
    for bad in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            best_so_far_reward(2, bad)
    with pytest.raises(ValueError, match="smaller"):
        EpisodeContext(4, 2, 1).features()
    with pytest.raises(ValueError, match="non-negative integer"):
        EpisodeContext(2, 4, -1).features()


def test_mode_defaults_and_invalid_configurations():
    legacy = parse_args([])
    assert (legacy.gamma, legacy.rollout_length, legacy.eval_steps) == (.95, 20, 20)
    flags = ["--training_reward_mode", BEST_SO_FAR_LOG_VOLUME,
             "--subcomplex_actor_type", "two_face_deep_sets", "--neighbor_mode", "two_neighbors",
             "--reward_function", "max_kcup", "--no-vertex_aug_enable", "--no-include_points_interior_to_facets"]
    args = parse_args(flags)
    assert (args.gamma, args.rollout_length, args.eval_steps) == (1, 128, 128)
    for extra in (["--gamma", ".95"], ["--rollout_length", "0"], ["--count_bonus_coef", "1"],
                  ["--subcomplex_actor_type", "mlp"], ["--vertex_aug_enable"]):
        with pytest.raises(ValueError):
            parse_args(flags + extra)


@pytest.mark.parametrize("encoder", ["deep_sets", "triangle_dual"])
def test_context_is_local_and_survives_batching_chunks_and_ppo(encoder):
    state, model = square_state(dimension=4), context_policy(encoder)
    contexts = [EpisodeContext(2, 2, 128), EpisodeContext(2, 8, 7), EpisodeContext(2, 4, 0)]
    states, actions = [state] * 3, [square_actions(state), square_actions(state), []]
    selection = batched_policy_action_selection(states, actions, model, device=DEVICE, episode_contexts=contexts)
    data = selection.data_list
    assert all(item.observation_schema_version == 2 for item in data)
    assert not torch.equal(data[0].episode_context, data[1].episode_context)
    legacy_data = build_cy_data_list([state], [actions[0]], observation_kind="two_face")[0]
    assert legacy_data.observation_schema_version == 1 and "episode_context" not in legacy_data
    with pytest.raises(ValueError, match="schema"):
        evaluate_policy_scores(states, actions, model, device=DEVICE)
    chunked = evaluate_policy_scores(states, actions, model, device=DEVICE, episode_contexts=contexts, max_graph_size=1)
    torch.testing.assert_close(chunked.value_tensor, selection.value_tensor)
    assert not torch.isclose(selection.value_tensor[0], selection.value_tensor[1])
    buffer = PPORolloutBuffer()
    step = _step(states, selection)
    buffer.append(step)
    prepared = buffer.prepare(bootstrap_value=torch.zeros(3), gamma=1, gae_lambda=.95, device=DEVICE)
    scored = evaluate_policy_actions_from_data_list(prepared.data_buffer_list, prepared.action_index_buffer_flat,
                                                   model, device=DEVICE)
    torch.testing.assert_close(scored.log_prob_tensor, selection.log_prob_tensor)
    for item, context in zip(prepared.data_buffer_list, contexts):
        torch.testing.assert_close(item.episode_context, torch.tensor([context.features()]))
    scorer = EvaluationPolicy(model, device="cpu")
    torch.testing.assert_close(torch.tensor(scorer.score_values(states, episode_contexts=contexts)),
                               selection.value_tensor.double())
    scorer.validate_context_algorithms(["rl_value_best_first", "rl_value_beam_search", "rl_policy_beam_search"])
    assert scorer.value_score_definition == "ln_best_objective_plus_discounted_remaining_best_gain"
    with pytest.raises(ValueError, match="max_kcup"):
        scorer.validate_context_algorithms(["rl_value_best_first"], "max_cy_volume")
    # Both heads actually train the context path.
    (scored.value_tensor.sum() + scored.log_prob_tensor[:2].sum()).backward()
    assert model.context_mlp[0].weight.grad[:, -3:].abs().sum() > 0


class EpisodeEngine:
    """Deterministic trajectory with real two-face observations and no geometry solver."""
    def __init__(self, *, early_dead_end=False):
        geometry = square_state(dimension=4)
        self.states = [SimpleNamespace(key=str(i), configuration=geometry.configuration,
            simplices=geometry.simplices, point_config_index=7, visitation=0) for i in range(5)]
        self.action = square_actions(geometry)[0]
        adjacency = [1, None if early_dead_end else 2, 3, 0, None]
        self.nodes_by_key = {}
        for state, target in zip(self.states, adjacency):
            transitions = {} if target is None else {self.action: SimpleNamespace(
                next_key=str(target), next_simplices=geometry.simplices)}
            self.nodes_by_key[state.key] = SimpleNamespace(transitions=transitions, candidate_actions=tuple(transitions))
        self.reward_function = get_reward("max_kcup")
        self.samples = 0
        self.objective_calls = []
        self.expanded = []

    def sample_initial_states(self, num_envs, **kwargs):
        self.samples += 1
        return [self.states[0], self.states[4]][:num_envs]

    def candidate_actions_for_states(self, states, **kwargs):
        self.expanded.extend(state.key for state in states)
        return [self.nodes_by_key[state.key].candidate_actions for state in states], self.expand_states([])

    def expand_states(self, states, **kwargs):
        return SimpleNamespace(expanded_count=0, discovered_count=0, used_multiprocessing=False)

    def materialize_transition(self, state, transition):
        return self.states[int(transition.next_key)]

    def objective(self, state):
        self.objective_calls.append(state.key)
        return [1., 4., 2., 8., 16.][int(state.key)]


def collect(engine, model, horizon=5, *, batch_objective_function=None, num_envs=2):
    return collect_policy_rollout(
        engine=engine, policy=model, rng=np.random.default_rng(0), device=DEVICE,
        initial_state_pool=engine.states, num_envs=num_envs, rollout_length=horizon, gamma=1., deterministic=True,
        use_multiprocessing=False, transition_pool=None, transition_mp_chunksize=1, transition_mp_min_batch=1,
        store_buffer=True, report_every=0, label="test", objective_function=engine.objective,
        batch_objective_function=batch_objective_function,
        objective_name="max_kcup", objective_goal="max", training_reward_mode=BEST_SO_FAR_LOG_VOLUME,
    )


def test_episode_budget_no_reset_no_extra_objectives_and_masked_losses():
    engine, model = EpisodeEngine(), context_policy()
    summary = collect(engine, model)
    assert summary.episode_lengths == [5, 1]
    assert summary.episode_terminal_reasons == ["budget_exhausted", "dead_end_current"]
    assert summary.objective_best_values == [8, 16] and summary.objective_final_values == [4, 16]
    assert summary.training_return_mean == pytest.approx(math.log(8) / 2)
    assert summary.finished_count == 2 and summary.all_step_reset_count == 0 and engine.samples == 1
    assert len(engine.objective_calls) == 7  # Two initial values + five transitions.
    assert engine.expanded.count("4") == 1
    buffer = summary.rollout_buffer
    assert torch.stack(buffer.reward_buffer).sum(0).tolist() == pytest.approx([math.log(8), 0])
    prepared = buffer.prepare(bootstrap_value=torch.full((2,), 1e6), gamma=1, gae_lambda=1, device=DEVICE)
    assert prepared.sample_valid_mask_flat.sum() == 6
    assert prepared.value_targets[0].tolist() == pytest.approx([math.log(8), 0], abs=1e-6)
    assert prepared.value_targets[-1, 0].item() == pytest.approx(0, abs=1e-6)
    assert buffer.data_buffer[2][0].episode_context.tolist()[0] == pytest.approx(
        EpisodeContext(2, 4, 3).features())
    # Poison padding to prove it is excluded from critic loss as well as actor loss.
    poisoned = copy.deepcopy(prepared)
    poisoned.value_targets.reshape(-1)[~prepared.sample_valid_mask_flat] = 1e6
    other = copy.deepcopy(model)
    stats = []
    for policy, batch in ((model, prepared), (other, poisoned)):
        torch.manual_seed(41)
        stats.append(train_policy_from_rollout(policy=policy, optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
            prepared_rollout=batch, device=DEVICE, num_epochs=1, batch_size=6, clip_coef=.2, value_coef=.5,
            entropy_coef=.001, max_grad_norm=1))
    assert stats[0] == stats[1] and stats[0].num_samples == 6
    for first, second in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(first, second)
    buffer.clear()
    assert not buffer.sample_valid_mask_buffer


def test_early_dead_end_stops_whole_batch_and_next_rollout_resets_best():
    engine, model = EpisodeEngine(early_dead_end=True), context_policy()
    first = collect(engine, model)
    second = collect(engine, model)
    for summary in (first, second):
        assert summary.episode_lengths == [1, 1]
        assert summary.episode_terminal_reasons == ["dead_end_next", "dead_end_current"]
        assert summary.training_return_mean == pytest.approx(math.log(4) / 2)
        assert len(summary.rollout_buffer.reward_buffer) == 1
    assert engine.samples == 2 and len(engine.objective_calls) == 6


@pytest.mark.parametrize("budget", [0, 2])
def test_search_context_is_per_session_and_uses_existing_query_budget(tmp_path, budget):
    pool = GraphPool([[1], [2], []], [8., 4., 16.])
    class ContextScorer(GraphPolicy):
        requires_episode_context = True
        contexts = []
        def score_actions(self, states, action_lists, *, episode_contexts=None):
            self.contexts.extend(episode_contexts)
            return super().score_actions(states, action_lists)
    policy = ContextScorer(pool)
    starts = pool.states[:2]
    engine = CYRandomRolloutEngine(base_states={state.key: state for state in starts}, initial_states=starts,
        polytope_by_index={0: starts[0].configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, reward_function=lambda a, b: 0,
        transition_pool=pool, history_path=str(tmp_path / "states.sqlite3"))
    try:
        result = run_batched_rollouts(starts, [RLStochasticPolicy(), RLStochasticPolicy()], engine,
            policy=policy, objective_function=lambda state: engine.objective_value(state, "max_kcup"),
            objective_goal="max", objective_budget=budget, seeds=[0, 1], start_indices=[0, 1])
        if budget == 0:
            assert not policy.contexts and not pool.expansion_calls
            assert [item.objective_queries for item in result] == [0, 0]
        else:
            assert policy.contexts == [EpisodeContext(8, 8, 2), EpisodeContext(4, 4, 2), EpisodeContext(4, 8, 1)]
            assert [item.objective_queries for item in result] == [2, 1]
        with pytest.raises(ValueError, match="max_kcup"):
            run_batched_rollouts([starts[0]], [RLValueBestFirst()], engine, policy=policy,
                objective_function=None, objective_name="max_cy_volume", objective_goal="max",
                objective_budget=budget, seeds=[0], start_indices=[0])
    finally:
        engine.close()


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("best_first", [False, True])
def test_context_search_scores_known_objectives_and_refreshes_queued_frontier(tmp_path, cache, best_first):
    pool = GraphPool([[1, 2], [3], [4], [5], [], []], [1., 2., 3., 10., 20., 30.])
    pool.batch_open = False
    contexts = []
    class ContextScorer(GraphPolicy):
        requires_episode_context = True

        def score_actions(self, states, action_lists, *, episode_contexts=None):
            assert all(context.best_volume >= context.current_volume for context in episode_contexts)
            return super().score_actions(states, action_lists)

        def score_values(self, states, *, episode_contexts=None):
            assert not pool.batch_open
            contexts.extend(episode_contexts)
            values = []
            for state, context in zip(states, episode_contexts):
                index = self.index(state)
                assert index in pool.objective_calls  # No critic inference before objective queries finish.
                assert context.current_volume == pool.values[index]
                if best_first:
                    value = 3 if index == 1 else (5 if index == 2 and context.best_volume >= 10 else
                                                  1 if index == 2 else 0)
                else:
                    value = .2 if index == 1 else 0
                values.append(value)
            return values

    source = pool.states[0]
    policy = ContextScorer(pool)
    engine = CYRandomRolloutEngine(base_states={source.key: source}, initial_states=[source],
        polytope_by_index={0: source.configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, reward_function=lambda a, b: 0,
        transition_pool=pool, state_cache_mode="lru" if cache else "none",
        cache_budget_bytes=1000000 if cache else 0, history_path=str(tmp_path / "states.sqlite3"))
    algorithm = RLValueBestFirst(2, value_discount=1) if best_first else RLValueBeamSearch(1, 2, 1)
    events, queries, expansions = [], [], []
    def batch_objectives(states):
        pool.batch_open = True
        try:
            yield from engine.objective_values(states, "max_kcup")
        finally:
            pool.batch_open = False
    try:
        result = run_batched_rollouts([source], [algorithm], engine, policy=policy,
            objective_function=lambda state: engine.objective_value(state, "max_kcup"), objective_goal="max",
            objective_budget=5 if best_first else 3, seeds=[0], start_indices=[0],
            batch_objective_function=batch_objectives,
            on_query=queries.append, on_expansion=expansions.append, search_observer=events.append)[0]
        assert contexts[:2] == [EpisodeContext(2, 3, 3 if best_first else 1),
                                EpisodeContext(3, 3, 3 if best_first else 1)]
        assert result.best_objective == (30 if best_first else 10)
        assert result.objective_queries == (5 if best_first else 3)
        assert len(queries) == len(pool.objective_calls) == result.objective_queries + 1
        assert result.transition_count == 0
        assert result.termination_reason == "budget_exhausted"
        if best_first:
            assert EpisodeContext(3, 10, 2) in contexts  # Queued node gets new record and budget.
            assert [event["state_key"] for event in expansions] == [pool.states[i].key for i in (0, 1, 2, 3)]
            assert any(event["event"] == "frontier_rescore" for event in events)
        scored = [event for event in events if event["event"] == "candidate_score"]
        assert scored[0]["score"] == pytest.approx(math.log(3) + (3 if best_first else .2))
        if not cache:
            assert not engine.nodes_by_key
    finally:
        engine.close()


@pytest.mark.parametrize("early_dead_end", [False, True])
def test_batched_rollout_matches_scalar_and_only_queries_selected_active_states(early_dead_end):
    class MultiEpisodeEngine(EpisodeEngine):
        batch_open = False

        def sample_initial_states(self, num_envs, **kwargs):
            self.samples += 1
            return [self.states[i] for i in (0, 2, 0, 4)]

        def expand_states(self, states, **kwargs):
            assert not self.batch_open, "Finish objective iterator before geometry expansion."
            return super().expand_states(states, **kwargs)

    scalar, batched = [MultiEpisodeEngine(early_dead_end=early_dead_end) for _ in range(2)]
    batches, closed = [], []
    def batch_query(states):
        batches.append([state.key for state in states])
        batched.batch_open = True
        try:
            yield from (batched.objective(state) for state in states)
        finally:
            batched.batch_open = False
            closed.append(True)
    reference = collect(scalar, context_policy(), num_envs=4)
    actual = collect(batched, context_policy(), num_envs=4, batch_objective_function=batch_query)
    assert batches[:2] == [["0", "2", "0", "4"], ["1", "3", "1"]]
    assert len(closed) == len(batches)
    assert all("4" not in batch for batch in batches[1:])  # No queries for the inactive slot.
    assert scalar.objective_calls == batched.objective_calls
    for name in ("objective_initial_values", "objective_final_values", "objective_best_values",
                 "episode_lengths", "episode_terminal_reasons", "training_return_mean"):
        assert getattr(reference, name) == getattr(actual, name)
    for name in ("reward_buffer", "done_buffer", "action_index_buffer", "log_prob_buffer", "value_buffer"):
        for expected, observed in zip(getattr(reference.rollout_buffer, name), getattr(actual.rollout_buffer, name)):
            torch.testing.assert_close(expected, observed)
    for expected_step, observed_step in zip(reference.rollout_buffer.data_buffer, actual.rollout_buffer.data_buffer):
        for expected, observed in zip(expected_step, observed_step):
            torch.testing.assert_close(expected.episode_context, observed.episode_context)
    # Cached geometry and batch size cannot alter the GAE targets.
    prepared = [result.rollout_buffer.prepare(bootstrap_value=torch.zeros(4), gamma=1,
                                             gae_lambda=.95, device=DEVICE) for result in (reference, actual)]
    torch.testing.assert_close(prepared[0].value_targets, prepared[1].value_targets)
    assert actual.objective_query_sec >= 0


@pytest.mark.parametrize("mode", ["short", "long", "raises", "bad_value"])
def test_objective_batch_closes_on_count_conversion_and_worker_errors(mode):
    closed = []
    def batch(states):
        try:
            yield 1.
            if mode == "raises":
                raise RuntimeError("worker failed")
            if mode == "bad_value":
                yield None
            if mode in ("long", "bad_value"):
                yield 2.
                yield 3.
        finally:
            closed.append(True)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        query_objective_values(["first", "second"], lambda _: 0, batch)
    assert closed == [True]
    assert query_objective_values([], None, batch) == []
    assert closed == [True]
