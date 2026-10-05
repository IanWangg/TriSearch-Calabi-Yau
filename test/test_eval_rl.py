"""RL search contracts on exact graphs, independent of geometry or checkpoints."""

from dataclasses import asdict, replace
from contextlib import closing
import math

import numpy as np
import pytest

from eval.algorithm import RLStochasticPolicy, RLPolicyBeamSearch, RLValueBeamSearch, RLValueBestFirst, get_algorithm
from eval.algorithm import RLMetricValueBeamSearch, RLMetricValueBestFirst
from eval.batched_rollout import run_batched_rollouts
from eval.rollout import run_rollout
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import canonical_simplices, CyStateRecord
from test_eval_rollout import GraphPool


class GraphPolicy:
    def __init__(self, pool, probabilities=None, values=None, forbid_actor=False):
        self.pool = pool
        self.probabilities = probabilities or {}
        self.values = values or {}
        self.forbid_actor = forbid_actor
        self.action_batches, self.value_batches = [], []

    def index(self, state):
        return self.pool.by_simplices[canonical_simplices(state.simplices)]

    def score_actions(self, states, action_lists):
        assert not self.forbid_actor
        self.action_batches.append([self.index(state) for state in states])
        return [np.log(np.asarray(self.probabilities.get(self.index(state),
                        np.ones(len(actions)) / len(actions)), dtype=float))
                for state, actions in zip(states, action_lists)]

    def score_values(self, states):
        self.value_batches.append([self.index(state) for state in states])
        return [self.values.get(self.index(state), 0.0) for state in states]


def run_rl_graph(tmp_path, factory, *, adjacency, values, budget=20, starts=(0,), cache=True,
                 probabilities=None, critic=None, forbid_actor=False, serial=False, goal="max",
                 objective_name="max_kcup", seeds=None, fail_at=None):
    pool = GraphPool(adjacency, values)
    policy = GraphPolicy(pool, probabilities, critic, forbid_actor)
    initial_states = [pool.states[index] for index in starts]
    engine = CYRandomRolloutEngine(
        base_states={state.key: state for state in initial_states}, initial_states=initial_states,
        polytope_by_index={0: pool.states[0].configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, reward_function=lambda a, b: 0.0,
        state_cache_mode="lru" if cache else "none", cache_budget_bytes=1000000 if cache else 0,
        transition_pool=pool, history_path=str(tmp_path / "states.sqlite3"),
    )
    events = dict(queries=[], expansions=[], transitions=[], rollouts=[])
    callbacks = dict(on_query=events["queries"].append, on_expansion=events["expansions"].append,
                     on_transition=events["transitions"].append)
    seeds = list(range(10, 10 + len(starts))) if seeds is None else seeds

    def batch_objectives(states):
        with closing(engine.objective_values(states, "max_kcup")) as results:
            for state in states:
                if fail_at == policy.index(state):
                    raise RuntimeError("intentional objective failure")
                yield next(results)

    kwargs = dict(policy=policy, objective_function=lambda state: engine.objective_value(state, "max_kcup"),
                  objective_goal=goal, objective_budget=budget, objective_name=objective_name,
                  batch_objective_function=batch_objectives, **callbacks)
    try:
        if serial:
            results = [run_rollout(state, factory(), engine, seed=seed, start_index=index, **kwargs)
                       for index, (state, seed) in enumerate(zip(initial_states, seeds))]
        else:
            results = run_batched_rollouts(initial_states, [factory() for _ in starts], engine,
                                          seeds=seeds, start_indices=list(range(len(starts))),
                                          on_rollout=events["rollouts"].append, **kwargs)
        if not cache:
            assert not engine.nodes_by_key
            assert engine.memory_stats()["objective_cache_bytes"] == 0
            assert len(pool.objective_calls) == sum(result.objective_queries + 1 for result in results)
        return results, pool, policy, events
    finally:
        engine.close()


def indices(pool, events, field="state_key"):
    by_key = {state.key: i for i, state in enumerate(pool.states)}
    return [by_key[event[field]] for event in events]


def test_stochastic_masks_visited_targets_even_with_tiny_remaining_probability(tmp_path):
    results, pool, policy, events = run_rl_graph(
        tmp_path, RLStochasticPolicy, adjacency=[[0, 1], [0, 1, 2], [0, 1, 2]], values=[1, 2, 3],
        probabilities={0: [1 - 1e-10, 1e-10], 1: [0.5, 0.5 - 1e-100, 1e-100]}, cache=False,
    )
    result = results[0]
    assert result.termination_reason == "no_unvisited_neighbors"
    assert result.objective_queries == result.transition_count == 2
    assert result.expansion_count == 3
    assert indices(pool, events["queries"]) == [0, 1, 2]
    assert [event["round_queries"] for event in events["expansions"]] == [1, 1, 0]
    assert not policy.value_batches


def test_stochastic_uses_conditioned_distribution_and_independent_rngs(tmp_path):
    seeds = list(range(64))
    _, pool, policy, events = run_rl_graph(
        tmp_path, RLStochasticPolicy, adjacency=[[0, 1, 2], [], []], values=[1, 2, 3], budget=1,
        starts=(0,) * len(seeds), seeds=seeds, probabilities={0: [0.999, 0.00025, 0.00075]},
    )
    expected = [1 + int(np.random.default_rng(seed).choice(2, p=[0.25, 0.75])) for seed in seeds]
    assert indices(pool, events["transitions"]) == expected
    assert len(policy.action_batches) == 1 and len(policy.action_batches[0]) == len(seeds)
    assert len(pool.objective_calls) == 3  # Physical cache/dedup; logical charges remain per start.


def test_policy_beam_uses_cumulative_probability_and_best_includes_pruned_proposals(tmp_path):
    results, pool, policy, events = run_rl_graph(
        tmp_path, lambda: RLPolicyBeamSearch(2),
        adjacency=[[1, 2, 3], [4, 5], [6, 7], [], [], [], [], []],
        values=[1, 2, 3, 1000, 4, 5, 6, 90],
        probabilities={0: [0.6, 0.3, 0.1], 1: [0.51, 0.49], 2: [0.9, 0.1]},
    )
    assert indices(pool, events["expansions"]) == [0, 1, 2, 4, 5]
    assert indices(pool, events["queries"]) == [0, 1, 2, 4, 5, 6, 7]
    assert results[0].best_objective == 90 and results[0].objective_queries == 6
    assert results[0].transition_count == 0 and not events["transitions"]
    assert policy.action_batches == [[0], [1, 2]]


@pytest.mark.parametrize("discount,expected", [(0.0, [3, 4]), (0.9, [4, 3])])
def test_value_beam_uses_absolute_log_metric_and_configured_discount(tmp_path, discount, expected):
    _, pool, _, events = run_rl_graph(
        tmp_path, lambda: RLValueBeamSearch(2, value_discount=discount),
        adjacency=[[1, 2], [3], [4], [], []], values=[1, 10, 2, 11, 5], critic={3: -2},
    )
    assert indices(pool, events["expansions"])[-2:] == expected


def test_value_all_neighbors_skips_actor_and_finishes_parent_over_budget(tmp_path):
    results, pool, policy, events = run_rl_graph(
        tmp_path, lambda: RLValueBeamSearch(1, -1),
        adjacency=[[1, 2, 3, 4], [], [], [], []], values=[1, 2, 3, 4, 5],
        budget=1, forbid_actor=True,
    )
    assert results[0].objective_queries == 4 and results[0].budget_overshoot == 3
    assert len(events["expansions"]) == 1
    assert policy.value_batches == [[1, 2, 3, 4]] and not policy.action_batches


def test_value_proposal_count_is_independent_of_beam_width(tmp_path):
    results, _, _, events = run_rl_graph(
        tmp_path, lambda: RLValueBeamSearch(1, 3), adjacency=[[1, 2, 3, 4], [], [], [], []],
        values=[1, 2, 3, 4, 5], budget=1,
    )
    assert results[0].objective_queries == 3 and results[0].budget_overshoot == 2
    assert events["expansions"][0]["candidate_count"] == 4


@pytest.mark.parametrize("factory", [lambda: RLPolicyBeamSearch(2), lambda: RLValueBeamSearch(2),
                                    RLValueBestFirst, lambda: RLValueBestFirst(-1)])
def test_beams_stop_before_next_parent_and_deduplicate_shared_children(tmp_path, factory):
    result, pool, _, events = run_rl_graph(
        tmp_path, factory, adjacency=[[0, 1, 2], [0, 2, 3, 4], [0, 3, 4, 5], [], [], []],
        values=[1, 3, 2, 4, 5, 6], budget=3,
    )
    assert result[0].objective_queries == 4 and result[0].budget_overshoot == 1
    assert indices(pool, events["expansions"]) == [0, 1]
    assert indices(pool, events["queries"]) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("factory", [RLStochasticPolicy, lambda: RLPolicyBeamSearch(2),
                                    lambda: RLValueBeamSearch(2), lambda: RLValueBeamSearch(2, -1),
                                    RLValueBestFirst, lambda: RLValueBestFirst(-1),
                                    lambda: RLMetricValueBeamSearch(2), lambda: RLMetricValueBeamSearch(2, -1),
                                    RLMetricValueBestFirst, lambda: RLMetricValueBestFirst(-1)])
def test_batch_matches_serial_cache_order_and_start_local_history(tmp_path, factory):
    kwargs = dict(adjacency=[[1, 2], [0, 2, 3], [0, 1, 3], [0]], values=[1, 3, 2, 5],
                  budget=10, starts=(0, 1, 2))
    batch, _, _, logs = run_rl_graph(tmp_path / "batch", factory, **kwargs)
    serial, _, _, serial_logs = run_rl_graph(tmp_path / "serial", factory, serial=True, **kwargs)
    cold, _, _, cold_logs = run_rl_graph(tmp_path / "cold", factory, cache=False, **kwargs)
    assert [asdict(row) for row in batch] == [asdict(row) for row in serial] == [asdict(row) for row in cold]
    for key in ("queries", "expansions", "transitions"):
        for start in range(3):
            select = lambda rows: [row for row in rows if row["start_index"] == start]
            assert select(logs[key]) == select(serial_logs[key]) == select(cold_logs[key])
    assert all(result.objective_queries <= 3 for result in batch)


@pytest.mark.parametrize("factory", [RLStochasticPolicy, lambda: RLPolicyBeamSearch(2), RLValueBeamSearch,
                                    RLValueBestFirst, lambda: RLValueBestFirst(-1),
                                    RLMetricValueBeamSearch, RLMetricValueBestFirst])
def test_zero_budget_and_dead_end(tmp_path, factory):
    results, pool, policy, events = run_rl_graph(tmp_path / "zero", factory, adjacency=[[]], values=[1], budget=0)
    assert results[0].termination_reason == "budget_exhausted"
    assert not events["expansions"] and not pool.expansion_calls and not policy.action_batches
    results, _, _, events = run_rl_graph(tmp_path / "dead", factory, adjacency=[[]], values=[1])
    assert results[0].objective_queries == 0 and len(events["expansions"]) == 1
    assert results[0].termination_reason == ("no_neighbors" if factory is RLStochasticPolicy else "frontier_exhausted")


@pytest.mark.parametrize("factory", [lambda: RLValueBeamSearch(1, -1), lambda: RLValueBestFirst(-1)])
def test_value_search_rejects_unsupported_objective(tmp_path, factory):
    with pytest.raises(ValueError, match="supports only"):
        run_rl_graph(tmp_path, factory, adjacency=[[1, 2], [], []], values=[5, 3, 8],
                     goal="min", objective_name="min_tri", forbid_actor=True)


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("factory", [RLPolicyBeamSearch, RLValueBestFirst, RLMetricValueBeamSearch, RLMetricValueBestFirst])
def test_batch_invalid_objective_keeps_rollout_context(tmp_path, invalid, factory):
    with pytest.raises(RuntimeError, match=f"algorithm={factory.name}, polytope=0, start=0, queries=1"):
        run_rl_graph(tmp_path, factory, adjacency=[[1], []], values=[1, invalid])


@pytest.mark.parametrize("count,expected", [(None, 4), (2, 2), (6, 6), (-1, 6)])
@pytest.mark.parametrize("name", ["rl_value_best_first", "rl_metric_value_best_first"])
def test_value_best_first_proposals_finish_parent_and_ignore_beam_width(tmp_path, count, expected, name):
    results, pool, policy, events = run_rl_graph(
        tmp_path, lambda: get_algorithm(name, beam_width=8, policy_proposal_count=count),
        adjacency=[[1, 2, 3, 4, 5, 6], [], [], [], [], [], []], values=[1, 2, 3, 4, 5, 6, 7],
        budget=1, forbid_actor=count == -1,
    )
    assert results[0].objective_queries == expected
    assert results[0].budget_overshoot == expected - 1
    assert results[0].transition_count == 0 and not events["transitions"]
    assert indices(pool, events["queries"]) == list(range(expected + 1))
    assert indices(pool, events["expansions"]) == [0]
    assert policy.value_batches == [list(range(1, expected + 1))]
    assert policy.action_batches == ([] if count == -1 else [[0]])


@pytest.mark.parametrize("count", [None, -1])
def test_value_best_first_keeps_global_frontier_and_recovers_from_dead_ends(tmp_path, count):
    results, pool, policy, events = run_rl_graph(
        tmp_path, lambda: RLValueBestFirst(count),
        adjacency=[[1, 2, 3], [4], [5], [], [], []], values=[1, 10, 3, 2, 11, 12],
        forbid_actor=count == -1,
    )
    # Child 4 outranks old siblings; after its dead end the frontier still reaches child 5.
    expected = [0, 1, 4, 2, 5, 3]
    assert indices(pool, events["expansions"]) == expected
    assert pool.expansion_calls == expected  # No free prefetch of the entire frontier.
    assert policy.value_batches == [[1, 2, 3], [4], [5]]
    assert results[0].termination_reason == "frontier_exhausted"
    assert results[0].objective_queries == 5 and results[0].expansion_count == 6
    assert results[0].best_objective == 12
    assert results[0].transition_count == 0 and not events["transitions"]


@pytest.mark.parametrize("discount,expected", [(0.0, [0, 1, 3, 2, 4]), (0.9, [0, 1, 2, 4, 3])])
def test_value_best_first_absolute_metric_and_discount(tmp_path, discount, expected):
    _, pool, _, events = run_rl_graph(
        tmp_path, lambda: RLValueBestFirst(value_discount=discount),
        adjacency=[[1, 2], [3], [4], [], []], values=[1, 10, 2, 11, 5], critic={3: -2},
    )
    assert indices(pool, events["expansions"]) == expected


def test_value_best_first_score_ties_keep_first_discovery_order(tmp_path):
    _, pool, _, events = run_rl_graph(
        tmp_path, RLValueBestFirst, adjacency=[[1, 2, 3], [4], [], [], []], values=[1] * 5,
    )
    assert indices(pool, events["expansions"]) == [0, 1, 2, 3, 4]


def test_value_best_first_shared_child_keeps_first_score_and_skips_cycles(tmp_path):
    results, pool, policy, events = run_rl_graph(
        tmp_path, lambda: RLValueBestFirst(-1),
        adjacency=[[0, 1, 2], [0, 2, 3, 3], [0, 3, 4], [], []], values=[1, 10, 2, 3, 2],
        forbid_actor=True,
    )
    # Child 3 is evaluated only once despite duplicate actions and another parent.
    assert indices(pool, events["expansions"]) == [0, 1, 3, 2, 4]
    assert indices(pool, events["queries"]) == [0, 1, 2, 3, 4]
    assert policy.value_batches == [[1, 2], [3], [4]]
    assert results[0].objective_queries == 4


def test_value_best_first_filters_seen_before_top_policy_and_can_revisit_unqueried_targets(tmp_path):
    _, pool, _, events = run_rl_graph(
        tmp_path, lambda: RLValueBestFirst(1),
        adjacency=[[1, 2], [0, 2, 3], [1, 3], []], values=[1, 2, 3, 4],
        probabilities={0: [0.9, 0.1], 1: [0.9, 0.05, 0.05], 2: [0.99, 0.01]},
    )
    assert indices(pool, events["queries"]) == [0, 1, 2, 3]
    assert [row["round_queries"] for row in events["expansions"]] == [1, 1, 1, 0]


def test_value_best_first_best_includes_unexpanded_proposals(tmp_path):
    results, pool, _, events = run_rl_graph(
        tmp_path, RLValueBestFirst, adjacency=[[1, 2], [], [3], []], values=[1, 100, 2, 3],
        critic={1: -10, 2: 2}, budget=3,
    )
    assert indices(pool, events["expansions"]) == [0, 2]
    assert results[0].best_objective == 100 and results[0].best_state_key == pool.states[1].key


def test_all_polytopes_share_inference_but_keep_separate_identity(tmp_path):
    pool = GraphPool([[1], [0], [3], [2]], [1, 2, 10, 20])
    second_config = replace(pool.states[2].configuration, index=1)
    for index in (2, 3):
        pool.states[index] = CyStateRecord(second_config, pool.states[index].simplices, "two_neighbors", True, True)
    states = [pool.states[0], pool.states[2]]
    policy = GraphPolicy(pool)
    engine = CYRandomRolloutEngine(
        base_states={state.key: state for state in states}, initial_states=states,
        polytope_by_index={state.point_config_index: state.configuration for state in states},
        neighbor_mode="two_neighbors", include_points_interior_to_facets=False,
        reward_function=lambda a, b: 0.0,
        transition_pool=pool, history_path=str(tmp_path / "multi.sqlite3"),
    )
    try:
        results = run_batched_rollouts(
            states, [RLStochasticPolicy(), RLStochasticPolicy()], engine, policy=policy,
            objective_function=lambda state: engine.objective_value(state, "max_kcup"),
            batch_objective_function=lambda entries: engine.objective_values(entries, "max_kcup"),
            objective_goal="max", objective_budget=5, seeds=[1, 2], start_indices=[0, 0],
        )
        assert policy.action_batches == [[0, 2], [1, 3]]
        assert [result.polytope_index for result in results] == [0, 1]
        assert [result.best_objective for result in results] == [2, 20]
        assert all(result.objective_queries == 1 for result in results)
    finally:
        engine.close()
