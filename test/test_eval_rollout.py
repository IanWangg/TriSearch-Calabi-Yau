"""Budget accounting on known graphs, through the actual cached rollout engine."""

from dataclasses import asdict
import math

import pytest

from eval.algorithm import BestFirstAlgorithm, BeamSearchAlgorithm, GreedyAlgorithm, RandomAlgorithm
from eval.rollout import run_rollout
from mdp.cy_graph import CYGraphTransition, CYStateExpansion
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices


class GraphPool:
    def __init__(self, adjacency, values):
        points = tuple((index, 0, 0, 0) for index in range(len(adjacency) + 5))
        configuration = CYPointConfiguration(0, points, points, tuple(range(len(points))), False)
        self.states = [CyStateRecord(configuration, frozenset({(0, 1, 2, 3, index + 4)}),
                                     "two_neighbors", True, True) for index in range(len(adjacency))]
        self.by_simplices = {canonical_simplices(state.simplices): index for index, state in enumerate(self.states)}
        self.adjacency, self.values = adjacency, values
        self.objective_calls = []
        self.expansion_calls = []
        self.expansion_events = []

    def imap(self, function, requests, chunksize=1):
        for request in requests:
            index = self.by_simplices[request["state"][1]]
            if request["operation"] == "objective":
                self.objective_calls.append(index)
                yield self.values[index]
                continue
            assert request["operation"] == "expand" and request["objective_mode"] is True
            self.expansion_calls.append(index)
            source = self.states[index]
            entries = []
            for target in self.adjacency[index]:
                destination = self.states[target]
                action = (0, 1, 2, target + 4)
                transition = CYGraphTransition(
                    next_key=destination.key, next_simplices=canonical_simplices(destination.simplices),
                    next_is_target=True, next_is_frst=True,
                )
                entries.append((action, transition))
            yield CYStateExpansion(source.key, source.point_config_index, canonical_simplices(source.simplices),
                                   tuple(action for action, _ in entries), frozenset(), tuple(entries))


def run_graph(tmp_path, algorithm, *, cache=True, budget=5, adjacency=None, values=None, seed=4, goal="max"):
    pool = GraphPool(adjacency or [[1, 2, 3], [0], [0], [0]], values or [10.0, 5.0, 5.0, 2.0])
    source = pool.states[0]
    engine = CYRandomRolloutEngine(
        base_states={source.key: source}, initial_states=[source],
        polytope_by_index={0: source.configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, reward_function=lambda a, b: 0.0,
        state_cache_mode="lru" if cache else "none", cache_budget_bytes=1000000 if cache else 0,
        transition_pool=pool, history_path=str(tmp_path / f"{algorithm.name}_{cache}_{budget}_{seed}.sqlite3"),
    )
    queries, transitions = [], []
    try:
        result = run_rollout(
            source, algorithm, engine, objective_function=lambda state: engine.objective_value(state, "max_kcup"),
            objective_goal=goal, objective_budget=budget, seed=seed,
            on_query=queries.append, on_transition=transitions.append,
            on_expansion=pool.expansion_events.append,
        )
        if not cache:
            assert not engine.state_cache.hot_states
            assert not engine.nodes_by_key
            assert engine.memory_stats()["objective_cache_bytes"] == 0
        return result, pool, queries, transitions
    finally:
        engine.close()


def test_random_queries_only_selected_neighbors_and_reuses_initial_value(tmp_path):
    result, pool, queries, transitions = run_graph(tmp_path, RandomAlgorithm(), budget=5)
    assert result.objective_queries == result.transition_count == 5
    assert [row["query_index"] for row in queries] == list(range(6))
    assert queries[0]["is_initial"] is True
    assert all(row["round_queries"] == 1 for row in transitions)
    assert len(pool.objective_calls) <= 4  # Unselected neighbors need no objective.
    first_destination = queries[1]["state_key"]
    assert first_destination == transitions[0]["state_key"]
    single, one_step_pool, _, _ = run_graph(tmp_path, RandomAlgorithm(), budget=1)
    assert len(one_step_pool.objective_calls) == 2  # Only the start and selected neighbor.
    assert single.objective_queries == 1


def test_greedy_finishes_round_and_moves_downhill_with_canonical_ties(tmp_path):
    result, pool, queries, transitions = run_graph(tmp_path, GreedyAlgorithm(), budget=5)
    assert [row["round_queries"] for row in transitions] == [3, 1, 3]
    assert result.objective_queries == 7 and result.budget_overshoot == 2
    assert result.transition_count == 3
    assert transitions[0]["state_key"] == pool.states[1].key  # Tie with state 2.
    assert transitions[-1]["objective"] == 5.0 < result.initial_objective
    assert result.best_objective == 10.0
    assert result.best_state_key == pool.states[0].key
    assert len(queries) == 8


def test_greedy_respects_minimization_and_exact_budget(tmp_path):
    result, pool, _, transitions = run_graph(tmp_path, GreedyAlgorithm(), budget=3, goal="min")
    assert transitions[-1]["state_key"] == pool.states[3].key
    assert result.best_objective == 2.0
    assert result.objective_queries == 3 and result.budget_overshoot == 0
    assert result.transition_count == 1


@pytest.mark.parametrize("algorithm", [RandomAlgorithm, GreedyAlgorithm])
def test_cache_changes_physical_work_but_not_budget_or_trajectory(tmp_path, algorithm):
    cached, warm, q1, t1 = run_graph(tmp_path, algorithm(), budget=9)
    uncached, cold, q2, t2 = run_graph(tmp_path, algorithm(), budget=9, cache=False)
    assert asdict(cached) == asdict(uncached)
    assert q1 == q2 and t1 == t2
    assert len(warm.objective_calls) < len(cold.objective_calls)
    assert len(warm.expansion_calls) < len(cold.expansion_calls)
    assert len(cold.objective_calls) == uncached.objective_queries + 1


def test_zero_budget_and_dead_end_only_evaluate_the_start(tmp_path):
    zero, pool, queries, _ = run_graph(tmp_path, GreedyAlgorithm(), budget=0)
    assert zero.objective_queries == zero.transition_count == 0
    assert not pool.expansion_calls and len(queries) == 1
    dead, _, _, _ = run_graph(tmp_path, GreedyAlgorithm(), adjacency=[[]], values=[3.0], budget=10)
    assert dead.termination_reason == "no_neighbors"
    assert dead.objective_queries == dead.transition_count == 0


def test_best_includes_evaluated_candidates_even_if_algorithm_does_not_select_them(tmp_path):
    class ProbeAlgorithm:
        name = "probe"

        def select_action(self, context):
            context.evaluate_action(context.actions[0])
            return context.evaluate_action(context.actions[1])

    result, pool, _, transitions = run_graph(tmp_path, ProbeAlgorithm(), values=[1.0, 20.0, 2.0, 3.0], budget=1)
    assert result.best_state_key == pool.states[1].key
    assert transitions[-1]["state_key"] == pool.states[2].key
    assert result.objective_queries == 2


def test_each_logical_query_is_charged_even_within_one_round(tmp_path):
    class RepeatAlgorithm:
        name = "repeat"

        def select_action(self, context):
            context.evaluate_action(context.actions[0])
            return context.evaluate_action(context.actions[0])

    result, pool, _, _ = run_graph(tmp_path, RepeatAlgorithm(), budget=1)
    assert result.objective_queries == 2 and result.transition_count == 1
    assert len(pool.objective_calls) == 2  # Start and one cached candidate.


@pytest.mark.parametrize("invalid", [math.nan, math.inf, 0.0, -1.0])
@pytest.mark.parametrize("algorithm", [GreedyAlgorithm, BestFirstAlgorithm, BeamSearchAlgorithm])
def test_invalid_kcup_objective_has_rollout_context(tmp_path, invalid, algorithm):
    with pytest.raises(RuntimeError, match=f"algorithm={algorithm.name}, polytope=0, start=0, queries=1"):
        run_graph(tmp_path, algorithm(), values=[1.0, invalid, 2.0, 3.0])


def test_best_first_returns_to_other_frontier_branches_after_dead_end(tmp_path):
    result, pool, queries, transitions = run_graph(
        tmp_path, BestFirstAlgorithm(), adjacency=[[1, 2], [0], [3], []],
        values=[1.0, 10.0, 9.0, 20.0], budget=3,
    )
    assert pool.expansion_calls == [0, 1, 2]
    assert result.best_state_key == pool.states[3].key and result.best_objective == 20.0
    assert result.objective_queries == result.expansion_count == 3
    assert result.transition_count == 0 and transitions == []
    assert [event["round_queries"] for event in pool.expansion_events] == [2, 0, 1]
    assert queries[-1]["source_key"] == pool.states[2].key
    assert [event["depth"] for event in pool.expansion_events] == [0, 1, 1]


def test_best_first_uses_global_priority_and_beam_waits_for_the_next_layer(tmp_path):
    adjacency = [[1, 2], [3], [4], [], []]
    values = [1.0, 10.0, 9.0, 100.0, 8.0]
    _, best_pool, _, _ = run_graph(tmp_path, BestFirstAlgorithm(), adjacency=adjacency, values=values, budget=10)
    _, beam_pool, _, _ = run_graph(tmp_path, BeamSearchAlgorithm(2), adjacency=adjacency, values=values, budget=10)
    assert best_pool.expansion_calls == [0, 1, 3, 2, 4]
    assert beam_pool.expansion_calls == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm])
def test_frontier_deduplicates_self_loops_cycles_and_shared_children(tmp_path, algorithm):
    result, pool, queries, _ = run_graph(
        tmp_path, algorithm(), adjacency=[[0, 1, 1, 2], [0, 2, 3], [0, 3], [1]],
        values=[10.0, 5.0, 5.0, 2.0], budget=20,
    )
    assert pool.objective_calls == [0, 1, 2, 3]
    assert len({event["state_key"] for event in queries}) == len(queries) == 4
    assert pool.expansion_calls == [0, 1, 2, 3]  # Ties follow first discovery.
    assert result.objective_queries == 3 and result.expansion_count == 4
    assert result.termination_reason == "frontier_exhausted"


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm])
def test_frontier_respects_minimization(tmp_path, algorithm):
    result, pool, _, _ = run_graph(tmp_path, algorithm(), goal="min", budget=20)
    assert pool.expansion_calls == [0, 3, 1, 2]
    assert result.best_objective == 2.0 and result.best_state_key == pool.states[3].key


def test_beam_keeps_layer_top_k_and_never_reintroduces_pruned_states(tmp_path):
    # State 3 is queried at depth 1, pruned, then encountered again from state 1.
    # State 4 would only be reachable by expanding that pruned state.
    result, pool, queries, _ = run_graph(
        tmp_path, BeamSearchAlgorithm(2), adjacency=[[1, 2, 3], [3], [], [4], []],
        values=[1.0, 10.0, 9.0, 8.0, 100.0], budget=20,
    )
    assert pool.expansion_calls == [0, 1, 2]
    assert pool.objective_calls == [0, 1, 2, 3]
    assert result.best_objective == 10.0 and result.objective_queries == 3
    assert len(queries) == 4 and result.termination_reason == "frontier_exhausted"


def test_beam_completes_one_parent_when_budget_runs_out_mid_layer(tmp_path):
    result, pool, queries, _ = run_graph(
        tmp_path, BeamSearchAlgorithm(2), adjacency=[[1, 2], [3, 4, 5], [6], [], [], [], []],
        values=[1.0, 10.0, 9.0, 3.0, 4.0, 30.0, 100.0], budget=3,
    )
    assert pool.expansion_calls == [0, 1]  # Do not process the other depth-1 parent.
    assert result.objective_queries == 5 and result.budget_overshoot == 2
    assert result.best_state_key == pool.states[5].key and result.best_objective == 30.0
    assert len(queries) == 6 and result.termination_reason == "budget_exhausted"


def test_beam_width_does_not_bound_neighbor_query_count(tmp_path):
    result, pool, _, _ = run_graph(
        tmp_path, BeamSearchAlgorithm(2), adjacency=[[1, 2, 3, 4, 5, 6], [], [], [], [], [], []],
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], budget=1,
    )
    assert result.objective_queries == 6 and result.budget_overshoot == 5 > 2**2
    assert pool.expansion_calls == [0] and result.best_objective == 7.0


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm])
def test_frontier_zero_budget_and_dead_end(tmp_path, algorithm):
    zero, pool, _, _ = run_graph(tmp_path, algorithm(), budget=0)
    assert zero.expansion_count == zero.objective_queries == zero.transition_count == 0
    assert not pool.expansion_calls and zero.termination_reason == "budget_exhausted"
    dead, pool, _, _ = run_graph(tmp_path, algorithm(), adjacency=[[]], values=[3.0], budget=10)
    assert dead.expansion_count == 1 and dead.objective_queries == 0
    assert dead.termination_reason == "frontier_exhausted"
    assert pool.expansion_events[0]["candidate_count"] == 0


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm])
def test_frontier_cache_eviction_preserves_pending_nodes_and_all_events(tmp_path, algorithm):
    args = dict(adjacency=[[1, 2], [0, 3], [0, 3], [1]], values=[1.0, 10.0, 9.0, 20.0], budget=10)
    cached, warm, q1, t1 = run_graph(tmp_path, algorithm(), **args)
    uncached, cold, q2, t2 = run_graph(tmp_path, algorithm(), cache=False, **args)
    assert asdict(cached) == asdict(uncached)
    assert q1 == q2 and t1 == t2 == []
    assert warm.expansion_events == cold.expansion_events
    assert len(cold.objective_calls) == uncached.objective_queries + 1


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm])
def test_frontier_seen_set_is_local_to_each_start_even_on_warm_engine(tmp_path, algorithm):
    pool = GraphPool([[1, 2], [0, 2], [0, 1]], [1.0, 2.0, 3.0])
    source = pool.states[0]
    engine = CYRandomRolloutEngine(
        base_states={source.key: source}, initial_states=[source], polytope_by_index={0: source.configuration},
        neighbor_mode="two_neighbors", include_points_interior_to_facets=False,
        reward_function=lambda a, b: 0.0, transition_pool=pool,
        history_path=str(tmp_path / "shared.sqlite3"),
    )
    try:
        for start, state in enumerate(pool.states[:2]):
            result = run_rollout(
                state, algorithm(), engine, objective_function=lambda state: engine.objective_value(state, "max_kcup"),
                objective_goal="max", objective_budget=20, seed=4, start_index=start,
            )
            assert result.objective_queries == 2 and result.expansion_count == 3
            assert result.best_objective == 3.0
        assert len(pool.objective_calls) == 3  # Second search is warm, but still charges its queries.
    finally:
        engine.close()


@pytest.mark.parametrize("algorithm", [RandomAlgorithm, GreedyAlgorithm, BestFirstAlgorithm, BeamSearchAlgorithm])
def test_result_records_expansions_and_omits_final_fields(tmp_path, algorithm):
    result, pool, _, _ = run_graph(tmp_path, algorithm(), budget=5)
    assert result.expansion_count == len(pool.expansion_events)
    assert not {"final_state_key", "final_objective"} & asdict(result).keys()
    assert [row["expansion_index"] for row in pool.expansion_events] == list(range(1, result.expansion_count + 1))
