"""Observation is inert; partial class expansion has a reproducible graph witness."""
from dataclasses import asdict
import copy

import pytest

from eval.algorithm import BestFirstAlgorithm, RLStochasticPolicy, RLValueBeamSearch, RLValueBestFirst
from eval.batched_rollout import run_batched_rollouts
from eval.rollout import run_rollout
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices
from test_eval_rollout import GraphPool
from test_eval_rl import GraphPolicy


class AliasPool(GraphPool):
    def __init__(self, adjacency=None, objectives=None, classes=None):
        super().__init__(adjacency or [[2,3,4,5,6,7], [2,3,4,5,6,7], [1], [], [], [], [], []],
                         objectives or [1,1,2,2,2,2,10,100])
        classes = classes or [4,4,5,6,7,8,9,10]
        points = tuple((i,0,0,0) for i in range(40))
        configuration = CYPointConfiguration(0, points, points, tuple(range(40)), False,
                                             ((1,2,*sorted(set(classes))),))
        self.states = [CyStateRecord(configuration, frozenset({(0,1,2,label,20+i)}),
                                     "two_neighbors", True, True) for i,label in enumerate(classes)]
        self.by_simplices = {canonical_simplices(s.simplices):i for i,s in enumerate(self.states)}


class RandomWithRngTrace(RLStochasticPolicy):
    def __init__(self):
        super().__init__()
        self.rng_states = []

    def propose(self, next_keys, log_probabilities, seen, rng, reserved=()):
        result = super().propose(next_keys, log_probabilities, seen, rng, reserved)
        self.rng_states.append(copy.deepcopy(rng.bit_generator.state))
        return result


def run_alias(factory, *, enabled=False, observer=False, pool=None, critic=None, start=0):
    pool = pool or AliasPool()
    engine = CYRandomRolloutEngine(
        base_states={pool.states[start].key:pool.states[start]}, initial_states=[pool.states[start]],
        polytope_by_index={0:pool.states[0].configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, transition_pool=pool, reward_function=lambda a,b:0.,
        state_cache_mode="none", cache_budget_bytes=0, two_face_state=enabled)
    policy = GraphPolicy(pool, values=critic or {0:5,1:5,2:10})
    algorithm = factory()
    queries, expansions, trace = [], [], []
    kwargs = dict(objective_function=lambda s:engine.objective_value(s,"max_kcup"),
                  batch_objective_function=lambda ss:engine.objective_values(ss,"max_kcup"),
                  objective_goal="max", objective_budget=20, policy=policy,
                  on_query=queries.append, on_expansion=expansions.append, two_face_state=enabled)
    def observe(event):
        trace.append(copy.deepcopy(event))
        event.clear()  # The observer cannot mutate live search objects through snapshots.
    try:
        if isinstance(algorithm, BestFirstAlgorithm):
            result = run_rollout(pool.states[start], algorithm, engine, seed=7, start_index=0, **kwargs)
        else:
            result = run_batched_rollouts([pool.states[start]], [algorithm], engine,
                seeds=[7], start_indices=[0], search_observer=observe if observer else None, **kwargs)[0]
        return dict(result=asdict(result),queries=queries,expansions=expansions,
                    objective_calls=pool.objective_calls,geometry_calls=pool.expansion_calls,
                    action_batches=policy.action_batches,value_batches=policy.value_batches,
                    rng_states=getattr(algorithm,"rng_states",None)), trace, pool
    finally:
        engine.close()


@pytest.mark.parametrize("factory", [lambda:RLValueBeamSearch(4,4), RLValueBestFirst, RandomWithRngTrace])
@pytest.mark.parametrize("enabled", [False,True])
def test_observer_preserves_results_events_rng_and_all_call_counts(factory,enabled):
    plain, _, _ = run_alias(factory,enabled=enabled)
    watched, trace, _ = run_alias(factory,enabled=enabled,observer=True)
    assert watched == plain
    assert trace and any(e["event"] == "proposal" for e in trace)
    assert all(isinstance(e,dict) for e in trace)


@pytest.mark.parametrize("factory", [lambda:RLValueBeamSearch(4,4), RLValueBestFirst])
def test_alias_reexpansion_reveals_policy_tail_with_identical_class_model_and_neighbors(factory):
    old, trace, pool = run_alias(factory,observer=True)
    new, _, _ = run_alias(factory,enabled=True)
    assert pool.states[0].two_face_key == pool.states[1].two_face_key
    assert pool.adjacency[0] == pool.adjacency[1]
    assert old["result"]["best_objective"] == 100
    assert new["result"]["best_objective"] == 2
    proposal = next(e for e in trace if e["event"] == "proposal" and e["parent_key"] == pool.states[1].key)
    assert [c["state_key"] for c in proposal["candidates"] if c["selected"]] == [pool.states[i].key for i in (6,7)]
    assert sum(c["seen"] for c in proposal["candidates"]) == 4
    # All-neighbor BeFS sees the good tail at the first representative, in either mode.
    for enabled in (False,True):
        control, _, _ = run_alias(BestFirstAlgorithm,enabled=enabled)
        assert control["result"]["best_objective"] == 100


def test_class_equal_children_with_different_critic_change_native_frontier_decision():
    def pool():
        return AliasPool([[2,4],[3,4],[],[],[]], [1,1,2,2,2], [4,4,5,5,6])
    a, trace_a, graph = run_alias(RLValueBestFirst,pool=pool(),critic={2:10,3:-10},observer=True,start=0)
    b, trace_b, _ = run_alias(RLValueBestFirst,pool=pool(),critic={2:10,3:-10},observer=True,start=1)
    assert graph.states[2].two_face_key == graph.states[3].two_face_key
    first_a = next(e for e in trace_a if e["event"] == "frontier")["selected"][0]["state_key"]
    first_b = next(e for e in trace_b if e["event"] == "frontier")["selected"][0]["state_key"]
    assert first_a == graph.states[2].key
    assert first_b == graph.states[4].key
