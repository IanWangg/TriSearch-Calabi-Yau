"""2-face evaluation identity, independent of full-FRST geometry identity."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.algorithm import (
    BestFirstAlgorithm, BeamSearchAlgorithm, GreedyAlgorithm, RandomAlgorithm,
    RLPolicyBeamSearch, RLStochasticPolicy, RLValueBeamSearch, RLValueBestFirst,
)
from eval.config import EvaluationSpec
from eval.rollout import run_rollout
from eval.results.plotting import combine_comparisons, read_comparison
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices, two_face_state_key
from scripts.eval_cy import parse_args
from test_eval_rollout import GraphPool
from test_eval_rl import GraphPolicy
from test_eval_comparison import make_run


class FaceGraphPool(GraphPool):
    def __init__(self):
        super().__init__([[1, 2, 3, 4], [2, 3, 4], [0, 1, 4, 5], [0, 1, 4, 5],
                          [3, 6, 5], [0, 4], [3, 5]], [1., 1., 2., 2., 3., 4., 3.])
        points = tuple((index, 0, 0, 0) for index in range(20))
        configuration = CYPointConfiguration(0, points, points, tuple(range(20)), False, ((1, 2, 4, 5, 6, 7),))
        self.states = [CyStateRecord(configuration, frozenset({(0, 1, 2, label, 10 + index)}),
                                     "two_neighbors", True, True)
                       for index, label in enumerate((4, 4, 5, 5, 6, 7, 6))]
        self.by_simplices = {canonical_simplices(state.simplices): index for index, state in enumerate(self.states)}


@contextmanager
def face_engine(cache=True, two_face_state=True):
    pool = FaceGraphPool()
    engine = CYRandomRolloutEngine(
        base_states={pool.states[0].key: pool.states[0]}, initial_states=[pool.states[0]],
        polytope_by_index={0: pool.states[0].configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, transition_pool=pool,
        reward_function=lambda a, b: 0.0,
        state_cache_mode="lru" if cache else "none", cache_budget_bytes=1000000 if cache else 0,
        two_face_state=two_face_state,
    )
    try:
        yield engine, pool
    finally:
        engine.close()


def face_rollout(algorithm, *, cache=True, enabled=True, budget=20, omit_option=False, starts=(0,)):
    with face_engine(cache, enabled) as (engine, pool):
        policy = GraphPolicy(pool)
        queries, expansions, transitions, results = [], [], [], []
        for start_index, index in enumerate(starts):
            results.append(run_rollout(
                pool.states[index], algorithm(), engine,
                objective_function=lambda state: engine.objective_value(state, "max_kcup"),
                batch_objective_function=lambda states: engine.objective_values(states, "max_kcup"),
                objective_goal="max", objective_budget=budget, seed=10 + start_index,
                start_index=start_index, policy=policy,
                on_query=queries.append, on_expansion=expansions.append, on_transition=transitions.append,
                **({} if omit_option else {"two_face_state": enabled}),
            ))
        return results, pool, policy, queries, expansions, transitions


def test_two_face_key_uses_ambient_faces_not_arbitrary_simplex_subsets():
    pool = FaceGraphPool()
    first, equivalent, different = pool.states[:3]
    assert first.key != equivalent.key and first.two_face_key == equivalent.two_face_key
    assert first == replace(first)  # Computing the cached identity does not change the descriptor's equality.
    assert first.two_face_key != different.two_face_key
    changed_order = replace(first.configuration, two_face_labels=((7, 6, 5, 4, 2, 1),))
    assert two_face_state_key(changed_order, reversed(tuple(first.simplices))) == first.two_face_key
    assert two_face_state_key(replace(first.configuration, index=1), first.simplices) != first.two_face_key
    with pytest.raises(ValueError, match="ambient 2-face labels"):
        two_face_state_key(replace(first.configuration, two_face_labels=()), first.simplices)


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm,
    lambda: RLValueBeamSearch(beam_width=2, policy_proposal_count=-1),
    lambda: RLValueBestFirst(policy_proposal_count=-1), lambda: RLPolicyBeamSearch(2), RLStochasticPolicy])
def test_two_face_search_deduplicates_aliases_cycles_shared_children_and_preserves_policy_inputs(algorithm):
    warm = face_rollout(algorithm, starts=(0, 1))
    cold = face_rollout(algorithm, cache=False, starts=(0, 1))
    assert warm[0] == cold[0] and warm[3:] == cold[3:]
    results, pool, policy, queries, _, _ = warm
    assert results[0].initial_state_key != results[1].initial_state_key
    assert results[0].initial_evaluation_state_key == results[1].initial_evaluation_state_key
    for index, result in enumerate(results):
        events = [event for event in queries if event["start_index"] == index]
        assert result.objective_queries == 3
        assert len({event["evaluation_state_key"] for event in events}) == len(events) == 4
        assert result.best_objective == 4.
        assert result.best_evaluation_state_key == pool.states[5].two_face_key
        assert all(event["best_evaluation_state_key"].startswith("two_face|") for event in events)
    assert all(len(next(iter(state.simplices))) == 5 for state in pool.states)
    assert len(cold[1].objective_calls) == 8  # No cross-start deduplication or cache when disabled.
    assert len(pool.objective_calls) == 4
    if policy.value_batches:
        assert 1 not in [index for batch in policy.value_batches for index in batch]  # Start alias is never a child.


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, BeamSearchAlgorithm,
    lambda: RLValueBeamSearch(beam_width=2, policy_proposal_count=-1),
    lambda: RLValueBestFirst(policy_proposal_count=-1)])
def test_two_face_parent_budget_and_default_false_compatibility(algorithm):
    zero, _, _, initial_queries, initial_expansions, _ = face_rollout(algorithm, budget=0)
    assert zero[0].objective_queries == zero[0].expansion_count == 0
    assert len(initial_queries) == 1 and not initial_expansions
    result, _, _, queries, _, _ = face_rollout(algorithm, budget=1)
    assert result[0].objective_queries == 2 and result[0].budget_overshoot == 1
    assert len(queries) == 3
    off = face_rollout(algorithm, enabled=False)
    default = face_rollout(algorithm, enabled=False, omit_option=True)
    assert off[0] == default[0] and off[3:] == default[3:]
    assert off[0][0].objective_queries > result[0].objective_queries
    assert "initial_evaluation_state_key" not in asdict(off[0][0])
    assert all("evaluation_state_key" not in event for event in off[3])


@pytest.mark.parametrize("algorithm,budget,queries", [(RandomAlgorithm, 3, 3), (GreedyAlgorithm, 1, 4)])
def test_two_face_walks_keep_repeated_query_charging(algorithm, budget, queries):
    warm, cold = face_rollout(algorithm, budget=budget), face_rollout(algorithm, cache=False, budget=budget)
    assert warm[0] == cold[0] and warm[3:] == cold[3:]
    assert warm[0][0].objective_queries == queries
    assert len(warm[3]) == queries + 1


def test_two_face_scalar_and_batch_objective_caches_merge_only_physical_work():
    with face_engine() as (engine, pool):
        assert list(engine.objective_values(pool.states[:4], "max_kcup")) == [1., 1., 2., 2.]
        assert pool.objective_calls == [0, 2]
        assert engine.objective_value(pool.states[1], "max_kcup") == 1.
        assert pool.objective_calls == [0, 2]
        with pytest.raises(ValueError, match="only max_kcup"):
            engine.objective_value(pool.states[0], "max_tri")
    with face_engine(cache=False) as (engine, pool):
        assert list(engine.objective_values(pool.states[:4], "max_kcup")) == [1., 1., 2., 2.]
        assert pool.objective_calls == [0, 1, 2, 3]


@pytest.mark.parametrize("algorithm", [BestFirstAlgorithm, RLValueBestFirst])
def test_two_face_rollout_rejects_mismatched_engine_before_queries(algorithm):
    with face_engine(two_face_state=False) as (engine, pool):
        with pytest.raises(ValueError, match="must match"):
            run_rollout(pool.states[0], algorithm(), engine,
                        objective_function=lambda state: engine.objective_value(state, "max_kcup"),
                        objective_goal="max", objective_budget=1, seed=0,
                        policy=GraphPolicy(pool), two_face_state=True)
        assert not pool.objective_calls


def test_two_face_config_cli_and_shared_setup_compatibility(tmp_path):
    spec = EvaluationSpec(1, 12, 2, 3)
    enabled = replace(spec, two_face_state=True)
    assert spec.setup_parameters() == enabled.setup_parameters()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(enabled.to_dict()))
    assert parse_args(["--config", str(config)]).two_face_state
    assert not parse_args(["--config", str(config), "--no_two_face_state"]).two_face_state
    config.write_text(json.dumps(spec.to_dict()))
    assert parse_args(["--config", str(config), "--two_face_state"]).two_face_state
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            replace(spec, two_face_state=value)
    with pytest.raises(ValueError, match="only max_kcup"):
        replace(enabled, reward_function="max_tri")


def test_two_face_reader_metric_validation_and_combination_compatibility(tmp_path):
    def enabled_run(path, algorithm):
        make_run(path, algorithms=(algorithm,))
        config = json.loads((path / "config.json").read_text())
        config["spec"]["two_face_state"] = True
        (path / "config.json").write_text(json.dumps(config))
        for filename in ("queries.jsonl", "rollouts.jsonl"):
            rows = [json.loads(line) for line in (path / filename).read_text().splitlines()]
            for row in rows:
                if filename == "queries.jsonl":
                    row["evaluation_state_key"] = "two_face|" + row["state_key"]
                else:
                    row["initial_evaluation_state_key"] = "two_face|" + row["initial_state_key"]
                row["best_evaluation_state_key"] = "two_face|" + row["best_state_key"]
            (path / filename).write_text("\n".join(map(json.dumps, rows)) + "\n")

    first, second = tmp_path / "first", tmp_path / "second"
    enabled_run(first, "random")
    enabled_run(second, "greedy")
    assert len(combine_comparisons([first, second]).rollouts) == 8
    events = [json.loads(line) for line in (second / "queries.jsonl").read_text().splitlines()]
    # A non-initial child pretends to have a different-volume initial state's class.
    candidate = next(row for row in events if row["query_index"] == 1)
    candidate["evaluation_state_key"] = "two_face|initial_1_1"
    candidate["best_evaluation_state_key"] = candidate["evaluation_state_key"]
    (second / "queries.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    with pytest.raises(ValueError, match="Inconsistent max_kcup metric"):
        read_comparison(second)

    third = tmp_path / "third"
    make_run(third, algorithms=("greedy",))
    with pytest.raises(ValueError, match="two_face_state"):
        combine_comparisons([first, third])
    # Historical absence is compatible with the explicit default false.
    fourth = tmp_path / "fourth"
    make_run(fourth, algorithms=("random",))
    config = json.loads((fourth / "config.json").read_text())
    del config["spec"]["two_face_state"]
    (fourth / "config.json").write_text(json.dumps(config))
    assert not combine_comparisons([third, fourth]).spec["two_face_state"]


def test_two_face_ga_retains_repeated_fitness_queries():
    pytest.importorskip("cyopt")
    from eval.algorithm.cyopt_ga import CyoptGAAlgorithm

    def run(cache):
        with face_engine(cache=cache) as (engine, pool):
            representatives = (pool.states[0], pool.states[2], pool.states[4], pool.states[5])
            encoding = SimpleNamespace(bounds=((0, 3),), initial_dna={pool.states[0].key: (0,)},
                                       metadata={"codebook_sha256": "faces"},
                                       decode=lambda dna, initial: representatives[dna[0]])
            events = []
            result = run_rollout(pool.states[0], CyoptGAAlgorithm(population_size=4, encoding=encoding), engine,
                                 objective_function=lambda state: engine.objective_value(state, "max_kcup"),
                                 objective_goal="max", objective_budget=12, seed=10, two_face_state=True,
                                 on_query=events.append)
            return result, events, len(pool.objective_calls)
    warm, cold = run(True), run(False)
    assert warm[:2] == cold[:2]
    assert warm[0].objective_queries == 12 and warm[0].budget_overshoot == 0
    assert cold[2] == 13 and warm[2] <= 4
    assert len({event["evaluation_state_key"] for event in warm[1]}) < len(warm[1])


def test_real_two_face_worker_metric_invariance_and_pipeline_pairing(tmp_path):
    from core.cytools_config import configure_cytools
    from eval.pipeline import run_evaluation
    from eval.setup import EvaluationSetup
    from mdp.cy_geometry_worker import _objective, _OBJECTIVES, _get_polytope

    configure_cytools()
    from cytools import Polytope

    fixture = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    row = json.loads(fixture.read_text().splitlines()[0])
    polytope = Polytope(row["vertices"])
    source = polytope.triangulate(simplices=row["frst_list"][0]["simplices"],
                                include_points_interior_to_facets=False, check_input_simplices=False)
    faces = tuple(sorted(tuple(sorted(map(int, face.labels))) for face in polytope.faces(2)))
    configuration = CYPointConfiguration(row["polytope_index"], tuple(map(tuple, row["vertices"])),
        tuple(tuple(map(int, point)) for point in polytope.points()), tuple(map(int, polytope.labels)), False, faces)
    first = CyStateRecord(configuration, frozenset(map(tuple, source.simplices())), "two_neighbors", True, True)
    equivalent = next(tri for tri in source.neighbor_triangulations(only_fine=True, only_regular=True, only_star=True)
                      if two_face_state_key(configuration, tri.simplices()) == first.two_face_key)
    second = CyStateRecord(configuration, frozenset(map(tuple, equivalent.simplices())), "two_neighbors", True, True)
    assert first.key != second.key and first.two_face_key == second.two_face_key
    expected = tuple((tuple(sorted(map(int, restriction.labels))), canonical_simplices(restriction.simplices()))
                     for restriction in source.restrict(as_poly=True))
    assert first.two_face_key == f"two_face|{configuration.index}:{tuple(sorted(expected))}"

    # Real KCUP solves on both full representatives, then the worker cache on the quotient key.
    actual_polytope = _get_polytope(configuration)
    _OBJECTIVES.clear()
    values = [_objective({"reward_name": "max_kcup"}, configuration, state.to_payload(), actual_polytope)
              for state in (first, second)]
    assert values[0] == pytest.approx(values[1], rel=1e-6)
    _OBJECTIVES.clear()
    cached = [_objective({"reward_name": "max_kcup", "two_face_state": True}, configuration,
                         state.to_payload(), actual_polytope) for state in (first, second)]
    assert cached[0] == cached[1] and len(_OBJECTIVES) == 1

    row["frst_list"] = [dict(simplices=canonical_simplices(state.simplices), triangulation_list=[]) for state in (first, second)]
    spec = EvaluationSpec(1, 12, 2, 1, algorithms=("best_first", "rl_value_best_first", "cyopt_ga"),
                          two_face_state=True, ga_population_size=4)
    setup = EvaluationSetup([row], {"parameters": spec.setup_parameters(), "source_metadata": {"fixture": fixture.name}},
                            tmp_path / "setup")

    class FullFRSTPolicy:
        def score_actions(self, states, actions):
            assert all(len(next(iter(state.simplices))) == 5 for state in states)
            return [np.full(len(row), -np.log(len(row))) for row in actions]

        def score_values(self, states):
            assert all(len(next(iter(state.simplices))) == 5 for state in states)
            return [0.] * len(states)

    warm = run_evaluation(spec, setup, output_dir=tmp_path / "warm", policy=FullFRSTPolicy())
    cold = run_evaluation(replace(spec, cache_states=False), setup, output_dir=tmp_path / "cold", policy=FullFRSTPolicy())
    assert warm.rollouts == cold.rollouts
    for output in (warm.output_dir, cold.output_dir):
        data = read_comparison(output)
        assert len(data.rollouts) == 6 and data.spec["two_face_state"]
        assert all(row["initial_evaluation_state_key"] == first.two_face_key for row in data.rollouts.values())
        config = json.loads((output / "config.json").read_text())
        assert config["state_representation"] == "two_face_restrictions"
        for algorithm in spec.algorithms:
            queries = [json.loads(line) for line in (output / "queries.jsonl").read_text().splitlines()
                       if json.loads(line)["algorithm"] == algorithm]
            assert len({row["source_key"] for row in queries if row["is_initial"]}) == 2
    assert not json.loads((cold.output_dir / "summary.json").read_text())["runtime_stats"]["best_first"]["objective_cache_bytes"]
