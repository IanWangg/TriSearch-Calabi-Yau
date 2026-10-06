"""Contracts for the upstream GA adapter, plus real DNA/FRST round trips."""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import pytest

pytest.importorskip("cyopt")

from eval.algorithm.cyopt_ga import CyoptGAAlgorithm
from eval.config import EvaluationSpec
from mdp.cy_state_record import CyStateRecord
from test_eval_rollout import run_graph


class ToyEncoding:
    bounds = ((0, 3),)

    def __init__(self, *, reject=False, broken=False):
        self.initial_dna = defaultdict(lambda: (0,))
        self.reject, self.broken = reject, broken
        self.metadata = {"codebook_sha256": "toy"}
        self.decode_calls = 0

    def decode(self, dna, initial):
        self.decode_calls += 1
        if self.broken:
            raise RuntimeError("unexpected decoder failure")
        if self.reject and dna != (0,):
            return None
        return CyStateRecord(initial.configuration, frozenset({(0, 1, 2, 3, dna[0] + 4)}),
                             "two_neighbors", True, True)


def toy_algorithm(**kwargs):
    return CyoptGAAlgorithm(population_size=4, encoding=ToyEncoding(**kwargs), max_stalled_generations=2)


@pytest.mark.parametrize("budget", [0, 1, 3, 13])
def test_ga_seed_budget_cache_and_actual_upstream_generations(tmp_path, monkeypatch, budget):
    from cyopt import GA

    generations = []
    upstream = GA._step

    def step(self, index):
        generations.append(index)
        return upstream(self, index)

    monkeypatch.setattr(GA, "_step", step)
    warm_algorithm, cold_algorithm = toy_algorithm(), toy_algorithm()
    warm = run_graph(tmp_path, warm_algorithm, budget=budget, values=[1., 2., 3., 4.])
    cold = run_graph(tmp_path, cold_algorithm, budget=budget, values=[1., 2., 3., 4.], cache=False)
    result, pool, queries, moves = warm
    assert result == cold[0] and queries == cold[2]
    assert pool.expansion_events == cold[1].expansion_events
    assert result.objective_queries == budget and result.budget_overshoot == 0
    assert result.transition_count == 0 and not moves and not pool.expansion_calls
    assert len(queries) == budget + 1
    assert result.initial_state_key == pool.states[0].key
    assert result.best_objective == max(row["objective"] for row in queries)
    assert len(cold[1].objective_calls) == budget + 1
    assert all(row["is_initial"] == (q == 0) for q, row in enumerate(queries))
    assert all(row["source_key"] is None and row["action"] is None and row["depth"] is None
               for row in queries[1:])
    if budget > 3:
        assert generations and len(pool.objective_calls) < len(cold[1].objective_calls)
        assert warm_algorithm.encoding.decode_calls < cold_algorithm.encoding.decode_calls


def test_ga_minimize_and_small_space_elite_clamp(tmp_path):
    algorithm = toy_algorithm()
    algorithm.elitism = 3
    algorithm.encoding.bounds = ((0, 1),)
    result, pool, _, _ = run_graph(tmp_path, algorithm, budget=8, goal="min", values=[10., 2., 5., 4.])
    assert result.best_objective == 2. and result.objective_queries == 8
    assert all(event["effective_elitism"] == 1 for event in pool.expansion_events)


def test_ga_rejects_only_explicit_infeasibility_and_stops_stalled_search(tmp_path):
    result, pool, queries, _ = run_graph(tmp_path, toy_algorithm(reject=True), budget=100)
    assert result.termination_reason == "no_feasible_offspring"
    assert len(queries) == result.objective_queries + 1
    assert all(row["objective"] == result.initial_objective for row in queries)
    rejected = [item for event in pool.expansion_events for item in event["rejected_candidates"]]
    assert rejected and all(item["reason"] == "non_solid_cone" for item in rejected)
    with pytest.raises(RuntimeError, match="unexpected decoder failure"):
        run_graph(tmp_path, toy_algorithm(broken=True), budget=3)


def test_ga_singleton_has_only_shared_initial_objective(tmp_path):
    algorithm = toy_algorithm()
    algorithm.encoding.bounds = ()
    result, _, queries, _ = run_graph(tmp_path, algorithm, budget=100)
    assert result.termination_reason == "dna_space_singleton"
    assert result.objective_queries == result.expansion_count == 0
    assert len(queries) == 1


@pytest.mark.parametrize("change", [dict(ga_population_size=3), dict(ga_elitism=0), dict(ga_elitism=50),
                                    dict(ga_mutation_rate=1.1), dict(ga_face_samples=0),
                                    dict(ga_max_stalled_generations=0), dict(reward_function="min_tri")])
def test_ga_config_rejects_invalid_settings(change):
    with pytest.raises(ValueError):
        EvaluationSpec(1, 12, 1, 3, algorithms=("cyopt_ga",), **change)


def test_real_cyopt_face_samples_preserve_ambient_labels():
    from cytools import Polytope
    from eval.algorithm.cyopt_ga import _label_face_triangulations
    from mdp.cy_state_record import canonical_simplices

    fixture = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    polytope = Polytope(json.loads(fixture.read_text().splitlines()[0])["vertices"])
    for face in polytope.faces(2):
        # grow_frt constructs precisely this local 2D polytope internally.
        local = Polytope(face.as_poly().points(optimal=True)).triangulate(
            make_star=False, include_points_interior_to_facets=True,
        )
        restored = _label_face_triangulations(face, [local])[0]
        by_coordinate = dict(zip((tuple(point) for point in face.points(optimal=True)), face.labels))
        labels = {label: by_coordinate[tuple(point)] for label, point in zip(local.labels, local.points())}
        expected = canonical_simplices([[labels[label] for label in simplex] for simplex in local.simplices()])
        assert canonical_simplices(restored.simplices()) == expected
        assert {label for simplex in restored.simplices() for label in simplex} == set(face.labels)
        assert restored.is_fine() and restored.is_regular()
        assert _label_face_triangulations(face, [restored])[0] is restored


def test_real_cyopt_encoding_queries_cache_equivalence_and_paired_start(tmp_path):
    from eval.pipeline import run_evaluation
    from eval.setup import EvaluationSetup, save_eval_setup
    from eval.results.plotting import read_comparison

    fixture = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    row = json.loads(fixture.read_text().splitlines()[0])
    spec = EvaluationSpec(1, 12, 1, 9, algorithms=("cyopt_ga",), ga_population_size=4)
    setup = EvaluationSetup([row], {"parameters": spec.setup_parameters(), "source_metadata": {"fixture": str(fixture)}})
    save_eval_setup(setup, tmp_path / "setup")
    warm = run_evaluation(spec, setup, output_dir=tmp_path / "warm")
    cold = run_evaluation(replace(spec, cache_states=False), setup, output_dir=tmp_path / "cold")
    assert warm.rollouts == cold.rollouts
    assert warm.rollouts[0].objective_queries == 9
    assert (warm.output_dir / "queries.jsonl").read_text() == (cold.output_dir / "queries.jsonl").read_text()
    assert (warm.output_dir / "expansions.jsonl").read_text() == (cold.output_dir / "expansions.jsonl").read_text()
    read_comparison(warm.output_dir)
    first = json.loads(next((warm.output_dir / "cyopt_encoding").glob("*.json")).read_text())
    second = json.loads(next((cold.output_dir / "cyopt_encoding").glob("*.json")).read_text())
    assert first["codebook_sha256"] == second["codebook_sha256"]
    assert first["starts"] == second["starts"]
    stats = json.loads((cold.output_dir / "summary.json").read_text())["runtime_stats"]["cyopt_ga"]
    assert stats["objective_cache_bytes"] == stats["hot_state_bytes"] == stats["resident_graph_nodes"] == 0
