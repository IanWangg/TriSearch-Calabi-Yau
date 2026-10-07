"""Real CY geometry smoke tests; network access is an explicit opt-in."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest

from eval.config import EvaluationSpec
from eval.pipeline import run_evaluation
from eval.setup import EvaluationSetup, load_eval_setup, prepare_eval_setup, save_eval_setup


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def real_setup(tmp_path):
    from core.cytools_config import configure_cytools

    configure_cytools()
    from cytools import Polytope

    fixture = Path(__file__).resolve().parents[1] / "data/cy/two_neighbors_h11_12.samples.jsonl"
    row = read_jsonl(fixture)[0]
    polytope = Polytope(row["vertices"])
    source = polytope.triangulate(simplices=row["frst_list"][0]["simplices"],
                                 include_points_interior_to_facets=False)
    neighbor = next(iter(source.neighbor_triangulations(two_neighbors=True)))
    # CYTools uses polytope labels here, just like the original sample row.
    row["frst_list"].append({"frst_index": 1, "simplices": neighbor.simplices().tolist(), "triangulation_list": []})
    spec = EvaluationSpec(1, 12, 2, 2, algorithms=("random", "greedy", "best_first", "beam_search"))
    setup = EvaluationSetup([row], {"parameters": spec.setup_parameters(), "source_metadata": {"fixture": fixture.name}})
    save_eval_setup(setup, tmp_path / "setup")
    return spec, setup


def test_real_kcup_pipeline_cache_equivalence_shared_starts_and_algorithm_order(tmp_path, real_setup):
    spec, setup = real_setup
    cached = run_evaluation(spec, setup, output_dir=tmp_path / "cached")
    off_spec = replace(spec, cache_states=False, algorithms=tuple(reversed(spec.algorithms)))
    uncached = run_evaluation(off_spec, load_eval_setup(setup.path), output_dir=tmp_path / "uncached")
    order = lambda result: (result.algorithm, result.start_index)
    assert [asdict(result) for result in sorted(cached.rollouts, key=order)] == [
        asdict(result) for result in sorted(uncached.rollouts, key=order)
    ]
    assert len(cached.rollouts) == 8
    for start in (0, 1):
        pair = [result for result in cached.rollouts if result.start_index == start]
        assert len({result.initial_state_key for result in pair}) == 1
        assert len({result.initial_objective for result in pair}) == 1
    random = [result for result in cached.rollouts if result.algorithm == "random"]
    assert all(result.objective_queries == result.transition_count == 2 for result in random)
    assert all(result.expansion_count >= 1 for result in cached.rollouts)
    assert all(result.transition_count == 0 for result in cached.rollouts
               if result.algorithm in ("best_first", "beam_search"))
    for output in (cached.output_dir, uncached.output_dir):
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "complete" and summary["num_rollouts"] == 8
        assert summary["format_version"] == 2
        assert json.loads((output / "config.json").read_text())["format_version"] == 2
        queries = read_jsonl(output / "queries.jsonl")
        assert len(queries) == summary["objective_queries"] + 8
        assert len(read_jsonl(output / "expansions.jsonl")) == summary["expansion_count"]
        assert len(read_jsonl(output / "transitions.jsonl")) == summary["transition_count"]
        rollouts = read_jsonl(output / "rollouts.jsonl")
        assert len(rollouts) == 8
        assert all("final_state_key" not in row and "final_objective" not in row for row in rollouts)
        assert not (output / "failure.json").exists()
    off_stats = json.loads((uncached.output_dir / "summary.json").read_text())["runtime_stats"]
    assert all(stats["resident_graph_nodes"] == stats["hot_state_bytes"] == stats["objective_cache_bytes"] == 0
               for stats in off_stats.values())
    # Algorithm execution order may change, but each start's complete event stream must not.
    for filename in ("queries.jsonl", "expansions.jsonl", "transitions.jsonl"):
        first = read_jsonl(cached.output_dir / filename)
        second = read_jsonl(uncached.output_dir / filename)
        for algorithm in spec.algorithms:
            for start in (0, 1):
                matches = lambda event: event["algorithm"] == algorithm and event["start_index"] == start
                assert list(filter(matches, first)) == list(filter(matches, second))


def test_pipeline_preserves_partial_results_and_closes_runtime_on_failure(monkeypatch, tmp_path, real_setup):
    import eval.pipeline as pipeline

    spec, setup = real_setup
    pools, engines = [], []
    original_pool, original_engine = pipeline.create_transition_pool, pipeline.CYRandomRolloutEngine

    def capture_pool(**kwargs):
        pool = original_pool(**kwargs)
        pools.append(pool)
        return pool

    def capture_engine(**kwargs):
        engine = original_engine(**kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setattr(pipeline, "create_transition_pool", capture_pool)
    monkeypatch.setattr(pipeline, "CYRandomRolloutEngine", capture_engine)

    class FailingAlgorithm:
        name = "failing"

        def select_action(self, context):
            context.evaluate_action(context.actions[0])
            raise RuntimeError("intentional algorithm failure")

    with pytest.raises(RuntimeError, match="intentional algorithm failure"):
        run_evaluation(replace(spec, algorithms=("failing",)), setup, output_dir=tmp_path / "failed",
                       algorithm_factories={"failing": FailingAlgorithm})
    failure = json.loads((tmp_path / "failed" / "failure.json").read_text())
    assert failure["status"] == "failed" and "polytope=13" in failure["error"]
    assert len(read_jsonl(tmp_path / "failed" / "queries.jsonl")) == 2
    expansions = read_jsonl(tmp_path / "failed" / "expansions.jsonl")
    assert len(expansions) == 1 and expansions[0]["status"] == "failed"
    assert not (tmp_path / "failed" / "summary.json").exists()
    assert all(pool._closed for pool in pools)
    assert all(engine.history is None for engine in engines)


@pytest.mark.parametrize("force_cpu,two_face_state", [(True, False), (False, False), (True, True)])
def test_real_rl_checkpoint_batches_all_starts_and_preserves_cache_semantics(tmp_path, real_setup, force_cpu, two_face_state):
    import torch
    from eval.algorithm import RL_ALGORITHM_NAMES
    from eval.policy import EvaluationPolicy

    if not force_cpu and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not (Path(EvaluationSpec.policy_checkpoint) / "latest.pth").exists():
        pytest.skip("Research checkpoint is not present in this checkout")
    spec, setup = real_setup
    spec = replace(spec, algorithms=RL_ALGORITHM_NAMES, beam_width=2, objective_budget=8,
                   force_cpu=force_cpu, two_face_state=two_face_state)
    policy = EvaluationPolicy.from_spec(spec)
    cached = run_evaluation(spec, setup, output_dir=tmp_path / "rl_cached", policy=policy)
    cold_spec = replace(spec, cache_states=False, algorithms=tuple(reversed(spec.algorithms)))
    cold = run_evaluation(cold_spec, setup, output_dir=tmp_path / "rl_cold", policy=policy)
    order = lambda result: (result.algorithm, result.start_index)

    def assert_same_records(first, second):
        for left, right in zip(first, second, strict=True):
            right = dict(right)
            # Equivalent full representatives can change the last floating-point
            # bits of a KCUP solve. All identities, actions and counts stay exact.
            if two_face_state:
                for field in ("objective", "initial_objective", "best_objective"):
                    if field in left:
                        assert left[field] == pytest.approx(right[field], rel=1e-6)
                        right[field] = left[field]
            assert left == right

    assert_same_records([asdict(row) for row in sorted(cached.rollouts, key=order)],
                        [asdict(row) for row in sorted(cold.rollouts, key=order)])
    for filename in ("queries.jsonl", "expansions.jsonl", "transitions.jsonl"):
        first, second = read_jsonl(cached.output_dir / filename), read_jsonl(cold.output_dir / filename)
        for algorithm in spec.algorithms:
            for start in range(spec.num_starts):
                select = lambda rows: [row for row in rows if row["algorithm"] == algorithm and row["start_index"] == start]
                assert_same_records(select(first), select(second))
    summary = json.loads((cached.output_dir / "summary.json").read_text())
    assert summary["num_rollouts"] == len(spec.algorithms) * spec.num_starts
    assert all(stats["max_logical_batch_states"] >= 2 for stats in summary["policy_stats"].values())
    config = json.loads((cached.output_dir / "config.json").read_text())
    assert config["policy"]["checkpoint"].endswith("latest.pth")
    assert config["policy"]["value_discount"] == 0.9
    assert config["policy"]["resolved_policy_proposal_counts"] == {
        "rl_stochastic_policy": 1, "rl_policy_beam_search": 2,
        "rl_value_beam_search": 2, "rl_value_best_first": 4,
        "rl_metric_value_beam_search": 2, "rl_metric_value_best_first": 4,
    }
    assert config["policy"]["value_score_definitions"]["rl_metric_value_beam_search"] == "ln_objective_plus_discounted_value"
    assert config["policy"]["value_score_definitions"]["rl_value_beam_search"] == "ln_objective_plus_discounted_value"
    assert len(config["policy"]["checkpoint_sha256"]) == 64
    stochastic = [row for row in read_jsonl(cached.output_dir / "queries.jsonl")
                  if row["algorithm"] == "rl_stochastic_policy"]
    for start in range(spec.num_starts):
        identity_field = "evaluation_state_key" if two_face_state else "state_key"
        keys = [row[identity_field] for row in stochastic if row["start_index"] == start]
        assert len(keys) == len(set(keys))
    if two_face_state:
        from eval.results.plotting import read_comparison

        for result in (cached, cold):
            assert len(read_comparison(result.output_dir).rollouts) == len(spec.algorithms) * spec.num_starts
    stats = json.loads((cold.output_dir / "summary.json").read_text())["runtime_stats"]
    assert all(row["resident_graph_nodes"] == row["objective_cache_bytes"] == row["hot_state_bytes"] == 0
               for row in stats.values())


@pytest.mark.parametrize("name", ["rl_value_beam_search", "rl_value_best_first",
                                 "rl_metric_value_beam_search", "rl_metric_value_best_first"])
def test_real_value_all_neighbors_uses_no_actor_and_batches_objectives(tmp_path, real_setup, name):
    from eval.policy import EvaluationPolicy

    if not (Path(EvaluationSpec.policy_checkpoint) / "latest.pth").exists():
        pytest.skip("Research checkpoint is not present in this checkout")
    spec, setup = real_setup
    spec = replace(spec, algorithms=(name,), beam_width=2, policy_proposal_count=-1,
                   transition_num_workers=2, force_cpu=True, cache_states=False)
    policy = EvaluationPolicy.from_spec(spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("Value search with all proposals must never compute logits")

    policy.score_actions = forbidden
    policy.model.get_value_and_logits = forbidden
    policy.model.subcomplex_decoder_head.forward = forbidden
    result = run_evaluation(spec, setup, output_dir=tmp_path / "rl_all", policy=policy)
    summary = json.loads((result.output_dir / "summary.json").read_text())
    stats = summary["policy_stats"][name]
    assert stats["action_batches"] == 0 and stats["value_batches"] == 1
    assert all(row.best_objective > 0 for row in result.rollouts)
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["policy"]["resolved_policy_proposal_counts"] == {name: -1}


@pytest.mark.parametrize("name", ["rl_stochastic_policy", "rl_value_best_first"])
def test_batched_rl_failure_preserves_completed_start_and_closes_runtime(monkeypatch, tmp_path, real_setup, name):
    import numpy as np
    import eval.pipeline as pipeline

    spec, setup = real_setup
    spec = replace(spec, algorithms=(name,), objective_budget=1, policy_proposal_count=1)
    pools, engines = [], []
    original_pool, original_engine = pipeline.create_transition_pool, pipeline.CYRandomRolloutEngine

    class UniformPolicy:
        def score_actions(self, states, actions):
            return [np.full(len(row), -np.log(len(row))) for row in actions]

        def score_values(self, states):
            return [0.0] * len(states)

    def capture_pool(**kwargs):
        pool = original_pool(**kwargs)
        pools.append(pool)
        return pool

    def capture_engine(**kwargs):
        engine = original_engine(**kwargs)
        original_values = engine.objective_values
        calls = 0

        def objective_values(states, name):
            nonlocal calls
            from contextlib import closing

            calls += 1
            with closing(original_values(states, name)) as values:
                for index, value in enumerate(values):
                    if calls > 1 and index == 1:
                        raise RuntimeError("intentional batched failure")
                    yield value

        engine.objective_values = objective_values
        engines.append(engine)
        return engine

    monkeypatch.setattr(pipeline, "create_transition_pool", capture_pool)
    monkeypatch.setattr(pipeline, "CYRandomRolloutEngine", capture_engine)
    output = tmp_path / "rl_failed"
    with pytest.raises(RuntimeError, match="start=1.*intentional batched failure"):
        run_evaluation(spec, setup, output_dir=output, policy=UniformPolicy())
    completed = read_jsonl(output / "rollouts.jsonl")
    assert len(completed) == 1 and completed[0]["start_index"] == 0
    assert read_jsonl(output / "queries.jsonl")[-1]["status"] == "failed"
    assert read_jsonl(output / "expansions.jsonl")[-1]["status"] == "failed"
    assert (output / "failure.json").exists() and not (output / "summary.json").exists()
    assert all(pool._closed for pool in pools)
    assert all(engine.history is None for engine in engines)


@pytest.mark.skipif(os.environ.get("CY_EVAL_HF_SMOKE") != "1", reason="Set CY_EVAL_HF_SMOKE=1 to download HF data.")
def test_hugging_face_kcup_smoke(tmp_path):
    spec = EvaluationSpec(1, 12, 1, 5, hf_cache_dir=str(tmp_path / "hf_cache"),
                          algorithms=("random", "greedy", "best_first", "beam_search"))
    setup = prepare_eval_setup(spec, output_dir=tmp_path / "hf_setup")
    assert setup.metadata["source_metadata"]["resolved_revision"]
    assert setup.metadata["source_metadata"]["source_hodge_filter"] == {"column": "h12", "value": 12}
    results = []
    for cache in (True, False):
        result = run_evaluation(replace(spec, cache_states=cache), setup, output_dir=tmp_path / f"cache_{cache}")
        assert len(result.rollouts) == 4
        assert all(rollout.initial_objective > 0 for rollout in result.rollouts)
        results.append(result.rollouts)
    assert results[0] == results[1]


@pytest.mark.parametrize("names,two_face_state", [(("random", "greedy"), False),
    (("rl_value_beam_search", "rl_value_best_first"), False), (("random", "greedy"), True)])
def test_parallel_cli_preserves_shared_starts_and_sequential_results(tmp_path, real_setup, names, two_face_state):
    from eval.parallel import run_parallel_evaluation
    from eval.results.plotting import read_comparison

    spec, setup = real_setup
    if "rl_value_best_first" in names and not (Path(EvaluationSpec.policy_checkpoint) / "latest.pth").exists():
        pytest.skip("Research checkpoint is not present in this checkout")
    spec = replace(spec, algorithms=names, force_cpu=True, two_face_state=two_face_state)
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({name: dict(cpu_count=2, transition_num_workers=1, memory_budget_gb=8)
                                     for name in spec.algorithms}))
    sequential = run_evaluation(spec, setup, output_dir=tmp_path / "sequential")
    parallel = run_parallel_evaluation(spec, setup, resources_path=resources, output_dir=tmp_path / "parallel")
    order = lambda result: (result.algorithm, result.polytope_index, result.start_index)
    assert sorted(parallel.rollouts, key=order) == sorted(sequential.rollouts, key=order)
    assert len(read_comparison(parallel.output_dir).rollouts) == 4
    manifest = json.loads((parallel.output_dir / "benchmark.json").read_text())
    assert manifest["status"] == "complete" and manifest["total_cpu_count"] == 4
    first, second = [set(job["resources"]["cpu_ids"]) for job in manifest["jobs"].values()]
    assert not first & second
    for name in spec.algorithms:
        config = json.loads((parallel.output_dir / "algorithms" / name / "config.json").read_text())
        assert config["spec"]["memory_budget_gb"] == 8
        assert config["setup_id"] == setup.setup_id
        assert config["spec"]["two_face_state"] is two_face_state


def test_parallel_failure_records_context_and_terminates_other_children(tmp_path, real_setup, monkeypatch):
    from eval.parallel import run_parallel_evaluation

    spec, setup = real_setup
    spec = replace(spec, algorithms=("random", "greedy"))
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({name: dict(cpu_count=2, transition_num_workers=1, memory_budget_gb=8)
                                     for name in spec.algorithms}))
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(children)
            self.code = 1 if not children else None
            self.terminated = False
            children.append(self)
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated, self.code = True, -15

        def wait(self, timeout):
            return self.code

    monkeypatch.setattr("eval.parallel.subprocess.Popen", Child)
    output = tmp_path / "failed_parallel"
    with pytest.raises(RuntimeError, match="random exited with 1"):
        run_parallel_evaluation(spec, setup, resources_path=resources, output_dir=output)
    assert children[1].terminated
    manifest = json.loads((output / "benchmark.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["jobs"]["random"]["status"] == "failed"
    assert manifest["jobs"]["greedy"]["status"] == "cancelled"
    assert (output / "failure.json").exists()
