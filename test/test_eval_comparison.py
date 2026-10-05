"""Configuration and offline comparison contracts, without network or geometry."""

from itertools import zip_longest
import json

import numpy as np
import pytest

from eval.config import EvaluationSpec
from eval.results.plotting import comparison_statistics, plot_evaluation, read_comparison
from scripts.eval_cy import parse_args


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_run(path, *, algorithms=("random", "greedy"), budget=3, version=2):
    path.mkdir()
    spec = EvaluationSpec(2, 21, 2, budget, algorithms=algorithms)
    write_json(path / "config.json", dict(format_version=version, spec=spec.to_dict(), setup_id="shared"))
    rollouts, traces = [], []
    for algorithm in algorithms:
        for polytope in range(2):
            for start in range(2):
                initial = float(1 + polytope * 2 + start)
                # Greedy's large final improvement is beyond the nominal budget.
                values = ([initial] if budget == 0 else
                          [initial, initial * 2] if algorithm == "random" else
                          [initial, initial * 2, initial * 3, initial * 4, initial * 100])
                count = len(values) - 1
                identity = dict(algorithm=algorithm, polytope_index=polytope, start_index=start, seed=0)
                initial_key = f"initial_{polytope}_{start}"
                trace = []
                for q, value in enumerate(values):
                    key = initial_key if q == 0 else f"{algorithm}_{polytope}_{start}_{q}"
                    trace.append(dict(**identity, query_index=q, is_initial=q == 0, status="ok",
                                      objective=value, best_objective=value, state_key=key, best_state_key=key))
                traces.append(trace)
                row = dict(**identity, objective_name="max_kcup", objective_goal="max", objective_budget=budget,
                           objective_queries=count, budget_overshoot=max(0, count - budget), transition_count=count,
                           initial_state_key=initial_key, best_state_key=trace[-1]["state_key"],
                           initial_objective=initial, best_objective=values[-1],
                           termination_reason="no_neighbors" if count < budget else "budget_exhausted")
                if version == 2:
                    row["expansion_count"] = count
                else:
                    row.update(final_objective=values[-1], final_state_key=trace[-1]["state_key"])
                rollouts.append(row)
    # RL logs interleave starts; the reader must not assume contiguous trajectories.
    events = [event for group in zip_longest(*traces) for event in group if event is not None]
    write_jsonl(path / "queries.jsonl", events)
    write_jsonl(path / "rollouts.jsonl", rollouts)
    summary = dict(status="complete", setup_id="shared", num_rollouts=len(rollouts),
                   **{field: sum(row[field] for row in rollouts)
                      for field in ("objective_queries", "budget_overshoot", "transition_count")})
    if version == 2:
        summary.update(format_version=2, expansion_count=sum(row["expansion_count"] for row in rollouts))
    write_json(path / "summary.json", summary)
    return spec


def test_config_precedence_and_original_cli(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, dict(num_polytopes=5, h11=21, num_starts=10, objective_budget=1000,
                            algorithms=["random", "greedy"], cache_states=False, force_cpu=True))
    args = parse_args(["--config", str(config), "--objective_budget", "3", "--cache_states", "--no_force_cpu"])
    assert (args.num_polytopes, args.h11, args.num_starts, args.objective_budget) == (5, 21, 10, 3)
    assert args.cache_states and not args.force_cpu and args.beam_width == 4
    original = parse_args(["--num_polytopes", "1", "--h11", "12", "--num_starts", "1", "--objective_budget", "0"])
    assert original.algorithms == ["random", "greedy"] and original.cache_states
    assert not original.skip_insufficient_starts
    write_json(config, {**json.loads(config.read_text()), "skip_insufficient_starts": True})
    assert parse_args(["--config", str(config)]).skip_insufficient_starts
    assert not parse_args(["--config", str(config), "--no_skip_insufficient_starts"]).skip_insufficient_starts


@pytest.mark.parametrize("change", [
    {"typo": 2}, {"h11": "21"}, {"h11": True}, {"num_polytopes": 0},
    {"algorithms": "random"}, {"algorithms": ["misspelled"]}, {"algorithms": ["random", "random"]},
    {"cache_states": 1}, {"reward_function": "missing"}, {"subcomplex_actor_type": "missing"},
    {"value_discount": 3}, {"value_discount": float("nan")}, {"seed": None},
])
def test_invalid_config_fails_before_evaluation(tmp_path, change):
    config = tmp_path / "config.json"
    write_json(config, {**EvaluationSpec(1, 21, 1, 3).to_dict(), **change})
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_config_missing_required_and_plot_preflight(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"h11": 21})
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
    write_json(config, EvaluationSpec(1, 21, 1, 3).to_dict())
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--plot_results", "--setup_only"])
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--plot_results", "--reward_function", "min_tri"])


@pytest.mark.parametrize("version", [1, 2])
def test_query_endpoint_excludes_overshoot_and_carries_successful_early_stop(tmp_path, version):
    run = tmp_path / "run"
    make_run(run, version=version)
    data = read_comparison(run)
    np.testing.assert_array_equal(data.curves["greedy", 0, 0], [1, 2, 3, 4])
    assert data.rollouts["greedy", 0, 0]["best_objective"] == 100
    np.testing.assert_array_equal(data.curves["random", 0, 0], [1, 2, 2, 2])
    per_polytope, overall = comparison_statistics(data)
    assert per_polytope["greedy", 0]["median"][-1] == 6
    assert per_polytope["greedy", 0]["q25"][-1] == 5
    assert overall["greedy"]["median"][-1] == 10
    assert overall["greedy"]["q25"][-1] == 7
    assert overall["greedy"]["q75"][-1] == 13


def test_zero_budget_and_initial_objective(tmp_path):
    run = tmp_path / "run"
    make_run(run, budget=0)
    data = read_comparison(run)
    assert all(len(curve) == 1 for curve in data.curves.values())
    _, overall = comparison_statistics(data)
    assert all(stats["median"][0] == 2.5 for stats in overall.values())


def test_combine_completed_runs_and_reject_unpaired_or_duplicate_inputs(tmp_path):
    from eval.results.plotting import combine_comparisons

    one, two = tmp_path / "one", tmp_path / "two"
    make_run(one, algorithms=("random",))
    make_run(two, algorithms=("greedy",))
    combined = combine_comparisons([one, two])
    assert combined.spec["algorithms"] == ["random", "greedy"]
    assert len(combined.rollouts) == 8
    with pytest.raises(ValueError, match="duplicate algorithms"):
        combine_comparisons([one, one])
    config = json.loads((two / "config.json").read_text())
    config["spec"]["seed"] = 3
    write_json(two / "config.json", config)
    with pytest.raises(ValueError, match="different seed"):
        combine_comparisons([one, two])


@pytest.mark.parametrize("change", ["checkpoint", "model"])
def test_combined_runs_reject_inconsistent_rl_models(tmp_path, change):
    from eval.results.plotting import combine_comparisons

    runs = [tmp_path / "one", tmp_path / "two"]
    for path, name in zip(runs, ["rl_stochastic_policy", "rl_policy_beam_search"]):
        make_run(path, algorithms=(name,))
        config = json.loads((path / "config.json").read_text())
        config["policy"] = {"checkpoint_sha256": "a" * 64}
        write_json(path / "config.json", config)
    assert combine_comparisons(runs).policy_checkpoint_sha256 == "a" * 64
    config = json.loads((runs[1] / "config.json").read_text())
    if change == "checkpoint":
        config["policy"]["checkpoint_sha256"] = "b" * 64
    else:
        config["spec"]["num_layers"] += 1
    write_json(runs[1] / "config.json", config)
    with pytest.raises(ValueError, match="different policy"):
        combine_comparisons(runs)


def test_aggregation_uses_absolute_volume_without_initial_normalization(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    data = read_comparison(run)
    for start in (0, 1):
        data.curves["greedy", 1, start][-1] *= 4
    _, overall = comparison_statistics(data)
    # Endpoints are 4, 8, 48, 64, even though the starts have different volumes.
    assert overall["greedy"]["median"][-1] == 28
    assert overall["greedy"]["q25"][-1] == 7
    assert overall["greedy"]["q75"][-1] == 52


@pytest.mark.parametrize("corruption", ["failure", "missing_summary", "missing_rollout", "unpaired", "missing_query",
                                       "failed_query", "duplicate_query", "wrong_best", "bad_termination", "version"])
def test_reader_rejects_incomplete_or_inconsistent_results(tmp_path, corruption):
    run = tmp_path / "run"
    make_run(run)
    if corruption == "failure":
        write_json(run / "failure.json", {"status": "failed"})
    elif corruption == "missing_summary":
        (run / "summary.json").unlink()
    elif corruption == "version":
        config = json.loads((run / "config.json").read_text())
        config["format_version"] = 99
        write_json(run / "config.json", config)
    elif corruption in ("unpaired", "missing_rollout", "bad_termination"):
        rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
        if corruption == "unpaired":
            rows[0]["initial_state_key"] = "different"
        elif corruption == "missing_rollout":
            rows.pop()
        else:
            rows[0]["termination_reason"] = "budget_exhausted"
        write_jsonl(run / "rollouts.jsonl", rows)
    else:
        rows = [json.loads(line) for line in (run / "queries.jsonl").read_text().splitlines()]
        if corruption == "missing_query":
            rows.pop()
        elif corruption == "failed_query":
            rows[0]["status"] = "failed"
        elif corruption == "duplicate_query":
            rows.append(rows[-1])
        else:
            rows[0]["best_objective"] *= 2
        write_jsonl(run / "queries.jsonl", rows)
    with pytest.raises(ValueError):
        read_comparison(run)


def test_parallel_comparison_requires_all_paired_children(tmp_path):
    root = tmp_path / "parallel"
    root.mkdir()
    for algorithm in ("random", "greedy"):
        make_run(root / algorithm, algorithms=(algorithm,))
    manifest = dict(format_version=1, status="complete", setup_id="shared",
                    spec=EvaluationSpec(2, 21, 2, 3).to_dict(),
                    jobs={name: dict(status="complete", run_dir=name) for name in ("random", "greedy")})
    write_json(root / "benchmark.json", manifest)
    assert len(read_comparison(root).rollouts) == 8
    manifest["jobs"]["greedy"]["status"] = "failed"
    write_json(root / "benchmark.json", manifest)
    with pytest.raises(ValueError, match="Incomplete parallel job"):
        read_comparison(root)


def test_headless_plot_outputs_match_statistics_and_preserve_raw_logs(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    output = plot_evaluation(run)
    for stem in ("best_volume_by_polytope", "best_volume_overall", "budget_endpoint_distribution"):
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
    import csv

    with (output / "benchmark_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    greedy = next(row for row in rows if row["algorithm"] == "greedy" and row["polytope_index"] == row["start_index"] == "0")
    assert float(greedy["best_at_budget"]) == 4 and float(greedy["best_complete_search"]) == 100
    assert len(rows) == 8
    assert "log_gain_at_budget" not in rows[0]
    assert not (output / "relative_improvement.png").exists()
    with (output / "algorithm_summary.csv").open() as stream:
        summaries = {row["algorithm"]: row for row in csv.DictReader(stream)}
    assert float(summaries["greedy"]["median_best_at_budget"]) == 10
    assert "geometric_mean_gain_factor" not in summaries["greedy"]
    assert json.loads((output / "plot_config.json").read_text())["format_version"] == 2
    assert before == {name: (run / name).read_bytes() for name in before}
    first_csv = (output / "benchmark_summary.csv").read_bytes()
    plot_evaluation(run, output)
    assert first_csv == (output / "benchmark_summary.csv").read_bytes()


def test_parallel_resource_plan_is_disjoint_and_rejects_overcommit(tmp_path, monkeypatch):
    from eval.parallel import plan_parallel_resources

    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {2, 3, 6, 7})
    resources = tmp_path / "resources.json"
    plan = {name: dict(cpu_count=2, transition_num_workers=1, memory_budget_gb=8) for name in ("random", "greedy")}
    write_json(resources, plan)
    spec = EvaluationSpec(1, 21, 1, 3)
    resolved = plan_parallel_resources(spec, resources)
    assert resolved["random"]["cpu_ids"] == [2, 3]
    assert resolved["greedy"]["cpu_ids"] == [6, 7]
    plan["greedy"]["cpu_count"] = 3
    write_json(resources, plan)
    with pytest.raises(ValueError, match="exceeds"):
        plan_parallel_resources(spec, resources)


def test_cpu_affinity_includes_threads_created_before_resource_setup():
    import subprocess
    import sys

    # Isolate affinity changes from pytest and start a real thread before pinning.
    code = """
import os
from pathlib import Path
import threading
from scripts.eval_cy import restrict_cpu_affinity
stop = threading.Event()
thread = threading.Thread(target=stop.wait)
thread.start()
try:
    assigned = {min(os.sched_getaffinity(0))}
    restrict_cpu_affinity(assigned)
    assert os.sched_getaffinity(thread.native_id) == assigned
    assert all(os.sched_getaffinity(int(task.name)) == assigned
               for task in Path('/proc/self/task').iterdir())
finally:
    stop.set()
    thread.join()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


@pytest.mark.parametrize("second", ["rl_value_beam_search", "rl_value_best_first"])
def test_parallel_resource_plan_rejects_shared_or_unavailable_gpu(tmp_path, monkeypatch, second):
    import torch
    from eval.parallel import plan_parallel_resources

    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    names = ("rl_policy_beam_search", second)
    plan = {name: dict(cpu_count=4, transition_num_workers=2, memory_budget_gb=8, gpu_index=0) for name in names}
    resources = tmp_path / "resources.json"
    write_json(resources, plan)
    spec = EvaluationSpec(1, 21, 1, 3, algorithms=names)
    with pytest.raises(ValueError, match="distinct"):
        plan_parallel_resources(spec, resources)
    plan[names[1]]["gpu_index"] = 2
    write_json(resources, plan)
    with pytest.raises(ValueError, match="unavailable"):
        plan_parallel_resources(spec, resources)
