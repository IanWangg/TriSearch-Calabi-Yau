"""Run paired CPU-only smoke, pilot and full Mori cap comparisons.

Use the sage interpreter with TRISEARCH_CPU_ONLY=1 and tools/cpu_runtime on
PYTHONPATH. All children inherit the CPU startup and use --force_cpu.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
sys.path.insert(0, str(REPO))

from eval.config import EvaluationSpec, load_evaluation_config
from eval.results.plotting import combine_labeled_comparisons, plot_evaluation, _write_csv
from eval.setup import EvaluationSetup, load_eval_setup, save_eval_setup

METHODS = ("rl_befs_incremental", "rl_befs_best_so_far", "vanilla_befs", "ga")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def analyze(run_dirs, output):
    data = combine_labeled_comparisons(run_dirs)
    plot_evaluation(next(iter(run_dirs.values())), output, labeled_run_dirs=run_dirs)
    rows = []
    for method in METHODS[:2]:
        for baseline in METHODS[2:]:
            for polytope in [*data.polytopes, None]:
                pairs = [(data.curves[method, p, s][-1], data.curves[baseline, p, s][-1])
                         for p in data.polytopes if polytope is None or p == polytope
                         for s in range(data.spec["num_starts"])]
                differences = [left - right for left, right in pairs]
                rows.append(dict(method=method, baseline=baseline,
                    polytope_index="all" if polytope is None else polytope,
                    num_pairs=len(pairs), wins=sum(delta < 0 for delta in differences),
                    ties=sum(delta == 0 for delta in differences), losses=sum(delta > 0 for delta in differences),
                    mean_difference=sum(differences)/len(differences)))
    _write_csv(output / "paired_vs_baselines.csv", rows)


def run_phase(phase, specs, setup, cpu_ids):
    smoke = phase == "smoke"
    root = REPO / "eval/smoke_tests" if smoke else STUDY
    results = root / "results" / ("mori_cap_20261029_smoke" if smoke else phase)
    if results.exists():
        raise FileExistsError(f"Refusing to overwrite {results}")
    results.mkdir(parents=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    budget = {"smoke": 32, "pilot": 128, "full": 1000}[phase]
    if smoke:
        rows = deepcopy(setup.rows[:1])
        rows[0]["frst_list"] = rows[0]["frst_list"][:1]
        metadata = deepcopy(setup.metadata)
        metadata["parameters"].update(num_polytopes=1, num_starts=1)
        metadata["derived_from_setup_id"] = setup.setup_id
        setup = EvaluationSetup(rows, metadata)
        save_eval_setup(setup, root / "setups/mori_cap_20261029_smoke")
    manifest = dict(phase=phase, status="running", budget=budget, setup_id=setup.setup_id,
                    cpu_only=True, started_at=datetime.now(timezone.utc).isoformat(), jobs={})
    manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest["cpu_startup"] = "tools/cpu_runtime/sitecustomize.py"
    manifest["code_sha256"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("core", "mdp", "models", "reward_functions", "eval", "scripts", "tools/cpu_runtime")
        for path in (REPO / directory).rglob("*.py") if "sweep" not in path.parts and "smoke_tests" not in path.parts}
    with zipfile.ZipFile(results / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for relative_path in manifest["code_sha256"]:
            snapshot.write(REPO / relative_path, relative_path)
        snapshot.write(__file__, "run_study.py")
    (results / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=REPO))
    processes, handles, run_dirs, started = {}, [], {}, {}
    try:
        for index, method in enumerate(METHODS):
            spec = specs[method]
            config = spec.to_dict()
            config.update(objective_budget=budget)
            if smoke:
                config.update(num_polytopes=1, num_starts=1)
            config_path = results / (method + "_spec.json")
            write_json(config_path, config)
            run_dir = results / method
            assigned = cpu_ids[index*5:(index+1)*5]
            command = [sys.executable, "scripts/eval_cy.py", "--config", str(config_path),
                       "--setup_path", str(setup.path), "--output_dir", str(run_dir),
                       "--force_cpu", "--cpu_ids", *map(str, assigned)]
            log_path = logs / f"{phase}_{method}.log"
            handle = log_path.open("w")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                       env=os.environ.copy(), start_new_session=True)
            started[method] = time.monotonic()
            processes[method] = process
            run_dirs[method] = str(run_dir)
            manifest["jobs"][method] = dict(pid=process.pid, command=command, cpu_ids=assigned,
                run_dir=str(run_dir), log=str(log_path), status="running")
        write_json(results / "study_run.json", manifest)
        while any(process.poll() is None for process in processes.values()):
            for method, process in processes.items():
                status = process.poll()
                if status is not None and manifest["jobs"][method]["status"] == "running":
                    manifest["jobs"][method]["wall_sec"] = time.monotonic() - started[method]
                    manifest["jobs"][method].update(exit_code=status, status="complete" if status == 0 else "failed")
                    if status:
                        raise RuntimeError(f"{phase}/{method} failed with exit code {status}; see its log")
            write_json(results / "study_run.json", manifest)
            time.sleep(2)
        for method, process in processes.items():
            if process.returncode:
                raise RuntimeError(f"{phase}/{method} failed with exit code {process.returncode}")
            if "wall_sec" not in manifest["jobs"][method]:
                manifest["jobs"][method]["wall_sec"] = time.monotonic() - started[method]
            manifest["jobs"][method].update(exit_code=0, status="complete")
        plot_root = root / "plots/mori_cap_20261029_smoke" if smoke else STUDY / "plots" / phase
        analyze(run_dirs, plot_root)
        timing_rows = []
        for method, run_dir in run_dirs.items():
            summary = json.loads((Path(run_dir) / "summary.json").read_text())
            stats = next(iter(summary["policy_stats"].values()), {})
            timing_rows.append(dict(method=method, total_process_wall_sec=manifest["jobs"][method]["wall_sec"],
                objective_queries=summary["objective_queries"], inference_sec=stats.get("inference_sec", 0),
                critic_states=stats.get("value_states", 0), actor_states=stats.get("action_states", 0),
                cuda_peak_allocated_bytes=stats.get("cuda_peak_allocated_bytes", 0)))
        _write_csv(plot_root / "runtime_summary.csv", timing_rows)
        manifest.update(status="complete", plots=str(plot_root), finished_at=datetime.now(timezone.utc).isoformat())
        print(f"Completed {phase}: {results}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        for method, process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for method, process in processes.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            manifest["jobs"][method].update(exit_code=process.returncode,
                status="complete" if process.returncode == 0 else "failed")
        raise
    finally:
        write_json(results / "study_run.json", manifest)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "full", "all"), default="all")
    parser.add_argument("--cpu_ids", type=int, nargs=20, required=True)
    args = parser.parse_args()
    if os.environ.get("TRISEARCH_CPU_ONLY") != "1":
        parser.error("Set TRISEARCH_CPU_ONLY=1 and PYTHONPATH to tools/cpu_runtime before starting Python")
    if len(set(args.cpu_ids)) != 20 or not set(args.cpu_ids) <= os.sched_getaffinity(0):
        parser.error("Provide 20 distinct available CPUs")
    specs = {method: EvaluationSpec(**{**load_evaluation_config(STUDY / "configs" / (method + ".json"))})
             for method in METHODS}
    for spec in specs.values():
        if not spec.force_cpu or spec.reward_function != "min_mori_cap":
            raise ValueError("This study requires CPU Mori cap specifications")
    checkpoints = json.loads((STUDY / "configs/checkpoint_manifest.json").read_text())
    for method, info in checkpoints.items():
        if hashlib.sha256((REPO / info["checkpoint"]).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {method}")
    setup = load_eval_setup(REPO / "eval/data/setups/h11_30_5_polytopes_5_starts_seed_0")
    for spec in specs.values():
        setup.validate(spec)
    for phase in ("smoke", "pilot", "full") if args.phase == "all" else (args.phase,):
        run_phase(phase, specs, setup, args.cpu_ids)


if __name__ == "__main__":
    main()
