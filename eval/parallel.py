"""Run independent algorithms with disjoint CPU affinity and explicit GPU/RAM budgets."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from eval.algorithm import RL_ALGORITHM_NAMES
from eval.config import EVAL_ROOT, EvaluationSpec


def plan_parallel_resources(spec: EvaluationSpec, resources_path: str | Path) -> dict:
    """Resolve CPU counts to disjoint affinity sets before starting any child."""
    resources = json.loads(Path(resources_path).expanduser().read_text())
    if not isinstance(resources, dict) or any(name not in resources for name in spec.algorithms):
        raise ValueError("Parallel resources must specify every selected algorithm.")
    available = sorted(os.sched_getaffinity(0))
    cursor, used_gpus, jobs = 0, set(), {}
    for name in spec.algorithms:
        requested = resources[name]
        allowed = {"cpu_count", "transition_num_workers", "memory_budget_gb", "gpu_index"}
        if not isinstance(requested, dict) or requested.keys() - allowed:
            raise ValueError(f"Invalid resource fields for {name}.")
        for field in ("cpu_count", "transition_num_workers"):
            if type(requested.get(field)) is not int or requested[field] <= 0:
                raise ValueError(f"{name}.{field} must be a positive integer.")
        cpu_count = requested["cpu_count"]
        if requested["transition_num_workers"] >= cpu_count:
            raise ValueError(f"{name} needs at least one CPU in addition to its geometry workers.")
        memory = requested.get("memory_budget_gb")
        if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
            raise ValueError(f"{name}.memory_budget_gb must be finite and positive.")
        gpu = requested.get("gpu_index")
        if name in RL_ALGORITHM_NAMES and not spec.force_cpu:
            if type(gpu) is not int or gpu < 0 or gpu in used_gpus:
                raise ValueError("Each parallel RL algorithm needs a distinct nonnegative gpu_index.")
            used_gpus.add(gpu)
        elif gpu is not None:
            raise ValueError(f"CPU-only algorithm {name} must not reserve a GPU.")
        if cursor + cpu_count > len(available):
            raise ValueError("Parallel resource plan exceeds available CPU affinity.")
        jobs[name] = {**requested, "gpu_index": gpu, "cpu_ids": available[cursor:cursor + cpu_count]}
        cursor += cpu_count
    if used_gpus:
        import torch

        if max(used_gpus) >= torch.cuda.device_count():
            raise ValueError("Parallel resource plan requests an unavailable CUDA device.")
    return jobs


def run_parallel_evaluation(spec, setup=None, *, resources_path, output_dir=None):
    """Reuse the ordinary eval CLI for each algorithm; never duplicate search logic."""
    from data.cy.pipeline import _write_json_atomic
    from eval.pipeline import EvaluationResult
    from eval.rollout import RolloutResult, TwoFaceRolloutResult
    from eval.setup import prepare_eval_setup, save_eval_setup

    resources = plan_parallel_resources(spec, resources_path)
    if setup is None:
        setup = prepare_eval_setup(spec)
    setup.validate(spec)
    save_eval_setup(setup, setup.path or EVAL_ROOT / "data/setups" / setup.setup_id)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = EVAL_ROOT / "results/runs" / f"benchmark_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir()
    (output_dir / "specs").mkdir()
    manifest = dict(format_version=1, status="running", spec=spec.to_dict(), setup_id=setup.setup_id,
                    setup_path=str(setup.path), resources_path=str(Path(resources_path).resolve()),
                    total_cpu_count=sum(row["cpu_count"] for row in resources.values()),
                    total_memory_budget_gb=sum(row["memory_budget_gb"] for row in resources.values()), jobs={})
    processes, streams, rollouts = {}, {}, []
    checkpoint_hashes = set()
    print(f"Parallel benchmark: {output_dir}", flush=True)
    try:
        for name, allocation in resources.items():
            is_rl = name in RL_ALGORITHM_NAMES
            child_spec = replace(spec, algorithms=(name,), transition_num_workers=allocation["transition_num_workers"],
                                 memory_budget_gb=allocation["memory_budget_gb"],
                                 force_cpu=spec.force_cpu or not is_rl,
                                 gpu_index=allocation["gpu_index"] if allocation["gpu_index"] is not None else 0)
            spec_path = output_dir / "specs" / f"{name}.json"
            _write_json_atomic(spec_path, child_spec.to_dict())
            run_dir = Path("algorithms") / name
            command = [sys.executable, "-u", str(EVAL_ROOT.parent / "scripts/eval_cy.py"),
                       "--config", str(spec_path), "--setup_path", str(setup.path),
                       "--output_dir", str(output_dir / run_dir),
                       "--cpu_ids", *map(str, allocation["cpu_ids"])]
            environment = os.environ.copy()
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
                environment[variable] = "1"
            if not is_rl or spec.force_cpu:
                environment["CUDA_VISIBLE_DEVICES"] = ""
            streams[name] = (output_dir / "logs" / f"{name}.log").open("w", encoding="utf-8")
            manifest["jobs"][name] = dict(status="running", run_dir=str(run_dir), resources=allocation,
                                           log_file=f"logs/{name}.log", command=command)
            processes[name] = subprocess.Popen(command, cwd=EVAL_ROOT.parent, env=environment,
                                               stdout=streams[name], stderr=subprocess.STDOUT, start_new_session=True)
            manifest["jobs"][name]["pid"] = processes[name].pid
            _write_json_atomic(output_dir / "benchmark.json", manifest)
            print(f"Started {name}: CPUs={allocation['cpu_ids']} GPU={allocation['gpu_index']} "
                  f"workers={allocation['transition_num_workers']} RAM={allocation['memory_budget_gb']} GiB", flush=True)
        pending = set(processes)
        while pending:
            for name in tuple(pending):
                code = processes[name].poll()
                if code is None:
                    continue
                job = manifest["jobs"][name]
                job["return_code"] = code
                if code != 0:
                    job["status"] = "failed"
                    raise RuntimeError(f"Parallel algorithm {name} exited with {code}; see {output_dir / job['log_file']}.")
                child_dir = output_dir / job["run_dir"]
                summary = json.loads((child_dir / "summary.json").read_text())
                config = json.loads((child_dir / "config.json").read_text())
                if (summary["status"] != "complete" or summary["setup_id"] != setup.setup_id
                        or summary["num_rollouts"] != spec.num_polytopes * spec.num_starts):
                    raise ValueError(f"Incomplete or mismatched output from {name}.")
                if name in RL_ALGORITHM_NAMES:
                    checkpoint_hashes.add(config["policy"]["checkpoint_sha256"])
                    if len(checkpoint_hashes) != 1:
                        raise ValueError("Parallel RL jobs loaded different checkpoint contents.")
                with (child_dir / "rollouts.jsonl").open() as stream:
                    result_type = TwoFaceRolloutResult if spec.two_face_state else RolloutResult
                    rollouts.extend(result_type(**json.loads(line)) for line in stream if line.strip())
                job["status"] = "complete"
                pending.remove(name)
                _write_json_atomic(output_dir / "benchmark.json", manifest)
                print(f"Completed {name}: {summary['num_rollouts']} rollouts, {summary['objective_queries']} queries", flush=True)
            if pending:
                time.sleep(1)
        manifest["status"] = "complete"
        manifest["num_rollouts"] = len(rollouts)
        manifest["checkpoint_sha256"] = next(iter(checkpoint_hashes), None)
        _write_json_atomic(output_dir / "benchmark.json", manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        for name, job in manifest["jobs"].items():
            process = processes.get(name)
            if process is not None and process.poll() is None:
                process.terminate()
                job["status"] = "cancelled"
            elif job["status"] == "running":
                job["status"] = "failed"
        _write_json_atomic(output_dir / "benchmark.json", manifest)
        _write_json_atomic(output_dir / "failure.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for stream in streams.values():
            stream.close()
    return EvaluationResult(output_dir, setup.setup_id, rollouts)
