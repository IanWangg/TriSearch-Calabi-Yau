"""Launch every (algorithm, seed, polytope) shard with at most N concurrent CPU processes.

Usage: run_grid.py --budget 2000 --seeds 0 1 2 --workers 16 [--algorithms ...] [--dry-run]
Each shard is one run_shard.py process (one geometry worker). Logs go to OUT/logs.
Shard setups must exist (make_setups.py --shards). Completed shards are skipped
when their result directory already contains summary.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ALGORITHMS, OUT, PYTHON, WORKTREE, result_dir, shard_setup_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--polytopes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num-starts", type=int, default=3)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="grid")
    args = parser.parse_args()
    if args.workers > 16:
        raise SystemExit("At most 16 geometry workers are allowed.")

    jobs = []
    for algorithm, seed, polytope in itertools.product(args.algorithms, args.seeds, args.polytopes):
        output = result_dir(algorithm, seed, polytope, args.budget)
        if (output / "summary.json").exists():
            continue
        setup = shard_setup_dir(seed, polytope, args.num_starts)
        command = [PYTHON, "-u", "-B", str(HERE / "run_shard.py"), "--algorithm", algorithm,
                   "--setup-dir", str(setup), "--output-dir", str(output), "--budget", str(args.budget)]
        jobs.append((f"{algorithm}_seed{seed}_poly{polytope}", command))
    print(f"{len(jobs)} shards to run with {args.workers} workers", flush=True)
    if args.dry_run:
        for name, command in jobs:
            print(name, " ".join(command))
        return

    logs = OUT / "logs" / args.tag
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), MOSEKLM_LICENSE_FILE=str(Path.home() / "mosek" / "mosek.lic"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    running, pending = [], list(jobs)
    summary = []
    started = time.time()
    while pending or running:
        while pending and len(running) < args.workers:
            name, command = pending.pop(0)
            handle = open(logs / f"{name}.log", "w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
            running.append((name, process, handle, time.time()))
        time.sleep(5)
        for item in list(running):
            name, process, handle, began = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            running.remove(item)
            summary.append({"job": name, "exit_code": code, "wall_sec": time.time() - began})
            print(f"done {name} exit={code} wall={time.time() - began:.0f}s elapsed={time.time() - started:.0f}s",
                  flush=True)
        (logs / "grid_status.json").write_text(json.dumps({"finished": summary, "pending": len(pending),
                                                           "running": [r[0] for r in running]}, indent=1))


if __name__ == "__main__":
    main()
