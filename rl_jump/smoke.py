"""Vectorized random-policy smoke for rl_jump: timings, infeasible fraction, cache hits, best dlnV.

Example (from the worktree root, PYTHONPATH=$PWD):
  python -m rl_jump.smoke --rows R/data/cy/favorable_v1/train/h11_20.samples.jsonl --lines 0,1,2,3 \
      --envs 8 --inline 1 --episodes 2 --horizon 20 --out_dir R/runs/rl_jump_smoke_20261104/h11_20
Env i uses polytope lines[i % len(lines)] and start i // len(lines). Each env: its own process, except
the last one with --inline 1, which runs in this process (so --envs 8 --inline 1 is 8 processes in total).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def _summarize(stats, episodes, policy_sec, policy_calls, env_steps, loop_wall, setup_seconds, batch_sizes):
    keys = ("queries", "objective_requests", "objective_cache_hits", "objective_failures", "objective_sec",
            "decode_calls", "decode_cache_hits", "decode_sec", "decode_infeasible", "drawn", "infeasible",
            "obs_sec", "obs_builds", "steps", "step_failures")
    total = {key: sum(s[key] for s in stats) for key in keys}
    steps = max(1, total["steps"])
    best = [e["dlnV_best"] for e in episodes]
    return dict(
        envs=len(stats), env_steps=env_steps, episodes=len(episodes),
        setup_seconds_max=max(setup_seconds), setup_seconds_mean=sum(setup_seconds) / len(setup_seconds),
        decode_ms_per_step=1000 * total["decode_sec"] / steps,
        decode_ms_per_call=1000 * total["decode_sec"] / max(1, total["decode_calls"]),
        decode_calls_per_step=total["decode_calls"] / steps,
        infeasible_fraction=total["infeasible"] / max(1, total["drawn"]),
        candidates_per_entered_state=(total["drawn"] - total["infeasible"]) / max(1, total["obs_builds"]),
        objective_ms_per_query=1000 * total["objective_sec"] / max(1, total["queries"]),
        objective_ms_per_step=1000 * total["objective_sec"] / steps,
        queries_per_step=total["queries"] / steps,
        objective_cache_hit_rate=total["objective_cache_hits"] / max(1, total["objective_requests"]),
        decode_cache_hit_rate=total["decode_cache_hits"] / max(1, total["decode_calls"]),
        obs_ms_per_step=1000 * total["obs_sec"] / steps,
        policy_ms_per_call=1000 * policy_sec / max(1, policy_calls),
        policy_ms_per_env_step=1000 * policy_sec / max(1, env_steps),
        policy_calls=policy_calls, mean_batch_envs=sum(batch_sizes) / max(1, len(batch_sizes)),
        steps_per_second=env_steps / loop_wall, loop_wall_seconds=loop_wall,
        objective_queries=total["queries"], step_failures=total["step_failures"],
        dlnV_best_mean=sum(best) / len(best), dlnV_best_max=max(best),
        dlnV_best_per_episode=best,
    )


def run(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from torch_geometric.data import Batch

    from rl_jump.policy import batched_move_logits, build_random_policy, sample_actions
    from rl_jump.vec import VecJumpEnv

    lines = [int(x) for x in args.lines.split(",")]
    specs, starts = [], []
    for i in range(args.envs):
        specs.append(dict(path=args.rows, line=lines[i % len(lines)],
                          env_kwargs=dict(K=args.K, seed=args.seed + i, horizon=args.horizon,
                                          near_fraction=args.near_fraction, samples=args.samples,
                                          max_points=args.max_points, codebook_seed=args.seed)))
        starts.append(i // len(lines))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vec = VecJumpEnv(specs, inline_last=bool(args.inline))
    try:
        setup_seconds = [float(s) for s in vec.setup_seconds]
        policy = build_random_policy(args.checkpoint, seed=args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        policy_sec, policy_calls, env_steps, batch_sizes = 0.0, 0, 0, []
        episodes, records = [], []
        loop_started = time.perf_counter()
        for episode in range(args.episodes):
            obs = vec.reset(starts)
            start_values = [o.value for o in obs]
            best = list(start_values)
            active = {i for i in range(args.envs) if obs[i].candidates}
            if len(active) < args.envs:
                raise RuntimeError("An env started with no feasible candidates; choose other starts.")
            while active:
                idx = sorted(active)
                batch = Batch.from_data_list([obs[i].data for i in idx])
                counts = [int(obs[i].data.num_candidates) for i in idx]
                started = time.perf_counter()
                logits = batched_move_logits(policy, batch, aggregation="mean2")
                actions = sample_actions(logits, counts, generator)
                policy_sec += time.perf_counter() - started
                policy_calls += 1
                batch_sizes.append(len(idx))
                results = vec.step({i: a for i, a in zip(idx, actions)})
                for i in idx:
                    new_obs, reward, done, info = results[i]
                    obs[i] = new_obs
                    env_steps += 1
                    best[i] = info["episode_best"]
                    record = dict(env=i, episode=episode, **{k: v for k, v in info.items()})
                    record["entry_stats"] = new_obs.stats
                    records.append(record)
                    if done:
                        active.discard(i)
            for i in range(args.envs):
                episodes.append(dict(env=i, episode=episode, start_index=starts[i], start_value=start_values[i],
                                     best_value=best[i],
                                     dlnV_best=math.log(best[i]) - math.log(start_values[i])))
        loop_wall = time.perf_counter() - loop_started
        stats = vec.stats()
    finally:
        vec.close()

    summary = _summarize(stats, episodes, policy_sec, policy_calls, env_steps, loop_wall, setup_seconds,
                         batch_sizes)
    summary.update(rows=args.rows, lines=lines, starts=starts, K=args.K, horizon=args.horizon,
                   near_fraction=args.near_fraction, samples=args.samples, max_points=args.max_points,
                   checkpoint=str(Path(args.checkpoint)), env_counters=stats, episodes=episodes)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with (out / "steps.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--lines", default="0,1,2,3", help="comma-separated row indices of the polytopes")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--inline", type=int, default=1, help="1: the last env runs in this process")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--near_fraction", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(root / "runs/inv_rl_full_20261104/checkpoints/final.pth"))
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)
    summary = run(args)
    keys = [k for k in summary if not isinstance(summary[k], (list, dict))]
    for key in keys:
        print(f"{key}: {summary[key]}", flush=True)
    print("dlnV_best_per_episode:", [round(v, 4) for v in summary["dlnV_best_per_episode"]], flush=True)
    return summary


if __name__ == "__main__":
    main()
