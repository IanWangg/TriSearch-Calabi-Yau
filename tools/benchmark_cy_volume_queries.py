"""Compare scalar and batched volume queries on identical finite PPO rollouts.

Each mode uses a fresh managed pool/cache and identical model/RNG seeds. No
optimizer updates occur. Collection setup is excluded from rollout timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.cy_episode_context import BEST_SO_FAR_LOG_VOLUME
from core.cy_managed_runtime import managed_rollout_runtime
from core.cy_policy_rollout import collect_policy_rollout
from core.cy_runtime_utils import resolve_training_device, set_seeds
from core.cy_training_config import configure_torch_cpu_threads, parse_args as training_args
from mdp.cy_rollout import CYRandomRolloutEngine, build_cy_rollout_collection, create_transition_pool, load_cy_sample_rows
from models.subcomplex_policy_factory import build_subcomplex_agent
from reward_functions import get_objective, get_reward


def run_mode(args, config, rows, mode, device):
    set_seeds(args.seed)
    with managed_rollout_runtime(config, create_transition_pool) as (pool, cache_bytes, register_engine):
        collection = build_cy_rollout_collection(rows, include_points_interior_to_facets=False,
            neighbor_mode="two_neighbors", transition_pool=pool, include_two_face_metadata=True)
        reward = get_reward("max_kcup")
        engine = CYRandomRolloutEngine(collection=collection, include_points_interior_to_facets=False,
            neighbor_mode="two_neighbors", reward_function=reward, transition_pool=pool,
            cache_budget_bytes=cache_bytes, action_order="canonical")
        register_engine(engine)
        policy = build_subcomplex_agent(model_type="egnn", subcomplex_actor_type="two_face_deep_sets",
            in_channels=4, hidden_channels=64, out_channels=64, num_layers=3, device=str(device),
            training_reward_mode=BEST_SO_FAR_LOG_VOLUME, episode_horizon=args.rollout_length).eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = pool.stats
        started = time.perf_counter()
        summary = collect_policy_rollout(engine=engine, policy=policy, rng=np.random.default_rng(args.seed),
            device=device, initial_state_pool=list(collection.initial_states), num_envs=args.num_states,
            rollout_length=args.rollout_length, gamma=1, deterministic=False, use_multiprocessing=True,
            transition_pool=pool, transition_mp_chunksize=1, transition_mp_min_batch=1,
            store_buffer=True, report_every=0, label=mode, objective_function=get_objective("max_kcup", reward=reward),
            batch_objective_function=(lambda states: engine.objective_values(states, "max_kcup")) if mode == "batch" else None,
            objective_name="max_kcup", objective_goal="max", training_reward_mode=BEST_SO_FAR_LOG_VOLUME)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        after = pool.stats
        buffer = summary.rollout_buffer
        result = dict(mode=mode, rollout_sec=elapsed, objective_query_sec=summary.objective_query_sec,
            candidate_expand_sec=summary.candidate_expand_sec, transition_apply_sec=summary.transition_apply_sec,
            environment_steps=sum(summary.episode_lengths), model_config=policy.model_config,
            initial_volumes=summary.objective_initial_values, best_volumes=summary.objective_best_values,
            final_volumes=summary.objective_final_values, final_keys=[state.key for state in summary.final_states],
            actions=[step.tolist() for step in buffer.action_index_buffer],
            rewards=[step.tolist() for step in buffer.reward_buffer],
            log_probabilities=[step.tolist() for step in buffer.log_prob_buffer],
            values=[step.tolist() for step in buffer.value_buffer],
            terminals=summary.episode_terminal_reasons,
            submitted_requests=after["submitted"] - before["submitted"],
            worker_pids=after["worker_pids"])
        buffer.clear()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default="data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_states", type=int, default=128)
    parser.add_argument("--rollout_length", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    if min(args.num_states, args.rollout_length, args.max_rows, args.workers) <= 0:
        parser.error("State, step, row and worker counts must be positive.")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output_dir must be inside the repository.")
    output.mkdir(parents=True, exist_ok=False)
    config = training_args(["--training_reward_mode", BEST_SO_FAR_LOG_VOLUME,
        "--subcomplex_actor_type", "two_face_deep_sets", "--neighbor_mode", "two_neighbors",
        "--reward_function", "max_kcup", "--no-vertex_aug_enable", "--no-include_points_interior_to_facets",
        "--use_multiprocessing", "--transition_num_workers", str(args.workers),
        "--memory_budget_gb", "16", "--runtime_cache_gb", "1"])
    configure_torch_cpu_threads(config)
    rows = load_cy_sample_rows(args.dataset_path, max_rows=args.max_rows)
    rows = [dict(row, frst_list=row["frst_list"][:1]) for row in rows]
    device = resolve_training_device(gpu_index=args.gpu_index, force_cpu=args.force_cpu)
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        dataset_sha256=hashlib.sha256(Path(args.dataset_path).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "core/cy_policy_inference.py", "core/cy_policy_rollout.py", "mdp/cy_rollout.py",
            "core/cy_process_runtime.py", "tools/benchmark_cy_volume_queries.py")},
        device=str(device), selection="first_initial_state_per_first_n_rows_then_seeded_sampling",
        notes="Separate cold runtimes; rollout timing excludes setup and optimizer updates.")
    (output / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    results = {}
    for mode in ("scalar", "batch"):
        results[mode] = run_mode(args, config, rows, mode, device)
        (output / f"{mode}.json").write_text(json.dumps(results[mode], indent=2, allow_nan=False) + "\n")
        print(f"{mode}: rollout={results[mode]['rollout_sec']:.3f}s "
              f"objective={results[mode]['objective_query_sec']:.3f}s", flush=True)
    for field in ("actions", "final_keys", "terminals"):
        if results["scalar"][field] != results["batch"][field]:
            raise AssertionError(f"Scalar/batch mismatch: {field}")
    for field in ("initial_volumes", "best_volumes", "final_volumes", "rewards", "log_probabilities", "values"):
        np.testing.assert_allclose(results["scalar"][field], results["batch"][field], rtol=1e-6, atol=1e-6,
                                   err_msg=f"Scalar/batch mismatch: {field}")
    report.update(trajectories_match=True,
        rollout_speedup=results["scalar"]["rollout_sec"] / results["batch"]["rollout_sec"],
        objective_speedup=results["scalar"]["objective_query_sec"] / results["batch"]["objective_query_sec"])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"trajectories_match=True rollout_speedup={report['rollout_speedup']:.2f} "
          f"objective_speedup={report['objective_speedup']:.2f}", flush=True)


if __name__ == "__main__":
    main()
