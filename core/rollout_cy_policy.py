import argparse
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from mdp.cy_rollout import (
    CYRandomRolloutEngine,
    build_cy_rollout_collection,
    create_transition_pool,
    load_cy_sample_rows,
    runtime_cache_hot_size,
    runtime_cache_total_unique_states,
)
from core.train_cy import normalize_subcomplex_actor_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl",
        help="Path to CY .samples.jsonl file.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on the number of polytopes loaded from the JSONL file.",
    )
    parser.add_argument(
        "--include_points_interior_to_facets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to cytools Polytope.triangulate(...).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel rollout states.")
    parser.add_argument("--rollout_steps", type=int, default=100, help="Number of rollout steps.")
    parser.add_argument(
        "--filter_actionable_initial_states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand the dataset initial states once and keep only states with at least one valid regular flip.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a process pool when expanding previously unseen states.",
    )
    parser.add_argument(
        "--transition_num_workers",
        type=int,
        default=0,
        help="Number of worker processes. <=0 uses os.cpu_count() in TransitionPool.",
    )
    parser.add_argument(
        "--transition_mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for expansion workers.",
    )
    parser.add_argument(
        "--transition_mp_chunksize",
        type=int,
        default=16,
        help="Chunksize for worker expansion batches.",
    )
    parser.add_argument(
        "--transition_mp_min_batch",
        type=int,
        default=1,
        help="Minimum number of unseen states before using multiprocessing.",
    )
    parser.add_argument(
        "--state_cache_mode",
        type=str,
        default="lru",
        choices=["full", "lru", "none"],
        help="Object-cache policy for materialized runtime states.",
    )
    parser.add_argument(
        "--max_hot_states",
        type=int,
        default=100000,
        help="Maximum runtime state objects kept in memory when --state_cache_mode=lru.",
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=10,
        help="Print rollout progress every N steps. <=0 disables periodic progress logs.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions instead of sampling from the untrained policy.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for GAE preparation.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda for PPO preparation.")
    parser.add_argument("--in_channels", type=int, default=3, help="Policy node feature dimension.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Policy hidden width.")
    parser.add_argument("--out_channels", type=int, default=64, help="Policy output width.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of EGNN layers.")
    parser.add_argument(
        "--subcomplex_actor_type",
        type=str,
        default="gnn",
        choices=["mlp", "gnn", "circuit_pool", "snn_simplex", "default"],
        help="Subcomplex actor architecture for the untrained policy.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help="CUDA device index for the untrained policy.",
    )
    parser.add_argument(
        "--force_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force CPU execution even when CUDA is available.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a very small rollout for a quick smoke test.",
    )
    return parser.parse_args()


def read_process_memory_gb() -> Tuple[float, float]:
    rss_kb = 0.0
    hwm_kb = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = float(line.split()[1])
    except OSError:
        return 0.0, 0.0
    return rss_kb / (1024.0 * 1024.0), hwm_kb / (1024.0 * 1024.0)


def increment_visitation(states: Iterable[object]) -> None:
    for state in states:
        if hasattr(state, "visitation"):
            state.visitation += 1


def resolve_policy_device(gpu_index: int, *, force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if torch.cuda.device_count() <= int(gpu_index):
        raise RuntimeError(
            f"Requested gpu_index={gpu_index}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    device = torch.device(f"cuda:{int(gpu_index)}")
    torch.cuda.set_device(device)
    return device


def main(args: argparse.Namespace) -> None:
    from core.cy_policy_rollout_utils import PPORolloutBuffer, evaluate_policy_values, rollout_step_with_policy
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dry_run:
        args.max_rows = 2 if args.max_rows is None else min(int(args.max_rows), 2)
        args.num_envs = min(int(args.num_envs), 8)
        args.rollout_steps = min(int(args.rollout_steps), 5)
        args.report_every = 1
        print(
            "Dry-run overrides: "
            f"max_rows={args.max_rows}, num_envs={args.num_envs}, rollout_steps={args.rollout_steps}"
        )

    device = resolve_policy_device(args.gpu_index, force_cpu=bool(args.force_cpu))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"
    else:
        device_label = str(device)
    print(f"Using policy device: {device_label}")

    dataset_path = str(Path(args.dataset_path).expanduser())
    print(f"Loading CY rollout dataset from {dataset_path}")
    rows = load_cy_sample_rows(dataset_path, max_rows=args.max_rows)
    print(f"Loaded {len(rows)} polytopes")

    build_start = time.perf_counter()
    collection = build_cy_rollout_collection(
        rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
    )
    build_sec = time.perf_counter() - build_start
    print(
        "Built rollout collection: "
        f"base_states={len(collection.base_states)}, "
        f"initial_states={len(collection.initial_states)}, "
        f"polytopes={len(collection.polytope_indices)}, "
        f"time={build_sec:.2f}s"
    )

    engine = CYRandomRolloutEngine(
        collection=collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
    )
    policy = EGNNSubcomplexAgent(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        share_encoder=True,
        mlp_hidden_channel_list=[64],
        act="silu",
        subcomplex_actor_type=normalize_subcomplex_actor_type(args.subcomplex_actor_type),
        device=str(device),
    ).to(device).eval()

    transition_pool = None
    if args.use_multiprocessing:
        transition_pool = create_transition_pool(
            num_workers=args.transition_num_workers,
            start_method=args.transition_mp_start_method,
        )
        print(
            "Multiprocessing enabled: "
            f"workers={args.transition_num_workers}, "
            f"start_method={args.transition_mp_start_method}, "
            f"chunksize={args.transition_mp_chunksize}, "
            f"min_batch={args.transition_mp_min_batch}"
        )

    try:
        initial_state_pool = list(collection.initial_states)
        if args.filter_actionable_initial_states:
            filter_start = time.perf_counter()
            initial_state_pool = engine.filter_actionable_initial_states(
                initial_state_pool,
                use_multiprocessing=args.use_multiprocessing,
                transition_pool=transition_pool,
                transition_mp_chunksize=args.transition_mp_chunksize,
                transition_mp_min_batch=args.transition_mp_min_batch,
            )
            filter_sec = time.perf_counter() - filter_start
            print(
                "Filtered initial state pool: "
                f"actionable={len(initial_state_pool)}/{len(collection.initial_states)}, "
                f"time={filter_sec:.2f}s"
            )

        if not initial_state_pool:
            raise ValueError("The initial state pool is empty after filtering.")

        states = engine.sample_initial_states(args.num_envs, rng=rng, initial_state_pool=initial_state_pool)
        rollout_buffer = PPORolloutBuffer()

        total_frt_hits = 0
        total_collapsed_hits = 0
        total_dead_end_hits = 0
        total_resets = 0
        total_expanded_states = 0
        total_discovered_states = 0
        total_mp_steps = 0
        total_candidate_expand_sec = 0.0
        total_policy_data_build_sec = 0.0
        total_policy_batch_transfer_sec = 0.0
        total_policy_value_inference_sec = 0.0
        total_policy_action_inference_sec = 0.0
        total_transition_apply_sec = 0.0
        total_candidates = 0
        total_valid_actions = 0

        rollout_start = time.perf_counter()
        for step_idx in range(int(args.rollout_steps)):
            increment_visitation(states)
            step_result = rollout_step_with_policy(
                engine,
                states,
                policy,
                rng=rng,
                device=device,
                initial_state_pool=initial_state_pool,
                deterministic=args.deterministic,
                use_multiprocessing=args.use_multiprocessing,
                transition_pool=transition_pool,
                transition_mp_chunksize=args.transition_mp_chunksize,
                transition_mp_min_batch=args.transition_mp_min_batch,
            )
            rollout_buffer.append(step_result)
            states = step_result.next_states

            total_frt_hits += int(step_result.frt_hits)
            total_collapsed_hits += int(step_result.collapsed_hits)
            total_dead_end_hits += int(step_result.dead_end_hits)
            total_resets += int(step_result.reset_count)
            total_expanded_states += int(step_result.expanded_states)
            total_discovered_states += int(step_result.discovered_states)
            total_mp_steps += int(step_result.used_multiprocessing)
            total_candidate_expand_sec += float(step_result.candidate_expand_sec)
            total_policy_data_build_sec += float(step_result.policy_data_build_sec)
            total_policy_batch_transfer_sec += float(step_result.policy_batch_transfer_sec)
            total_policy_value_inference_sec += float(step_result.policy_value_inference_sec)
            total_policy_action_inference_sec += float(step_result.policy_action_inference_sec)
            total_transition_apply_sec += float(step_result.transition_apply_sec)
            total_candidates += sum(len(actions) for actions in step_result.action_candidates)
            total_valid_actions += int(step_result.valid_action_mask.sum().item())

            should_report = args.report_every > 0 and (
                step_idx == 0
                or (step_idx + 1) % int(args.report_every) == 0
                or (step_idx + 1) == int(args.rollout_steps)
            )
            if should_report:
                rss_gb, hwm_gb = read_process_memory_gb()
                avg_reward = float(np.mean(step_result.rewards)) if step_result.rewards else 0.0
                done_fraction = float(np.mean(step_result.dones)) if step_result.dones else 0.0
                valid_action_fraction = (
                    float(step_result.valid_action_mask.float().mean().item())
                    if step_result.valid_action_mask.numel() > 0
                    else 0.0
                )
                mean_candidates = (
                    float(np.mean([len(actions) for actions in step_result.action_candidates]))
                    if step_result.action_candidates
                    else 0.0
                )
                policy_total_sec = (
                    step_result.policy_data_build_sec
                    + step_result.policy_batch_transfer_sec
                    + step_result.policy_value_inference_sec
                    + step_result.policy_action_inference_sec
                )
                gpu_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
                print(
                    f"step={step_idx + 1}/{args.rollout_steps} "
                    f"avg_reward={avg_reward:.4f} "
                    f"done_fraction={done_fraction:.4f} "
                    f"valid_action_fraction={valid_action_fraction:.4f} "
                    f"mean_candidates={mean_candidates:.2f} "
                    f"resets={step_result.reset_count} "
                    f"expanded={step_result.expanded_states} "
                    f"discovered={step_result.discovered_states} "
                    f"candidate_expand_sec={step_result.candidate_expand_sec:.4f} "
                    f"policy_total_sec={policy_total_sec:.4f} "
                    f"transition_apply_sec={step_result.transition_apply_sec:.4f} "
                    f"graph_nodes={engine.graph_node_count()} "
                    f"graph_edges={engine.graph_edge_count()} "
                    f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
                    f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
                    f"rss_gb={rss_gb:.2f} "
                    f"hwm_gb={hwm_gb:.2f} "
                    f"gpu_peak_mem_mb={gpu_peak_mem_mb:.1f}"
                )

        rollout_sec = time.perf_counter() - rollout_start

        bootstrap_candidate_start = time.perf_counter()
        bootstrap_action_lists, bootstrap_expand_summary = engine.candidate_actions_for_states(
            states,
            use_multiprocessing=args.use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=args.transition_mp_chunksize,
            transition_mp_min_batch=args.transition_mp_min_batch,
        )
        bootstrap_candidate_sec = time.perf_counter() - bootstrap_candidate_start
        bootstrap_value_result = evaluate_policy_values(
            states,
            bootstrap_action_lists,
            policy,
            device=device,
        )
        ppo_prepare_start = time.perf_counter()
        prepared_rollout = rollout_buffer.prepare(
            bootstrap_value=bootstrap_value_result.value_tensor,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=device,
        )
        ppo_prepare_sec = time.perf_counter() - ppo_prepare_start

        rss_gb, hwm_gb = read_process_memory_gb()
        env_steps = int(args.num_envs) * int(args.rollout_steps)
        env_steps_per_sec = env_steps / rollout_sec if rollout_sec > 0 else 0.0
        graph_stats = engine.graph_stats_by_polytope()
        expanded_nodes = sum(stats["expanded_nodes"] for stats in graph_stats.values())
        mean_candidates = total_candidates / env_steps if env_steps > 0 else 0.0
        valid_action_fraction = total_valid_actions / env_steps if env_steps > 0 else 0.0
        policy_total_sec = (
            total_policy_data_build_sec
            + total_policy_batch_transfer_sec
            + total_policy_value_inference_sec
            + total_policy_action_inference_sec
        )
        gpu_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        print("Policy rollout summary")
        print(
            f"env_steps={env_steps} "
            f"rollout_sec={rollout_sec:.2f} "
            f"env_steps_per_sec={env_steps_per_sec:.2f}"
        )
        print(
            f"mean_candidates={mean_candidates:.4f} "
            f"valid_action_fraction={valid_action_fraction:.4f} "
            f"frt_hits={total_frt_hits} "
            f"collapsed_hits={total_collapsed_hits} "
            f"dead_end_hits={total_dead_end_hits} "
            f"resets={total_resets}"
        )
        print(
            f"candidate_expand_sec={total_candidate_expand_sec:.2f} "
            f"policy_data_build_sec={total_policy_data_build_sec:.2f} "
            f"policy_batch_transfer_sec={total_policy_batch_transfer_sec:.2f} "
            f"policy_value_inference_sec={total_policy_value_inference_sec:.2f} "
            f"policy_action_inference_sec={total_policy_action_inference_sec:.2f} "
            f"policy_total_sec={policy_total_sec:.2f} "
            f"transition_apply_sec={total_transition_apply_sec:.2f}"
        )
        print(
            f"bootstrap_candidate_sec={bootstrap_candidate_sec:.4f} "
            f"bootstrap_build_sec={bootstrap_value_result.data_build_sec:.4f} "
            f"bootstrap_transfer_sec={bootstrap_value_result.batch_transfer_sec:.4f} "
            f"bootstrap_inference_sec={bootstrap_value_result.inference_sec:.4f} "
            f"ppo_prepare_sec={ppo_prepare_sec:.4f}"
        )
        print(
            f"ppo_samples={prepared_rollout.action_buffer_flat.size(0)} "
            f"ppo_valid_samples={int(prepared_rollout.valid_mask_flat.sum().item())} "
            f"rollout_length={prepared_rollout.reward_buffer_tensor.size(0)} "
            f"num_envs={prepared_rollout.reward_buffer_tensor.size(1)} "
            f"action_width={prepared_rollout.action_buffer_flat.size(1)}"
        )
        print(
            f"graph_nodes={engine.graph_node_count()} "
            f"graph_edges={engine.graph_edge_count()} "
            f"expanded_nodes={expanded_nodes} "
            f"expanded_states={total_expanded_states} "
            f"discovered_states={total_discovered_states} "
            f"mp_steps={total_mp_steps}/{args.rollout_steps} "
            f"bootstrap_mp={int(bootstrap_expand_summary.used_multiprocessing)}"
        )
        print(
            f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
            f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
            f"rss_gb={rss_gb:.2f} "
            f"hwm_gb={hwm_gb:.2f} "
            f"gpu_peak_mem_mb={gpu_peak_mem_mb:.1f}"
        )
    finally:
        if transition_pool is not None:
            transition_pool.shutdown()


if __name__ == "__main__":
    main(parse_args())
