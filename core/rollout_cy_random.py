from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from core.cy_evaluation_config import add_managed_runtime_arguments, managed_rollout_runtime
from core.cy_runtime_utils import (
    increment_visitation,
    read_process_memory_gb,
    set_seeds,
)
from mdp.cy_rollout import (
    CYRandomRolloutEngine as _CYRandomRolloutEngine,
    build_cy_rollout_collection,
    create_transition_pool,
    load_cy_sample_rows,
    runtime_cache_hot_size,
    runtime_cache_total_unique_states,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_managed_runtime_arguments(parser)
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
    parser.add_argument("--rollout_steps", type=int, default=100, help="Number of random-policy rollout steps.")
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
        help="Number of worker processes. 0 selects up to eight within CPU and memory limits.",
    )
    parser.add_argument(
        "--transition_mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Managed geometry requires spawn; other retained choices raise an explicit error.",
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
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a very small rollout for a quick smoke test.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    with managed_rollout_runtime(args, create_transition_pool) as runtime:
        _run_rollout(args, *runtime)


def _run_rollout(args, transition_pool, cache_budget_bytes, register_engine) -> None:
    set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        args.max_rows = 2 if args.max_rows is None else min(int(args.max_rows), 2)
        args.num_envs = min(int(args.num_envs), 8)
        args.rollout_steps = min(int(args.rollout_steps), 5)
        args.report_every = 1
        print(
            "Dry-run overrides: "
            f"max_rows={args.max_rows}, num_envs={args.num_envs}, rollout_steps={args.rollout_steps}"
        )

    dataset_path = str(Path(args.dataset_path).expanduser())
    print(f"Loading CY rollout dataset from {dataset_path}")
    rows = load_cy_sample_rows(dataset_path, max_rows=args.max_rows)
    print(f"Loaded {len(rows)} polytopes")

    build_start = time.perf_counter()
    collection = build_cy_rollout_collection(
        rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        transition_pool=transition_pool,
    )
    build_sec = time.perf_counter() - build_start
    print(
        "Built rollout collection: "
        f"base_states={len(collection.base_states)}, "
        f"initial_states={len(collection.initial_states)}, "
        f"polytopes={len(collection.polytope_indices)}, "
        f"time={build_sec:.2f}s"
    )

    engine = _CYRandomRolloutEngine(
        collection=collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
        transition_pool=transition_pool,
        cache_budget_bytes=cache_budget_bytes,
    )
    register_engine(engine)

    if args.use_multiprocessing:
        print(
            "Managed geometry workers: "
            f"workers={transition_pool.num_workers}, "
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

        states = engine.sample_initial_states(
            args.num_envs,
            rng=rng,
            initial_state_pool=initial_state_pool,
        )
        total_frt_hits = 0
        total_collapsed_hits = 0
        total_dead_end_hits = 0
        total_resets = 0
        total_expanded_states = 0
        total_discovered_states = 0
        total_mp_steps = 0

        rollout_start = time.perf_counter()
        for step_idx in range(int(args.rollout_steps)):
            increment_visitation(states)
            step_result = engine.rollout_step(
                states,
                rng=rng,
                initial_state_pool=initial_state_pool,
                use_multiprocessing=args.use_multiprocessing,
                transition_pool=transition_pool,
                transition_mp_chunksize=args.transition_mp_chunksize,
                transition_mp_min_batch=args.transition_mp_min_batch,
            )
            states = step_result.next_states

            total_frt_hits += int(step_result.frt_hits)
            total_collapsed_hits += int(step_result.collapsed_hits)
            total_dead_end_hits += int(step_result.dead_end_hits)
            total_resets += int(step_result.reset_count)
            total_expanded_states += int(step_result.expanded_states)
            total_discovered_states += int(step_result.discovered_states)
            total_mp_steps += int(step_result.used_multiprocessing)

            should_report = args.report_every > 0 and (
                step_idx == 0
                or (step_idx + 1) % int(args.report_every) == 0
                or (step_idx + 1) == int(args.rollout_steps)
            )
            if should_report:
                rss_gb, hwm_gb = read_process_memory_gb()
                avg_reward = float(np.mean(step_result.rewards)) if step_result.rewards else 0.0
                done_fraction = float(np.mean(step_result.dones)) if step_result.dones else 0.0
                print(
                    f"step={step_idx + 1}/{args.rollout_steps} "
                    f"avg_reward={avg_reward:.4f} "
                    f"done_fraction={done_fraction:.4f} "
                    f"resets={step_result.reset_count} "
                    f"expanded={step_result.expanded_states} "
                    f"discovered={step_result.discovered_states} "
                    f"graph_nodes={engine.graph_node_count()} "
                    f"graph_edges={engine.graph_edge_count()} "
                    f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
                    f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
                    f"rss_gb={rss_gb:.2f} "
                    f"hwm_gb={hwm_gb:.2f}"
                )

        rollout_sec = time.perf_counter() - rollout_start
        rss_gb, hwm_gb = read_process_memory_gb()
        env_steps = int(args.num_envs) * int(args.rollout_steps)
        env_steps_per_sec = env_steps / rollout_sec if rollout_sec > 0 else 0.0
        graph_stats = engine.graph_stats_by_polytope()
        expanded_nodes = sum(stats["expanded_nodes"] for stats in graph_stats.values())

        print("Rollout summary")
        print(
            f"env_steps={env_steps} "
            f"rollout_sec={rollout_sec:.2f} "
            f"env_steps_per_sec={env_steps_per_sec:.2f}"
        )
        print(
            f"graph_nodes={engine.graph_node_count()} "
            f"graph_edges={engine.graph_edge_count()} "
            f"expanded_nodes={expanded_nodes}"
        )
        print(
            f"frt_hits={total_frt_hits} "
            f"collapsed_hits={total_collapsed_hits} "
            f"dead_end_hits={total_dead_end_hits} "
            f"resets={total_resets}"
        )
        print(
            f"expanded_states={total_expanded_states} "
            f"discovered_states={total_discovered_states} "
            f"mp_steps={total_mp_steps}/{args.rollout_steps}"
        )
        print(
            f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
            f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
            f"rss_gb={rss_gb:.2f} "
            f"hwm_gb={hwm_gb:.2f}"
        )
        print(
            f"managed_workers={transition_pool.num_workers} "
            f"owned_rss_gb={transition_pool.memory_snapshot.get('rss_bytes', 0) / 1024**3:.2f} "
            f"resident_graph_nodes={engine.memory_stats()['resident_graph_nodes']}"
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main(parse_args())
