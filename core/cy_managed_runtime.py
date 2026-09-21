"""Shared managed geometry lifecycle and resource controls for CY entrypoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import math
import os


def add_managed_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory_budget_gb", type=float, default=64.0,
                        help="Combined trainer and descendant process memory budget in GiB.")
    parser.add_argument("--runtime_cache_gb", type=float, default=16.0,
                        help="Combined retained-cache allowance in GiB; individual caches use fixed shares.")
    parser.add_argument("--transition_task_timeout_sec", type=float, default=300.0,
                        help="Deadline for one immutable geometry request.")
    parser.add_argument("--policy_max_graph_size", type=int, default=250000,
                        help="Maximum graph work per physical policy batch.")
    parser.add_argument("--profile_cuda_timing", action=argparse.BooleanOptionalAction,
                        default=False, help="Synchronize CUDA when collecting detailed timing measurements.")
    parser.add_argument("--action_order", choices=("canonical", "native"), default="canonical",
                        help="Canonical order gives worker-independent action indices; native retains backend order.")


@contextmanager
def managed_rollout_runtime(args: argparse.Namespace, create_pool):
    """Own the pool and engines across collection/model setup and rollout errors."""
    from core.cy_data_utils import configure_cy_data_tensor_caches
    from core.cy_policy_inference import configure_policy_execution
    from mdp.cy_geometry_worker import configure_geometry_worker

    memory_budget = float(getattr(args, "memory_budget_gb", 64.0))
    cache_allowance = float(getattr(args, "runtime_cache_gb", 16.0))
    if not math.isfinite(memory_budget) or memory_budget <= 0:
        raise ValueError("memory_budget_gb must be finite and positive.")
    if not math.isfinite(cache_allowance) or cache_allowance < 0:
        raise ValueError("runtime_cache_gb must be finite and non-negative.")
    cache_bytes = int(min(cache_allowance, memory_budget) * 1024**3)
    requested_workers = args.transition_num_workers if args.use_multiprocessing else 1
    allocation_workers = int(requested_workers) if int(requested_workers) > 0 else min(8, os.cpu_count() or 1)
    configuration_cache_bytes = min(64 * 1024**2, cache_bytes // (32 * (allocation_workers + 1)))
    cache_bytes -= configuration_cache_bytes * (allocation_workers + 1)
    engine_cache_bytes = tensor_cache_bytes = cache_bytes // 4
    entry_limit = getattr(args, "shared_cache_max_entries", None)
    entry_limit = int(entry_limit) if entry_limit is not None and int(entry_limit) > 0 else None
    configure_cy_data_tensor_caches(max_bytes=tensor_cache_bytes,
                                   max_entries=entry_limit)
    configure_policy_execution(max_graph_size=getattr(args, "policy_max_graph_size", 250000),
                               synchronize_timing=getattr(args, "profile_cuda_timing", False))
    engines = []

    def reclaim():
        from mdp.cy_geometry_worker import clear_geometry_caches

        for engine in engines:
            engine.prune_runtime_caches(pressure=True)
        configure_cy_data_tensor_caches(max_bytes=0)
        configure_cy_data_tensor_caches(max_bytes=tensor_cache_bytes, max_entries=entry_limit)
        clear_geometry_caches()

    with ExitStack() as resources:
        transition_pool = create_pool(
            num_workers=requested_workers,
            start_method=args.transition_mp_start_method,
            memory_budget_gb=memory_budget,
            task_timeout_sec=getattr(args, "transition_task_timeout_sec", 300.0),
            reclaim_callback=reclaim,
            initializer=configure_geometry_worker,
            initargs=(cache_bytes // (4 * allocation_workers),),
            configuration_cache_bytes=configuration_cache_bytes,
        )
        resources.callback(transition_pool.shutdown)
        max_rss = getattr(args, "max_rss_gb", None)
        if max_rss is not None:
            max_rss = float(max_rss)
            if not math.isfinite(max_rss) or max_rss <= 0:
                raise ValueError("max_rss_gb must be finite and positive.")
            from core.cy_process_runtime import MemoryBudgetExceeded

            original_check = transition_pool.check_memory

            def check_memory():
                snapshot = original_check()
                with open("/proc/self/statm", encoding="ascii") as stream:
                    rss_bytes = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                if rss_bytes >= max_rss * 1024**3:
                    raise MemoryBudgetExceeded(f"Trainer RSS exceeds max_rss_gb={max_rss:g}.")
                return snapshot

            transition_pool.check_memory = check_memory

        def register_engine(engine):
            engine.action_order = getattr(args, "action_order", "canonical")
            engines.append(engine)
            resources.callback(engine.close)

        try:
            yield transition_pool, engine_cache_bytes, register_engine
        finally:
            configure_cy_data_tensor_caches(max_bytes=0)
            configure_cy_data_tensor_caches(max_bytes=tensor_cache_bytes, max_entries=entry_limit)
