from __future__ import annotations

import argparse
import os
import time
import signal
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import numpy as np

from core.cy_checkpointing import (
    find_latest_policy_checkpoint,
    save_iteration_checkpoints,
    save_policy_checkpoint,
)
from core.cy_data_utils import (
    get_cy_data_tensor_cache_stats,
    get_cy_data_tensor_cache_sizes,
    infer_dataset_coordinate_dim,
    mean_vertex_count,
    prune_cy_data_tensor_caches,
    resolve_policy_in_channels,
    split_rows_by_vertex_count,
)
from core.cy_policy_rollout_utils import (
    collect_policy_rollout,
    evaluate_policy_values,
    format_rollout_summary,
    summarize_objective_performance,
    train_policy_from_rollout,
)
from core.cy_runtime_utils import (
    build_checkpoint_dir,
    load_policy_checkpoint,
    memory_guard_triggered,
    read_process_memory_gb,
    resolve_training_device,
    set_seeds,
)
from core.cy_training_config import (
    CYTrainingConfig,
    apply_dry_run_overrides,
    build_training_variant_suffix,
    configure_torch_cpu_threads,
    validate_count_bonus_args,
    validate_cy_volume_reward_transform_args,
    validate_neighbor_mode_args,
    validate_similarity_aug_args,
)
from core.cy_training_metrics import (
    build_iteration_metrics_record,
    init_wandb_run,
    write_iteration_metrics_record,
)
from core.naming_utils import append_coordinate_dim_suffix
from mdp.cy_rollout import (
    CYRandomRolloutEngine,
    build_cy_rollout_collection,
    create_transition_pool,
    get_cy_shared_cache_sizes,
    load_cy_sample_rows,
    prune_cy_shared_caches,
    runtime_cache_hot_size,
    runtime_cache_total_unique_states,
)
from models.subcomplex_policy_config import (
    DEFAULT_SUBCOMPLEX_ACTOR_TYPE,
    normalize_subcomplex_actor_type,
    value_feature_source_for_subcomplex_actor,
)
from reward_functions import get_objective, get_reward, infer_goal
from core.cy_managed_runtime import managed_rollout_runtime
from core.cy_process_runtime import MemoryBudgetExceeded


class TrainingInterrupted(RuntimeError):
    pass

def maybe_filter_initial_state_pool(
    *,
    engine: CYRandomRolloutEngine,
    initial_state_pool: Sequence[Any],
    use_filter: bool,
    use_multiprocessing: bool,
    transition_pool: Any,
    transition_mp_chunksize: int,
    transition_mp_min_batch: int,
    label: str,
) -> List[Any]:
    source_pool = list(initial_state_pool)
    if not use_filter:
        return source_pool

    filter_start = time.perf_counter()
    filtered_pool = engine.filter_actionable_initial_states(
        source_pool,
        use_multiprocessing=use_multiprocessing,
        transition_pool=transition_pool,
        transition_mp_chunksize=transition_mp_chunksize,
        transition_mp_min_batch=transition_mp_min_batch,
    )
    filter_sec = time.perf_counter() - filter_start
    print(
        f"{label} initial state filter: actionable={len(filtered_pool)}/{len(source_pool)} "
        f"time={filter_sec:.2f}s"
    )
    if not filtered_pool:
        raise ValueError(f"{label} initial state pool is empty after filtering.")
    return filtered_pool

def main(args: argparse.Namespace) -> None:
    old_handler = signal.getsignal(signal.SIGTERM)

    def interrupt_training(signum, frame):
        raise TrainingInterrupted("Training received SIGTERM.")

    signal.signal(signal.SIGTERM, interrupt_training)
    try:
        with managed_rollout_runtime(args, create_transition_pool) as resources:
            _run_training(args, *resources)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def _run_training(args: argparse.Namespace, transition_pool, cache_budget_bytes, register_engine) -> None:
    from models.subcomplex_policy_factory import build_subcomplex_agent
    from core.cy_runtime_utils import build_checkpoint_dir

    torch_thread_config = configure_torch_cpu_threads(args)
    validate_similarity_aug_args(
        name="vertex_aug",
        aug_prob=float(args.vertex_aug_prob),
        scale_min=float(args.vertex_aug_scale_min),
        scale_max=float(args.vertex_aug_scale_max),
        shift_std=float(args.vertex_aug_shift_std),
        reflect_prob=float(args.vertex_aug_reflect_prob),
    )
    validate_count_bonus_args(args)
    validate_neighbor_mode_args(args)
    validate_cy_volume_reward_transform_args(args)
    set_seeds(args.seed)
    if args.dry_run:
        apply_dry_run_overrides(args)
    args = CYTrainingConfig.from_namespace(args)

    reward_function = (
        get_reward(
            args.reward_function,
            cy_volume_reward_transform=args.cy_volume_reward_transform,
        )
        if args.reward_function is not None
        else None
    )
    objective_function = (
        get_objective(args.reward_function, reward=reward_function)
        if args.reward_function is not None
        else None
    )
    objective_goal = (
        infer_goal(args.reward_function) if args.reward_function is not None else None
    )
    if reward_function is None:
        print("Using CY sampling reward.")
    else:
        print(
            f"Using triangulation objective: reward={args.reward_function} "
            f"goal={objective_goal} "
            f"cy_volume_reward_transform={args.cy_volume_reward_transform}"
        )

    device = resolve_training_device(gpu_index=args.gpu_index, force_cpu=bool(args.force_cpu))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Using training device: {device}")
    print(
        "Using PyTorch CPU threads: "
        f"intra_op={torch_thread_config['torch_num_threads']} "
        f"interop={torch_thread_config['torch_num_interop_threads']}"
    )

    dataset_path = str(Path(args.dataset_path).expanduser())
    print(f"Loading CY dataset from {dataset_path}")
    rows = load_cy_sample_rows(dataset_path, max_rows=args.max_rows)
    dataset_coordinate_dim = infer_dataset_coordinate_dim(rows)
    resolved_in_channels = resolve_policy_in_channels(rows, args.in_channels)
    split = split_rows_by_vertex_count(rows, num_eval_polytopes=args.num_eval_polytopes)
    print(
        "Dataset split: "
        f"train_polytopes={len(split.train_polytope_indices)} "
        f"eval_polytopes={len(split.eval_polytope_indices)} "
        f"coord_dim={dataset_coordinate_dim} "
        f"train_mean_vertices={mean_vertex_count(split.train_rows):.2f} "
        f"eval_mean_vertices={mean_vertex_count(split.eval_rows):.2f}"
    )
    print(
        f"Hardest eval polytopes: {split.eval_polytope_indices[: min(10, len(split.eval_polytope_indices))]}"
    )

    build_start = time.perf_counter()
    train_collection = build_cy_rollout_collection(
        split.train_rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        neighbor_mode=args.neighbor_mode,
        transition_pool=transition_pool,
    )
    eval_collection = build_cy_rollout_collection(
        split.eval_rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        neighbor_mode=args.neighbor_mode,
        transition_pool=transition_pool,
    )
    build_sec = time.perf_counter() - build_start
    print(
        "Built CY collections: "
        f"train_initial_states={len(train_collection.initial_states)} "
        f"eval_initial_states={len(eval_collection.initial_states)} "
        f"time={build_sec:.2f}s"
    )

    train_engine = CYRandomRolloutEngine(
        collection=train_collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
        reward_function=reward_function,
        neighbor_mode=args.neighbor_mode,
        cache_budget_bytes=cache_budget_bytes,
        transition_pool=transition_pool,
    )
    register_engine(train_engine)
    eval_engine = CYRandomRolloutEngine(
        collection=eval_collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
        reward_function=reward_function,
        neighbor_mode=args.neighbor_mode,
        cache_budget_bytes=cache_budget_bytes,
        transition_pool=transition_pool,
    )
    register_engine(eval_engine)
    subcomplex_actor_type = normalize_subcomplex_actor_type(
        getattr(args, "subcomplex_actor_type", DEFAULT_SUBCOMPLEX_ACTOR_TYPE)
    )
    value_feature_source = value_feature_source_for_subcomplex_actor(subcomplex_actor_type)

    objective_prefix = f"{args.reward_function}_" if args.reward_function else ""
    checkpoint_dir = build_checkpoint_dir(
        checkpoint_path=args.checkpoint_path,
        default_dir=append_coordinate_dim_suffix(
            (
                f"ckpt/cy_{objective_prefix}subcomplex_ppo_improved_"
                f"{int(args.num_states)}state_{int(args.rollout_length)}rollout"
                f"{build_training_variant_suffix(args)}"
                f"{'_' + args.name_suffix if args.name_suffix else ''}"
            ),
            dataset_coordinate_dim,
        ),
    )
    policy = build_subcomplex_agent(
        model_type="egnn",
        in_channels=resolved_in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        share_encoder=True,
        mlp_hidden_channel_list=[64],
        act="silu",
        subcomplex_actor_type=subcomplex_actor_type,
        device=str(device),
    ).to(device)
    print(f"Using policy in_channels={resolved_in_channels}")
    print(f"Using subcomplex_actor_type={subcomplex_actor_type}")
    print(f"Using value_feature_source={value_feature_source}")
    latest_policy_checkpoint = find_latest_policy_checkpoint(checkpoint_dir)
    if latest_policy_checkpoint is not None:
        print(f"Loading policy checkpoint from {latest_policy_checkpoint}")
        load_policy_checkpoint(policy, str(latest_policy_checkpoint), map_location=device)
        print(
            "Loaded policy checkpoint. Optimizer state and iteration counters are "
            "not restored because existing checkpoints contain policy weights only."
        )
    else:
        print(f"No existing policy checkpoint found in {checkpoint_dir}; starting fresh.")
    if args.vertex_aug_enable:
        print(
            "Using rollout vertex augmentation: "
            f"prob={args.vertex_aug_prob}, "
            f"scale=[{args.vertex_aug_scale_min}, {args.vertex_aug_scale_max}], "
            f"shift_std={args.vertex_aug_shift_std}, "
            f"reflect_prob={args.vertex_aug_reflect_prob}"
        )
    if float(args.count_bonus_coef) > 0.0:
        print(
            "Using training count bonus: "
            f"coef={float(args.count_bonus_coef)}, "
            f"exponent={float(args.count_bonus_exponent)}"
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))

    if args.use_wandb:
        init_wandb_run(
            args,
            extra_config={
                "coordinate_dim": dataset_coordinate_dim,
                "resolved_in_channels": resolved_in_channels,
                "subcomplex_actor_type": subcomplex_actor_type,
                "value_feature_source": value_feature_source,
                "objective_goal": objective_goal,
                "train_polytopes": len(split.train_polytope_indices),
                "eval_polytopes": len(split.eval_polytope_indices),
                "train_mean_vertices": mean_vertex_count(split.train_rows),
                "eval_mean_vertices": mean_vertex_count(split.eval_rows),
            },
        )

    # Compact collections own all data needed beyond initialization.
    del rows, split
    iteration_metrics_stream = None
    if args.iteration_metrics_path is not None:
        iteration_metrics_path = Path(args.iteration_metrics_path).expanduser()
        iteration_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        iteration_metrics_stream = iteration_metrics_path.open("w", encoding="utf-8")
        print(f"Writing iteration metrics to {iteration_metrics_path}")
    print(f"Managed geometry: trainer_pid={os.getpid()} guardian_pid={transition_pool.guardian_pid} "
          f"workers={len(transition_pool.worker_pids)} budget_gb={args.memory_budget_gb:g} "
          f"action_order={args.action_order}")

    try:
        train_initial_state_pool = maybe_filter_initial_state_pool(
            engine=train_engine,
            initial_state_pool=train_collection.initial_states,
            use_filter=bool(args.filter_actionable_initial_states),
            use_multiprocessing=bool(args.use_multiprocessing),
            transition_pool=transition_pool,
            transition_mp_chunksize=int(args.transition_mp_chunksize),
            transition_mp_min_batch=int(args.transition_mp_min_batch),
            label="Train",
        )
        eval_initial_state_pool = maybe_filter_initial_state_pool(
            engine=eval_engine,
            initial_state_pool=eval_collection.initial_states,
            use_filter=bool(args.filter_actionable_initial_states),
            use_multiprocessing=bool(args.use_multiprocessing),
            transition_pool=transition_pool,
            transition_mp_chunksize=int(args.transition_mp_chunksize),
            transition_mp_min_batch=int(args.transition_mp_min_batch),
            label="Eval",
        )

        original_train_state_count = len(train_collection.base_states)
        count_visit_counts_by_key = (train_engine.history.counts("visitation")
                                    if train_engine.history is not None else {})
        for iteration in range(int(args.num_iterations)):
            transition_pool.check_memory()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            iter_start = time.perf_counter()
            print(f"Iteration {iteration + 1}/{args.num_iterations}")

            policy.eval()
            rollout_start = time.perf_counter()
            rollout_summary = collect_policy_rollout(
                engine=train_engine,
                policy=policy,
                rng=np.random.default_rng(args.seed + iteration),
                device=device,
                initial_state_pool=train_initial_state_pool,
                num_envs=int(args.num_states),
                rollout_length=int(args.rollout_length),
                gamma=float(args.gamma),
                deterministic=bool(args.deterministic_rollout),
                use_multiprocessing=bool(args.use_multiprocessing),
                transition_pool=transition_pool,
                transition_mp_chunksize=int(args.transition_mp_chunksize),
                transition_mp_min_batch=int(args.transition_mp_min_batch),
                store_buffer=True,
                report_every=int(args.report_every),
                label="rollout",
                count_bonus_coef=float(args.count_bonus_coef),
                count_bonus_exponent=float(args.count_bonus_exponent),
                visit_counts_by_key=count_visit_counts_by_key,
                vertex_aug_enable=bool(args.vertex_aug_enable),
                vertex_aug_prob=float(args.vertex_aug_prob),
                vertex_aug_scale_min=float(args.vertex_aug_scale_min),
                vertex_aug_scale_max=float(args.vertex_aug_scale_max),
                vertex_aug_shift_std=float(args.vertex_aug_shift_std),
                vertex_aug_reflect_prob=float(args.vertex_aug_reflect_prob),
                objective_function=objective_function,
                objective_name=args.reward_function,
                objective_goal=objective_goal,
            )
            rollout_sec = time.perf_counter() - rollout_start
            print(
                format_rollout_summary(
                    label="Rollout",
                    summary=rollout_summary,
                    num_envs=int(args.num_states),
                    rollout_length=int(args.rollout_length),
                )
            )

            bootstrap_start = time.perf_counter()
            bootstrap_action_lists, bootstrap_expand_summary = train_engine.candidate_actions_for_states(
                rollout_summary.final_states,
                use_multiprocessing=bool(args.use_multiprocessing),
                transition_pool=transition_pool,
                transition_mp_chunksize=int(args.transition_mp_chunksize),
                transition_mp_min_batch=int(args.transition_mp_min_batch),
            )
            bootstrap_value_result = evaluate_policy_values(
                rollout_summary.final_states,
                bootstrap_action_lists,
                policy,
                device=device,
                trajectory_transforms=rollout_summary.trajectory_transforms,
            )
            bootstrap_sec = time.perf_counter() - bootstrap_start

            if rollout_summary.rollout_buffer is None:
                raise RuntimeError("Training rollout did not retain PPO rollout data.")
            prepare_start = time.perf_counter()
            transition_pool.check_memory()
            prepared_rollout = rollout_summary.rollout_buffer.prepare(
                bootstrap_value=bootstrap_value_result.value_tensor,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                device=device,
            )
            prepare_sec = time.perf_counter() - prepare_start

            train_start = time.perf_counter()
            train_stats = train_policy_from_rollout(
                policy=policy,
                optimizer=optimizer,
                prepared_rollout=prepared_rollout,
                device=device,
                num_epochs=int(args.num_epochs),
                batch_size=int(args.batch_size),
                clip_coef=float(args.clip_coef),
                value_coef=float(args.value_coef),
                entropy_coef=float(args.entropy_coef),
                max_grad_norm=float(args.max_grad_norm),
                memory_guard=transition_pool.check_memory,
            )
            train_sec = time.perf_counter() - train_start
            rollout_summary.rollout_buffer.clear()
            rollout_summary.rollout_buffer = None
            del prepared_rollout
            train_engine.release_active_states()
            transition_pool.check_memory()

            eval_summary = None
            eval_sec = 0.0
            if int(args.eval_interval) > 0 and iteration % int(args.eval_interval) == 0:
                policy.eval()
                eval_start = time.perf_counter()
                eval_summary = collect_policy_rollout(
                    engine=eval_engine,
                    policy=policy,
                    rng=np.random.default_rng(args.seed + 100000 + iteration),
                    device=device,
                    initial_state_pool=eval_initial_state_pool,
                    num_envs=int(args.num_eval_states),
                    rollout_length=int(args.eval_steps),
                    gamma=float(args.gamma),
                    deterministic=bool(args.deterministic_eval),
                    use_multiprocessing=bool(args.use_multiprocessing),
                    transition_pool=transition_pool,
                    transition_mp_chunksize=int(args.transition_mp_chunksize),
                    transition_mp_min_batch=int(args.transition_mp_min_batch),
                    store_buffer=False,
                    report_every=0,
                    label="eval",
                    objective_function=objective_function,
                    objective_name=args.reward_function,
                    objective_goal=objective_goal,
                )
                eval_sec = time.perf_counter() - eval_start
                eval_engine.release_active_states()
                print(
                    format_rollout_summary(
                        label="Eval",
                        summary=eval_summary,
                        num_envs=int(args.num_eval_states),
                        rollout_length=int(args.eval_steps),
                    )
                )

            if int(args.cache_prune_interval) > 0 and (iteration + 1) % int(args.cache_prune_interval) == 0:
                if args.shared_cache_keep_mode == "all":
                    keep_keys = None
                else:
                    keep_keys = (
                        set(train_engine.state_cache.base_states.keys())
                        | set(train_engine.state_cache.hot_states.keys())
                        | set(eval_engine.state_cache.base_states.keys())
                        | set(eval_engine.state_cache.hot_states.keys())
                    )
                prune_cy_shared_caches(
                    keep_keys=keep_keys,
                    max_entries=args.shared_cache_max_entries,
                )
                prune_cy_data_tensor_caches(
                    keep_keys=keep_keys,
                    max_entries=args.shared_cache_max_entries,
                )

            rss_gb, hwm_gb = read_process_memory_gb()
            job_memory = transition_pool.check_memory()
            resident_memory = train_engine.memory_stats()
            if train_engine.history is not None:
                train_engine.history.flush()
            shared_cache_sizes = get_cy_shared_cache_sizes()
            data_cache_sizes = get_cy_data_tensor_cache_sizes()
            iteration_sec = time.perf_counter() - iter_start
            total_unique_train_states = runtime_cache_total_unique_states(train_engine.state_cache)
            total_discovered_train_states = total_unique_train_states - original_train_state_count
            gpu_peak_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )

            print(
                "Train: "
                f"policy_loss={train_stats.policy_loss:.6f} "
                f"value_loss={train_stats.value_loss:.6f} "
                f"explained_variance={train_stats.explained_variance:.6f} "
                f"clip_ratio={train_stats.clip_ratio:.6f}"
            )
            print(
                "System: "
                f"rss_gb={rss_gb:.2f} "
                f"hwm_gb={hwm_gb:.2f} "
                f"gpu_peak_mem_mb={gpu_peak_mem_mb:.1f} "
                f"graph_nodes={train_engine.graph_node_count()} "
                f"graph_edges={train_engine.graph_edge_count()} "
                f"cached_states={total_unique_train_states} "
                f"discovered_states={total_discovered_train_states} "
                f"hot_cache={runtime_cache_hot_size(train_engine.state_cache)} "
                f"shared_subcomplex_cache={shared_cache_sizes['subcomplex']} "
                f"shared_transition_cache={shared_cache_sizes['subcomplex_transition']} "
                f"data_graph_cache={data_cache_sizes['graph']} "
                f"data_subcomplex_cache={data_cache_sizes['subcomplex']} "
                f"count_bonus_tracked_states={len(count_visit_counts_by_key)}"
                f" resident_graph_nodes={resident_memory['resident_graph_nodes']}"
                f" resident_graph_mb={resident_memory['resident_graph_bytes'] / 1024**2:.1f}"
                f" job_memory={job_memory}"
            )
            print(
                "Timing: "
                f"rollout_sec={rollout_sec:.2f} "
                f"bootstrap_sec={bootstrap_sec:.2f} "
                f"prepare_sec={prepare_sec:.2f} "
                f"train_sec={train_sec:.2f} "
                f"eval_sec={eval_sec:.2f} "
                f"iteration_sec={iteration_sec:.2f}"
            )

            if iteration_metrics_stream is not None:
                iteration_record = build_iteration_metrics_record(
                    iteration=iteration,
                    reward_function=args.reward_function,
                    cy_volume_reward_transform=args.cy_volume_reward_transform,
                    rollout_summary=rollout_summary,
                    eval_summary=eval_summary,
                    train_stats=train_stats,
                    deterministic_rollout=bool(args.deterministic_rollout),
                    deterministic_eval=bool(args.deterministic_eval),
                    rollout_sec=rollout_sec,
                    bootstrap_sec=bootstrap_sec,
                    prepare_sec=prepare_sec,
                    train_sec=train_sec,
                    eval_sec=eval_sec,
                    iteration_sec=iteration_sec,
                )
                iteration_record["memory"] = {"job": job_memory, "train": resident_memory,
                    "eval": eval_engine.memory_stats(), "tensor_caches": get_cy_data_tensor_cache_stats(),
                    "workers": transition_pool.stats}
                iteration_record["action_order"] = args.action_order
                write_iteration_metrics_record(iteration_metrics_stream, iteration_record)

            if args.use_wandb:
                import wandb

                payload = {
                    "rollout/return": rollout_summary.return_mean,
                    "rollout/return_std": rollout_summary.return_std,
                    "rollout/return_min": rollout_summary.return_min,
                    "rollout/return_max": rollout_summary.return_max,
                    "rollout/training_return": rollout_summary.training_return_mean,
                    "rollout/success_rate": rollout_summary.success_rate,
                    "rollout/discounted_reward": rollout_summary.discounted_reward,
                    "rollout/training_discounted_reward": rollout_summary.training_discounted_reward,
                    "rollout/intrinsic_bonus_mean": rollout_summary.intrinsic_bonus_mean,
                    "rollout/finished_fraction": rollout_summary.finished_fraction,
                    "rollout/finished_count": rollout_summary.finished_count,
                    "rollout/frt_hits": rollout_summary.frt_hits,
                    "rollout/collapsed_hits": rollout_summary.collapsed_hits,
                    "rollout/dead_end_hits": rollout_summary.dead_end_hits,
                    "rollout/all_step_resets": rollout_summary.all_step_reset_count,
                    "rollout/all_step_frt_hits": rollout_summary.all_step_frt_hits,
                    "rollout/all_step_collapsed_hits": rollout_summary.all_step_collapsed_hits,
                    "rollout/all_step_dead_end_hits": rollout_summary.all_step_dead_end_hits,
                    "rollout/expanded_states": rollout_summary.expanded_states,
                    "rollout/discovered_states": rollout_summary.discovered_states,
                    "rollout/total_num_states_visited": total_unique_train_states,
                    "rollout/total_num_states_discovered": total_discovered_train_states,
                    "rollout/count_bonus_tracked_states": len(count_visit_counts_by_key),
                    "train/total_loss": train_stats.total_loss,
                    "train/policy_loss": train_stats.policy_loss,
                    "train/value_loss": train_stats.value_loss,
                    "train/entropy_loss": train_stats.entropy_loss,
                    "train/explained_variance": train_stats.explained_variance,
                    "train/clip_ratio": train_stats.clip_ratio,
                    "train/num_samples": train_stats.num_samples,
                    "train/num_valid_action_samples": train_stats.num_valid_action_samples,
                    "system/rss_gb": rss_gb,
                    "system/hwm_gb": hwm_gb,
                    "system/gpu_peak_mem_mb": gpu_peak_mem_mb,
                    "system/hot_cache_size": runtime_cache_hot_size(train_engine.state_cache),
                    "system/shared_subcomplex_cache": shared_cache_sizes["subcomplex"],
                    "system/shared_neighbour_cache": shared_cache_sizes["neighbour_flip"],
                    "system/shared_transition_cache": shared_cache_sizes["subcomplex_transition"],
                    "system/shared_neighbour_obj_cache": shared_cache_sizes["subcomplex_neighbour"],
                    "system/data_graph_cache": data_cache_sizes["graph"],
                    "system/data_subcomplex_cache": data_cache_sizes["subcomplex"],
                    "timing/rollout_sec": rollout_sec,
                    "timing/bootstrap_sec": bootstrap_sec,
                    "timing/bootstrap_expand_mp": float(bootstrap_expand_summary.used_multiprocessing),
                    "timing/bootstrap_value_build_sec": bootstrap_value_result.data_build_sec,
                    "timing/bootstrap_value_transfer_sec": bootstrap_value_result.batch_transfer_sec,
                    "timing/bootstrap_value_inference_sec": bootstrap_value_result.inference_sec,
                    "timing/prepare_sec": prepare_sec,
                    "timing/train_sec": train_sec,
                    "timing/eval_sec": eval_sec,
                    "timing/iteration_sec": iteration_sec,
                    "timing/rollout_candidate_expand_sec": rollout_summary.candidate_expand_sec,
                    "timing/rollout_policy_data_build_sec": rollout_summary.policy_data_build_sec,
                    "timing/rollout_policy_batch_transfer_sec": rollout_summary.policy_batch_transfer_sec,
                    "timing/rollout_policy_value_inference_sec": rollout_summary.policy_value_inference_sec,
                    "timing/rollout_policy_action_inference_sec": rollout_summary.policy_action_inference_sec,
                    "timing/rollout_transition_apply_sec": rollout_summary.transition_apply_sec,
                }
                rollout_objective_metrics = summarize_objective_performance(rollout_summary)
                payload.update(
                    {
                        f"rollout/objective_{name}": value
                        for name, value in rollout_objective_metrics.items()
                    }
                )
                if eval_summary is not None:
                    payload.update(
                        {
                            "eval/return_mean": eval_summary.return_mean,
                            "eval/return_std": eval_summary.return_std,
                            "eval/return_min": eval_summary.return_min,
                            "eval/return_max": eval_summary.return_max,
                            "eval/success_rate": eval_summary.success_rate,
                            "eval/discounted_reward": eval_summary.discounted_reward,
                            "eval/finished_fraction": eval_summary.finished_fraction,
                            "eval/finished_count": eval_summary.finished_count,
                            "eval/frt_hits": eval_summary.frt_hits,
                            "eval/collapsed_hits": eval_summary.collapsed_hits,
                            "eval/dead_end_hits": eval_summary.dead_end_hits,
                            "eval/all_step_resets": eval_summary.all_step_reset_count,
                            "eval/all_step_frt_hits": eval_summary.all_step_frt_hits,
                            "eval/all_step_collapsed_hits": eval_summary.all_step_collapsed_hits,
                            "eval/all_step_dead_end_hits": eval_summary.all_step_dead_end_hits,
                        }
                    )
                    eval_objective_metrics = summarize_objective_performance(eval_summary)
                    payload.update(
                        {
                            f"eval/objective_{name}": value
                            for name, value in eval_objective_metrics.items()
                        }
                    )
                wandb.log(payload, step=iteration)

            save_iteration_checkpoints(
                policy=policy,
                checkpoint_dir=checkpoint_dir,
                iteration=iteration,
                save_interval=int(args.save_interval),
                latest_interval=int(args.latest_checkpoint_interval),
            )

            if memory_guard_triggered(max_rss_gb=args.max_rss_gb, rss_gb=rss_gb):
                guard_path = os.path.join(checkpoint_dir, f"oom_guard_iter{iteration + 1}.pth")
                save_policy_checkpoint(policy, guard_path)
                save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "latest.pth"))
                print(
                    "Memory guard triggered: "
                    f"rss_gb={rss_gb:.2f} >= max_rss_gb={args.max_rss_gb}. "
                    f"Saved {guard_path} and latest.pth, then exiting."
                )
                return

        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "final.pth"))
        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "latest.pth"))
    except (MemoryBudgetExceeded, TrainingInterrupted) as exc:
        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "oom_guard_interrupted.pth"))
        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "latest.pth"))
        print(f"Training stopped safely: {exc}. Saved current policy weights before cleanup.")
    finally:
        if iteration_metrics_stream is not None:
            iteration_metrics_stream.close()
        if args.use_wandb:
            import wandb

            wandb.finish()
