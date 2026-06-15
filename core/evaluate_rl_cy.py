from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence

import numpy as np
import torch

for _message in (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
):
    warnings.filterwarnings("ignore", message=_message, category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"\n\*+\nWarning: You have enabled experimental features of CYTools\.",
    category=UserWarning,
)

from core.training_types import CYDatasetSplit, FirstEpisodeTracker, PolicyRolloutSummary
from core.cy_policy_rollout_utils import format_rollout_summary, increment_visitation
from core.cy_data_utils import mean_vertex_count, split_rows_by_vertex_count
from core.train_cy import maybe_filter_initial_state_pool, normalize_subcomplex_actor_type
from core.cy_runtime_utils import resolve_training_device, set_seeds
from core.vertex_preprocessing import (
    SUPPORTED_PREPROCESSING,
    VertexPreprocessor,
    maybe_create_vertex_preprocessor,
    normalize_preprocessing_mode,
)
from mdp.cy_rollout import (
    CYRandomRolloutEngine,
    build_cy_rollout_collection,
    create_transition_pool,
    get_rollout_memory_stats,
    load_cy_sample_rows,
    maybe_compact_rollout_memory,
)

if TYPE_CHECKING:
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent


def rollout_step_with_policy(*args: Any, **kwargs: Any) -> Any:
    from core.cy_policy_rollout_utils import rollout_step_with_policy as _rollout_step_with_policy

    return _rollout_step_with_policy(*args, **kwargs)


from core.cy_runtime_utils import load_policy_checkpoint  # noqa: E402,F401


def _format_memory_stats(stats: dict[str, int]) -> str:
    return (
        f"graph_nodes={stats['graph_nodes']} "
        f"runtime_graph_nodes={stats['runtime_graph_nodes']} "
        f"graph_edges={stats['graph_edges']} "
        f"cached_states={stats['cached_states']} "
        f"hot_cache={stats['hot_cache']} "
        f"shared_subcomplex={stats['shared_subcomplex']} "
        f"shared_neighbour_flip={stats['shared_neighbour_flip']} "
        f"shared_subcomplex_transition={stats['shared_subcomplex_transition']} "
        f"shared_subcomplex_neighbour={stats['shared_subcomplex_neighbour']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CY subcomplex policy checkpoint on the full eval split.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint saved by main_rl_egnn_subcomplex_cy_improved_fixed.py.",
    )
    parser.add_argument(
        "--random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a uniformly random policy. Overrides checkpoint/model arguments.",
    )
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
    parser.add_argument(
        "--num_eval_polytopes",
        type=int,
        default=20,
        help="Number of hardest polytopes, by N-lattice vertex count, reserved for evaluation.",
    )
    parser.add_argument(
        "--polytope_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit eval polytope indices. Overrides --num_eval_polytopes when provided.",
    )
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use greedy action selection during evaluation.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="none",
        choices=SUPPORTED_PREPROCESSING,
        help="Optional eval-time coordinate preprocessing applied before policy inference.",
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="Model input width. Defaults to the dataset vertex coordinate dimension.",
    )
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--subcomplex_actor_type",
        type=str,
        default="gnn",
        choices=["mlp", "gnn", "circuit_pool", "snn_simplex", "default"],
        help="Subcomplex actor architecture for loading CY RL checkpoints.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="CUDA device index when CUDA is available.",
    )
    parser.add_argument(
        "--force_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force CPU execution even when CUDA is available.",
    )

    parser.add_argument(
        "--filter_actionable_initial_states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter eval initial states to those with at least one valid action.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multiprocessing workers when expanding unseen CY states.",
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
        default=32,
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
        "--graph_max_nodes",
        type=int,
        default=50000,
        help="Compact the runtime graph at safe outer boundaries when runtime node count exceeds this threshold. <=0 disables.",
    )
    parser.add_argument(
        "--shared_cache_max_entries",
        type=int,
        default=50000,
        help="Cap each CY shared cache dictionary at safe outer boundaries. <=0 disables.",
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=0,
        help="Print rollout progress every N steps. <=0 disables per-step progress logs.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional path to save the evaluation summary JSON.",
    )
    return parser.parse_args()


def resolve_dataset_split(
    rows: Sequence[dict],
    *,
    num_eval_polytopes: int,
    polytope_indices: Sequence[int] | None,
) -> CYDatasetSplit:
    if polytope_indices is None:
        return split_rows_by_vertex_count(rows, num_eval_polytopes=num_eval_polytopes)

    vertex_count_by_polytope: dict[int, int] = {}
    for row in rows:
        polytope_index = int(row["polytope_index"])
        vertex_count_by_polytope.setdefault(polytope_index, len(row.get("vertices", ())))

    requested_eval_indices = normalize_polytope_indices(polytope_indices)

    missing_indices = [index for index in requested_eval_indices if index not in vertex_count_by_polytope]
    if missing_indices:
        raise ValueError(f"Requested eval polytope indices are not in the dataset: {missing_indices}")

    sorted_polytopes = sorted(
        vertex_count_by_polytope,
        key=lambda polytope_index: (-vertex_count_by_polytope[polytope_index], polytope_index),
    )
    eval_polytope_set = set(requested_eval_indices)
    train_polytope_indices = [index for index in sorted_polytopes if index not in eval_polytope_set]
    train_rows = [row for row in rows if int(row["polytope_index"]) not in eval_polytope_set]
    eval_rows = [row for row in rows if int(row["polytope_index"]) in eval_polytope_set]
    return CYDatasetSplit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_polytope_indices=train_polytope_indices,
        eval_polytope_indices=requested_eval_indices,
    )


def normalize_polytope_indices(polytope_indices: Sequence[int]) -> List[int]:
    normalized_indices: List[int] = []
    seen_indices: set[int] = set()
    for polytope_index in polytope_indices:
        resolved_index = int(polytope_index)
        if resolved_index in seen_indices:
            continue
        seen_indices.add(resolved_index)
        normalized_indices.append(resolved_index)
    return normalized_indices


from core.cy_data_utils import infer_dataset_coordinate_dim, resolve_policy_in_channels  # noqa: E402,F401


def resolve_eval_vertex_preprocessor(
    *,
    random_policy: bool,
    preprocessing: str,
) -> VertexPreprocessor | None:
    resolved_mode = normalize_preprocessing_mode(preprocessing)
    if random_policy and resolved_mode != "none":
        raise ValueError("--preprocessing requires policy evaluation; random rollout does not use model inputs.")
    return maybe_create_vertex_preprocessor(resolved_mode)


def attach_rollout_length_record(
    summary: PolicyRolloutSummary,
    rollout_lengths: Sequence[int],
) -> PolicyRolloutSummary:
    resolved_lengths = [int(length) for length in rollout_lengths]
    summary.rollout_lengths = resolved_lengths
    if resolved_lengths:
        summary.rollout_length_mean = float(np.mean(resolved_lengths))
        summary.rollout_length_min = int(min(resolved_lengths))
        summary.rollout_length_max = int(max(resolved_lengths))
    else:
        summary.rollout_length_mean = 0.0
        summary.rollout_length_min = 0
        summary.rollout_length_max = 0
    return summary


def collect_policy_rollout_over_initial_states(
    *,
    engine: CYRandomRolloutEngine,
    policy: EGNNSubcomplexAgent,
    rng: np.random.Generator,
    device: torch.device,
    initial_states: Sequence[Any],
    rollout_length: int,
    gamma: float,
    deterministic: bool,
    use_multiprocessing: bool,
    transition_pool: Any,
    transition_mp_chunksize: int,
    transition_mp_min_batch: int,
    report_every: int,
    label: str,
    vertex_preprocessor: VertexPreprocessor | None = None,
) -> PolicyRolloutSummary:
    full_initial_states = list(initial_states)
    if not full_initial_states:
        raise ValueError("initial_states must be non-empty.")

    tracker = FirstEpisodeTracker.create(num_envs=len(full_initial_states), gamma=gamma)
    active_states = list(full_initial_states)
    active_indices = list(range(len(full_initial_states)))
    final_states = list(full_initial_states)
    rollout_lengths = np.zeros(len(full_initial_states), dtype=np.int64)

    total_frt_hits = 0
    total_collapsed_hits = 0
    total_dead_end_hits = 0
    total_expanded_states = 0
    total_discovered_states = 0
    total_mp_steps = 0
    total_candidates = 0
    total_valid_actions = 0
    total_candidate_expand_sec = 0.0
    total_policy_data_build_sec = 0.0
    total_policy_batch_transfer_sec = 0.0
    total_policy_value_inference_sec = 0.0
    total_policy_action_inference_sec = 0.0
    total_transition_apply_sec = 0.0

    for step_index in range(int(rollout_length)):
        if not active_states:
            break

        increment_visitation(active_states)
        step_result = rollout_step_with_policy(
            engine,
            active_states,
            policy,
            rng=rng,
            device=device,
            initial_state_pool=full_initial_states,
            deterministic=deterministic,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
            vertex_preprocessor=vertex_preprocessor,
        )

        full_rewards = np.zeros(len(full_initial_states), dtype=np.float64)
        full_dones = np.zeros(len(full_initial_states), dtype=bool)
        full_terminal_reasons = np.full(len(full_initial_states), "continue", dtype=object)

        next_active_states: List[Any] = []
        next_active_indices: List[int] = []
        for local_idx, global_idx in enumerate(active_indices):
            rollout_lengths[global_idx] = step_index + 1
            full_rewards[global_idx] = float(step_result.rewards[local_idx])
            full_dones[global_idx] = bool(step_result.dones[local_idx])
            full_terminal_reasons[global_idx] = step_result.terminal_reasons[local_idx]

            if step_result.dones[local_idx]:
                final_states[global_idx] = step_result.transitioned_states[local_idx]
            else:
                next_state = step_result.next_states[local_idx]
                final_states[global_idx] = next_state
                next_active_states.append(next_state)
                next_active_indices.append(global_idx)

        tracker.update(
            rewards=full_rewards,
            dones=full_dones,
            terminal_reasons=full_terminal_reasons,
            step_index=step_index,
        )

        active_states = next_active_states
        active_indices = next_active_indices
        total_frt_hits += int(step_result.frt_hits)
        total_collapsed_hits += int(step_result.collapsed_hits)
        total_dead_end_hits += int(step_result.dead_end_hits)
        total_expanded_states += int(step_result.expanded_states)
        total_discovered_states += int(step_result.discovered_states)
        total_mp_steps += int(step_result.used_multiprocessing)
        total_candidates += sum(len(actions) for actions in step_result.action_candidates)
        total_valid_actions += int(step_result.valid_action_mask.sum().item())
        total_candidate_expand_sec += float(step_result.candidate_expand_sec)
        total_policy_data_build_sec += float(step_result.policy_data_build_sec)
        total_policy_batch_transfer_sec += float(step_result.policy_batch_transfer_sec)
        total_policy_value_inference_sec += float(step_result.policy_value_inference_sec)
        total_policy_action_inference_sec += float(step_result.policy_action_inference_sec)
        total_transition_apply_sec += float(step_result.transition_apply_sec)

        should_report = report_every > 0 and (
            step_index == 0
            or (step_index + 1) % int(report_every) == 0
            or (step_index + 1) == int(rollout_length)
            or not active_states
        )
        if should_report:
            step_reward = float(np.mean(step_result.rewards)) if step_result.rewards else 0.0
            done_fraction = float(np.mean(step_result.dones)) if step_result.dones else 0.0
            print(
                f"{label} step={step_index + 1}/{rollout_length} "
                f"reward_mean={step_reward:.4f} "
                f"done_fraction={done_fraction:.4f} "
                f"first_episode_finished={tracker.finished_fraction():.4f} "
                f"active_states={len(active_states)}"
            )

    return attach_rollout_length_record(
        PolicyRolloutSummary(
            final_states=final_states,
            rollout_buffer=None,
            success_rate=tracker.success_rate(),
            discounted_reward=tracker.mean_discounted_reward(),
            finished_fraction=tracker.finished_fraction(),
            finished_count=tracker.finished_count(),
            frt_hits=tracker.success_count(),
            collapsed_hits=tracker.collapsed_count(),
            dead_end_hits=tracker.dead_end_count(),
            all_step_reset_count=0,
            all_step_frt_hits=total_frt_hits,
            all_step_collapsed_hits=total_collapsed_hits,
            all_step_dead_end_hits=total_dead_end_hits,
            expanded_states=total_expanded_states,
            discovered_states=total_discovered_states,
            multiprocessing_steps=total_mp_steps,
            total_candidates=total_candidates,
            total_valid_actions=total_valid_actions,
            candidate_expand_sec=total_candidate_expand_sec,
            policy_data_build_sec=total_policy_data_build_sec,
            policy_batch_transfer_sec=total_policy_batch_transfer_sec,
            policy_value_inference_sec=total_policy_value_inference_sec,
            policy_action_inference_sec=total_policy_action_inference_sec,
            transition_apply_sec=total_transition_apply_sec,
        ),
        rollout_lengths=rollout_lengths.tolist(),
    )


def collect_random_rollout_over_initial_states(
    *,
    engine: CYRandomRolloutEngine,
    rng: np.random.Generator,
    initial_states: Sequence[Any],
    rollout_length: int,
    gamma: float,
    use_multiprocessing: bool,
    transition_pool: Any,
    transition_mp_chunksize: int,
    transition_mp_min_batch: int,
    report_every: int,
    label: str,
) -> PolicyRolloutSummary:
    full_initial_states = list(initial_states)
    if not full_initial_states:
        raise ValueError("initial_states must be non-empty.")

    tracker = FirstEpisodeTracker.create(num_envs=len(full_initial_states), gamma=gamma)
    active_states = list(full_initial_states)
    active_indices = list(range(len(full_initial_states)))
    final_states = list(full_initial_states)
    rollout_lengths = np.zeros(len(full_initial_states), dtype=np.int64)

    total_frt_hits = 0
    total_collapsed_hits = 0
    total_dead_end_hits = 0
    total_expanded_states = 0
    total_discovered_states = 0
    total_mp_steps = 0
    total_candidates = 0
    total_valid_actions = 0

    for step_index in range(int(rollout_length)):
        if not active_states:
            break

        increment_visitation(active_states)
        step_result = engine.rollout_step(
            active_states,
            rng=rng,
            initial_state_pool=full_initial_states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )

        action_lists = [
            engine.nodes_by_key[str(state.key)].candidate_actions
            for state in step_result.input_states
        ]

        full_rewards = np.zeros(len(full_initial_states), dtype=np.float64)
        full_dones = np.zeros(len(full_initial_states), dtype=bool)
        full_terminal_reasons = np.full(len(full_initial_states), "continue", dtype=object)

        next_active_states: List[Any] = []
        next_active_indices: List[int] = []
        for local_idx, global_idx in enumerate(active_indices):
            rollout_lengths[global_idx] = step_index + 1
            full_rewards[global_idx] = float(step_result.rewards[local_idx])
            full_dones[global_idx] = bool(step_result.dones[local_idx])
            full_terminal_reasons[global_idx] = step_result.terminal_reasons[local_idx]

            if step_result.dones[local_idx]:
                final_states[global_idx] = step_result.transitioned_states[local_idx]
            else:
                next_state = step_result.next_states[local_idx]
                final_states[global_idx] = next_state
                next_active_states.append(next_state)
                next_active_indices.append(global_idx)

        tracker.update(
            rewards=full_rewards,
            dones=full_dones,
            terminal_reasons=full_terminal_reasons,
            step_index=step_index,
        )

        active_states = next_active_states
        active_indices = next_active_indices
        total_frt_hits += int(step_result.frt_hits)
        total_collapsed_hits += int(step_result.collapsed_hits)
        total_dead_end_hits += int(step_result.dead_end_hits)
        total_expanded_states += int(step_result.expanded_states)
        total_discovered_states += int(step_result.discovered_states)
        total_mp_steps += int(step_result.used_multiprocessing)
        total_candidates += sum(len(actions) for actions in action_lists)
        total_valid_actions += sum(int(len(actions) > 0) for actions in action_lists)

        should_report = report_every > 0 and (
            step_index == 0
            or (step_index + 1) % int(report_every) == 0
            or (step_index + 1) == int(rollout_length)
            or not active_states
        )
        if should_report:
            step_reward = float(np.mean(step_result.rewards)) if step_result.rewards else 0.0
            done_fraction = float(np.mean(step_result.dones)) if step_result.dones else 0.0
            print(
                f"{label} step={step_index + 1}/{rollout_length} "
                f"reward_mean={step_reward:.4f} "
                f"done_fraction={done_fraction:.4f} "
                f"first_episode_finished={tracker.finished_fraction():.4f} "
                f"active_states={len(active_states)}"
            )

    return attach_rollout_length_record(
        PolicyRolloutSummary(
            final_states=final_states,
            rollout_buffer=None,
            success_rate=tracker.success_rate(),
            discounted_reward=tracker.mean_discounted_reward(),
            finished_fraction=tracker.finished_fraction(),
            finished_count=tracker.finished_count(),
            frt_hits=tracker.success_count(),
            collapsed_hits=tracker.collapsed_count(),
            dead_end_hits=tracker.dead_end_count(),
            all_step_reset_count=0,
            all_step_frt_hits=total_frt_hits,
            all_step_collapsed_hits=total_collapsed_hits,
            all_step_dead_end_hits=total_dead_end_hits,
            expanded_states=total_expanded_states,
            discovered_states=total_discovered_states,
            multiprocessing_steps=total_mp_steps,
            total_candidates=total_candidates,
            total_valid_actions=total_valid_actions,
            candidate_expand_sec=0.0,
            policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0,
            policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0,
            transition_apply_sec=0.0,
        ),
        rollout_lengths=rollout_lengths.tolist(),
    )


def build_summary_payload(
    *,
    checkpoint_path: str | None,
    policy_mode: str,
    preprocessing: str,
    device: torch.device,
    eval_initial_states: Sequence[Any],
    eval_polytope_indices: Sequence[int],
    eval_summary: PolicyRolloutSummary,
    eval_steps: int,
    eval_sec: float,
    eval_mean_vertices: float,
    graph_node_count: int,
    graph_edge_count: int,
    cached_states: int,
    hot_cache_size: int,
    shared_cache_sizes: dict[str, int],
) -> dict[str, Any]:
    rollout_lengths = [int(length) for length in getattr(eval_summary, "rollout_lengths", ())]
    rollout_length_mean = float(getattr(eval_summary, "rollout_length_mean", 0.0))
    rollout_length_min = int(getattr(eval_summary, "rollout_length_min", 0))
    rollout_length_max = int(getattr(eval_summary, "rollout_length_max", 0))
    return {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "preprocessing": normalize_preprocessing_mode(preprocessing),
        "device": str(device),
        "eval_initial_states": len(eval_initial_states),
        "eval_polytopes": len(eval_polytope_indices),
        "eval_polytope_indices": [int(index) for index in eval_polytope_indices],
        "eval_mean_vertices": float(eval_mean_vertices),
        "eval_steps": int(eval_steps),
        "rollout_lengths": rollout_lengths,
        "rollout_length_mean": rollout_length_mean,
        "rollout_length_min": rollout_length_min,
        "rollout_length_max": rollout_length_max,
        "eval_sec": float(eval_sec),
        "graph_node_count": int(graph_node_count),
        "graph_edge_count": int(graph_edge_count),
        "cached_states": int(cached_states),
        "hot_cache_size": int(hot_cache_size),
        "shared_cache_sizes": {
            "subcomplex": int(shared_cache_sizes["subcomplex"]),
            "neighbour_flip": int(shared_cache_sizes["neighbour_flip"]),
            "subcomplex_transition": int(shared_cache_sizes["subcomplex_transition"]),
            "subcomplex_neighbour": int(shared_cache_sizes["subcomplex_neighbour"]),
        },
        "success_rate": float(eval_summary.success_rate),
        "discounted_reward": float(eval_summary.discounted_reward),
        "finished_fraction": float(eval_summary.finished_fraction),
        "finished_count": int(eval_summary.finished_count),
        "frt_hits": int(eval_summary.frt_hits),
        "collapsed_hits": int(eval_summary.collapsed_hits),
        "dead_end_hits": int(eval_summary.dead_end_hits),
        "all_step_resets": int(eval_summary.all_step_reset_count),
        "all_step_frt_hits": int(eval_summary.all_step_frt_hits),
        "all_step_collapsed_hits": int(eval_summary.all_step_collapsed_hits),
        "all_step_dead_end_hits": int(eval_summary.all_step_dead_end_hits),
        "expanded_states": int(eval_summary.expanded_states),
        "discovered_states": int(eval_summary.discovered_states),
        "multiprocessing_steps": int(eval_summary.multiprocessing_steps),
        "total_candidates": int(eval_summary.total_candidates),
        "total_valid_actions": int(eval_summary.total_valid_actions),
        "candidate_expand_sec": float(eval_summary.candidate_expand_sec),
        "policy_data_build_sec": float(eval_summary.policy_data_build_sec),
        "policy_batch_transfer_sec": float(eval_summary.policy_batch_transfer_sec),
        "policy_value_inference_sec": float(eval_summary.policy_value_inference_sec),
        "policy_action_inference_sec": float(eval_summary.policy_action_inference_sec),
        "transition_apply_sec": float(eval_summary.transition_apply_sec),
    }


def main(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    resolved_preprocessing = normalize_preprocessing_mode(args.preprocessing)
    vertex_preprocessor = resolve_eval_vertex_preprocessor(
        random_policy=bool(args.random),
        preprocessing=resolved_preprocessing,
    )
    if args.random:
        device = torch.device("cpu")
        checkpoint_path = None
        print("Using uniformly random policy.")
    else:
        from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent

        if args.checkpoint_path is None:
            raise ValueError("--checkpoint_path is required unless --random is set.")
        device = resolve_training_device(gpu_index=args.gpu_index, force_cpu=bool(args.force_cpu))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Using evaluation device: {device}")

        checkpoint_path = str(Path(args.checkpoint_path).expanduser())
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_path = str(Path(args.dataset_path).expanduser())
    print(f"Loading CY dataset from {dataset_path}")
    rows = load_cy_sample_rows(dataset_path, max_rows=args.max_rows)
    dataset_coordinate_dim = infer_dataset_coordinate_dim(rows)
    split = resolve_dataset_split(
        rows,
        num_eval_polytopes=args.num_eval_polytopes,
        polytope_indices=args.polytope_indices,
    )
    print(
        "Dataset split: "
        f"train_polytopes={len(split.train_polytope_indices)} "
        f"eval_polytopes={len(split.eval_polytope_indices)} "
        f"coord_dim={dataset_coordinate_dim} "
        f"preprocessing={resolved_preprocessing} "
        f"train_mean_vertices={mean_vertex_count(split.train_rows):.2f} "
        f"eval_mean_vertices={mean_vertex_count(split.eval_rows):.2f}"
    )
    if args.polytope_indices is None:
        print(
            "Using eval polytopes selected by "
            f"--num_eval_polytopes={int(args.num_eval_polytopes)}: {split.eval_polytope_indices}"
        )
    else:
        selected_eval_indices = normalize_polytope_indices(args.polytope_indices)
        print(
            f"Using explicit eval polytope indices: {selected_eval_indices}"
        )

    build_start = time.perf_counter()
    eval_collection = build_cy_rollout_collection(
        split.eval_rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
    )
    build_sec = time.perf_counter() - build_start
    print(
        "Built CY eval collection: "
        f"eval_initial_states={len(eval_collection.initial_states)} "
        f"time={build_sec:.2f}s"
    )

    eval_engine = CYRandomRolloutEngine(
        collection=eval_collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
    )
    policy = None
    if not args.random:
        resolved_in_channels = resolve_policy_in_channels(rows, args.in_channels)
        subcomplex_actor_type = normalize_subcomplex_actor_type(
            getattr(args, "subcomplex_actor_type", "gnn")
        )
        policy = EGNNSubcomplexAgent(
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
        load_policy_checkpoint(policy, checkpoint_path, map_location=device)
        print(f"Using policy in_channels={resolved_in_channels}")
        print(f"Using subcomplex_actor_type={subcomplex_actor_type}")
        print(f"Loaded checkpoint: {checkpoint_path}")
        if vertex_preprocessor is not None:
            print(f"Applying eval preprocessing: {vertex_preprocessor.mode}")

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
        print(f"Evaluating on {len(eval_initial_state_pool)} eval initial states.")

        memory_guard = maybe_compact_rollout_memory(
            eval_engine,
            graph_max_nodes=int(args.graph_max_nodes),
            shared_cache_max_entries=int(args.shared_cache_max_entries),
        )
        if memory_guard["compacted_graph"] or memory_guard["pruned_shared"]:
            before = memory_guard["before"]
            after = memory_guard["after"]
            print(
                "Eval memory_guard "
                f"compacted_graph={memory_guard['compacted_graph']} "
                f"pruned_shared={memory_guard['pruned_shared']} "
                f"runtime_graph_nodes={before['runtime_graph_nodes']}->{after['runtime_graph_nodes']} "
                f"cached_states={before['cached_states']}->{after['cached_states']} "
                f"hot_cache={before['hot_cache']}->{after['hot_cache']} "
                f"shared_subcomplex={before['shared_subcomplex']}->{after['shared_subcomplex']}"
            )

        eval_start = time.perf_counter()
        if args.random:
            eval_summary = collect_random_rollout_over_initial_states(
                engine=eval_engine,
                rng=np.random.default_rng(args.seed + 100000),
                initial_states=eval_initial_state_pool,
                rollout_length=int(args.eval_steps),
                gamma=float(args.gamma),
                use_multiprocessing=bool(args.use_multiprocessing),
                transition_pool=transition_pool,
                transition_mp_chunksize=int(args.transition_mp_chunksize),
                transition_mp_min_batch=int(args.transition_mp_min_batch),
                report_every=int(args.report_every),
                label="eval",
            )
        else:
            policy.eval()
            eval_summary = collect_policy_rollout_over_initial_states(
                engine=eval_engine,
                policy=policy,
                rng=np.random.default_rng(args.seed + 100000),
                device=device,
                initial_states=eval_initial_state_pool,
                rollout_length=int(args.eval_steps),
                gamma=float(args.gamma),
                deterministic=bool(args.deterministic_eval),
                use_multiprocessing=bool(args.use_multiprocessing),
                transition_pool=transition_pool,
                transition_mp_chunksize=int(args.transition_mp_chunksize),
                transition_mp_min_batch=int(args.transition_mp_min_batch),
                report_every=int(args.report_every),
                label="eval",
                vertex_preprocessor=vertex_preprocessor,
            )
        eval_sec = time.perf_counter() - eval_start

        print(
            format_rollout_summary(
                label="Eval",
                summary=eval_summary,
                num_envs=len(eval_initial_state_pool),
                rollout_length=int(args.eval_steps),
            )
        )
        print(
            "Rollout length: "
            f"mean={float(getattr(eval_summary, 'rollout_length_mean', 0.0)):.2f} "
            f"min={int(getattr(eval_summary, 'rollout_length_min', 0))} "
            f"max={int(getattr(eval_summary, 'rollout_length_max', 0))}"
        )
        print(
            "System: "
            f"{_format_memory_stats(get_rollout_memory_stats(eval_engine))} "
            f"eval_sec={eval_sec:.2f}"
        )

        if args.summary_path is not None:
            memory_stats = get_rollout_memory_stats(eval_engine)
            payload = build_summary_payload(
                checkpoint_path=checkpoint_path,
                policy_mode="random" if args.random else "policy",
                preprocessing=resolved_preprocessing,
                device=device,
                eval_initial_states=eval_initial_state_pool,
                eval_polytope_indices=split.eval_polytope_indices,
                eval_summary=eval_summary,
                eval_steps=int(args.eval_steps),
                eval_sec=eval_sec,
                eval_mean_vertices=mean_vertex_count(split.eval_rows),
                graph_node_count=eval_engine.graph_node_count(),
                graph_edge_count=eval_engine.graph_edge_count(),
                cached_states=memory_stats["cached_states"],
                hot_cache_size=memory_stats["hot_cache"],
                shared_cache_sizes={
                    "subcomplex": memory_stats["shared_subcomplex"],
                    "neighbour_flip": memory_stats["shared_neighbour_flip"],
                    "subcomplex_transition": memory_stats["shared_subcomplex_transition"],
                    "subcomplex_neighbour": memory_stats["shared_subcomplex_neighbour"],
                },
            )
            summary_path = Path(args.summary_path).expanduser()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Saved summary to {summary_path}")
    finally:
        if transition_pool is not None:
            transition_pool.shutdown()


if __name__ == "__main__":
    main(parse_args())
