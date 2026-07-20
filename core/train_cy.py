from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, TextIO

import numpy as np
import torch

from core.naming_utils import append_coordinate_dim_suffix
from mdp.cy_rollout import (
    CYRandomRolloutEngine,
    create_transition_pool,
    load_cy_sample_rows,
    build_cy_rollout_collection,
    runtime_cache_hot_size,
    runtime_cache_total_unique_states,
)
from core.training_types import (
    CYDatasetSplit,
    FirstEpisodeTracker,
    PolicyRolloutSummary,
    PPOTrainStats,
)
from core.cy_runtime_utils import (
    memory_guard_triggered,
    read_process_memory_gb,
    resolve_training_device,
    set_seeds,
)
from core.cy_data_utils import (
    polytope_vertex_count,
    split_rows_by_vertex_count,
    mean_vertex_count,
    infer_dataset_coordinate_dim,
    resolve_policy_in_channels,
    get_cy_data_tensor_cache_sizes,
    prune_cy_data_tensor_caches,
)
from core.cy_policy_rollout_utils import (
    format_rollout_summary,
    increment_visitation,
    normalize_advantages_masked,
    compute_explained_variance,
    collect_policy_rollout,
    summarize_objective_performance,
    train_policy_from_rollout,
)
from reward_functions import (
    CY_VOLUME_REWARD_TRANSFORMS,
    SUPPORTED_REWARDS,
    get_objective,
    get_reward,
    infer_goal,
)
from mdp.cy_triangulation_state import NEIGHBOR_MODES

if TYPE_CHECKING:
    from core.cy_policy_rollout_utils import PPORolloutBuffer, PreparedPPORolloutBatch
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved PPO training loop for CY subcomplex policies.",
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
    parser.add_argument(
        "--neighbor_mode",
        type=str,
        choices=NEIGHBOR_MODES,
        default="regular",
        help="Use ordinary regular neighbors or CYTools FRST two-neighbors.",
    )
    parser.add_argument(
        "--reward_function",
        "--reward",
        dest="reward_function",
        choices=SUPPORTED_REWARDS,
        default=None,
        help="Optional triangulation objective. Omit to keep CY sampling rewards.",
    )
    parser.add_argument(
        "--cy_volume_reward_transform",
        type=str,
        choices=CY_VOLUME_REWARD_TRANSFORMS,
        default="raw",
        help=(
            "Transform for max_cy_volume transition rewards. 'raw' uses "
            "V_next - V_current; 'log' uses log(V_next) - log(V_current)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num_eval_polytopes",
        type=int,
        default=20,
        help="Number of hardest polytopes, by N-lattice vertex count, reserved for evaluation.",
    )

    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--num_states",
        "--num_envs",
        dest="num_states",
        type=int,
        default=128,
        help="Number of parallel rollout environments.",
    )
    parser.add_argument("--rollout_length", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--count_bonus_coef",
        type=float,
        default=0.0,
        help="Training-only intrinsic reward coefficient for destination-state visitation counts.",
    )
    parser.add_argument(
        "--count_bonus_exponent",
        type=float,
        default=0.5,
        help="Exponent in count bonus coef / (count + 1) ** exponent.",
    )

    parser.add_argument(
        "--deterministic_rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use greedy action selection during training rollouts.",
    )
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use greedy action selection during evaluation.",
    )
    parser.add_argument("--num_eval_states", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=100)

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
        help="Subcomplex actor architecture.",
    )
    parser.add_argument(
        "--vertex_aug_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable one sampled trajectory-level similarity transform per rollout slot during rollout and value bootstrap.",
    )
    parser.add_argument(
        "--vertex_aug_prob",
        type=float,
        default=1.0,
        help="Per-trajectory probability of applying rollout augmentation.",
    )
    parser.add_argument(
        "--vertex_aug_scale_min",
        type=float,
        default=0.9,
        help="Lower bound of log-uniform isotropic scale factor.",
    )
    parser.add_argument(
        "--vertex_aug_scale_max",
        type=float,
        default=1.1,
        help="Upper bound of log-uniform isotropic scale factor.",
    )
    parser.add_argument(
        "--vertex_aug_shift_std",
        type=float,
        default=0.05,
        help="Std of random translation, relative to graph radius.",
    )
    parser.add_argument(
        "--vertex_aug_reflect_prob",
        type=float,
        default=0.1,
        help="Probability of applying a random axis reflection.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
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
        "--torch_num_threads",
        type=int,
        default=1,
        help=(
            "PyTorch intra-op CPU threads in the main training process. "
            "<=0 leaves the PyTorch default unchanged."
        ),
    )
    parser.add_argument(
        "--torch_num_interop_threads",
        type=int,
        default=1,
        help=(
            "PyTorch inter-op CPU threads in the main training process. "
            "<=0 leaves the PyTorch default unchanged."
        ),
    )

    parser.add_argument(
        "--filter_actionable_initial_states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter initial states to those with at least one valid action before training/eval sampling.",
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
        "--cache_prune_interval",
        type=int,
        default=20,
        help="Prune shared CY caches every N iterations. <=0 disables pruning.",
    )
    parser.add_argument(
        "--shared_cache_keep_mode",
        type=str,
        default="active",
        choices=["all", "active"],
        help="Retention policy for shared CY caches when pruning.",
    )
    parser.add_argument(
        "--shared_cache_max_entries",
        type=int,
        default=50000,
        help="Upper bound for each CY shared cache dictionary after pruning.",
    )
    parser.add_argument(
        "--max_rss_gb",
        type=float,
        default=None,
        help="If set, save a guard checkpoint and stop when process RSS exceeds this threshold.",
    )

    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--latest_checkpoint_interval",
        type=int,
        default=10,
        help="Save ckpt/latest.pth every N iterations. <=0 disables periodic latest writes.",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--iteration_metrics_path",
        type=str,
        default=None,
        help="Optional JSONL path flushed after every completed PPO iteration.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default=None,
        help="Suffix to append to the checkpoint and wandb run names.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="calabi_yau_rl_training")
    parser.add_argument(
        "--report_every",
        type=int,
        default=0,
        help="Deprecated compatibility option; training rollouts are summarized once after completion.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a short sanity-check training pass.",
    )
    parser.add_argument(
        "--dry_run_row_limit",
        type=int,
        default=16,
        help="Maximum number of dataset rows kept in dry-run mode.",
    )
    return parser.parse_args(argv)


def save_policy_checkpoint(policy: EGNNSubcomplexAgent, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(policy.state_dict(), str(tmp_path))
    os.replace(tmp_path, path)


def save_iteration_checkpoints(
    *,
    policy: EGNNSubcomplexAgent,
    checkpoint_dir: str,
    iteration: int,
    save_interval: int,
    latest_interval: int,
) -> None:
    iteration_one_based = int(iteration) + 1
    if save_interval > 0 and iteration_one_based % int(save_interval) == 0:
        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, f"{iteration_one_based}.pth"))
    if latest_interval > 0 and iteration_one_based % int(latest_interval) == 0:
        save_policy_checkpoint(policy, os.path.join(checkpoint_dir, "latest.pth"))


def validate_cy_volume_reward_transform_args(args: argparse.Namespace) -> None:
    transform = str(getattr(args, "cy_volume_reward_transform", "raw")).strip().lower()
    if transform not in CY_VOLUME_REWARD_TRANSFORMS:
        raise ValueError(
            f"Unknown cy_volume_reward_transform '{transform}'. "
            f"Expected one of: {', '.join(CY_VOLUME_REWARD_TRANSFORMS)}."
        )
    if transform != "raw" and getattr(args, "reward_function", None) != "max_cy_volume":
        raise ValueError(
            "--cy_volume_reward_transform log requires --reward max_cy_volume."
        )


def _return_metrics_payload(summary: PolicyRolloutSummary) -> Dict[str, float]:
    return {
        "mean": float(summary.return_mean),
        "std": float(summary.return_std),
        "min": float(summary.return_min),
        "max": float(summary.return_max),
        "discounted_mean": float(summary.discounted_reward),
        "training_mean": float(summary.training_return_mean),
        "training_discounted_mean": float(summary.training_discounted_reward),
    }


def build_raw_volume_metrics(summary: PolicyRolloutSummary) -> Dict[str, Any]:
    if summary.objective_name != "max_cy_volume":
        raise ValueError("Raw volume metrics require objective_name='max_cy_volume'.")

    initial_values = [float(value) for value in summary.objective_initial_values or ()]
    final_values = [float(value) for value in summary.objective_final_values or ()]
    best_values = [float(value) for value in summary.objective_best_values or ()]
    if not initial_values or not (
        len(initial_values) == len(final_values) == len(best_values)
    ):
        raise ValueError("Raw volume arrays must be non-empty and have equal lengths.")

    improvements = [
        best_volume - initial_volume
        for initial_volume, best_volume in zip(initial_values, best_values)
    ]
    slots = [
        {
            "slot": slot,
            "initial_volume": initial_volume,
            "final_volume": final_volume,
            "best_volume": best_volume,
            "best_volume_improvement": improvement,
        }
        for slot, (initial_volume, final_volume, best_volume, improvement) in enumerate(
            zip(initial_values, final_values, best_values, improvements)
        )
    ]
    return {
        "slots": slots,
        "initial_mean": float(np.mean(initial_values)),
        "final_mean": float(np.mean(final_values)),
        "best_mean": float(np.mean(best_values)),
        "mean_best_volume_improvement": float(np.mean(improvements)),
        "improved_fraction": float(np.mean(np.asarray(improvements) > 0.0)),
    }


def _rollout_iteration_metrics_payload(
    summary: PolicyRolloutSummary,
    *,
    deterministic: bool,
    elapsed_sec: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "deterministic": bool(deterministic),
        "return": _return_metrics_payload(summary),
        "elapsed_sec": float(elapsed_sec),
    }
    if summary.objective_name == "max_cy_volume":
        payload["raw_volume"] = build_raw_volume_metrics(summary)
    return payload


def build_iteration_metrics_record(
    *,
    iteration: int,
    reward_function: str | None,
    cy_volume_reward_transform: str,
    rollout_summary: PolicyRolloutSummary,
    eval_summary: PolicyRolloutSummary | None,
    train_stats: PPOTrainStats,
    deterministic_rollout: bool,
    deterministic_eval: bool,
    rollout_sec: float,
    bootstrap_sec: float,
    prepare_sec: float,
    train_sec: float,
    eval_sec: float,
    iteration_sec: float,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "iteration": int(iteration) + 1,
        "reward_function": reward_function,
        "cy_volume_reward_transform": str(cy_volume_reward_transform),
        "train": _rollout_iteration_metrics_payload(
            rollout_summary,
            deterministic=deterministic_rollout,
            elapsed_sec=rollout_sec,
        ),
        "eval": (
            None
            if eval_summary is None
            else _rollout_iteration_metrics_payload(
                eval_summary,
                deterministic=deterministic_eval,
                elapsed_sec=eval_sec,
            )
        ),
        "ppo": {
            "total_loss": float(train_stats.total_loss),
            "policy_loss": float(train_stats.policy_loss),
            "value_loss": float(train_stats.value_loss),
            "entropy_loss": float(train_stats.entropy_loss),
            "explained_variance": float(train_stats.explained_variance),
            "clip_ratio": float(train_stats.clip_ratio),
            "num_samples": int(train_stats.num_samples),
            "num_valid_action_samples": int(train_stats.num_valid_action_samples),
        },
        "timing": {
            "rollout_sec": float(rollout_sec),
            "bootstrap_sec": float(bootstrap_sec),
            "prepare_sec": float(prepare_sec),
            "train_sec": float(train_sec),
            "eval_sec": float(eval_sec),
            "iteration_sec": float(iteration_sec),
        },
    }


def write_iteration_metrics_record(
    metrics_stream: TextIO,
    record: Dict[str, Any],
) -> None:
    metrics_stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
    metrics_stream.write("\n")
    metrics_stream.flush()


def build_wandb_run_name(args: argparse.Namespace) -> str:
    objective_token = (
        f"{args.reward_function}-" if getattr(args, "reward_function", None) else ""
    )
    run_name = (
        f"algo-cy-{objective_token}egnn-subcomplex-ppo-improved__"
        f"hardest-eval-{int(args.num_eval_polytopes)}__"
        f"epochs-per-iter-{int(args.num_epochs)}"
    )
    if args.name_suffix:
        run_name = f"{run_name}__{args.name_suffix}"
    return run_name


def init_wandb_run(args: argparse.Namespace, *, extra_config: Dict[str, Any]) -> None:
    import wandb

    config = dict(vars(args))
    config.update(extra_config)
    wandb.init(
        project=args.wandb_project,
        name=build_wandb_run_name(args),
        config=config,
    )


def prune_cy_shared_caches(*, keep_keys: Iterable[str] | None, max_entries: int | None) -> None:
    from mdp.cy_triangulation_state import CYTriangulationState

    max_entries_int = None if max_entries is None else max(1, int(max_entries))
    keep_key_set = None if keep_keys is None else set(keep_keys)

    for cache_name in (
        "_SHARED_SUBCOMPLEX_CACHE",
        "_SHARED_NEIGHBOUR_FLIP_CACHE",
        "_SHARED_SUBCOMPLEX_TRANSITION_CACHE",
        "_SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE",
    ):
        cache_dict = getattr(CYTriangulationState, cache_name)
        if keep_key_set is not None:
            for key in list(cache_dict.keys()):
                if key not in keep_key_set:
                    cache_dict.pop(key, None)

        if max_entries_int is not None and len(cache_dict) > max_entries_int:
            overflow = len(cache_dict) - max_entries_int
            for key in list(cache_dict.keys())[:overflow]:
                cache_dict.pop(key, None)


def get_cy_shared_cache_sizes() -> Dict[str, int]:
    from mdp.cy_triangulation_state import CYTriangulationState

    return {
        "subcomplex": len(CYTriangulationState._SHARED_SUBCOMPLEX_CACHE),
        "neighbour_flip": len(CYTriangulationState._SHARED_NEIGHBOUR_FLIP_CACHE),
        "subcomplex_transition": len(CYTriangulationState._SHARED_SUBCOMPLEX_TRANSITION_CACHE),
        "subcomplex_neighbour": len(CYTriangulationState._SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE),
    }


def validate_similarity_aug_args(
    *,
    name: str,
    aug_prob: float,
    scale_min: float,
    scale_max: float,
    shift_std: float,
    reflect_prob: float,
) -> None:
    if not (0.0 <= float(aug_prob) <= 1.0):
        raise ValueError(f"{name}_prob must be in [0, 1], got {aug_prob}.")
    if float(scale_min) <= 0.0:
        raise ValueError(f"{name}_scale_min must be > 0, got {scale_min}.")
    if float(scale_max) < float(scale_min):
        raise ValueError(
            f"{name}_scale_max must be >= {name}_scale_min, got "
            f"{scale_max} < {scale_min}."
        )
    if float(shift_std) < 0.0:
        raise ValueError(f"{name}_shift_std must be >= 0, got {shift_std}.")
    if not (0.0 <= float(reflect_prob) <= 1.0):
        raise ValueError(
            f"{name}_reflect_prob must be in [0, 1], got {reflect_prob}."
        )


def validate_count_bonus_args(args: argparse.Namespace) -> None:
    count_bonus_coef = float(args.count_bonus_coef)
    count_bonus_exponent = float(args.count_bonus_exponent)
    if count_bonus_coef < 0.0:
        raise ValueError(f"count_bonus_coef must be >= 0, got {count_bonus_coef}.")
    if count_bonus_exponent <= 0.0:
        raise ValueError(
            f"count_bonus_exponent must be > 0, got {count_bonus_exponent}."
        )


def validate_neighbor_mode_args(args: argparse.Namespace) -> None:
    if (
        str(getattr(args, "neighbor_mode", "regular")) == "two_neighbors"
        and bool(args.include_points_interior_to_facets)
    ):
        raise ValueError(
            "--neighbor_mode two_neighbors requires "
            "--no-include_points_interior_to_facets because CYTools constructs "
            "two-neighbor representatives on that point configuration."
        )


def configure_torch_cpu_threads(args: argparse.Namespace) -> Dict[str, int]:
    torch_num_threads = int(getattr(args, "torch_num_threads", 1))
    torch_num_interop_threads = int(getattr(args, "torch_num_interop_threads", 1))

    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    if torch_num_interop_threads > 0:
        current_interop_threads = int(torch.get_num_interop_threads())
        if current_interop_threads != torch_num_interop_threads:
            try:
                torch.set_num_interop_threads(torch_num_interop_threads)
            except RuntimeError as exc:
                if int(torch.get_num_interop_threads()) != torch_num_interop_threads:
                    raise RuntimeError(
                        "Unable to set PyTorch inter-op threads. "
                        "Call --torch_num_interop_threads before PyTorch parallel work starts, "
                        "or use --torch_num_interop_threads 0 to keep the current setting."
                    ) from exc

    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def format_float_suffix(value: float) -> str:
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p").replace("+", "")


def normalize_subcomplex_actor_type(subcomplex_actor_type: str) -> str:
    resolved_actor_type = str(subcomplex_actor_type).strip().lower()
    if resolved_actor_type == "default":
        resolved_actor_type = "gnn"
    if resolved_actor_type not in ("mlp", "gnn", "circuit_pool", "snn_simplex"):
        raise ValueError(
            f"Unsupported subcomplex_actor_type '{subcomplex_actor_type}'. "
            "Expected one of: mlp, gnn, circuit_pool, snn_simplex, default."
        )
    return resolved_actor_type


def build_training_variant_suffix(args: argparse.Namespace) -> str:
    suffix_parts: List[str] = []
    subcomplex_actor_type = normalize_subcomplex_actor_type(
        getattr(args, "subcomplex_actor_type", "gnn")
    )
    if subcomplex_actor_type != "mlp":
        suffix_parts.append(f"actor_{subcomplex_actor_type}")

    if bool(getattr(args, "vertex_aug_enable", False)):
        suffix_parts.append("rollout_aug")

    count_bonus_coef = float(getattr(args, "count_bonus_coef", 0.0))
    if count_bonus_coef > 0.0:
        count_bonus_exponent = float(getattr(args, "count_bonus_exponent", 0.5))
        suffix_parts.append(
            "count_bonus"
            f"{format_float_suffix(count_bonus_coef)}"
            "_exp"
            f"{format_float_suffix(count_bonus_exponent)}"
        )

    if str(getattr(args, "neighbor_mode", "regular")) == "two_neighbors":
        suffix_parts.append("two_neighbors")

    if not suffix_parts:
        return ""
    return "_" + "_".join(suffix_parts)


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


def apply_dry_run_overrides(args: argparse.Namespace) -> None:
    args.max_rows = min(int(args.dry_run_row_limit), int(args.max_rows)) if args.max_rows is not None else int(args.dry_run_row_limit)
    args.num_iterations = min(int(args.num_iterations), 1)
    args.num_epochs = min(int(args.num_epochs), 1)
    args.num_states = min(int(args.num_states), 8)
    args.rollout_length = min(int(args.rollout_length), 4)
    args.num_eval_states = min(int(args.num_eval_states), 8)
    args.eval_steps = min(int(args.eval_steps), 4)
    args.batch_size = min(int(args.batch_size), 16)
    args.eval_interval = 1
    args.save_interval = 0
    args.latest_checkpoint_interval = 0
    args.report_every = 1
    args.use_wandb = False
    print(
        "Dry-run overrides: "
        f"max_rows={args.max_rows}, iterations={args.num_iterations}, epochs={args.num_epochs}, "
        f"num_states={args.num_states}, rollout_length={args.rollout_length}, "
        f"num_eval_states={args.num_eval_states}, eval_steps={args.eval_steps}, "
        f"batch_size={args.batch_size}"
    )


def main(args: argparse.Namespace) -> None:
    from core.cy_policy_rollout_utils import evaluate_policy_values
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent
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
    )
    eval_collection = build_cy_rollout_collection(
        split.eval_rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        neighbor_mode=args.neighbor_mode,
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
    )
    eval_engine = CYRandomRolloutEngine(
        collection=eval_collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
        reward_function=reward_function,
        neighbor_mode=args.neighbor_mode,
    )
    subcomplex_actor_type = normalize_subcomplex_actor_type(
        getattr(args, "subcomplex_actor_type", "gnn")
    )

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
    print(f"Using policy in_channels={resolved_in_channels}")
    print(f"Using subcomplex_actor_type={subcomplex_actor_type}")
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
                "objective_goal": objective_goal,
                "train_polytopes": len(split.train_polytope_indices),
                "eval_polytopes": len(split.eval_polytope_indices),
                "train_mean_vertices": mean_vertex_count(split.train_rows),
                "eval_mean_vertices": mean_vertex_count(split.eval_rows),
            },
        )

    transition_pool = None
    iteration_metrics_stream = None
    if args.iteration_metrics_path is not None:
        iteration_metrics_path = Path(args.iteration_metrics_path).expanduser()
        iteration_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        iteration_metrics_stream = iteration_metrics_path.open("w", encoding="utf-8")
        print(f"Writing iteration metrics to {iteration_metrics_path}")
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
        count_visit_counts_by_key: Dict[str, int] = {}
        for iteration in range(int(args.num_iterations)):
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
            )
            train_sec = time.perf_counter() - train_start

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
    finally:
        if iteration_metrics_stream is not None:
            iteration_metrics_stream.close()
        if transition_pool is not None:
            transition_pool.shutdown()
        if args.use_wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main(parse_args())
