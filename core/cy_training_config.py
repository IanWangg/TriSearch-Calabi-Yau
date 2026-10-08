from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any, List, Sequence

import torch

from mdp.cy_state_record import NEIGHBOR_MODES
from models.subcomplex_policy_config import (
    DEFAULT_SUBCOMPLEX_ACTOR_TYPE,
    SUPPORTED_SUBCOMPLEX_ACTOR_TYPES,
    normalize_subcomplex_actor_type,
    observation_kind_for_subcomplex_actor,
    value_feature_source_for_subcomplex_actor,
)
from reward_functions import CY_VOLUME_REWARD_TRANSFORMS, SUPPORTED_REWARDS


@dataclass(frozen=True)
class CYTrainingConfig:
    dataset_path: str
    max_rows: int | None
    include_points_interior_to_facets: bool
    neighbor_mode: str
    reward_function: str | None
    cy_volume_reward_transform: str
    seed: int
    num_eval_polytopes: int
    num_iterations: int
    num_epochs: int
    num_states: int
    rollout_length: int
    gamma: float
    gae_lambda: float
    batch_size: int
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    count_bonus_coef: float
    count_bonus_exponent: float
    deterministic_rollout: bool
    deterministic_eval: bool
    num_eval_states: int
    eval_steps: int
    eval_interval: int
    in_channels: int | None
    hidden_channels: int
    out_channels: int
    num_layers: int
    subcomplex_actor_type: str
    vertex_aug_enable: bool
    vertex_aug_prob: float
    vertex_aug_scale_min: float
    vertex_aug_scale_max: float
    vertex_aug_shift_std: float
    vertex_aug_reflect_prob: float
    lr: float
    gpu_index: int
    force_cpu: bool
    torch_num_threads: int
    torch_num_interop_threads: int
    filter_actionable_initial_states: bool
    use_multiprocessing: bool
    transition_num_workers: int
    transition_mp_start_method: str
    transition_mp_chunksize: int
    transition_mp_min_batch: int
    state_cache_mode: str
    max_hot_states: int
    cache_prune_interval: int
    shared_cache_keep_mode: str
    shared_cache_max_entries: int
    max_rss_gb: float | None
    memory_budget_gb: float
    runtime_cache_gb: float
    transition_task_timeout_sec: float
    policy_max_graph_size: int
    profile_cuda_timing: bool
    action_order: str
    save_interval: int
    latest_checkpoint_interval: int
    checkpoint_path: str | None
    iteration_metrics_path: str | None
    name_suffix: str | None
    use_wandb: bool
    wandb_project: str
    report_every: int
    dry_run: bool
    dry_run_row_limit: int

    @property
    def observation_kind(self) -> str:
        return observation_kind_for_subcomplex_actor(self.subcomplex_actor_type)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CYTrainingConfig":
        values: dict[str, Any] = vars(args)
        expected = {field.name for field in fields(cls)}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "Training arguments do not match CYTrainingConfig: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return cls(**values)

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
        default=DEFAULT_SUBCOMPLEX_ACTOR_TYPE,
        choices=SUPPORTED_SUBCOMPLEX_ACTOR_TYPES,
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
        help="Geometry workers; <=0 selects at most eight within CPU and RAM limits.",
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
        default=1,
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
    parser.add_argument("--memory_budget_gb", type=float, default=64.0,
                        help="Total RAM operating budget for trainer, guardian and all workers (GiB).")
    parser.add_argument("--runtime_cache_gb", type=float, default=16.0,
                        help="Combined cache allowance within the job RAM budget (GiB).")
    parser.add_argument("--transition_task_timeout_sec", type=float, default=300.0,
                        help="Deadline for a geometry request, including native subprocesses.")
    parser.add_argument("--policy_max_graph_size", type=int, default=250000,
                        help="Maximum graph work per physical policy batch; zero disables chunking.")
    parser.add_argument("--profile_cuda_timing", action="store_true",
                        help="Synchronize CUDA for detailed phase timing (adds overhead).")
    parser.add_argument("--action_order", choices=("canonical", "native"), default="canonical",
                        help="Canonical order makes seeded rollout indices independent of worker scheduling.")

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


def validate_two_face_training_args(args: argparse.Namespace) -> None:
    if observation_kind_for_subcomplex_actor(args.subcomplex_actor_type) != "two_face":
        return
    if args.neighbor_mode != "two_neighbors" or args.include_points_interior_to_facets:
        raise ValueError("two_face_deep_sets training requires two_neighbors without facet-interior points.")
    if args.reward_function != "max_kcup" or args.in_channels not in (None, 4):
        raise ValueError("two_face_deep_sets training requires 4D max_kcup data.")
    if args.vertex_aug_enable or args.count_bonus_coef != 0:
        raise ValueError("two_face_deep_sets v1 requires augmentation disabled and count_bonus_coef=0.")


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


def build_training_variant_suffix(args: argparse.Namespace) -> str:
    suffix_parts: List[str] = []
    subcomplex_actor_type = normalize_subcomplex_actor_type(
        getattr(args, "subcomplex_actor_type", DEFAULT_SUBCOMPLEX_ACTOR_TYPE)
    )
    if subcomplex_actor_type != "mlp":
        suffix_parts.append(f"actor_{subcomplex_actor_type}")
    value_feature_source = value_feature_source_for_subcomplex_actor(subcomplex_actor_type)
    if value_feature_source != "egnn":
        suffix_parts.append(f"value_{value_feature_source}")

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
