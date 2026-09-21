from __future__ import annotations

import json
import argparse
from typing import Any, TextIO

import numpy as np

from core.training_types import PPOTrainStats, PolicyRolloutSummary

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


def _return_metrics_payload(summary: PolicyRolloutSummary) -> dict[str, float]:
    return {
        "mean": float(summary.return_mean),
        "std": float(summary.return_std),
        "min": float(summary.return_min),
        "max": float(summary.return_max),
        "discounted_mean": float(summary.discounted_reward),
        "training_mean": float(summary.training_return_mean),
        "training_discounted_mean": float(summary.training_discounted_reward),
    }


def build_raw_volume_metrics(summary: PolicyRolloutSummary) -> dict[str, Any]:
    if summary.objective_name not in {"max_cy_volume", "max_kcup"}:
        raise ValueError(
            "Raw volume metrics require objective_name='max_cy_volume' or 'max_kcup'."
        )

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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "deterministic": bool(deterministic),
        "return": _return_metrics_payload(summary),
        "elapsed_sec": float(elapsed_sec),
    }
    if summary.objective_name in {"max_cy_volume", "max_kcup"}:
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
) -> dict[str, Any]:
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
    record: dict[str, Any],
) -> None:
    metrics_stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
    metrics_stream.write("\n")
    metrics_stream.flush()
