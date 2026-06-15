"""Unified naming conventions for experiments, checkpoints, and eval results.

Training experiment tag:
    {domain}_{reward}_{action_type}_{algo}_{states}s_{rollout}r_seed{seed}

Checkpoint paths:
    ckpt/{training_tag}/iter{iteration:06d}.pt
    ckpt/{training_tag}/latest.pt

Evaluation experiment tag:
    {domain}_{reward}_{eval_method}_{dataset_tag}_steps{max_steps}_seed{seed}

Eval result paths:
    eval_results/{domain}_{reward}/{eval_tag}[_ckpt{iteration:06d}].npz
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def build_training_tag(
    *,
    domain: str,
    reward: str,
    action_type: str = "subcomplex",
    algo: str = "ppo",
    num_states: int,
    rollout_length: int,
    seed: int = 0,
) -> str:
    return (
        f"{domain}_{reward}_{action_type}_{algo}"
        f"_{num_states}s_{rollout_length}r_seed{seed}"
    )


def append_coordinate_dim_suffix(
    path: str,
    coordinate_dim: int,
    *,
    default_coordinate_dim: int = 3,
) -> str:
    resolved_dim = int(coordinate_dim)
    if resolved_dim == int(default_coordinate_dim):
        return path

    path_obj = Path(path)
    return str(path_obj.with_name(f"{path_obj.name}_d{resolved_dim}"))


def build_checkpoint_path(
    training_tag: str,
    *,
    iteration: Optional[int] = None,
    latest: bool = False,
    base_dir: str = "ckpt",
) -> str:
    if latest:
        return os.path.join(base_dir, training_tag, "latest.pt")
    if iteration is None:
        raise ValueError("Provide iteration or set latest=True.")
    return os.path.join(base_dir, training_tag, f"iter{iteration:06d}.pt")


def build_eval_tag(
    *,
    domain: str,
    reward: str,
    eval_method: str,
    dataset_tag: str,
    max_steps: int,
    seed: int = 0,
) -> str:
    return (
        f"{domain}_{reward}_{eval_method}"
        f"_{dataset_tag}_steps{max_steps}_seed{seed}"
    )


def build_eval_result_path(
    eval_tag: str,
    *,
    domain: str,
    reward: str,
    checkpoint_iteration: Optional[int] = None,
    base_dir: str = "eval_results",
) -> str:
    subdir = os.path.join(base_dir, f"{domain}_{reward}")
    suffix = f"_ckpt{checkpoint_iteration:06d}" if checkpoint_iteration is not None else ""
    return os.path.join(subdir, f"{eval_tag}{suffix}.npz")
