from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def read_process_memory_gb() -> tuple[float, float]:
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


def memory_guard_triggered(*, max_rss_gb: float | None, rss_gb: float) -> bool:
    if max_rss_gb is None:
        return False
    return float(rss_gb) >= float(max_rss_gb)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_training_device(*, gpu_index: int, force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if int(gpu_index) < 0 or int(gpu_index) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested gpu_index={gpu_index}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    device = torch.device(f"cuda:{int(gpu_index)}")
    torch.cuda.set_device(device)
    return device


def build_checkpoint_dir(checkpoint_path: Optional[str], default_dir: str) -> str:
    checkpoint_dir = checkpoint_path if checkpoint_path is not None else default_dir
    checkpoint_dir = os.path.expanduser(str(checkpoint_dir))
    if not os.path.isabs(checkpoint_dir):
        normalized = os.path.normpath(checkpoint_dir)
        if not (normalized == "ckpt" or normalized.startswith(f"ckpt{os.sep}")):
            checkpoint_dir = os.path.join("ckpt", normalized)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory at {checkpoint_dir}")
    else:
        print(f"Checkpoint directory already exists at {checkpoint_dir}")
    return checkpoint_dir


def _resolve_policy_device(policy) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_policy_checkpoint(policy, checkpoint_path: str, map_location=None):
    if map_location is None:
        map_location = _resolve_policy_device(policy)
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=map_location)
    policy.load_state_dict(state_dict)
    policy.to(map_location)
    policy.eval()
    return policy
