from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any

import torch


def validate_policy_model_config(policy: Any, checkpoint_dir, *, allow_create: bool = False) -> None:
    """Require exact architecture metadata for new policies; preserve legacy loads."""
    directory = Path(checkpoint_dir)
    path = directory / "model_config.json"
    expected = getattr(policy, "model_config", None)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Checkpoint model_config mismatch: {path}")
        return
    if expected is None:
        return
    if not allow_create or any(directory.glob("*.pth")):
        raise ValueError(f"Missing model_config.json for two_face checkpoint directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    # Publish a complete file without replacing an existing configuration.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix="model_config_", suffix=".tmp",
                                     encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(expected, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            validate_policy_model_config(policy, directory)
    finally:
        temporary.unlink()


def save_policy_checkpoint(policy: Any, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_policy_model_config(policy, path.parent, allow_create=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(policy.state_dict(), str(tmp_path))
    os.replace(tmp_path, path)


def _checkpoint_iteration(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    if stem.startswith("oom_guard_iter"):
        suffix = stem.removeprefix("oom_guard_iter")
        if suffix.isdigit():
            return int(suffix)
    return None


def find_latest_policy_checkpoint(checkpoint_dir: str) -> Path | None:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        return None

    latest_path = checkpoint_path / "latest.pth"
    if latest_path.is_file():
        return latest_path

    candidates = [path for path in checkpoint_path.glob("*.pth") if path.is_file()]
    if not candidates:
        return None

    iteration_candidates = [
        (iteration, path)
        for path in candidates
        if (iteration := _checkpoint_iteration(path)) is not None
    ]
    if iteration_candidates:
        return max(iteration_candidates, key=lambda item: item[0])[1]

    return max(candidates, key=lambda path: path.stat().st_mtime)


def save_iteration_checkpoints(
    *,
    policy: Any,
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
