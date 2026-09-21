"""Naming helpers used by current CY experiments."""

from __future__ import annotations

from pathlib import Path


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
