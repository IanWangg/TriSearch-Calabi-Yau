"""Resolve saved research inputs after relocating the historical data archive."""

from pathlib import Path


def resolve_archived_path(path, *, historical_root, current_root=None):
    """Map recorded checkout paths in memory; never rewrite original evidence."""
    path = Path(path).expanduser()
    roots = {"historical": Path(historical_root).expanduser().resolve(),
             "current": Path(current_root if current_root is not None else historical_root).expanduser().resolve()}
    if not path.is_absolute():
        if path.parts and path.parts[0] in roots:
            path = roots[path.parts[0]].joinpath(*path.parts[1:])
        else:
            path = roots["historical"] / path
    else:
        text = path.as_posix()
        worktree_marker = "/.worktrees/best_so_far_reward/"
        checkout_marker = "/TriSearch-Calabi-Yau/"
        if worktree_marker in text and not path.is_relative_to(roots["current"]):
            path = roots["current"] / text.split(worktree_marker, 1)[1]
        elif worktree_marker not in text and checkout_marker in text and not path.is_relative_to(roots["historical"]):
            path = roots["historical"] / text.split(checkout_marker, 1)[1]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing archived research input: {path}. Supply --historical_root / --current_root "
            "pointing to the local data archives; generated data is not included in Git.")
    return path
