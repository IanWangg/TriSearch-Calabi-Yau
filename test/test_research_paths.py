"""Relocated archives keep recorded evidence intact and fail on missing inputs."""

import pytest

from eval.research_paths import resolve_archived_path


@pytest.mark.parametrize("recorded,root", [
    ("historical/eval/data/setup", "historical"),
    ("current/eval/data/setup", "current"),
    ("/old/TriSearch-Calabi-Yau/eval/data/setup", "historical"),
    ("/old/TriSearch-Calabi-Yau/.worktrees/best_so_far_reward/eval/data/setup", "current"),
])
def test_archived_paths_resolve_with_explicit_roots(tmp_path, recorded, root):
    roots = {name: tmp_path / name for name in ("historical", "current")}
    expected = roots[root] / "eval/data/setup"
    expected.mkdir(parents=True)
    assert resolve_archived_path(recorded, historical_root=roots["historical"],
                                 current_root=roots["current"]) == expected


def test_missing_archive_input_reports_the_required_roots(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing archived research input"):
        resolve_archived_path("eval/missing", historical_root=tmp_path)


@pytest.mark.parametrize("archive", ["TriSearch-Calabi-Yau/archive",
                                     "TriSearch-Calabi-Yau/.worktrees/best_so_far_reward/archive"])
def test_already_relocated_inputs_inside_a_checkout_are_not_remapped_twice(tmp_path, archive):
    root = tmp_path / archive
    expected = root / "eval/data/setup"
    expected.mkdir(parents=True)
    assert resolve_archived_path(expected, historical_root=root, current_root=root) == expected
