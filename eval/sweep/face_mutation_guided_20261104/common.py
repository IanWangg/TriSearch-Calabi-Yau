"""Shared settings for the guided face-move study (learned scorer vs random control; CPU only).

Setups are the B1 shards (same polytopes, starts and seed); they are referenced, never regenerated.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
MAIN = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
OUT = MAIN / "runs" / "face_mutation_guided_20261104"
B1_OUT = MAIN / "runs" / "face_mutation_20261104"
SETUP_ROOT = B1_OUT / "setups"
PYTHON = "/home/yiranwang/anaconda3/envs/sage/bin/python"
HF_REVISION = "60c0e119a03608418df538191f65da3f43b5b819"
HF_CACHE = str(MAIN / "eval" / "data" / "cache")
ALGORITHMS = ("face_mutation_befs_guided", "face_mutation_befs_random64")


def base_spec_kwargs(**overrides):
    """Same settings as the B1 h11 = 50 face-mutation study."""
    kwargs = dict(
        h11=50, favorable=True, hf_revision=HF_REVISION, hf_cache_dir=HF_CACHE,
        reward_function="max_kcup", cache_states=True, runtime_cache_gb=1.0, memory_budget_gb=8.0,
        transition_num_workers=1, max_hot_states=100000, force_cpu=True,
        ga_population_size=50, ga_mutation_rate=0.1, ga_elitism=1, ga_max_stalled_generations=20,
        ga_face_max_points=12, ga_face_samples=1000, beam_width=4, value_discount=0.9,
        policy_proposal_count=4, two_face_state=False,
    )
    kwargs.update(overrides)
    return kwargs


def setup_dir(seed: int, num_polytopes: int, num_starts: int) -> Path:
    return SETUP_ROOT / f"seed{seed}_p{num_polytopes}_s{num_starts}"


def shard_setup_dir(seed: int, polytope_index: int, num_starts: int) -> Path:
    return SETUP_ROOT / f"seed{seed}_poly{polytope_index}_s{num_starts}"


def result_dir(algorithm: str, seed: int, polytope_index: int, budget: int) -> Path:
    return OUT / "results" / f"{algorithm}_budget{budget}_seed{seed}_poly{polytope_index}"
