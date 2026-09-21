from __future__ import annotations

import argparse
from typing import List, Sequence

from core.cy_data_utils import split_rows_by_vertex_count
from core.cy_managed_runtime import add_managed_runtime_arguments, managed_rollout_runtime
from core.training_types import CYDatasetSplit
from core.vertex_preprocessing import (
    SUPPORTED_PREPROCESSING,
    VertexPreprocessor,
    maybe_create_vertex_preprocessor,
    normalize_preprocessing_mode,
)
from mdp.cy_state_record import NEIGHBOR_MODES
from models.subcomplex_policy_config import DEFAULT_SUBCOMPLEX_ACTOR_TYPE
from reward_functions import CY_VOLUME_REWARD_TRANSFORMS, SUPPORTED_REWARDS
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CY subcomplex policy checkpoint on the full eval split.",
    )
    add_managed_runtime_arguments(parser)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint saved by scripts/train_cy.py.",
    )
    parser.add_argument(
        "--random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a uniformly random policy. Overrides checkpoint/model arguments.",
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
        help="Optional triangulation objective. Omit to evaluate CY sampling.",
    )
    parser.add_argument(
        "--cy_volume_reward_transform",
        type=str,
        choices=CY_VOLUME_REWARD_TRANSFORMS,
        default="raw",
        help=(
            "Transform for max_cy_volume transition rewards. Raw CY volumes are "
            "still reported in the objective summary."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num_eval_polytopes",
        type=int,
        default=20,
        help="Number of hardest polytopes, by N-lattice vertex count, reserved for evaluation.",
    )
    parser.add_argument(
        "--polytope_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit eval polytope indices. Overrides --num_eval_polytopes when provided.",
    )
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use greedy action selection during evaluation.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="none",
        choices=SUPPORTED_PREPROCESSING,
        help="Optional eval-time coordinate preprocessing applied before policy inference.",
    )

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
        choices=["mlp", "gnn", "circuit_pool", "snn_simplex", "default"],
        help="Subcomplex actor architecture for loading CY RL checkpoints.",
    )
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
        "--filter_actionable_initial_states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter eval initial states to those with at least one valid action.",
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
        help="Number of worker processes. 0 selects up to eight within CPU and memory limits.",
    )
    parser.add_argument(
        "--transition_mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Managed geometry requires spawn; other retained choices raise an explicit error.",
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
        "--graph_max_nodes",
        type=int,
        default=50000,
        help="Compact the runtime graph at safe outer boundaries when runtime node count exceeds this threshold. <=0 disables.",
    )
    parser.add_argument(
        "--shared_cache_max_entries",
        type=int,
        default=50000,
        help="Cap each CY shared cache dictionary at safe outer boundaries. <=0 disables.",
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=0,
        help="Print rollout progress every N steps. <=0 disables per-step progress logs.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional path to save the evaluation summary JSON.",
    )
    return parser.parse_args(argv)


def resolve_dataset_split(
    rows: Sequence[dict],
    *,
    num_eval_polytopes: int,
    polytope_indices: Sequence[int] | None,
) -> CYDatasetSplit:
    if polytope_indices is None:
        return split_rows_by_vertex_count(rows, num_eval_polytopes=num_eval_polytopes)

    vertex_count_by_polytope: dict[int, int] = {}
    for row in rows:
        polytope_index = int(row["polytope_index"])
        vertex_count_by_polytope.setdefault(polytope_index, len(row.get("vertices", ())))

    requested_eval_indices = normalize_polytope_indices(polytope_indices)

    missing_indices = [index for index in requested_eval_indices if index not in vertex_count_by_polytope]
    if missing_indices:
        raise ValueError(f"Requested eval polytope indices are not in the dataset: {missing_indices}")

    sorted_polytopes = sorted(
        vertex_count_by_polytope,
        key=lambda polytope_index: (-vertex_count_by_polytope[polytope_index], polytope_index),
    )
    eval_polytope_set = set(requested_eval_indices)
    train_polytope_indices = [index for index in sorted_polytopes if index not in eval_polytope_set]
    train_rows = [row for row in rows if int(row["polytope_index"]) not in eval_polytope_set]
    eval_rows = [row for row in rows if int(row["polytope_index"]) in eval_polytope_set]
    return CYDatasetSplit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_polytope_indices=train_polytope_indices,
        eval_polytope_indices=requested_eval_indices,
    )


def normalize_polytope_indices(polytope_indices: Sequence[int]) -> List[int]:
    normalized_indices: List[int] = []
    seen_indices: set[int] = set()
    for polytope_index in polytope_indices:
        resolved_index = int(polytope_index)
        if resolved_index in seen_indices:
            continue
        seen_indices.add(resolved_index)
        normalized_indices.append(resolved_index)
    return normalized_indices


def resolve_eval_vertex_preprocessor(
    *,
    random_policy: bool,
    preprocessing: str,
) -> VertexPreprocessor | None:
    resolved_mode = normalize_preprocessing_mode(preprocessing)
    if random_policy and resolved_mode != "none":
        raise ValueError("--preprocessing requires policy evaluation; random rollout does not use model inputs.")
    return maybe_create_vertex_preprocessor(resolved_mode)
