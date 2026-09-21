from __future__ import annotations

import warnings
from typing import Any

for _message in (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
):
    warnings.filterwarnings("ignore", message=_message, category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"\n\*+\nWarning: You have enabled experimental features of CYTools\.",
    category=UserWarning,
)

import core.cy_evaluation as _evaluation
from core.cy_evaluation import (
    attach_objective_record,
    attach_rollout_length_record,
    build_summary_payload,
    collect_random_rollout_over_initial_states,
    load_policy_checkpoint,
    main,
)
from core.cy_evaluation_config import (
    normalize_polytope_indices,
    parse_args,
    resolve_dataset_split,
    resolve_eval_vertex_preprocessor,
)
from core.cy_data_utils import infer_dataset_coordinate_dim, resolve_policy_in_channels
from core.cy_policy_rollout_utils import rollout_step_with_policy as _rollout_step_with_policy
from core.training_types import PolicyRolloutSummary


def rollout_step_with_policy(*args: Any, **kwargs: Any) -> Any:
    return _rollout_step_with_policy(*args, **kwargs)


def collect_policy_rollout_over_initial_states(*args: Any, **kwargs: Any) -> PolicyRolloutSummary:
    original = _evaluation.rollout_step_with_policy
    _evaluation.rollout_step_with_policy = rollout_step_with_policy
    try:
        return _evaluation.collect_policy_rollout_over_initial_states(*args, **kwargs)
    finally:
        _evaluation.rollout_step_with_policy = original


__all__ = [
    "PolicyRolloutSummary",
    "attach_objective_record",
    "attach_rollout_length_record",
    "build_summary_payload",
    "collect_policy_rollout_over_initial_states",
    "collect_random_rollout_over_initial_states",
    "infer_dataset_coordinate_dim",
    "load_policy_checkpoint",
    "main",
    "normalize_polytope_indices",
    "parse_args",
    "resolve_dataset_split",
    "resolve_eval_vertex_preprocessor",
    "resolve_policy_in_channels",
    "rollout_step_with_policy",
]


if __name__ == "__main__":
    main(parse_args())
