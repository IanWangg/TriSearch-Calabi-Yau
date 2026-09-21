from __future__ import annotations

import core.cy_policy_inference as _policy_inference
import core.cy_policy_rollout as _policy_rollout

from core.cy_policy_inference import (
    PolicyActionEvaluationResult,
    PolicyActionSelectionResult,
    PolicyRolloutStepResult,
    PolicyValueResult,
    batched_policy_action_selection,
    build_cy_data_list,
    evaluate_policy_actions,
    evaluate_policy_actions_from_data_list,
    evaluate_policy_values,
    infer_batch_subcomplex_width,
)
from core.cy_policy_rollout import (
    compute_cy_state_count_bonus,
    format_rollout_summary,
    get_cy_state_visit_count,
    rollout_return_statistics,
    sample_cy_trajectory_transforms,
    summarize_objective_performance,
)
from core.cy_ppo import (
    PPORolloutBuffer,
    PreparedPPORolloutBatch,
    compute_explained_variance,
    compute_gae_with_dones,
    flatten_action_buffer,
    flatten_buffer,
    normalize_advantages_masked,
    train_policy_from_rollout,
)
from core.cy_runtime_utils import increment_visitation


def rollout_step_with_policy(*args, **kwargs):
    original = _policy_inference.batched_policy_action_selection
    _policy_inference.batched_policy_action_selection = batched_policy_action_selection
    try:
        return _policy_inference.rollout_step_with_policy(*args, **kwargs)
    finally:
        _policy_inference.batched_policy_action_selection = original


def collect_policy_rollout(*args, **kwargs):
    original = _policy_rollout.rollout_step_with_policy
    _policy_rollout.rollout_step_with_policy = rollout_step_with_policy
    try:
        return _policy_rollout.collect_policy_rollout(*args, **kwargs)
    finally:
        _policy_rollout.rollout_step_with_policy = original

__all__ = [
    "PPORolloutBuffer",
    "PolicyActionEvaluationResult",
    "PolicyActionSelectionResult",
    "PolicyRolloutStepResult",
    "PolicyValueResult",
    "PreparedPPORolloutBatch",
    "batched_policy_action_selection",
    "build_cy_data_list",
    "collect_policy_rollout",
    "compute_cy_state_count_bonus",
    "compute_explained_variance",
    "compute_gae_with_dones",
    "evaluate_policy_actions",
    "evaluate_policy_actions_from_data_list",
    "evaluate_policy_values",
    "flatten_action_buffer",
    "flatten_buffer",
    "format_rollout_summary",
    "get_cy_state_visit_count",
    "increment_visitation",
    "infer_batch_subcomplex_width",
    "normalize_advantages_masked",
    "rollout_return_statistics",
    "rollout_step_with_policy",
    "sample_cy_trajectory_transforms",
    "summarize_objective_performance",
    "train_policy_from_rollout",
]
