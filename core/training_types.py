from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence

import numpy as np

if TYPE_CHECKING:
    from core.cy_policy_rollout_utils import PPORolloutBuffer


@dataclass(frozen=True)
class CYDatasetSplit:
    train_rows: List[dict]
    eval_rows: List[dict]
    train_polytope_indices: List[int]
    eval_polytope_indices: List[int]


@dataclass
class FirstEpisodeTracker:
    gamma: float
    discounted_rewards: np.ndarray
    active_mask: np.ndarray
    finished_mask: np.ndarray
    success_mask: np.ndarray
    collapsed_mask: np.ndarray
    dead_end_mask: np.ndarray

    @classmethod
    def create(cls, *, num_envs: int, gamma: float) -> "FirstEpisodeTracker":
        resolved_num_envs = max(0, int(num_envs))
        return cls(
            gamma=float(gamma),
            discounted_rewards=np.zeros(resolved_num_envs, dtype=np.float64),
            active_mask=np.ones(resolved_num_envs, dtype=bool),
            finished_mask=np.zeros(resolved_num_envs, dtype=bool),
            success_mask=np.zeros(resolved_num_envs, dtype=bool),
            collapsed_mask=np.zeros(resolved_num_envs, dtype=bool),
            dead_end_mask=np.zeros(resolved_num_envs, dtype=bool),
        )

    def update(
        self,
        *,
        rewards: Sequence[float],
        dones: Sequence[bool],
        terminal_reasons: Sequence[str],
        step_index: int,
    ) -> None:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        done_array = np.asarray(dones, dtype=bool)
        reason_array = np.asarray(terminal_reasons, dtype=object)
        if rewards_array.shape[0] != self.discounted_rewards.shape[0]:
            raise ValueError("rewards length does not match tracker size.")
        if done_array.shape[0] != rewards_array.shape[0]:
            raise ValueError("dones length does not match tracker size.")
        if reason_array.shape[0] != rewards_array.shape[0]:
            raise ValueError("terminal_reasons length does not match tracker size.")

        active_now = self.active_mask.copy()
        if not np.any(active_now):
            return

        self.discounted_rewards[active_now] += (self.gamma ** int(step_index)) * rewards_array[active_now]
        success_now = active_now & (reason_array == "frt_or_frst")
        collapsed_now = active_now & (reason_array == "single_simplex")
        dead_end_now = active_now & np.isin(reason_array, ("dead_end_current", "dead_end_next"))
        self.success_mask[success_now] = True
        self.collapsed_mask[collapsed_now] = True
        self.dead_end_mask[dead_end_now] = True

        finished_now = active_now & done_array
        self.finished_mask[finished_now] = True
        self.active_mask[finished_now] = False

    def success_rate(self) -> float:
        if self.success_mask.size == 0:
            return 0.0
        return float(self.success_mask.mean())

    def mean_discounted_reward(self) -> float:
        if self.discounted_rewards.size == 0:
            return 0.0
        return float(self.discounted_rewards.mean())

    def success_count(self) -> int:
        return int(self.success_mask.sum())

    def collapsed_count(self) -> int:
        return int(self.collapsed_mask.sum())

    def dead_end_count(self) -> int:
        return int(self.dead_end_mask.sum())

    def finished_count(self) -> int:
        return int(self.finished_mask.sum())

    def finished_fraction(self) -> float:
        if self.finished_mask.size == 0:
            return 0.0
        return float(self.finished_mask.mean())


@dataclass
class PolicyRolloutSummary:
    final_states: List[Any]
    rollout_buffer: PPORolloutBuffer | None
    success_rate: float
    discounted_reward: float
    finished_fraction: float
    finished_count: int
    frt_hits: int
    collapsed_hits: int
    dead_end_hits: int
    all_step_reset_count: int
    all_step_frt_hits: int
    all_step_collapsed_hits: int
    all_step_dead_end_hits: int
    expanded_states: int
    discovered_states: int
    multiprocessing_steps: int
    total_candidates: int
    total_valid_actions: int
    candidate_expand_sec: float
    policy_data_build_sec: float
    policy_batch_transfer_sec: float
    policy_value_inference_sec: float
    policy_action_inference_sec: float
    transition_apply_sec: float
    intrinsic_bonus_mean: float = 0.0
    training_discounted_reward: float = 0.0
    trajectory_transforms: List[Any] | None = None
    objective_name: str | None = None
    objective_goal: str | None = None
    objective_initial_values: List[float] | None = None
    objective_final_values: List[float] | None = None
    objective_best_values: List[float] | None = None
    return_mean: float = 0.0
    return_std: float = 0.0
    return_min: float = 0.0
    return_max: float = 0.0
    training_return_mean: float = 0.0


@dataclass(frozen=True)
class PPOTrainStats:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    explained_variance: float
    clip_ratio: float
    num_samples: int
    num_valid_action_samples: int
