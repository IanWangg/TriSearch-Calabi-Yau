from __future__ import annotations

from typing import Any, Callable, List, Sequence

import numpy as np
import torch

from core.cy_policy_inference import rollout_step_with_policy
from core.cy_ppo import PPORolloutBuffer
from core.cy_runtime_utils import increment_visitation
from core.training_types import FirstEpisodeTracker, PolicyRolloutSummary
from core.vertex_augmentation import SimilarityTransform, sample_similarity_transform
from mdp.cy_rollout import CYRandomRolloutEngine

def format_rollout_summary(
    *,
    label: str,
    summary: PolicyRolloutSummary,
    num_envs: int,
    rollout_length: int,
) -> str:
    env_steps = max(1, int(num_envs) * int(rollout_length))
    mean_candidates = float(summary.total_candidates) / env_steps
    valid_action_fraction = float(summary.total_valid_actions) / env_steps
    parts = [
        f"{label}: return={summary.return_mean:.4f}",
        f"return_std={summary.return_std:.4f}",
        f"return_min={summary.return_min:.4f}",
        f"return_max={summary.return_max:.4f}",
        f"discounted_reward={summary.discounted_reward:.4f}",
        f"success_rate={summary.success_rate:.4f}",
    ]
    if abs(float(summary.intrinsic_bonus_mean)) > 0.0:
        parts.extend(
            [
                f"training_return={summary.training_return_mean:.4f}",
                f"training_discounted_reward={summary.training_discounted_reward:.4f}",
                f"intrinsic_bonus_mean={summary.intrinsic_bonus_mean:.4f}",
            ]
        )
    parts.extend(
        [
            f"finished_fraction={summary.finished_fraction:.4f}",
            f"finished_count={summary.finished_count}",
            f"mean_candidates={mean_candidates:.4f}",
            f"valid_action_fraction={valid_action_fraction:.4f}",
            f"frt_hits={summary.frt_hits}",
            f"collapsed_hits={summary.collapsed_hits}",
            f"dead_end_hits={summary.dead_end_hits}",
            f"all_step_resets={summary.all_step_reset_count}",
            f"expanded_states={summary.expanded_states}",
            f"discovered_states={summary.discovered_states}",
        ]
    )
    objective_metrics = summarize_objective_performance(summary)
    if objective_metrics:
        parts.extend(
            [
                f"objective_initial_mean={objective_metrics['initial_mean']:.4f}",
                f"objective_final_mean={objective_metrics['final_mean']:.4f}",
                f"objective_best_mean={objective_metrics['best_mean']:.4f}",
                f"objective_mean_improvement={objective_metrics['mean_improvement']:.4f}",
                f"objective_improved_fraction={objective_metrics['improved_fraction']:.4f}",
            ]
        )
    return " ".join(parts)


def rollout_return_statistics(return_values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(return_values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_objective_performance(summary: PolicyRolloutSummary) -> dict[str, float]:
    if summary.objective_name is None:
        return {}

    initial_values = list(summary.objective_initial_values or ())
    final_values = list(summary.objective_final_values or ())
    best_values = list(summary.objective_best_values or ())
    if not initial_values or not (
        len(initial_values) == len(final_values) == len(best_values)
    ):
        raise ValueError("Objective metric arrays must be non-empty and have equal lengths.")

    if summary.objective_goal == "min":
        improvements = [
            float(initial) - float(best)
            for initial, best in zip(initial_values, best_values)
        ]
    elif summary.objective_goal == "max":
        improvements = [
            float(best) - float(initial)
            for initial, best in zip(initial_values, best_values)
        ]
    else:
        raise ValueError(f"Unsupported objective goal '{summary.objective_goal}'.")

    return {
        "initial_mean": float(np.mean(initial_values)),
        "final_mean": float(np.mean(final_values)),
        "best_mean": float(np.mean(best_values)),
        "mean_improvement": float(np.mean(improvements)),
        "improved_fraction": float(np.mean(np.asarray(improvements) > 0.0)),
    }


def _state_key(state: Any) -> str:
    return str(getattr(state, "key", state))


def get_cy_state_visit_count(
    state: Any,
    visit_counts_by_key: dict[str, int] | None = None,
) -> int:
    if visit_counts_by_key is not None:
        return int(visit_counts_by_key.get(_state_key(state), 0))
    return int(getattr(state, "visitation", 0))


def compute_cy_state_count_bonus(
    *,
    input_states: Sequence[Any],
    transitioned_states: Sequence[Any],
    visit_counts_by_key: dict[str, int] | None,
    coef: float,
    exponent: float,
) -> List[float]:
    if float(coef) <= 0.0:
        return [0.0 for _ in transitioned_states]

    bonus_values: List[float] = []
    for input_state, transitioned_state in zip(input_states, transitioned_states):
        input_key = _state_key(input_state)
        transitioned_key = _state_key(transitioned_state)
        if transitioned_key == input_key:
            bonus_values.append(0.0)
            continue

        visit_count = get_cy_state_visit_count(transitioned_state, visit_counts_by_key)
        bonus_values.append(float(coef) / ((float(visit_count) + 1.0) ** float(exponent)))
    return bonus_values


def sample_cy_trajectory_transforms(
    states: Sequence[Any],
    *,
    aug_prob: float,
    scale_min: float,
    scale_max: float,
    shift_std: float,
    reflect_prob: float,
) -> List[SimilarityTransform]:
    transforms: List[SimilarityTransform] = []
    for state in states:
        vertices_tensor = torch.as_tensor(getattr(state, "vertices"), dtype=torch.float)
        transforms.append(
            sample_similarity_transform(
                vertices_tensor,
                aug_prob=float(aug_prob),
                scale_min=float(scale_min),
                scale_max=float(scale_max),
                shift_std=float(shift_std),
                reflect_prob=float(reflect_prob),
            )
        )
    return transforms


def collect_policy_rollout(
    *,
    engine: CYRandomRolloutEngine,
    policy: EGNNSubcomplexAgent,
    rng: np.random.Generator,
    device: torch.device,
    initial_state_pool: Sequence[Any],
    num_envs: int,
    rollout_length: int,
    gamma: float,
    deterministic: bool,
    use_multiprocessing: bool,
    transition_pool: Any,
    transition_mp_chunksize: int,
    transition_mp_min_batch: int,
    store_buffer: bool,
    report_every: int,
    label: str,
    count_bonus_coef: float = 0.0,
    count_bonus_exponent: float = 0.5,
    visit_counts_by_key: dict[str, int] | None = None,
    vertex_aug_enable: bool = False,
    vertex_aug_prob: float = 1.0,
    vertex_aug_scale_min: float = 0.9,
    vertex_aug_scale_max: float = 1.1,
    vertex_aug_shift_std: float = 0.05,
    vertex_aug_reflect_prob: float = 0.1,
    objective_function: Callable[[Any], float] | None = None,
    objective_name: str | None = None,
    objective_goal: str | None = None,
) -> PolicyRolloutSummary:
    states = engine.sample_initial_states(num_envs, rng=rng, initial_state_pool=initial_state_pool)
    if objective_function is not None and objective_goal not in {"min", "max"}:
        raise ValueError("objective_goal must be 'min' or 'max' with objective_function.")
    objective_initial_values = (
        [float(objective_function(state)) for state in states]
        if objective_function is not None
        else None
    )
    objective_final_values = (
        list(objective_initial_values) if objective_initial_values is not None else None
    )
    objective_best_values = (
        list(objective_initial_values) if objective_initial_values is not None else None
    )
    objective_first_episode_active = [True for _ in states]
    trajectory_transforms = None
    if bool(vertex_aug_enable):
        trajectory_transforms = sample_cy_trajectory_transforms(
            states,
            aug_prob=float(vertex_aug_prob),
            scale_min=float(vertex_aug_scale_min),
            scale_max=float(vertex_aug_scale_max),
            shift_std=float(vertex_aug_shift_std),
            reflect_prob=float(vertex_aug_reflect_prob),
        )
    tracker = FirstEpisodeTracker.create(num_envs=len(states), gamma=gamma)
    training_tracker = FirstEpisodeTracker.create(num_envs=len(states), gamma=gamma)
    rollout_buffer = PPORolloutBuffer() if store_buffer else None
    use_count_bonus = float(count_bonus_coef) > 0.0

    total_frt_hits = 0
    total_collapsed_hits = 0
    total_dead_end_hits = 0
    total_resets = 0
    total_expanded_states = 0
    total_discovered_states = 0
    total_mp_steps = 0
    total_candidates = 0
    total_valid_actions = 0
    total_candidate_expand_sec = 0.0
    total_policy_data_build_sec = 0.0
    total_policy_batch_transfer_sec = 0.0
    total_policy_value_inference_sec = 0.0
    total_policy_action_inference_sec = 0.0
    total_transition_apply_sec = 0.0
    total_intrinsic_bonus = 0.0
    total_intrinsic_bonus_count = 0
    rollout_returns = np.zeros(len(states), dtype=np.float64)
    training_rollout_returns = np.zeros(len(states), dtype=np.float64)

    for step_index in range(int(rollout_length)):
        pool = transition_pool or getattr(engine, "transition_pool", None)
        if pool is not None and hasattr(pool, "check_memory"):
            pool.check_memory()
        increment_visitation(
            states,
            visit_counts_by_key=visit_counts_by_key if use_count_bonus else None,
        )
        step_result = rollout_step_with_policy(
            engine,
            states,
            policy,
            rng=rng,
            device=device,
            initial_state_pool=initial_state_pool,
            deterministic=deterministic,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
            trajectory_transforms=trajectory_transforms,
        )
        intrinsic_bonus = compute_cy_state_count_bonus(
            input_states=getattr(step_result, "input_states", states),
            transitioned_states=getattr(step_result, "transitioned_states", step_result.next_states),
            visit_counts_by_key=visit_counts_by_key,
            coef=float(count_bonus_coef),
            exponent=float(count_bonus_exponent),
        )
        if len(intrinsic_bonus) != len(step_result.rewards):
            raise ValueError("count bonus length does not match reward length.")
        training_rewards = [
            float(extrinsic_reward) + float(bonus)
            for extrinsic_reward, bonus in zip(step_result.rewards, intrinsic_bonus)
        ]
        step_result.intrinsic_bonus = intrinsic_bonus
        step_result.training_rewards = training_rewards
        rollout_returns += np.asarray(step_result.rewards, dtype=np.float64)
        training_rollout_returns += np.asarray(training_rewards, dtype=np.float64)

        if objective_function is not None:
            for idx, transitioned_state in enumerate(step_result.transitioned_states):
                if not objective_first_episode_active[idx]:
                    continue
                objective_value = float(objective_function(transitioned_state))
                objective_final_values[idx] = objective_value
                if objective_goal == "min":
                    objective_best_values[idx] = min(
                        objective_best_values[idx], objective_value
                    )
                else:
                    objective_best_values[idx] = max(
                        objective_best_values[idx], objective_value
                    )
                if step_result.dones[idx]:
                    objective_first_episode_active[idx] = False
        total_intrinsic_bonus += float(sum(intrinsic_bonus))
        total_intrinsic_bonus_count += len(intrinsic_bonus)

        if rollout_buffer is not None:
            rollout_buffer.append(step_result)

        tracker.update(
            rewards=step_result.rewards,
            dones=step_result.dones,
            terminal_reasons=step_result.terminal_reasons,
            step_index=step_index,
        )
        training_tracker.update(
            rewards=training_rewards,
            dones=step_result.dones,
            terminal_reasons=step_result.terminal_reasons,
            step_index=step_index,
        )

        states = step_result.next_states
        total_frt_hits += int(step_result.frt_hits)
        total_collapsed_hits += int(step_result.collapsed_hits)
        total_dead_end_hits += int(step_result.dead_end_hits)
        total_resets += int(step_result.reset_count)
        total_expanded_states += int(step_result.expanded_states)
        total_discovered_states += int(step_result.discovered_states)
        total_mp_steps += int(step_result.used_multiprocessing)
        total_candidates += sum(len(actions) for actions in step_result.action_candidates)
        total_valid_actions += int(step_result.valid_action_mask.sum().item())
        total_candidate_expand_sec += float(step_result.candidate_expand_sec)
        total_policy_data_build_sec += float(step_result.policy_data_build_sec)
        total_policy_batch_transfer_sec += float(step_result.policy_batch_transfer_sec)
        total_policy_value_inference_sec += float(step_result.policy_value_inference_sec)
        total_policy_action_inference_sec += float(step_result.policy_action_inference_sec)
        total_transition_apply_sec += float(step_result.transition_apply_sec)

    return_stats = rollout_return_statistics(rollout_returns)
    training_return_stats = rollout_return_statistics(training_rollout_returns)

    return PolicyRolloutSummary(
        final_states=states,
        rollout_buffer=rollout_buffer,
        success_rate=tracker.success_rate(),
        discounted_reward=tracker.mean_discounted_reward(),
        finished_fraction=tracker.finished_fraction(),
        finished_count=tracker.finished_count(),
        frt_hits=tracker.success_count(),
        collapsed_hits=tracker.collapsed_count(),
        dead_end_hits=tracker.dead_end_count(),
        all_step_reset_count=total_resets,
        all_step_frt_hits=total_frt_hits,
        all_step_collapsed_hits=total_collapsed_hits,
        all_step_dead_end_hits=total_dead_end_hits,
        expanded_states=total_expanded_states,
        discovered_states=total_discovered_states,
        multiprocessing_steps=total_mp_steps,
        total_candidates=total_candidates,
        total_valid_actions=total_valid_actions,
        candidate_expand_sec=total_candidate_expand_sec,
        policy_data_build_sec=total_policy_data_build_sec,
        policy_batch_transfer_sec=total_policy_batch_transfer_sec,
        policy_value_inference_sec=total_policy_value_inference_sec,
        policy_action_inference_sec=total_policy_action_inference_sec,
        transition_apply_sec=total_transition_apply_sec,
        intrinsic_bonus_mean=(
            total_intrinsic_bonus / max(1, total_intrinsic_bonus_count)
        ),
        training_discounted_reward=training_tracker.mean_discounted_reward(),
        trajectory_transforms=trajectory_transforms,
        objective_name=objective_name if objective_function is not None else None,
        objective_goal=objective_goal if objective_function is not None else None,
        objective_initial_values=objective_initial_values,
        objective_final_values=objective_final_values,
        objective_best_values=objective_best_values,
        return_mean=return_stats["mean"],
        return_std=return_stats["std"],
        return_min=return_stats["min"],
        return_max=return_stats["max"],
        training_return_mean=training_return_stats["mean"],
    )
