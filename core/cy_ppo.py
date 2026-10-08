from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

from core.cy_policy_inference import (
    PolicyRolloutStepResult,
    evaluate_policy_actions,
    evaluate_policy_actions_from_data_list,
    build_cy_data_list,
    policy_data_chunks,
    policy_observation_kind,
    validate_policy_observations,
)
from core.training_types import PPOTrainStats
from mdp.cy_graph import CanonicalAction

@dataclass(frozen=True)
class PreparedPPORolloutBatch:
    state_buffer_list: List[Any]
    candidate_buffer_list: List[Tuple[CanonicalAction, ...]]
    action_buffer_flat: torch.Tensor
    action_index_buffer_flat: torch.Tensor
    log_prob_buffer_flat: torch.Tensor
    entropy_buffer_flat: torch.Tensor
    reward_buffer_tensor: torch.Tensor
    value_buffer_tensor: torch.Tensor
    done_buffer_tensor: torch.Tensor
    valid_mask_flat: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor
    data_buffer_list: List[Data] | None = None
    observation_kind: str = "full_triangulation"


@dataclass
class PPORolloutBuffer:
    state_buffer: List[List[Any]] = field(default_factory=list)
    candidate_buffer: List[List[Tuple[CanonicalAction, ...]]] = field(default_factory=list)
    data_buffer: List[List[Data] | None] = field(default_factory=list)
    action_buffer: List[torch.Tensor] = field(default_factory=list)
    action_index_buffer: List[torch.Tensor] = field(default_factory=list)
    log_prob_buffer: List[torch.Tensor] = field(default_factory=list)
    entropy_buffer: List[torch.Tensor] = field(default_factory=list)
    value_buffer: List[torch.Tensor] = field(default_factory=list)
    reward_buffer: List[torch.Tensor] = field(default_factory=list)
    done_buffer: List[torch.Tensor] = field(default_factory=list)
    valid_mask_buffer: List[torch.Tensor] = field(default_factory=list)
    observation_kind: str | None = None

    def append(self, step_result: PolicyRolloutStepResult) -> None:
        kinds = {getattr(data, "observation_kind", "full_triangulation")
                 for data in (step_result.data_list or [])}
        declared = getattr(step_result, "observation_kind", None)
        if declared is not None:
            kinds.add(declared)
        if not kinds:
            kinds.add(self.observation_kind or "full_triangulation")
        if len(kinds) != 1 or (self.observation_kind is not None and self.observation_kind not in kinds):
            raise ValueError("Cannot mix observation schemas in a PPO rollout buffer.")
        self.observation_kind = kinds.pop()
        reward_values = (
            step_result.training_rewards
            if step_result.training_rewards is not None
            else step_result.rewards
        )
        # CPU observations contain everything PPO uses. Keeping their source states
        # as well retains the entire CYTools neighbour graph for the rollout lifetime.
        has_data = step_result.data_list is not None
        self.state_buffer.append([None] * len(step_result.input_states) if has_data else list(step_result.input_states))
        self.candidate_buffer.append([()] * len(step_result.input_states) if has_data else list(step_result.action_candidates))
        self.data_buffer.append(
            None if not has_data else [data.detach().cpu() for data in step_result.data_list]
        )
        self.action_buffer.append(step_result.actions_tensor.detach().cpu())
        self.action_index_buffer.append(step_result.action_index_tensor.detach().cpu())
        self.log_prob_buffer.append(step_result.log_prob_tensor.detach().cpu())
        self.entropy_buffer.append(step_result.entropy_tensor.detach().cpu())
        self.value_buffer.append(step_result.value_tensor.detach().cpu())
        self.reward_buffer.append(torch.tensor(reward_values, dtype=torch.float))
        self.done_buffer.append(torch.tensor(step_result.dones, dtype=torch.float))
        self.valid_mask_buffer.append(step_result.valid_action_mask.detach().cpu())

    def clear(self) -> None:
        """Release observations immediately after the associated PPO update."""
        for name in self.__dataclass_fields__:
            if name != "observation_kind":
                getattr(self, name).clear()
        self.observation_kind = None

    def prepare(
        self,
        *,
        bootstrap_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> PreparedPPORolloutBatch:
        if not self.state_buffer:
            raise ValueError("Cannot prepare PPO tensors from an empty rollout buffer.")

        rollout_length = len(self.state_buffer)
        num_states = len(self.state_buffer[0])

        # Legacy callers can still append steps without prepared observations.
        # In a mixed buffer, rebuild only those steps, retaining augmented tensors
        # already supplied by the caller.
        if any(step_data is not None for step_data in self.data_buffer):
            for index, step_data in enumerate(self.data_buffer):
                if step_data is None:
                    self.data_buffer[index] = build_cy_data_list(
                        self.state_buffer[index], self.candidate_buffer[index],
                        include_simplex_topology=self.observation_kind != "two_face",
                        observation_kind=self.observation_kind or "full_triangulation",
                    )
                    self.state_buffer[index] = [None] * num_states
                    self.candidate_buffer[index] = [()] * num_states

        reward_buffer_tensor = torch.stack(self.reward_buffer).float().to(device)
        value_buffer_tensor = torch.stack(self.value_buffer).float().to(device)
        done_buffer_tensor = torch.stack(self.done_buffer).float().to(device)
        log_prob_buffer_tensor = torch.stack(self.log_prob_buffer).float().to(device)
        entropy_buffer_tensor = torch.stack(self.entropy_buffer).float().to(device)
        action_index_buffer_tensor = torch.stack(self.action_index_buffer).long().to(device)
        valid_mask_tensor = torch.stack(self.valid_mask_buffer).bool().to(device)

        advantages, value_targets = compute_gae_with_dones(
            reward_buffer_tensor=reward_buffer_tensor,
            value_buffer_tensor=value_buffer_tensor,
            done_buffer_tensor=done_buffer_tensor,
            bootstrap_value=bootstrap_value,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )

        return PreparedPPORolloutBatch(
            observation_kind=self.observation_kind or "full_triangulation",
            state_buffer_list=flatten_buffer(self.state_buffer, rollout_length, num_states),
            candidate_buffer_list=flatten_buffer(self.candidate_buffer, rollout_length, num_states),
            action_buffer_flat=flatten_action_buffer(self.action_buffer, rollout_length, num_states, device=device),
            action_index_buffer_flat=action_index_buffer_tensor.reshape(-1),
            log_prob_buffer_flat=log_prob_buffer_tensor.reshape(-1),
            entropy_buffer_flat=entropy_buffer_tensor.reshape(-1),
            reward_buffer_tensor=reward_buffer_tensor,
            value_buffer_tensor=value_buffer_tensor,
            done_buffer_tensor=done_buffer_tensor,
            valid_mask_flat=valid_mask_tensor.reshape(-1),
            advantages=advantages,
            value_targets=value_targets,
            data_buffer_list=(
                flatten_buffer(self.data_buffer, rollout_length, num_states)
                if all(step_data is not None for step_data in self.data_buffer)
                else None
            ),
        )

def compute_gae_with_dones(
    reward_buffer_tensor: torch.Tensor,
    value_buffer_tensor: torch.Tensor,
    done_buffer_tensor: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        next_value = bootstrap_value.detach().to(
            device=value_buffer_tensor.device,
            dtype=value_buffer_tensor.dtype,
        )
        gae = torch.zeros_like(next_value)
        advantages = torch.zeros_like(reward_buffer_tensor)
        rollout_length = reward_buffer_tensor.size(0)
        for t in reversed(range(rollout_length)):
            non_terminal = 1.0 - done_buffer_tensor[t]
            delta = reward_buffer_tensor[t] + gamma * next_value * non_terminal - value_buffer_tensor[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = value_buffer_tensor[t]
        value_targets = advantages + value_buffer_tensor
    return advantages, value_targets


def flatten_buffer(buffer: List[List[Any]], rollout_length: int, num_states: int) -> List[Any]:
    return [buffer[t][i] for t in range(rollout_length) for i in range(num_states)]


def flatten_action_buffer(
    action_buffer: List[torch.Tensor],
    rollout_length: int,
    num_states: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    flattened_actions: List[torch.Tensor] = []
    for t in range(rollout_length):
        for i in range(num_states):
            flattened_actions.append(action_buffer[t][i].detach())
    return pad_sequence(flattened_actions, batch_first=True, padding_value=-1).to(
        device=device,
        dtype=torch.long,
    )


def normalize_advantages_masked(advantages: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    normalized = torch.zeros_like(advantages)
    if advantages.numel() == 0:
        return normalized
    if not bool(valid_mask.any().item()):
        return normalized

    valid_advantages = advantages[valid_mask]
    mean = valid_advantages.mean()
    std = valid_advantages.std(unbiased=False)
    normalized[valid_mask] = (valid_advantages - mean) / (std + float(eps))
    return normalized


def compute_explained_variance(value_estimate: torch.Tensor, value_target: torch.Tensor, eps: float = 1e-8) -> float:
    target = value_target.detach()
    pred = value_estimate.detach()
    if target.numel() == 0:
        return 0.0
    target_var = torch.var(target, unbiased=False)
    if float(target_var.item()) <= float(eps):
        return 0.0
    residual_var = torch.var(target - pred, unbiased=False)
    return float((1.0 - residual_var / (target_var + float(eps))).item())

def train_policy_from_rollout(
    *,
    policy: EGNNSubcomplexAgent,
    optimizer: torch.optim.Optimizer,
    prepared_rollout: PreparedPPORolloutBatch,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    max_graph_size: int | None = None,
    memory_guard=None,
) -> PPOTrainStats:
    if prepared_rollout.observation_kind != policy_observation_kind(policy):
        raise ValueError("PPO rollout observation schema does not match the policy.")
    if prepared_rollout.data_buffer_list is not None:
        validate_policy_observations(prepared_rollout.data_buffer_list, policy)
    state_buffer_list = prepared_rollout.state_buffer_list
    candidate_buffer_list = prepared_rollout.candidate_buffer_list
    data_buffer_list = prepared_rollout.data_buffer_list
    action_index_buffer_flat = prepared_rollout.action_index_buffer_flat.detach()
    old_log_prob_flat = prepared_rollout.log_prob_buffer_flat.detach()
    value_targets_flat = prepared_rollout.value_targets.reshape(-1).detach()
    old_value_estimate_flat = prepared_rollout.value_buffer_tensor.reshape(-1).detach()
    valid_mask_flat = prepared_rollout.valid_mask_flat.detach()
    valid_mask_cpu = valid_mask_flat.cpu().tolist()
    advantages_flat = normalize_advantages_masked(
        prepared_rollout.advantages.reshape(-1).detach(),
        valid_mask_flat,
    )

    num_samples = len(state_buffer_list)
    if num_samples == 0:
        return PPOTrainStats(
            total_loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy_loss=0.0,
            explained_variance=0.0,
            clip_ratio=0.0,
            num_samples=0,
            num_valid_action_samples=0,
        )

    total_loss_sum = 0.0
    total_value_loss_sum = 0.0
    total_policy_loss_sum = 0.0
    total_entropy_loss_sum = 0.0
    total_value_weight = 0
    total_policy_weight = 0
    total_clip_count = 0.0
    total_valid_action_samples = 0

    policy.train()
    for _epoch in range(int(num_epochs)):
        permutation = torch.randperm(num_samples).tolist()
        for start in range(0, num_samples, int(batch_size)):
            if memory_guard is not None:
                memory_guard()
            mini_batch_indices = permutation[start : start + int(batch_size)]
            if data_buffer_list is not None:
                mini_data = [data_buffer_list[idx] for idx in mini_batch_indices]
            else:
                mini_data = build_cy_data_list(
                    [state_buffer_list[idx] for idx in mini_batch_indices],
                    [candidate_buffer_list[idx] for idx in mini_batch_indices],
                    observation_kind=prepared_rollout.observation_kind,
                    include_simplex_topology=(getattr(policy, "value_feature_source", "") == "snn_simplex"
                                              or getattr(policy, "subcomplex_actor_type", "") == "snn_simplex"),
                )
            batch_count = len(mini_batch_indices)
            actionable_count = sum(
                bool(valid_mask_cpu[idx]) and int(data.num_available_subcomplexes) > 0
                for idx, data in zip(mini_batch_indices, mini_data)
            )
            optimizer.zero_grad(set_to_none=True)
            logical_loss = 0.0
            for positions in policy_data_chunks(mini_data, max_graph_size=max_graph_size, policy=policy):
                if memory_guard is not None:
                    memory_guard()
                indices = [mini_batch_indices[position] for position in positions]
                evaluation = evaluate_policy_actions_from_data_list(
                    [mini_data[position] for position in positions],
                    action_index_buffer_flat[indices], policy, device=device,
                )
                actionable_mask = valid_mask_flat[indices] & evaluation.valid_action_mask
                value_estimate = evaluation.value_tensor
                ratio = torch.exp(evaluation.log_prob_tensor[actionable_mask] - old_log_prob_flat[indices][actionable_mask])
                advantage_values = advantages_flat[indices][actionable_mask]
                unclipped = -ratio * advantage_values
                clipped = -torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef)) * advantage_values
                # Every physical chunk uses the original logical denominator.
                # The optimizer and gradient clipping run once per logical batch.
                policy_loss = torch.max(unclipped, clipped).sum() / max(1, actionable_count)
                entropy_loss = evaluation.entropy_tensor[actionable_mask].sum() / max(1, actionable_count)
                value_loss = 0.5 * (value_estimate - value_targets_flat[indices]).pow(2).sum() / batch_count
                loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_loss
                loss.backward()
                logical_loss += float(loss.detach().item())
                total_value_loss_sum += float(value_loss.detach().item()) * batch_count
                total_policy_loss_sum += float(policy_loss.detach().item()) * actionable_count
                total_entropy_loss_sum += float(entropy_loss.detach().item()) * actionable_count
                total_clip_count += float((torch.abs(ratio.detach() - 1.0) > float(clip_coef)).sum().item())
                del evaluation, value_estimate, loss, policy_loss, value_loss, entropy_loss, ratio, unclipped, clipped

            nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
            optimizer.step()
            total_loss_sum += logical_loss * batch_count
            total_value_weight += batch_count
            total_policy_weight += actionable_count
            total_valid_action_samples += actionable_count

    explained_variance = compute_explained_variance(old_value_estimate_flat, value_targets_flat)
    avg_total_loss = total_loss_sum / max(1, total_value_weight)
    avg_value_loss = total_value_loss_sum / max(1, total_value_weight)
    avg_policy_loss = total_policy_loss_sum / max(1, total_policy_weight)
    avg_entropy_loss = total_entropy_loss_sum / max(1, total_policy_weight)
    clip_ratio = total_clip_count / max(1, total_valid_action_samples)
    return PPOTrainStats(
        total_loss=avg_total_loss,
        policy_loss=avg_policy_loss,
        value_loss=avg_value_loss,
        entropy_loss=avg_entropy_loss,
        explained_variance=explained_variance,
        clip_ratio=clip_ratio,
        num_samples=num_samples,
        num_valid_action_samples=int(valid_mask_flat.sum().item()),
    )
