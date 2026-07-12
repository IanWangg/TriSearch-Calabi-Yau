from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data

from mdp.cy_rollout import CYRandomRolloutEngine, CanonicalAction
from core.cy_data_utils import create_data_from_cy_state_with_subcomplex
from core.training_types import (
    FirstEpisodeTracker,
    PolicyRolloutSummary,
    PPOTrainStats,
)
from core.vertex_augmentation import (
    SimilarityTransform,
    apply_similarity_transform,
    sample_similarity_transform,
)
from core.vertex_preprocessing import VertexPreprocessor

if TYPE_CHECKING:
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ensure_policy_device(policy: Any, device: torch.device) -> Any:
    parameters = getattr(policy, "parameters", None)
    to_method = getattr(policy, "to", None)
    if parameters is None or not callable(parameters) or to_method is None or not callable(to_method):
        return policy

    needs_move = False
    for parameter in policy.parameters():
        if parameter.device != device:
            needs_move = True
            break
    if not needs_move:
        buffers = getattr(policy, "buffers", None)
        if buffers is not None and callable(buffers):
            for buffer in policy.buffers():
                if buffer.device != device:
                    needs_move = True
                    break

    if not needs_move:
        return policy
    return policy.to(device)


def _policy_uses_simplex_topology(policy: Any) -> bool:
    return str(getattr(policy, "subcomplex_actor_type", "")).strip().lower() == "snn_simplex"


def infer_batch_subcomplex_width(
    states: Sequence[Any],
    action_lists: Sequence[Sequence[CanonicalAction]],
) -> int:
    width_candidates: List[int] = []
    for state, actions in zip(states, action_lists):
        inferred_min_width = 0
        simplices = tuple(getattr(state, "simplices", ()))
        if simplices:
            inferred_min_width = len(next(iter(simplices))) + 1
        inferred_action_width = max((len(action) for action in actions), default=0)
        width_candidates.append(max(inferred_min_width, inferred_action_width))
    return max(width_candidates, default=0)


def build_cy_data_list(
    states: Sequence[Any],
    action_lists: Sequence[Sequence[CanonicalAction]],
    *,
    subcomplex_width: Optional[int] = None,
    vertex_preprocessor: VertexPreprocessor | None = None,
    trajectory_transforms: Sequence[SimilarityTransform] | None = None,
    include_simplex_topology: bool = False,
) -> List[Data]:
    if len(states) != len(action_lists):
        raise ValueError("states and action_lists must have the same length.")
    if trajectory_transforms is not None and len(trajectory_transforms) != len(states):
        raise ValueError("trajectory_transforms must have one transform per state.")
    if len(states) == 0:
        return []

    width = infer_batch_subcomplex_width(states, action_lists) if subcomplex_width is None else int(subcomplex_width)
    data_list = [
        create_data_from_cy_state_with_subcomplex(
            state,
            subcomplex_width=width,
            ensure_actions_ready=False,
            subcomplex_actions=actions,
            vertex_preprocessor=vertex_preprocessor,
            include_simplex_topology=include_simplex_topology,
        )
        for state, actions in zip(states, action_lists)
    ]
    if trajectory_transforms is None:
        return data_list

    for data, transform in zip(data_list, trajectory_transforms):
        data.x = apply_similarity_transform(data.x, transform)
    return data_list


@dataclass(frozen=True)
class PolicyActionSelectionResult:
    action_lists: List[Tuple[CanonicalAction, ...]]
    action_index_tensor: torch.Tensor
    actions_tensor: torch.Tensor
    log_prob_tensor: torch.Tensor
    entropy_tensor: torch.Tensor
    value_tensor: torch.Tensor
    valid_action_mask: torch.Tensor
    subcomplex_width: int
    data_list: List[Data]
    data_build_sec: float
    batch_transfer_sec: float
    value_inference_sec: float
    policy_inference_sec: float
    num_actionable: int


@dataclass
class PolicyRolloutStepResult:
    input_states: List[Any]
    transitioned_states: List[Any]
    next_states: List[Any]
    rewards: List[float]
    dones: List[bool]
    chosen_actions: List[Optional[CanonicalAction]]
    terminal_reasons: List[str]
    action_candidates: List[Tuple[CanonicalAction, ...]]
    action_index_tensor: torch.Tensor
    actions_tensor: torch.Tensor
    log_prob_tensor: torch.Tensor
    entropy_tensor: torch.Tensor
    value_tensor: torch.Tensor
    valid_action_mask: torch.Tensor
    reset_count: int
    frt_hits: int
    collapsed_hits: int
    dead_end_hits: int
    expanded_states: int
    discovered_states: int
    used_multiprocessing: bool
    candidate_expand_sec: float
    policy_data_build_sec: float
    policy_batch_transfer_sec: float
    policy_value_inference_sec: float
    policy_action_inference_sec: float
    transition_apply_sec: float
    data_list: List[Data] | None = None
    intrinsic_bonus: List[float] | None = None
    training_rewards: List[float] | None = None


@dataclass(frozen=True)
class PolicyValueResult:
    value_tensor: torch.Tensor
    data_build_sec: float
    batch_transfer_sec: float
    inference_sec: float


@dataclass(frozen=True)
class PolicyActionEvaluationResult:
    value_tensor: torch.Tensor
    log_prob_tensor: torch.Tensor
    entropy_tensor: torch.Tensor
    valid_action_mask: torch.Tensor
    data_build_sec: float
    batch_transfer_sec: float
    value_inference_sec: float
    policy_inference_sec: float


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

    def append(self, step_result: PolicyRolloutStepResult) -> None:
        reward_values = (
            step_result.training_rewards
            if step_result.training_rewards is not None
            else step_result.rewards
        )
        self.state_buffer.append(list(step_result.input_states))
        self.candidate_buffer.append(list(step_result.action_candidates))
        self.data_buffer.append(None if step_result.data_list is None else list(step_result.data_list))
        self.action_buffer.append(step_result.actions_tensor.detach().cpu())
        self.action_index_buffer.append(step_result.action_index_tensor.detach().cpu())
        self.log_prob_buffer.append(step_result.log_prob_tensor.detach().cpu())
        self.entropy_buffer.append(step_result.entropy_tensor.detach().cpu())
        self.value_buffer.append(step_result.value_tensor.detach().cpu())
        self.reward_buffer.append(torch.tensor(reward_values, dtype=torch.float))
        self.done_buffer.append(torch.tensor(step_result.dones, dtype=torch.float))
        self.valid_mask_buffer.append(step_result.valid_action_mask.detach().cpu())

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


def batched_policy_action_selection(
    states: Sequence[Any],
    action_lists: Sequence[Sequence[CanonicalAction]],
    policy: Any,
    *,
    device: torch.device,
    deterministic: bool = False,
    vertex_preprocessor: VertexPreprocessor | None = None,
    trajectory_transforms: Sequence[SimilarityTransform] | None = None,
) -> PolicyActionSelectionResult:
    if len(states) != len(action_lists):
        raise ValueError("states and action_lists must have the same length.")
    if len(states) == 0:
        raise ValueError("states must be non-empty.")

    policy = _ensure_policy_device(policy, device)
    include_simplex_topology = _policy_uses_simplex_topology(policy)
    candidate_lists = [tuple(tuple(int(v) for v in action) for action in actions) for actions in action_lists]

    data_build_start = time.perf_counter()
    subcomplex_width = infer_batch_subcomplex_width(states, candidate_lists)
    full_data_list = build_cy_data_list(
        states,
        candidate_lists,
        subcomplex_width=subcomplex_width,
        vertex_preprocessor=vertex_preprocessor,
        trajectory_transforms=trajectory_transforms,
        include_simplex_topology=include_simplex_topology,
    )
    data_build_sec = time.perf_counter() - data_build_start

    num_states = len(states)
    action_index_tensor = torch.full((num_states,), -1, dtype=torch.long, device=device)
    actions_tensor = torch.full((num_states, subcomplex_width), -1, dtype=torch.long, device=device)
    log_prob_tensor = torch.zeros(num_states, dtype=torch.float, device=device)
    entropy_tensor = torch.zeros(num_states, dtype=torch.float, device=device)
    valid_action_mask = torch.zeros(num_states, dtype=torch.bool, device=device)

    actionable_indices = [idx for idx, actions in enumerate(candidate_lists) if len(actions) > 0]
    batch_transfer_sec = 0.0
    value_inference_sec = 0.0
    policy_inference_sec = 0.0

    if len(actionable_indices) == num_states:
        full_batch = Batch.from_data_list(full_data_list)
        transfer_start = time.perf_counter()
        full_batch = full_batch.to(device)
        _synchronize_device(device)
        batch_transfer_sec += time.perf_counter() - transfer_start

        with torch.inference_mode():
            _synchronize_device(device)
            infer_start = time.perf_counter()
            value_tensor, logits_padded = policy.get_value_and_logits(full_batch)
            action_dist = torch.distributions.Categorical(logits=logits_padded)
            if deterministic:
                chosen_indices = torch.argmax(logits_padded, dim=1)
            else:
                chosen_indices = action_dist.sample()
            chosen_log_prob = action_dist.log_prob(chosen_indices)
            chosen_entropy = action_dist.entropy()
            _synchronize_device(device)
            policy_inference_sec += time.perf_counter() - infer_start

        valid_action_mask[:] = True
        action_index_tensor[:] = chosen_indices
        log_prob_tensor[:] = chosen_log_prob
        entropy_tensor[:] = chosen_entropy
    else:
        full_batch = Batch.from_data_list(full_data_list)
        transfer_start = time.perf_counter()
        full_batch = full_batch.to(device)
        _synchronize_device(device)
        batch_transfer_sec += time.perf_counter() - transfer_start

        with torch.inference_mode():
            _synchronize_device(device)
            infer_start = time.perf_counter()
            value_tensor = policy.get_value(full_batch)
            _synchronize_device(device)
            value_inference_sec += time.perf_counter() - infer_start

        if actionable_indices:
            actionable_data_list = [full_data_list[idx] for idx in actionable_indices]
            actionable_batch = Batch.from_data_list(actionable_data_list)
            transfer_start = time.perf_counter()
            actionable_batch = actionable_batch.to(device)
            _synchronize_device(device)
            batch_transfer_sec += time.perf_counter() - transfer_start

            with torch.inference_mode():
                _synchronize_device(device)
                infer_start = time.perf_counter()
                _actionable_values, logits_padded = policy.get_value_and_logits(actionable_batch)
                action_dist = torch.distributions.Categorical(logits=logits_padded)
                if deterministic:
                    chosen_indices = torch.argmax(logits_padded, dim=1)
                else:
                    chosen_indices = action_dist.sample()
                chosen_log_prob = action_dist.log_prob(chosen_indices)
                chosen_entropy = action_dist.entropy()
                _synchronize_device(device)
                policy_inference_sec += time.perf_counter() - infer_start

            for local_idx, global_idx in enumerate(actionable_indices):
                valid_action_mask[global_idx] = True
                action_index_tensor[global_idx] = chosen_indices[local_idx]
                log_prob_tensor[global_idx] = chosen_log_prob[local_idx]
                entropy_tensor[global_idx] = chosen_entropy[local_idx]

    for global_idx in actionable_indices:
        action_idx = int(action_index_tensor[global_idx].item())
        selected_action = candidate_lists[global_idx][action_idx]
        if selected_action:
            actions_tensor[global_idx, : len(selected_action)] = torch.tensor(
                selected_action,
                dtype=torch.long,
                device=device,
            )

    return PolicyActionSelectionResult(
        action_lists=candidate_lists,
        action_index_tensor=action_index_tensor,
        actions_tensor=actions_tensor,
        log_prob_tensor=log_prob_tensor,
        entropy_tensor=entropy_tensor,
        value_tensor=value_tensor,
        valid_action_mask=valid_action_mask,
        subcomplex_width=subcomplex_width,
        data_list=full_data_list,
        data_build_sec=data_build_sec,
        batch_transfer_sec=batch_transfer_sec,
        value_inference_sec=value_inference_sec,
        policy_inference_sec=policy_inference_sec,
        num_actionable=len(actionable_indices),
    )


def rollout_step_with_policy(
    engine: CYRandomRolloutEngine,
    states: Sequence[Any],
    policy: Any,
    *,
    rng: np.random.Generator,
    device: torch.device,
    initial_state_pool: Sequence[Any],
    deterministic: bool = False,
    use_multiprocessing: bool = False,
    transition_pool: Any = None,
    transition_mp_chunksize: int = 32,
    transition_mp_min_batch: int = 32,
    vertex_preprocessor: VertexPreprocessor | None = None,
    trajectory_transforms: Sequence[SimilarityTransform] | None = None,
) -> PolicyRolloutStepResult:
    current_states = list(states)

    candidate_expand_start = time.perf_counter()
    action_lists, expand_summary = engine.candidate_actions_for_states(
        current_states,
        use_multiprocessing=use_multiprocessing,
        transition_pool=transition_pool,
        transition_mp_chunksize=transition_mp_chunksize,
        transition_mp_min_batch=transition_mp_min_batch,
    )
    candidate_expand_sec = time.perf_counter() - candidate_expand_start

    selection = batched_policy_action_selection(
        current_states,
        action_lists,
        policy,
        device=device,
        deterministic=deterministic,
        vertex_preprocessor=vertex_preprocessor,
        trajectory_transforms=trajectory_transforms,
    )

    transition_apply_start = time.perf_counter()
    transitioned_states: List[Any] = []
    next_states: List[Any] = list(current_states)
    rewards = [0.0 for _ in current_states]
    dones = [False for _ in current_states]
    terminal_reasons = ["continue" for _ in current_states]
    chosen_actions: List[Optional[CanonicalAction]] = []
    frt_hits = 0
    collapsed_hits = 0
    dead_end_hits = 0

    unique_nonterminal_next_keys: dict[str, None] = {}
    reward_function = getattr(engine, "reward_function", None)
    objective_mode = reward_function is not None
    for idx, state in enumerate(current_states):
        if not bool(selection.valid_action_mask[idx].item()):
            transitioned_states.append(state)
            dones[idx] = True
            terminal_reasons[idx] = "dead_end_current"
            chosen_actions.append(None)
            dead_end_hits += 1
            continue

        action_idx = int(selection.action_index_tensor[idx].item())
        selected_action = selection.action_lists[idx][action_idx]
        chosen_actions.append(selected_action)

        transition = engine.nodes_by_key[str(state.key)].transitions[selected_action]
        if not objective_mode and transition.next_is_target is True:
            transitioned_states.append(state)
            rewards[idx] = 1.0
            dones[idx] = True
            terminal_reasons[idx] = "frt_or_frst"
            frt_hits += 1
            continue

        if not objective_mode and len(transition.next_simplices) <= 1:
            transitioned_states.append(state)
            rewards[idx] = -1.0
            dones[idx] = True
            terminal_reasons[idx] = "single_simplex"
            collapsed_hits += 1
            continue

        next_state = engine.materialize_state(transition.next_key)
        transitioned_states.append(next_state)
        next_states[idx] = next_state
        if objective_mode:
            rewards[idx] = float(reward_function(state, next_state))
        elif engine.is_target_state_fn(next_state):
            rewards[idx] = 1.0
            dones[idx] = True
            terminal_reasons[idx] = "frt_or_frst"
            frt_hits += 1
            continue
        unique_nonterminal_next_keys.setdefault(str(next_state.key), None)

    nonterminal_next_states = [engine.materialize_state(key) for key in unique_nonterminal_next_keys]
    next_expand_summary = engine.expand_states(
        nonterminal_next_states,
        use_multiprocessing=use_multiprocessing,
        transition_pool=transition_pool,
        transition_mp_chunksize=transition_mp_chunksize,
        transition_mp_min_batch=transition_mp_min_batch,
    )

    for idx, transitioned_state in enumerate(next_states):
        if dones[idx]:
            continue
        if len(engine.nodes_by_key[str(transitioned_state.key)].candidate_actions) == 0:
            dones[idx] = True
            terminal_reasons[idx] = "dead_end_next"
            dead_end_hits += 1

    reset_indices = [idx for idx, done in enumerate(dones) if done]
    if reset_indices:
        reset_states = engine.sample_initial_states(
            len(reset_indices),
            rng=rng,
            initial_state_pool=initial_state_pool,
        )
        for idx, reset_state in zip(reset_indices, reset_states):
            next_states[idx] = reset_state

    transition_apply_sec = time.perf_counter() - transition_apply_start

    return PolicyRolloutStepResult(
        input_states=current_states,
        transitioned_states=transitioned_states,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
        chosen_actions=chosen_actions,
        terminal_reasons=terminal_reasons,
        action_candidates=selection.action_lists,
        action_index_tensor=selection.action_index_tensor,
        actions_tensor=selection.actions_tensor,
        log_prob_tensor=selection.log_prob_tensor,
        entropy_tensor=selection.entropy_tensor,
        value_tensor=selection.value_tensor,
        valid_action_mask=selection.valid_action_mask,
        reset_count=len(reset_indices),
        frt_hits=frt_hits,
        collapsed_hits=collapsed_hits,
        dead_end_hits=dead_end_hits,
        expanded_states=expand_summary.expanded_count + next_expand_summary.expanded_count,
        discovered_states=expand_summary.discovered_count + next_expand_summary.discovered_count,
        used_multiprocessing=expand_summary.used_multiprocessing or next_expand_summary.used_multiprocessing,
        candidate_expand_sec=candidate_expand_sec,
        policy_data_build_sec=selection.data_build_sec,
        policy_batch_transfer_sec=selection.batch_transfer_sec,
        policy_value_inference_sec=selection.value_inference_sec,
        policy_action_inference_sec=selection.policy_inference_sec,
        transition_apply_sec=transition_apply_sec,
        data_list=selection.data_list,
    )


def evaluate_policy_values(
    states: Sequence[Any],
    action_lists: Sequence[Sequence[CanonicalAction]],
    policy: Any,
    *,
    device: torch.device,
    vertex_preprocessor: VertexPreprocessor | None = None,
    trajectory_transforms: Sequence[SimilarityTransform] | None = None,
) -> PolicyValueResult:
    if len(states) != len(action_lists):
        raise ValueError("states and action_lists must have the same length.")
    if len(states) == 0:
        raise ValueError("states must be non-empty.")

    policy = _ensure_policy_device(policy, device)
    data_build_start = time.perf_counter()
    data_list = build_cy_data_list(
        states,
        action_lists,
        vertex_preprocessor=vertex_preprocessor,
        trajectory_transforms=trajectory_transforms,
    )
    data_build_sec = time.perf_counter() - data_build_start

    batch = Batch.from_data_list(data_list)
    transfer_start = time.perf_counter()
    batch = batch.to(device)
    _synchronize_device(device)
    batch_transfer_sec = time.perf_counter() - transfer_start

    with torch.inference_mode():
        _synchronize_device(device)
        infer_start = time.perf_counter()
        value_tensor = policy.get_value(batch)
        _synchronize_device(device)
        inference_sec = time.perf_counter() - infer_start

    return PolicyValueResult(
        value_tensor=value_tensor,
        data_build_sec=data_build_sec,
        batch_transfer_sec=batch_transfer_sec,
        inference_sec=inference_sec,
    )


def _data_num_available_subcomplexes(data: Data) -> int:
    num_available = getattr(data, "num_available_subcomplexes")
    if isinstance(num_available, torch.Tensor):
        return int(num_available.reshape(-1)[0].item())
    return int(num_available)


def _data_subcomplex_width(data: Data) -> int:
    subcomplex_vertices = getattr(data, "subcomplex_vertices")
    if not isinstance(subcomplex_vertices, torch.Tensor):
        raise TypeError("`subcomplex_vertices` must be a torch.Tensor.")
    if subcomplex_vertices.dim() == 1:
        return int(subcomplex_vertices.numel())
    if subcomplex_vertices.dim() == 2:
        return int(subcomplex_vertices.size(1))
    raise ValueError(
        "`subcomplex_vertices` must be 1D or 2D, got shape "
        f"{tuple(subcomplex_vertices.shape)}."
    )


def _copy_data_with_subcomplex_vertices(data: Data, subcomplex_vertices: torch.Tensor) -> Data:
    copied = Data(
        x=data.x,
        edge_index=data.edge_index,
        subcomplex_vertices=subcomplex_vertices,
        num_available_subcomplexes=data.num_available_subcomplexes,
    )
    if hasattr(data, "simplex_vertices"):
        copied.simplex_vertices = data.simplex_vertices
    if hasattr(data, "num_top_simplices"):
        copied.num_top_simplices = data.num_top_simplices
    for attr_name in (
        "snn_laplacian_row",
        "snn_laplacian_col",
        "snn_laplacian_value",
        "num_snn_laplacian_entries",
        "snn_candidate",
        "snn_simplex",
        "num_snn_candidate_simplex_memberships",
    ):
        if hasattr(data, attr_name):
            setattr(copied, attr_name, getattr(data, attr_name))
    copied.edge_attr = getattr(data, "edge_attr", None)
    copied.num_edges = getattr(data, "num_edges", data.edge_index.size(1))
    return copied


def _pad_data_list_subcomplex_width(data_list: Sequence[Data]) -> List[Data]:
    if not data_list:
        return []

    max_width = max(_data_subcomplex_width(data) for data in data_list)
    padded_data_list: List[Data] = []
    for data in data_list:
        subcomplex_vertices = data.subcomplex_vertices
        if subcomplex_vertices.dim() == 1:
            subcomplex_vertices = subcomplex_vertices.view(1, -1)
        width = int(subcomplex_vertices.size(1))
        if width == max_width:
            padded_data_list.append(data)
            continue

        padded = torch.full(
            (int(subcomplex_vertices.size(0)), int(max_width)),
            -1,
            dtype=subcomplex_vertices.dtype,
            device=subcomplex_vertices.device,
        )
        if width > 0 and subcomplex_vertices.size(0) > 0:
            padded[:, :width] = subcomplex_vertices
        padded_data_list.append(_copy_data_with_subcomplex_vertices(data, padded))
    return padded_data_list


def evaluate_policy_actions_from_data_list(
    data_list: Sequence[Data],
    action_indices: torch.Tensor,
    policy: Any,
    *,
    device: torch.device,
) -> PolicyActionEvaluationResult:
    if len(data_list) == 0:
        raise ValueError("data_list must be non-empty.")

    policy = _ensure_policy_device(policy, device)
    full_data_list = _pad_data_list_subcomplex_width(data_list)
    action_indices = action_indices.to(device=device, dtype=torch.long).view(-1)
    if action_indices.size(0) != len(full_data_list):
        raise ValueError("action_indices must have one element per state.")

    num_states = len(full_data_list)
    log_prob_tensor = torch.zeros(num_states, dtype=torch.float, device=device)
    entropy_tensor = torch.zeros(num_states, dtype=torch.float, device=device)
    valid_action_mask = torch.zeros(num_states, dtype=torch.bool, device=device)

    actionable_indices = [
        idx for idx, data in enumerate(full_data_list) if _data_num_available_subcomplexes(data) > 0
    ]
    batch_transfer_sec = 0.0
    value_inference_sec = 0.0
    policy_inference_sec = 0.0

    if len(actionable_indices) == num_states:
        full_batch = Batch.from_data_list(full_data_list)
        transfer_start = time.perf_counter()
        full_batch = full_batch.to(device)
        _synchronize_device(device)
        batch_transfer_sec += time.perf_counter() - transfer_start

        _synchronize_device(device)
        infer_start = time.perf_counter()
        value_tensor, logits_padded = policy.get_value_and_logits(full_batch)
        chosen_log_prob, chosen_entropy = policy.get_log_prob(logits_padded, action_indices)
        _synchronize_device(device)
        policy_inference_sec += time.perf_counter() - infer_start

        valid_action_mask[:] = True
        log_prob_tensor[:] = chosen_log_prob
        entropy_tensor[:] = chosen_entropy
    else:
        full_batch = Batch.from_data_list(full_data_list)
        transfer_start = time.perf_counter()
        full_batch = full_batch.to(device)
        _synchronize_device(device)
        batch_transfer_sec += time.perf_counter() - transfer_start

        _synchronize_device(device)
        infer_start = time.perf_counter()
        value_tensor = policy.get_value(full_batch)
        _synchronize_device(device)
        value_inference_sec += time.perf_counter() - infer_start

        if actionable_indices:
            actionable_data_list = [full_data_list[idx] for idx in actionable_indices]
            actionable_batch = Batch.from_data_list(actionable_data_list)
            transfer_start = time.perf_counter()
            actionable_batch = actionable_batch.to(device)
            _synchronize_device(device)
            batch_transfer_sec += time.perf_counter() - transfer_start

            actionable_indices_tensor = action_indices[actionable_indices]
            _synchronize_device(device)
            infer_start = time.perf_counter()
            _actionable_values, logits_padded = policy.get_value_and_logits(actionable_batch)
            chosen_log_prob, chosen_entropy = policy.get_log_prob(
                logits_padded,
                actionable_indices_tensor,
            )
            _synchronize_device(device)
            policy_inference_sec += time.perf_counter() - infer_start

            for local_idx, global_idx in enumerate(actionable_indices):
                valid_action_mask[global_idx] = True
                log_prob_tensor[global_idx] = chosen_log_prob[local_idx]
                entropy_tensor[global_idx] = chosen_entropy[local_idx]

    return PolicyActionEvaluationResult(
        value_tensor=value_tensor,
        log_prob_tensor=log_prob_tensor,
        entropy_tensor=entropy_tensor,
        valid_action_mask=valid_action_mask,
        data_build_sec=0.0,
        batch_transfer_sec=batch_transfer_sec,
        value_inference_sec=value_inference_sec,
        policy_inference_sec=policy_inference_sec,
    )


def evaluate_policy_actions(
    states: Sequence[Any],
    action_lists: Sequence[Sequence[CanonicalAction]],
    action_indices: torch.Tensor,
    policy: Any,
    *,
    device: torch.device,
    vertex_preprocessor: VertexPreprocessor | None = None,
    trajectory_transforms: Sequence[SimilarityTransform] | None = None,
) -> PolicyActionEvaluationResult:
    if len(states) != len(action_lists):
        raise ValueError("states and action_lists must have the same length.")
    if len(states) == 0:
        raise ValueError("states must be non-empty.")

    policy = _ensure_policy_device(policy, device)
    include_simplex_topology = _policy_uses_simplex_topology(policy)
    candidate_lists = [tuple(tuple(int(v) for v in action) for action in actions) for actions in action_lists]
    action_indices = action_indices.to(device=device, dtype=torch.long).view(-1)
    if action_indices.size(0) != len(states):
        raise ValueError("action_indices must have one element per state.")

    data_build_start = time.perf_counter()
    full_data_list = build_cy_data_list(
        states,
        candidate_lists,
        vertex_preprocessor=vertex_preprocessor,
        trajectory_transforms=trajectory_transforms,
        include_simplex_topology=include_simplex_topology,
    )
    data_build_sec = time.perf_counter() - data_build_start

    evaluation = evaluate_policy_actions_from_data_list(
        full_data_list,
        action_indices,
        policy,
        device=device,
    )
    return PolicyActionEvaluationResult(
        value_tensor=evaluation.value_tensor,
        log_prob_tensor=evaluation.log_prob_tensor,
        entropy_tensor=evaluation.entropy_tensor,
        valid_action_mask=evaluation.valid_action_mask,
        data_build_sec=data_build_sec,
        batch_transfer_sec=evaluation.batch_transfer_sec,
        value_inference_sec=evaluation.value_inference_sec,
        policy_inference_sec=evaluation.policy_inference_sec,
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


def increment_visitation(
    states: Iterable[Any],
    visit_counts_by_key: dict[str, int] | None = None,
) -> None:
    for state in states:
        if hasattr(state, "visitation"):
            state.visitation += 1
        if visit_counts_by_key is not None:
            key = _state_key(state)
            visit_counts_by_key[key] = int(visit_counts_by_key.get(key, 0)) + 1


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
) -> PPOTrainStats:
    state_buffer_list = prepared_rollout.state_buffer_list
    candidate_buffer_list = prepared_rollout.candidate_buffer_list
    data_buffer_list = prepared_rollout.data_buffer_list
    action_index_buffer_flat = prepared_rollout.action_index_buffer_flat.detach()
    old_log_prob_flat = prepared_rollout.log_prob_buffer_flat.detach()
    value_targets_flat = prepared_rollout.value_targets.reshape(-1).detach()
    old_value_estimate_flat = prepared_rollout.value_buffer_tensor.reshape(-1).detach()
    valid_mask_flat = prepared_rollout.valid_mask_flat.detach()
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
            mini_batch_indices = permutation[start : start + int(batch_size)]
            action_index_batch = action_index_buffer_flat[mini_batch_indices].detach()
            old_log_prob_batch = old_log_prob_flat[mini_batch_indices].detach()
            target_values = value_targets_flat[mini_batch_indices].detach()
            advantage_batch = advantages_flat[mini_batch_indices].detach()
            valid_mask_batch = valid_mask_flat[mini_batch_indices].detach()

            if data_buffer_list is not None:
                evaluation = evaluate_policy_actions_from_data_list(
                    [data_buffer_list[idx] for idx in mini_batch_indices],
                    action_index_batch,
                    policy,
                    device=device,
                )
            else:
                mini_states = [state_buffer_list[idx] for idx in mini_batch_indices]
                mini_candidates = [candidate_buffer_list[idx] for idx in mini_batch_indices]
                evaluation = evaluate_policy_actions(
                    mini_states,
                    mini_candidates,
                    action_index_batch,
                    policy,
                    device=device,
                )
            actionable_mask = valid_mask_batch & evaluation.valid_action_mask
            value_estimate = evaluation.value_tensor

            if bool(actionable_mask.any().item()):
                new_log_prob = evaluation.log_prob_tensor[actionable_mask]
                old_log_prob = old_log_prob_batch[actionable_mask]
                advantage_values = advantage_batch[actionable_mask]
                entropy_values = evaluation.entropy_tensor[actionable_mask]

                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -ratio * advantage_values
                clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef))
                clipped = -clipped_ratio * advantage_values
                policy_loss = torch.max(unclipped, clipped).mean()
                entropy_loss = entropy_values.mean()

                clipped_samples = (torch.abs(ratio - 1.0) > float(clip_coef)).float()
                total_clip_count += float(clipped_samples.sum().item())
                total_valid_action_samples += int(actionable_mask.sum().item())
            else:
                policy_loss = value_estimate.new_zeros(())
                entropy_loss = value_estimate.new_zeros(())

            value_loss = 0.5 * (value_estimate - target_values).pow(2).mean()
            loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
            optimizer.step()

            batch_count = len(mini_batch_indices)
            total_loss_sum += float(loss.item()) * batch_count
            total_value_loss_sum += float(value_loss.item()) * batch_count
            total_value_weight += batch_count

            actionable_count = int(actionable_mask.sum().item())
            total_policy_loss_sum += float(policy_loss.item()) * actionable_count
            total_entropy_loss_sum += float(entropy_loss.item()) * actionable_count
            total_policy_weight += actionable_count

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
