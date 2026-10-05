from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from core.cy_data_utils import create_data_from_cy_state_with_subcomplex, get_cached_transformed_vertices
from core.vertex_augmentation import SimilarityTransform
from core.vertex_preprocessing import VertexPreprocessor
from mdp.cy_graph import CanonicalAction

if TYPE_CHECKING:
    from mdp.cy_rollout import CYRandomRolloutEngine

_POLICY_MAX_GRAPH_SIZE = 250_000
_SYNCHRONIZE_TIMING = False


def configure_policy_execution(*, max_graph_size: int = 250_000, synchronize_timing: bool = False) -> None:
    """Bound simultaneous graph work; enable CUDA timing barriers only on request."""
    if int(max_graph_size) <= 0:
        raise ValueError("max_graph_size must be positive.")
    global _POLICY_MAX_GRAPH_SIZE, _SYNCHRONIZE_TIMING
    _POLICY_MAX_GRAPH_SIZE = int(max_graph_size)
    _SYNCHRONIZE_TIMING = bool(synchronize_timing)


def _policy_requires_logical_batch(policy: Any) -> bool:
    if not bool(getattr(policy, "training", False)):
        return False
    modules = getattr(policy, "modules", lambda: ())()
    has_training_batch_norm = any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training for module in modules)
    independent_actor = (type(policy).__module__ in ("models.egnn_subcomplex_predictor", "models.gcn_subcomplex_predictor")
                         and getattr(policy, "subcomplex_actor_type", "") in ("gnn", "circuit_pool", "snn_simplex"))
    return has_training_batch_norm and not independent_actor


def policy_data_chunks(data_list: Sequence[Data], *, max_graph_size: int | None = None, policy: Any = None) -> Iterator[List[int]]:
    """Yield ordered physical batches; a single graph is indivisible.

    Count vertices, edges, candidate clique edges, and sparse simplex work to
    account for both the vertex encoder and either candidate decoder.
    """
    limit = _POLICY_MAX_GRAPH_SIZE if max_graph_size is None else int(max_graph_size)
    if limit <= 0:
        raise ValueError("max_graph_size must be positive.")
    # Known CY actors bypass projection BatchNorm. Other training policies need
    # the complete logical batch to preserve normalization and gradients.
    if _policy_requires_logical_batch(policy):
        if data_list:
            yield list(range(len(data_list)))
        return
    chunk: List[int] = []
    cost = 0
    for index, data in enumerate(data_list):
        candidates = data.subcomplex_vertices
        width = int(candidates.size(-1))
        graph_cost = int(data.x.size(0) + data.edge_index.size(1))
        graph_cost += int(candidates.size(0)) * max(1, width * width)
        for name in ("simplex_vertices", "snn_laplacian_row", "snn_candidate"):
            value = getattr(data, name, None)
            if isinstance(value, torch.Tensor):
                graph_cost += value.numel()
        if chunk and cost + graph_cost > limit:
            yield chunk
            chunk, cost = [], 0
        chunk.append(index)
        cost += graph_cost
    if chunk:
        yield chunk

def _synchronize_device(device: torch.device) -> None:
    if _SYNCHRONIZE_TIMING and device.type == "cuda":
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
    return (
        str(getattr(policy, "subcomplex_actor_type", "")).strip().lower() == "snn_simplex"
        or str(getattr(policy, "value_feature_source", "")).strip().lower() == "snn_simplex"
    )


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
        data.x = get_cached_transformed_vertices(data.x, transform)
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
    action_indices_cpu: List[int] | None = None


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


def _forward_policy_data(data_list: Sequence[Data], policy: Any, *, device: torch.device,
                         value_only: bool = False, max_graph_size: int | None = None):
    values, logits = [], []
    transfer_sec = inference_sec = 0.0
    max_candidates = max(_data_num_available_subcomplexes(data) for data in data_list)
    for indices in policy_data_chunks(data_list, policy=policy, max_graph_size=max_graph_size):
        batch = Batch.from_data_list([data_list[index] for index in indices])
        start = time.perf_counter()
        batch = batch.to(device)
        _synchronize_device(device)
        transfer_sec += time.perf_counter() - start
        start = time.perf_counter()
        if value_only:
            values.append(policy.get_value(batch))
        elif _policy_requires_logical_batch(policy) and any(_data_num_available_subcomplexes(data_list[index]) == 0 for index in indices):
            # The legacy mixed path normalized projected policy embeddings over
            # actionable states only. Preserve that population while training.
            values.append(policy.get_value(batch))
            actionable = [position for position, index in enumerate(indices)
                          if _data_num_available_subcomplexes(data_list[index]) > 0]
            chunk_logits = values[-1].new_full((len(indices), max_candidates), float("-inf"))
            if actionable:
                actionable_batch = Batch.from_data_list([data_list[indices[position]] for position in actionable]).to(device)
                _unused_values, actionable_logits = policy.get_value_and_logits(actionable_batch)
                chunk_logits[actionable] = actionable_logits
                del actionable_batch, _unused_values
            logits.append(chunk_logits)
        else:
            chunk_values, chunk_logits = policy.get_value_and_logits(batch)
            values.append(chunk_values)
            logits.append(torch.nn.functional.pad(chunk_logits, (0, max_candidates - chunk_logits.size(1)), value=float("-inf")))
        _synchronize_device(device)
        inference_sec += time.perf_counter() - start
        del batch
    return torch.cat(values), None if value_only else torch.cat(logits), transfer_sec, inference_sec


@dataclass(frozen=True)
class PolicyScoreResult:
    value_tensor: torch.Tensor
    logits_tensor: torch.Tensor | None
    data_build_sec: float
    batch_transfer_sec: float
    inference_sec: float
    physical_batches: int


def evaluate_policy_scores(
    states: Sequence[Any], action_lists: Sequence[Sequence[CanonicalAction]], policy: Any,
    *, device: torch.device, value_only: bool = False, max_graph_size: int | None = None,
) -> PolicyScoreResult:
    """Shared inference without sampling; callers own search rules and RNGs.

    Value-only callers can pass empty action lists: the critic depends on the
    state topology, not its outgoing actions. No neighbor enumeration is needed.
    """
    if len(states) != len(action_lists):
        raise ValueError("states and action_lists must have the same length.")
    if not states:
        return PolicyScoreResult(torch.empty(0, device=device), None, 0.0, 0.0, 0.0, 0)
    policy = _ensure_policy_device(policy, device)
    start = time.perf_counter()
    data = build_cy_data_list(states, action_lists, include_simplex_topology=_policy_uses_simplex_topology(policy))
    build_sec = time.perf_counter() - start
    with torch.inference_mode():
        values, logits, transfer_sec, inference_sec = _forward_policy_data(
            data, policy, device=device, value_only=value_only, max_graph_size=max_graph_size,
        )
    batches = sum(1 for _ in policy_data_chunks(data, policy=policy, max_graph_size=max_graph_size))
    return PolicyScoreResult(values, logits, build_sec, transfer_sec, inference_sec, batches)


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

    with torch.inference_mode():
        value_tensor, all_logits, batch_transfer_sec, policy_inference_sec = _forward_policy_data(
            full_data_list, policy, device=device,
        )
        if actionable_indices:
            # Make one categorical draw, after all chunks, preserving environment
            # order and RNG consumption independently of the physical batch size.
            logits_padded = all_logits[actionable_indices]
            action_dist = torch.distributions.Categorical(logits=logits_padded)
            chosen_indices = torch.argmax(logits_padded, dim=1) if deterministic else action_dist.sample()
            valid_action_mask[actionable_indices] = True
            action_index_tensor[actionable_indices] = chosen_indices
            log_prob_tensor[actionable_indices] = action_dist.log_prob(chosen_indices)
            entropy_tensor[actionable_indices] = action_dist.entropy()

    chosen_indices_cpu = action_index_tensor.detach().cpu().tolist()
    actions_cpu = torch.full((num_states, subcomplex_width), -1, dtype=torch.long, device="cpu")
    for global_idx in actionable_indices:
        action_idx = chosen_indices_cpu[global_idx]
        selected_action = candidate_lists[global_idx][action_idx]
        if selected_action:
            actions_cpu[global_idx, : len(selected_action)] = torch.tensor(
                selected_action,
                dtype=torch.long,
                device="cpu",
            )
    actions_tensor = actions_cpu.to(device)

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
        action_indices_cpu=chosen_indices_cpu,
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

    unique_nonterminal_next_keys: dict[str, Any] = {}
    reward_function = getattr(engine, "reward_function", None)
    objective_mode = reward_function is not None
    action_indices_cpu = getattr(selection, "action_indices_cpu", None)
    if action_indices_cpu is None:
        action_indices_cpu = selection.action_index_tensor.detach().cpu().tolist()
    for idx, state in enumerate(current_states):
        if action_indices_cpu[idx] < 0:
            transitioned_states.append(state)
            dones[idx] = True
            terminal_reasons[idx] = "dead_end_current"
            chosen_actions.append(None)
            dead_end_hits += 1
            continue

        action_idx = action_indices_cpu[idx]
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

        num_next_simplices = (transition.num_next_simplices(state.simplices)
                             if hasattr(transition, "num_next_simplices") else len(transition.next_simplices))
        if not objective_mode and num_next_simplices <= 1:
            transitioned_states.append(state)
            rewards[idx] = -1.0
            dones[idx] = True
            terminal_reasons[idx] = "single_simplex"
            collapsed_hits += 1
            continue

        next_state = (engine.materialize_transition(state, transition)
                      if hasattr(engine, "materialize_transition") else engine.materialize_state(transition.next_key))
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
        unique_nonterminal_next_keys.setdefault(str(next_state.key), next_state)

    nonterminal_next_states = list(unique_nonterminal_next_keys.values())
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
    include_simplex_topology = _policy_uses_simplex_topology(policy)
    data_build_start = time.perf_counter()
    data_list = build_cy_data_list(
        states,
        action_lists,
        vertex_preprocessor=vertex_preprocessor,
        trajectory_transforms=trajectory_transforms,
        include_simplex_topology=include_simplex_topology,
    )
    data_build_sec = time.perf_counter() - data_build_start

    with torch.inference_mode():
        value_tensor, _logits, batch_transfer_sec, inference_sec = _forward_policy_data(
            data_list, policy, device=device, value_only=True,
        )

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

    value_tensor, logits_padded, batch_transfer_sec, policy_inference_sec = _forward_policy_data(
        full_data_list, policy, device=device,
    )
    if actionable_indices:
        chosen_log_prob, chosen_entropy = policy.get_log_prob(
            logits_padded[actionable_indices], action_indices[actionable_indices],
        )
        valid_action_mask[actionable_indices] = True
        log_prob_tensor[actionable_indices] = chosen_log_prob
        entropy_tensor[actionable_indices] = chosen_entropy

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
