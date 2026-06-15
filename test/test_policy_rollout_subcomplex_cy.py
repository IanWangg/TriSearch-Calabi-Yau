from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from core.cy_policy_rollout_utils import (
    PPORolloutBuffer,
    PolicyRolloutStepResult,
    batched_policy_action_selection,
    build_cy_data_list,
    compute_cy_state_count_bonus,
    compute_gae_with_dones,
    rollout_step_with_policy,
)
from core.vertex_augmentation import SimilarityTransform
from core.vertex_preprocessing import VertexPreprocessor
from mdp.cy_rollout import CYRandomRolloutEngine
from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


def _simplices_key(simplices: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(simplex)) for simplex in simplices))


def _edges_from_simplices(simplices: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, int], ...]:
    edges = set()
    for simplex in simplices:
        simplex = tuple(simplex)
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add(tuple(sorted((int(simplex[i]), int(simplex[j])))))
    return tuple(sorted(edges))


class SimpleGraphState:
    def __init__(self, *, key: str, point_config_index: int, vertices, simplices):
        self.key = key
        self.point_config_index = point_config_index
        self.vertices = [list(point) for point in vertices]
        self.simplices = _simplices_key(simplices)
        self.edges = _edges_from_simplices(self.simplices)


class FakeCYState(SimpleGraphState):
    def __init__(
        self,
        *,
        key: str,
        point_config_index: int,
        vertices,
        simplices,
        transition_by_action: Dict[Tuple[int, ...], tuple],
        ambiguous_actions: Iterable[Tuple[int, ...]] = (),
        is_target: bool = False,
    ):
        super().__init__(
            key=key,
            point_config_index=point_config_index,
            vertices=vertices,
            simplices=simplices,
        )
        self.actions_ready = False
        self.available_subcomplex_actions = tuple(transition_by_action.keys())
        self.ambiguous_subcomplex_actions = frozenset(ambiguous_actions)
        self._transition_by_action = dict(transition_by_action)
        self.find_calls = 0
        self.visitation = 0
        self.is_target = bool(is_target)

    def find_available_actions(self):
        self.find_calls += 1
        self.actions_ready = True

    def get_available_subcomplex_actions(self):
        return tuple(self.available_subcomplex_actions)

    def get_transition_output_from_subcomplex_action(self, action):
        canonical = tuple(int(v) for v in action)
        return self._transition_by_action[canonical]


def _build_fake_state_factory(state_by_simplices):
    def _factory(point_config_index, simplices):
        key = (int(point_config_index), _simplices_key(simplices))
        return state_by_simplices[key]

    return _factory


def _build_agent():
    torch.manual_seed(0)
    return EGNNSubcomplexAgent(
        in_channels=3,
        out_channels=16,
        hidden_channels=16,
        num_layers=2,
        share_encoder=True,
        mlp_hidden_channel_list=[16],
        use_projection=True,
        act="silu",
        device="cpu",
    ).eval()


def test_batched_policy_action_selection_handles_mixed_actionability():
    state_a = SimpleGraphState(
        key="a",
        point_config_index=3,
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        simplices=((0, 1, 2), (0, 2, 3)),
    )
    state_b = SimpleGraphState(
        key="b",
        point_config_index=3,
        vertices=[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        simplices=((0, 1, 3), (1, 2, 3)),
    )
    action_lists = [
        [(0, 1, 2), (0, 2, 3)],
        [],
    ]
    device = torch.device("cpu")
    model = _build_agent()

    result = batched_policy_action_selection(
        [state_a, state_b],
        action_lists,
        model,
        device=device,
        deterministic=True,
    )

    assert tuple(result.value_tensor.shape) == (2,)
    assert tuple(result.action_index_tensor.shape) == (2,)
    assert tuple(result.actions_tensor.shape) == (2, 4)
    assert result.num_actionable == 1
    assert result.valid_action_mask.tolist() == [True, False]
    assert int(result.action_index_tensor[1].item()) == -1
    assert result.actions_tensor[1].tolist() == [-1, -1, -1, -1]
    assert float(result.log_prob_tensor[1].item()) == 0.0
    assert float(result.entropy_tensor[1].item()) == 0.0

    data_list = build_cy_data_list([state_a], [action_lists[0]], subcomplex_width=result.subcomplex_width)
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
    expected_idx = int(torch.argmax(logits, dim=1).item())
    expected_action = list(action_lists[0][expected_idx]) + [-1]
    assert result.actions_tensor[0].tolist() == expected_action


def test_build_cy_data_list_applies_vertex_preprocessing():
    state = SimpleGraphState(
        key="prep",
        point_config_index=5,
        vertices=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ],
        simplices=((0, 1, 2), (0, 1, 3)),
    )

    data_list = build_cy_data_list(
        [state],
        [[(0, 1, 2, 3)]],
        vertex_preprocessor=VertexPreprocessor("rms_radius"),
    )

    assert len(data_list) == 1
    assert torch.allclose(data_list[0].x.mean(dim=0), torch.zeros(3), atol=1e-6)


def test_build_cy_data_list_applies_trajectory_transforms():
    state = SimpleGraphState(
        key="aug",
        point_config_index=5,
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices=((0, 1, 2), (0, 1, 3)),
    )
    transform = SimilarityTransform(
        matrix=2.0 * torch.eye(3, dtype=torch.float),
        bias=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float),
    )

    data_list = build_cy_data_list(
        [state],
        [[(0, 1, 2, 3)]],
        trajectory_transforms=[transform],
    )

    expected = torch.tensor(state.vertices, dtype=torch.float) * 2.0 + transform.bias
    assert len(data_list) == 1
    assert torch.allclose(data_list[0].x, expected)


def test_rollout_step_with_policy_matches_engine_terminal_and_reset_rules():
    point_config_index = 7
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ]
    a_simplices = ((0, 1, 2), (0, 2, 4))
    b_simplices = ((0, 1, 3), (0, 3, 4))
    c_simplices = ((0, 1, 4), (1, 3, 4))
    dead_simplices = ((0, 2, 3), (2, 3, 4))

    state_a = FakeCYState(
        key="a",
        point_config_index=point_config_index,
        vertices=vertices,
        simplices=a_simplices,
        transition_by_action={(0, 1, 2, 4): (frozenset(b_simplices), frozenset(), "b")},
    )
    state_b = FakeCYState(
        key="b",
        point_config_index=point_config_index,
        vertices=vertices,
        simplices=b_simplices,
        transition_by_action={(0, 1, 3, 4): (frozenset(c_simplices), frozenset(), "c")},
    )
    state_c = FakeCYState(
        key="c",
        point_config_index=point_config_index,
        vertices=vertices,
        simplices=c_simplices,
        transition_by_action={},
        is_target=True,
    )
    state_dead = FakeCYState(
        key="dead",
        point_config_index=point_config_index,
        vertices=vertices,
        simplices=dead_simplices,
        transition_by_action={},
    )

    state_factory = _build_fake_state_factory(
        {
            (point_config_index, _simplices_key(a_simplices)): state_a,
            (point_config_index, _simplices_key(b_simplices)): state_b,
            (point_config_index, _simplices_key(c_simplices)): state_c,
            (point_config_index, _simplices_key(dead_simplices)): state_dead,
        }
    )
    engine = CYRandomRolloutEngine(
        base_states={"a": state_a, "b": state_b, "c": state_c, "dead": state_dead},
        initial_states=[state_a],
        state_factory=state_factory,
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )
    model = _build_agent()
    rng = np.random.default_rng(0)

    step = rollout_step_with_policy(
        engine,
        [state_b, state_dead],
        model,
        rng=rng,
        device=torch.device("cpu"),
        initial_state_pool=[state_a],
        deterministic=True,
    )

    assert step.dones == [True, True]
    assert step.rewards == [1.0, 0.0]
    assert step.terminal_reasons == ["frt_or_frst", "dead_end_current"]
    assert step.chosen_actions == [(0, 1, 3, 4), None]
    assert [state.key for state in step.next_states] == ["a", "a"]
    assert tuple(step.value_tensor.shape) == (2,)
    assert step.valid_action_mask.tolist() == [True, False]
    assert step.reset_count == 2
    assert step.frt_hits == 1
    assert step.dead_end_hits == 1


def test_ppo_rollout_buffer_prepare_pads_actions_and_respects_dones():
    buffer = PPORolloutBuffer()
    buffer.append(
        PolicyRolloutStepResult(
            input_states=[SimpleNamespace(key="s0"), SimpleNamespace(key="s1")],
            transitioned_states=[],
            next_states=[],
            rewards=[1.0, 2.0],
            dones=[False, True],
            chosen_actions=[(1, 2), None],
            terminal_reasons=["continue", "dead_end_current"],
            action_candidates=[((1, 2),), tuple()],
            action_index_tensor=torch.tensor([0, -1], dtype=torch.long),
            actions_tensor=torch.tensor([[1, 2, -1], [-1, -1, -1]], dtype=torch.long),
            log_prob_tensor=torch.tensor([0.1, 0.0], dtype=torch.float),
            entropy_tensor=torch.tensor([0.2, 0.0], dtype=torch.float),
            value_tensor=torch.tensor([0.3, 0.4], dtype=torch.float),
            valid_action_mask=torch.tensor([True, False]),
            reset_count=1,
            frt_hits=0,
            collapsed_hits=0,
            dead_end_hits=1,
            expanded_states=0,
            discovered_states=0,
            used_multiprocessing=False,
            candidate_expand_sec=0.0,
            policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0,
            policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0,
            transition_apply_sec=0.0,
        )
    )
    buffer.append(
        PolicyRolloutStepResult(
            input_states=[SimpleNamespace(key="s2"), SimpleNamespace(key="s3")],
            transitioned_states=[],
            next_states=[],
            rewards=[3.0, 4.0],
            dones=[False, False],
            chosen_actions=[(5,), (6, 7)],
            terminal_reasons=["continue", "continue"],
            action_candidates=[((5,),), ((6, 7),)],
            action_index_tensor=torch.tensor([0, 0], dtype=torch.long),
            actions_tensor=torch.tensor([[5, -1], [6, 7]], dtype=torch.long),
            log_prob_tensor=torch.tensor([0.5, 0.6], dtype=torch.float),
            entropy_tensor=torch.tensor([0.7, 0.8], dtype=torch.float),
            value_tensor=torch.tensor([0.9, 1.0], dtype=torch.float),
            valid_action_mask=torch.tensor([True, True]),
            reset_count=0,
            frt_hits=0,
            collapsed_hits=0,
            dead_end_hits=0,
            expanded_states=0,
            discovered_states=0,
            used_multiprocessing=False,
            candidate_expand_sec=0.0,
            policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0,
            policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0,
            transition_apply_sec=0.0,
        )
    )

    bootstrap_value = torch.tensor([0.5, 0.6], dtype=torch.float)
    prepared = buffer.prepare(
        bootstrap_value=bootstrap_value,
        gamma=0.9,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )

    expected_advantages, expected_value_targets = compute_gae_with_dones(
        reward_buffer_tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float),
        value_buffer_tensor=torch.tensor([[0.3, 0.4], [0.9, 1.0]], dtype=torch.float),
        done_buffer_tensor=torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float),
        bootstrap_value=bootstrap_value,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert prepared.action_buffer_flat.tolist() == [
        [1, 2, -1],
        [-1, -1, -1],
        [5, -1, -1],
        [6, 7, -1],
    ]
    assert prepared.action_index_buffer_flat.tolist() == [0, -1, 0, 0]
    assert prepared.valid_mask_flat.tolist() == [True, False, True, True]
    assert tuple(prepared.reward_buffer_tensor.shape) == (2, 2)
    assert tuple(prepared.value_buffer_tensor.shape) == (2, 2)
    assert torch.allclose(prepared.advantages, expected_advantages)
    assert torch.allclose(prepared.value_targets, expected_value_targets)
    assert len(prepared.state_buffer_list) == 4
    assert len(prepared.candidate_buffer_list) == 4


def test_compute_cy_state_count_bonus_uses_destination_pre_visit_counts():
    current = SimpleNamespace(key="current", visitation=10)
    new_destination = SimpleNamespace(key="new", visitation=0)
    repeated_destination = SimpleNamespace(key="repeated", visitation=0)

    bonus = compute_cy_state_count_bonus(
        input_states=[current, current, current],
        transitioned_states=[new_destination, repeated_destination, current],
        visit_counts_by_key={"repeated": 3},
        coef=4.0,
        exponent=1.0,
    )

    assert bonus == pytest.approx([4.0, 1.0, 0.0])


def test_ppo_rollout_buffer_prepare_uses_training_rewards_when_present():
    buffer = PPORolloutBuffer()
    buffer.append(
        PolicyRolloutStepResult(
            input_states=[SimpleNamespace(key="s0")],
            transitioned_states=[],
            next_states=[],
            rewards=[0.0],
            dones=[False],
            chosen_actions=[(1,)],
            terminal_reasons=["continue"],
            action_candidates=[((1,),)],
            action_index_tensor=torch.tensor([0], dtype=torch.long),
            actions_tensor=torch.tensor([[1]], dtype=torch.long),
            log_prob_tensor=torch.tensor([0.1], dtype=torch.float),
            entropy_tensor=torch.tensor([0.2], dtype=torch.float),
            value_tensor=torch.tensor([0.3], dtype=torch.float),
            valid_action_mask=torch.tensor([True]),
            reset_count=0,
            frt_hits=0,
            collapsed_hits=0,
            dead_end_hits=0,
            expanded_states=0,
            discovered_states=0,
            used_multiprocessing=False,
            candidate_expand_sec=0.0,
            policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0,
            policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0,
            transition_apply_sec=0.0,
            intrinsic_bonus=[2.5],
            training_rewards=[2.5],
        )
    )

    prepared = buffer.prepare(
        bootstrap_value=torch.tensor([0.0], dtype=torch.float),
        gamma=0.9,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )

    assert prepared.reward_buffer_tensor.tolist() == [[2.5]]


def test_ppo_rollout_buffer_prepare_preserves_rollout_data_list():
    data_a = Data(
        x=torch.zeros((2, 3), dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        subcomplex_vertices=torch.tensor([[0, 1, -1]], dtype=torch.long),
        num_available_subcomplexes=1,
    )
    data_b = Data(
        x=torch.ones((2, 3), dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        subcomplex_vertices=torch.empty((0, 3), dtype=torch.long),
        num_available_subcomplexes=0,
    )

    buffer = PPORolloutBuffer()
    buffer.append(
        PolicyRolloutStepResult(
            input_states=[SimpleNamespace(key="s0"), SimpleNamespace(key="s1")],
            transitioned_states=[],
            next_states=[],
            rewards=[1.0, 0.0],
            dones=[False, True],
            chosen_actions=[(0, 1), None],
            terminal_reasons=["continue", "dead_end_current"],
            action_candidates=[((0, 1),), tuple()],
            action_index_tensor=torch.tensor([0, -1], dtype=torch.long),
            actions_tensor=torch.tensor([[0, 1, -1], [-1, -1, -1]], dtype=torch.long),
            log_prob_tensor=torch.tensor([0.1, 0.0], dtype=torch.float),
            entropy_tensor=torch.tensor([0.2, 0.0], dtype=torch.float),
            value_tensor=torch.tensor([0.3, 0.4], dtype=torch.float),
            valid_action_mask=torch.tensor([True, False]),
            reset_count=1,
            frt_hits=0,
            collapsed_hits=0,
            dead_end_hits=1,
            expanded_states=0,
            discovered_states=0,
            used_multiprocessing=False,
            candidate_expand_sec=0.0,
            policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0,
            policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0,
            transition_apply_sec=0.0,
            data_list=[data_a, data_b],
        )
    )

    prepared = buffer.prepare(
        bootstrap_value=torch.tensor([0.0, 0.0], dtype=torch.float),
        gamma=0.9,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )

    assert prepared.data_buffer_list is not None
    assert prepared.data_buffer_list[0] is data_a
    assert prepared.data_buffer_list[1] is data_b
