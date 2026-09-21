from __future__ import annotations

import copy
import gc
from itertools import combinations
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch

from core.cy_data_utils import configure_cy_data_tensor_caches, get_cy_data_tensor_cache_stats
from core.cy_policy_inference import (
    batched_policy_action_selection,
    build_cy_data_list,
    configure_policy_execution,
    evaluate_policy_actions_from_data_list,
    policy_data_chunks,
)
from core.cy_ppo import PPORolloutBuffer, train_policy_from_rollout
from core.snn_simplex_topology import _build_candidate_simplex_memberships
from core.vertex_augmentation import SimilarityTransform
from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent
from models.gcn_subcomplex_predictor import GCNSubcomplexAgent


class GeometryState:
    def __init__(self, key="state"):
        self.key = key
        self.point_config_index = 703
        self.vertices = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
        self.simplices = ((0, 1, 2), (0, 2, 3))
        self.edges = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))


@pytest.fixture(autouse=True)
def _reset_limits():
    torch.set_default_device("cpu")
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()
    configure_policy_execution()
    yield
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()
    configure_policy_execution()


def _policy(model_type=EGNNSubcomplexAgent, actor="snn_simplex"):
    torch.manual_seed(23)
    return model_type(
        in_channels=3, out_channels=8, hidden_channels=8, num_layers=1,
        mlp_hidden_channel_list=[8], subcomplex_actor_type=actor, device="cpu",
    ).eval()


def _step(states, selection):
    return SimpleNamespace(
        input_states=states, action_candidates=selection.action_lists, data_list=selection.data_list,
        actions_tensor=selection.actions_tensor, action_index_tensor=selection.action_index_tensor,
        log_prob_tensor=selection.log_prob_tensor, entropy_tensor=selection.entropy_tensor,
        value_tensor=selection.value_tensor, valid_action_mask=selection.valid_action_mask,
        training_rewards=None, rewards=[float(index % 3 - 1) for index in range(len(states))],
        dones=[False] * len(states),
    )


def test_cpu_observation_buffer_releases_source_geometry_and_survives_eviction():
    state = GeometryState()
    reference = weakref.ref(state)
    selection = batched_policy_action_selection([state], [[(0, 1, 2)]], _policy(), device=torch.device("cpu"))
    data = selection.data_list[0]
    expected_x = data.x.clone()
    buffer = PPORolloutBuffer()
    step = _step([state], selection)
    buffer.append(step)
    del state, step, selection
    gc.collect()
    assert reference() is None
    assert buffer.state_buffer == [[None]]
    assert buffer.candidate_buffer == [[()]]
    configure_cy_data_tensor_caches(max_bytes=0)
    prepared = buffer.prepare(bootstrap_value=torch.zeros(1), gamma=0.9, gae_lambda=0.95, device=torch.device("cpu"))
    torch.testing.assert_close(prepared.data_buffer_list[0].x, expected_x)
    data_ref = weakref.ref(data)
    del data, prepared
    buffer.clear()
    gc.collect()
    assert data_ref() is None
    with pytest.raises(ValueError, match="empty rollout"):
        buffer.prepare(bootstrap_value=torch.zeros(1), gamma=0.9, gae_lambda=0.95, device=torch.device("cpu"))


def test_coordinates_and_trajectory_transform_share_storage_across_states():
    states = [GeometryState("one"), GeometryState("two")]
    plain = build_cy_data_list(states, [[(0, 1, 2)], [(0, 2, 3)]])
    assert plain[0].x.data_ptr() == plain[1].x.data_ptr()
    transform = SimilarityTransform(matrix=2 * torch.eye(3), bias=torch.ones(1, 3))
    first = build_cy_data_list(states[:1], [[(0, 1, 2)]], trajectory_transforms=[transform])
    second = build_cy_data_list(states[1:], [[(0, 2, 3)]], trajectory_transforms=[transform])
    assert first[0].x.data_ptr() == second[0].x.data_ptr()
    torch.testing.assert_close(first[0].x, 2 * plain[0].x + 1)


@pytest.mark.parametrize("managed", [False, True])
def test_tensor_cache_separates_identical_state_keys_with_different_points(managed):
    from mdp.cy_state_record import CYPointConfiguration, CyStateRecord

    first = GeometryState("same_key")
    second = GeometryState("same_key")
    second.vertices = tuple(tuple(2 * coordinate + 3 for coordinate in point) for point in first.vertices)
    if managed:
        def record(state):
            configuration = CYPointConfiguration(
                state.point_config_index, state.vertices, state.vertices,
                tuple(range(len(state.vertices))), True,
            )
            return CyStateRecord(configuration, frozenset(state.simplices))
        first, second = record(first), record(second)
    assert first.key == second.key
    assert first.point_config_index == second.point_config_index
    assert first.simplices == second.simplices
    first_data = build_cy_data_list([first], [[(0, 1, 2)]], include_simplex_topology=True)[0]
    second_data = build_cy_data_list([second], [[(0, 1, 2)]], include_simplex_topology=True)[0]
    torch.testing.assert_close(first_data.x, torch.tensor(first.vertices, dtype=torch.float))
    torch.testing.assert_close(second_data.x, torch.tensor(second.vertices, dtype=torch.float))
    assert first_data.x.data_ptr() != second_data.x.data_ptr()
    # Revisiting the first configuration must still find its own observation.
    revisited = build_cy_data_list([first], [[(0, 1, 2)]], include_simplex_topology=True)[0]
    assert revisited.x.data_ptr() == first_data.x.data_ptr()
    assert get_cy_data_tensor_cache_stats()["graph"]["entries"] == 2


def test_tensor_cache_admission_is_byte_bounded():
    configure_cy_data_tensor_caches(max_bytes=12_000, max_entries=2)
    retained = None
    for index in range(15):
        retained = build_cy_data_list([GeometryState(str(index))], [[(0, 1, 2)]], include_simplex_topology=True)
        stats = get_cy_data_tensor_cache_stats()
        assert sum(cache["bytes"] for cache in stats.values()) <= 12_000
        assert all(cache["entries"] <= 2 for cache in stats.values())
    assert sum(cache["evictions"] + cache["bypasses"] for cache in stats.values()) > 0
    assert retained[0].simplex_vertices.tolist() == [[0, 1, 2], [0, 2, 3]]


@pytest.mark.parametrize("model_type", [EGNNSubcomplexAgent, GCNSubcomplexAgent])
@pytest.mark.parametrize("actor", ["gnn", "mlp", "circuit_pool", "snn_simplex"])
def test_empty_actions_and_mixed_joint_pass_preserve_critic(model_type, actor):
    model = _policy(model_type, actor)
    states = [GeometryState("first"), GeometryState("second")]
    actions = [[(0, 1, 2), (0, 2, 3)], []]
    data = build_cy_data_list(states, actions, include_simplex_topology=actor == "snn_simplex")
    result = evaluate_policy_actions_from_data_list(data, torch.tensor([1, -1]), model, device=torch.device("cpu"))
    single_values = torch.cat([model.get_value(Batch.from_data_list([entry])) for entry in data])
    torch.testing.assert_close(result.value_tensor, single_values, atol=1e-6, rtol=1e-6)
    assert result.valid_action_mask.tolist() == [True, False]
    empty = batched_policy_action_selection(states, [[], []], model, device=torch.device("cpu"))
    assert empty.action_index_tensor.tolist() == [-1, -1]
    assert torch.isfinite(empty.value_tensor).all()
    assert empty.log_prob_tensor.tolist() == [0., 0.]


def test_snn_slow_and_cached_paths_compute_terminal_simplex_critic():
    model = _policy()
    data = build_cy_data_list([GeometryState()], [[]], include_simplex_topology=True)
    expected = model.get_value(Batch.from_data_list(data))
    for name in model.snn_simplex_actor._CACHED_TOPOLOGY_ATTRS:
        del data[0][name]
    actual = model.get_value(Batch.from_data_list(data))
    torch.testing.assert_close(actual, expected)


def test_sampling_rng_and_order_are_independent_of_physical_chunks():
    model = _policy()
    states = [GeometryState(str(index)) for index in range(5)]
    actions = [[(0, 1, 2), (0, 2, 3)], [], [(0, 1, 2)], [], [(0, 1, 2), (0, 2, 3)]]
    torch.manual_seed(83)
    full = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    full_rng = torch.get_rng_state()
    configure_policy_execution(max_graph_size=1)
    torch.manual_seed(83)
    chunked = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    assert torch.equal(full_rng, torch.get_rng_state())
    assert full.action_index_tensor.tolist() == chunked.action_index_tensor.tolist()
    torch.testing.assert_close(full.value_tensor, chunked.value_tensor, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full.log_prob_tensor, chunked.log_prob_tensor, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("actor", ["gnn", "circuit_pool", "snn_simplex"])
@pytest.mark.parametrize("model_type", [EGNNSubcomplexAgent, GCNSubcomplexAgent])
def test_ppo_physical_chunks_preserve_logical_update_and_gradients(actor, model_type):
    model = _policy(model_type, actor=actor)
    chunked_model = copy.deepcopy(model)
    states = [GeometryState(str(index)) for index in range(4)]
    actions = [[(0, 1, 2), (0, 2, 3)], [], [(0, 1, 2)], [(0, 1, 2), (0, 2, 3)]]
    selection = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    buffer = PPORolloutBuffer()
    buffer.append(_step(states, selection))
    prepared = buffer.prepare(bootstrap_value=torch.tensor([.1, .2, -.1, .4]), gamma=.9, gae_lambda=.95, device=torch.device("cpu"))
    kwargs = dict(prepared_rollout=prepared, device=torch.device("cpu"), num_epochs=2, batch_size=4,
                  clip_coef=.2, value_coef=.5, entropy_coef=.01, max_grad_norm=.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    chunked_optimizer = torch.optim.SGD(chunked_model.parameters(), lr=.05)
    torch.manual_seed(89)
    full = train_policy_from_rollout(policy=model, optimizer=optimizer, max_graph_size=10**9, **kwargs)
    torch.manual_seed(89)
    chunked = train_policy_from_rollout(policy=chunked_model, optimizer=chunked_optimizer, max_graph_size=1, **kwargs)
    for parameter, chunked_parameter in zip(model.parameters(), chunked_model.parameters()):
        torch.testing.assert_close(parameter, chunked_parameter, atol=2e-6, rtol=2e-6)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, chunked_parameter.grad, atol=2e-6, rtol=2e-6)
    assert chunked.total_loss == pytest.approx(full.total_loss, abs=2e-6)
    assert chunked.policy_loss == pytest.approx(full.policy_loss, abs=2e-6)
    assert chunked.value_loss == pytest.approx(full.value_loss, abs=2e-6)


def test_training_batch_norm_retains_logical_batch():
    model = _policy(actor="mlp").train()
    data = build_cy_data_list([GeometryState("a"), GeometryState("b")], [[(0, 1, 2)], [(0, 1, 2)]])
    assert list(policy_data_chunks(data, max_graph_size=1, policy=model)) == [[0, 1]]
    assert list(policy_data_chunks(data, max_graph_size=1, policy=model.eval())) == [[0], [1]]


@pytest.mark.parametrize("model_type", [EGNNSubcomplexAgent, GCNSubcomplexAgent])
def test_mixed_projected_actor_keeps_legacy_batch_norm_population(model_type):
    model = _policy(model_type, actor="mlp").train()
    reference = copy.deepcopy(model)
    states = [GeometryState("one"), GeometryState("two"), GeometryState("three")]
    states[1].vertices = tuple(tuple(3 * coordinate + 2 for coordinate in point) for point in states[1].vertices)
    data = build_cy_data_list(states, [[(0, 1, 2), (0, 2, 3)], [], [(0, 1, 2)]])
    expected_values = reference.get_value(Batch.from_data_list(data))
    _, expected_logits = reference.get_value_and_logits(Batch.from_data_list([data[0], data[2]]))
    expected_log_prob, _ = reference.get_log_prob(expected_logits, torch.tensor([1, 0]))
    result = evaluate_policy_actions_from_data_list(data, torch.tensor([1, -1, 0]), model, device=torch.device("cpu"))
    torch.testing.assert_close(result.value_tensor, expected_values)
    torch.testing.assert_close(result.log_prob_tensor[[0, 2]], expected_log_prob)
    (expected_values.sum() + expected_log_prob.sum()).backward()
    (result.value_tensor.sum() + result.log_prob_tensor.sum()).backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)
    for actual, expected in zip(model.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected)


def test_trainer_tensor_modules_do_not_import_geometry():
    import os
    # Other geometry tests import sage.all, which sets this environment flag
    # and explicitly asks every child mpmath interpreter to import Sage.
    environment = os.environ.copy()
    environment.pop("MPMATH_SAGE", None)
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import core.cy_ppo; import core.cy_data_utils; "
         "assert not any(name == 'sage' or name.startswith('sage.') or name == 'cytools' or name.startswith('cytools.') for name in sys.modules)"],
        capture_output=True, text=True, timeout=60, env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_indexed_simplex_memberships_preserve_exact_order_and_coface_rule():
    simplices = list(combinations(range(8), 4))
    # Exercise indexed full simplices, lower-dimensional cofaces, duplicate
    # simplex rows, and a large candidate that takes the incidence path.
    simplices.append(simplices[3])
    candidates = [(0, 1, 2, 3, 4), (0, 2, 5), (0, 1), tuple(range(12))]
    expected_candidates, expected_simplices = [], []
    for candidate_id, vertices in enumerate(candidates):
        contained = [index for index, simplex in enumerate(simplices) if set(simplex).issubset(vertices)]
        if not contained and len(vertices) < 4:
            contained = [index for index, simplex in enumerate(simplices) if len(set(simplex).intersection(vertices)) >= max(1, len(vertices) - 1)]
        expected_candidates.extend([candidate_id] * len(contained))
        expected_simplices.extend(contained)
    padded = torch.full((len(candidates), 12), -1, dtype=torch.long)
    for row, vertices in enumerate(candidates):
        padded[row, :len(vertices)] = torch.tensor(vertices)
    actual_candidates, actual_simplices = _build_candidate_simplex_memberships(
        simplex_vertices=torch.tensor(simplices), subcomplex_vertices=padded,
    )
    assert actual_candidates.tolist() == expected_candidates
    assert actual_simplices.tolist() == expected_simplices
