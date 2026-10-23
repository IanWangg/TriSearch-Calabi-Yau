"""Completion and permutation invariance, batching, and learnable heads."""

from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Batch

from core.cy_two_face_data import create_two_face_data_from_state
from models.subcomplex_policy_factory import build_subcomplex_agent
from test_cy_two_face_data import square_state, square_actions, real_representatives


def two_face_policy(dimension=3):
    torch.manual_seed(17)
    return build_subcomplex_agent(model_type="egnn", subcomplex_actor_type="two_face_deep_sets",
        in_channels=dimension, hidden_channels=8, out_channels=8, num_layers=2).eval()


def test_batch_independence_action_subsets_and_empty_forward():
    state, small = square_state(), square_state(one_face=True)
    data = create_two_face_data_from_state(state, square_actions(state))
    subset = create_two_face_data_from_state(state, square_actions(state)[:1])
    empty = create_two_face_data_from_state(state)
    smaller = create_two_face_data_from_state(small, square_actions(small))
    model = two_face_policy()
    single_value, single_logits = model.get_value_and_logits(Batch.from_data_list([data]))
    batch = Batch.from_data_list([data, empty, smaller, subset])
    values, logits = model.get_value_and_logits(batch)
    torch.testing.assert_close(values[[0, 1, 3]], single_value.expand(3))
    torch.testing.assert_close(logits[0], single_logits[0])
    torch.testing.assert_close(logits[3, :1], single_logits[0, :1])
    assert torch.isneginf(logits[1]).all() and torch.isneginf(logits[2, 1:]).all()
    selected, indices, forward_values, log_prob, entropy = model(batch)
    assert selected.shape == (4, 4) and selected[1].tolist() == [-1] * 4
    assert indices[1] == -1 and log_prob[1] == entropy[1] == 0
    torch.testing.assert_close(forward_values, values)
    no_actions = Batch.from_data_list([empty, empty])
    assert model.get_value_and_logits(no_actions)[1].shape == (2, 0)
    assert model(no_actions)[1].tolist() == [-1, -1]
    # Critic does not require any candidate fields or invoke the actor.
    del no_actions.action_face, no_actions.removed_triangle_ids, no_actions.added_triangle_vertices
    del no_actions.subcomplex_vertices, no_actions.num_available_subcomplexes
    torch.testing.assert_close(model.get_value(no_actions), single_value.expand(2))


def test_node_face_triangle_and_action_permutations():
    state = square_state()
    original = create_two_face_data_from_state(state, square_actions(state))
    model = two_face_policy()
    expected_value, expected_logits = model.get_value_and_logits(Batch.from_data_list([original]))
    data = original.clone()
    node_order = torch.tensor([6, 3, 1, 5, 0, 7, 4, 2])
    inverse_node = torch.argsort(node_order)
    data.x, data.node_face = data.x[node_order], data.node_face[node_order]
    for field in ("edge_index", "triangle_vertices", "added_triangle_vertices"):
        data[field] = inverse_node[data[field]]
    for field in ("node_face", "triangle_face", "action_face"):
        data[field] = 1 - data[field]
    triangle_order = torch.tensor([3, 0, 2, 1])
    inverse_triangle = torch.argsort(triangle_order)
    data.triangle_vertices = data.triangle_vertices[triangle_order].flip(-1)
    data.triangle_face = data.triangle_face[triangle_order]
    data.triangle_edge_index = inverse_triangle[data.triangle_edge_index]
    data.removed_triangle_ids = inverse_triangle[data.removed_triangle_ids].flip(-1)
    data.added_triangle_vertices = data.added_triangle_vertices.flip((1, 2))
    value, logits = model.get_value_and_logits(Batch.from_data_list([data]))
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(logits, expected_logits)
    reversed_data = create_two_face_data_from_state(state, list(reversed(square_actions(state))))
    value, logits = model.get_value_and_logits(Batch.from_data_list([reversed_data]))
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(logits, expected_logits.flip(1))


def test_backwards_reaches_shared_encoder_and_both_heads():
    state = square_state()
    model = two_face_policy().train()
    batch = Batch.from_data_list([create_two_face_data_from_state(state, square_actions(state))])
    value, logits = model.get_value_and_logits(batch)
    (value.square().sum() + logits.square().sum()).backward()
    for module in (model.point_egnn, model.triangle_mlp, model.triangle_dual, model.face_mlp,
                   model.rho, model.actor_mlp, model.value_mlp):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum()) for grad in grads) > 0


def test_real_completion_values_and_aligned_logits(real_representatives):
    from mdp.cy_geometry_worker import execute_geometry_request
    _, first, second = real_representatives
    actions = []
    for state in (first, second):
        expansion = execute_geometry_request(dict(operation="expand", configuration=state.configuration,
            state=state.to_payload(), objective_mode=True, action_order="canonical"))
        actions.append(expansion.candidate_actions)
    assert set(actions[0]) == set(actions[1])
    actions[1] = tuple(reversed(actions[1]))
    batch = Batch.from_data_list([create_two_face_data_from_state(state, candidate)
                                 for state, candidate in zip((first, second), actions)])
    model = two_face_policy(4)
    value, logits = model.get_value_and_logits(batch)
    torch.testing.assert_close(value[0], value[1])
    aligned = torch.tensor([actions[1].index(action) for action in actions[0]])
    torch.testing.assert_close(logits[0], logits[1, aligned])
    torch.testing.assert_close(logits[0].softmax(0), logits[1, aligned].softmax(0))


def test_reject_unsupported_model_combinations():
    with pytest.raises(ValueError, match="model_type"):
        build_subcomplex_agent(model_type="gcn", subcomplex_actor_type="two_face_deep_sets",
            in_channels=3, hidden_channels=8, out_channels=8, num_layers=1)
