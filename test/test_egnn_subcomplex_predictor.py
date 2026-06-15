import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pytest
from torch_geometric.data import Batch, Data

from models.egnn import EGNN
from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent, EGNNSubcomplexPredictor
from models.gcn_subcomplex_predictor import GCNSubcomplexAgent
from models.subcomplex_policy_factory import build_subcomplex_agent


def _ring_edges(num_nodes: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src.extend([i, j])
        dst.extend([j, i])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_graph(node_coords, subcomplex_vertices, simplex_vertices=None):
    x = torch.tensor(node_coords, dtype=torch.float)
    subcomplex = torch.tensor(subcomplex_vertices, dtype=torch.long)
    data = Data(
        x=x,
        edge_index=_ring_edges(x.size(0)),
        subcomplex_vertices=subcomplex,
        num_available_subcomplexes=subcomplex.size(0),
    )
    if simplex_vertices is not None:
        simplex_tensor = torch.tensor(simplex_vertices, dtype=torch.long)
        data.simplex_vertices = simplex_tensor
        data.num_top_simplices = simplex_tensor.size(0)
    return data


def _build_batch():
    graph_1 = _build_graph(
        node_coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        subcomplex_vertices=[
            [0, 1, -1],
            [1, 2, -1],
            [0, 2, 3],
        ],
    )
    graph_2 = _build_graph(
        node_coords=[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
        ],
        subcomplex_vertices=[
            [0, 3, 4],
            [1, 2, -1],
        ],
    )
    return Batch.from_data_list([graph_1, graph_2])


def _build_snn_batch():
    graph_1 = _build_graph(
        node_coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        subcomplex_vertices=[
            [0, 1, -1],
            [1, 2, -1],
            [0, 2, 3],
        ],
        simplex_vertices=[
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [0, 2],
        ],
    )
    graph_2 = _build_graph(
        node_coords=[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
        ],
        subcomplex_vertices=[
            [0, 3, 4],
            [1, 2, -1],
        ],
        simplex_vertices=[
            [0, 3],
            [3, 4],
            [1, 2],
            [2, 4],
        ],
    )
    return Batch.from_data_list([graph_1, graph_2])


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


def _build_predictor():
    torch.manual_seed(0)
    return EGNNSubcomplexPredictor(
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


def _build_gcn_agent():
    torch.manual_seed(0)
    return GCNSubcomplexAgent(
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


def _build_gnn_agent():
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
        subcomplex_actor_type="gnn",
        device="cpu",
    ).eval()


def _build_gcn_gnn_agent():
    torch.manual_seed(0)
    return GCNSubcomplexAgent(
        in_channels=3,
        out_channels=16,
        hidden_channels=16,
        num_layers=2,
        share_encoder=True,
        mlp_hidden_channel_list=[16],
        use_projection=True,
        act="silu",
        subcomplex_actor_type="gnn",
        device="cpu",
    ).eval()


def _build_circuit_pool_agent():
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
        subcomplex_actor_type="circuit_pool",
        device="cpu",
    ).eval()


def _build_snn_simplex_agent():
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
        subcomplex_actor_type="snn_simplex",
        device="cpu",
    ).eval()


def _build_gcn_snn_simplex_agent():
    torch.manual_seed(0)
    return GCNSubcomplexAgent(
        in_channels=3,
        out_channels=16,
        hidden_channels=16,
        num_layers=2,
        share_encoder=True,
        mlp_hidden_channel_list=[16],
        use_projection=True,
        act="silu",
        subcomplex_actor_type="snn_simplex",
        device="cpu",
    ).eval()


def _loop_decode_and_pool_egnn(
    model,
    node_embeddings,
    node_coord,
    subcomplex_vertices,
    num_available_subcomplexes,
    node_ptr,
):
    candidate_graph_index = torch.repeat_interleave(
        torch.arange(num_available_subcomplexes.numel(), dtype=torch.long),
        num_available_subcomplexes,
    )
    node_offsets = node_ptr[:-1].to(dtype=torch.long)
    pooled_inputs = []
    pooled_batch = []
    for candidate_idx, vertices in enumerate(subcomplex_vertices):
        valid_vertices = vertices[vertices >= 0]
        graph_idx = candidate_graph_index[candidate_idx]
        global_vertices = valid_vertices + node_offsets[graph_idx]
        sub_h = node_embeddings[global_vertices]
        sub_x = node_coord[global_vertices]
        sub_edges = model._build_complete_subcomplex_edges(
            num_nodes=valid_vertices.numel(),
            device=node_embeddings.device,
        )
        sub_h, _ = model.subcomplex_decoder(
            h=sub_h,
            x=sub_x,
            edges=sub_edges,
            edge_attr=None,
        )
        pooled_inputs.append(sub_h)
        pooled_batch.append(
            torch.full(
                (valid_vertices.numel(),),
                candidate_idx,
                device=node_embeddings.device,
                dtype=torch.long,
            )
        )

    return (
        gnn.pool.global_max_pool(
            torch.cat(pooled_inputs, dim=0),
            torch.cat(pooled_batch, dim=0),
            size=subcomplex_vertices.size(0),
        ),
        candidate_graph_index,
    )


def _loop_decode_and_pool_gcn(
    model,
    node_embeddings,
    subcomplex_vertices,
    num_available_subcomplexes,
    node_ptr,
):
    candidate_graph_index = torch.repeat_interleave(
        torch.arange(num_available_subcomplexes.numel(), dtype=torch.long),
        num_available_subcomplexes,
    )
    node_offsets = node_ptr[:-1].to(dtype=torch.long)
    pooled_inputs = []
    pooled_batch = []
    for candidate_idx, vertices in enumerate(subcomplex_vertices):
        valid_vertices = vertices[vertices >= 0]
        graph_idx = candidate_graph_index[candidate_idx]
        global_vertices = valid_vertices + node_offsets[graph_idx]
        sub_h = node_embeddings[global_vertices]
        sub_edges = model._build_complete_subcomplex_edges(
            num_nodes=valid_vertices.numel(),
            device=node_embeddings.device,
        )
        sub_h = model._decode_subcomplex(sub_h, sub_edges)
        pooled_inputs.append(sub_h)
        pooled_batch.append(
            torch.full(
                (valid_vertices.numel(),),
                candidate_idx,
                device=node_embeddings.device,
                dtype=torch.long,
            )
        )

    return (
        gnn.pool.global_max_pool(
            torch.cat(pooled_inputs, dim=0),
            torch.cat(pooled_batch, dim=0),
            size=subcomplex_vertices.size(0),
        ),
        candidate_graph_index,
    )


def test_subcomplex_logits_shape_and_padding():
    model = _build_agent()
    batch = _build_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_gnn_subcomplex_logits_shape_and_padding():
    model = _build_gnn_agent()
    batch = _build_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_circuit_pool_subcomplex_logits_shape_and_padding():
    model = _build_circuit_pool_agent()
    batch = _build_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_snn_simplex_subcomplex_logits_shape_and_padding():
    model = _build_snn_simplex_agent()
    batch = _build_snn_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_gcn_snn_simplex_subcomplex_logits_shape_and_padding():
    model = _build_gcn_snn_simplex_agent()
    batch = _build_snn_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_gcn_subcomplex_logits_shape_and_padding():
    model = _build_gcn_agent()
    batch = _build_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_gcn_gnn_subcomplex_logits_shape_and_padding():
    model = _build_gcn_gnn_agent()
    batch = _build_batch()

    with torch.no_grad():
        value, logits = model.get_value_and_logits(batch)

    assert tuple(value.shape) == (2,)
    assert tuple(logits.shape) == (2, 3)
    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isfinite(logits[1, :2]).all()
    assert torch.isneginf(logits[1, 2])


def test_egnn_gnn_vectorized_decode_matches_loop_reference():
    model = _build_gnn_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, _, z_remove_raw, _ = model.encode_projection(
            h=batch.x,
            x=batch.x,
            edges=batch.edge_index,
            edge_attr=None,
            return_z_before_proj=True,
        )
        subcomplex_vertices, num_available_subcomplexes = model._extract_batched_subcomplex_data(
            batch=batch,
            device=z_remove_raw.device,
        )
        vectorized_features, vectorized_graph_index = model._decode_and_pool_batched_subcomplex_embeddings(
            node_embeddings=z_remove_raw,
            node_coord=batch.x,
            subcomplex_vertices=subcomplex_vertices,
            num_available_subcomplexes=num_available_subcomplexes,
            node_ptr=batch.ptr,
        )
        loop_features, loop_graph_index = _loop_decode_and_pool_egnn(
            model,
            node_embeddings=z_remove_raw,
            node_coord=batch.x,
            subcomplex_vertices=subcomplex_vertices,
            num_available_subcomplexes=num_available_subcomplexes,
            node_ptr=batch.ptr,
        )

    assert torch.equal(vectorized_graph_index, loop_graph_index)
    assert torch.allclose(vectorized_features, loop_features, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        model.subcomplex_decoder_head(vectorized_features),
        model.subcomplex_decoder_head(loop_features),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gcn_gnn_vectorized_decode_matches_loop_reference():
    model = _build_gcn_gnn_agent()
    batch = _build_batch()

    with torch.no_grad():
        z_raw = model.encode(batch.x, batch.edge_index)
        subcomplex_vertices, num_available_subcomplexes = model._extract_batched_subcomplex_data(
            batch=batch,
            device=z_raw.device,
        )
        vectorized_features, vectorized_graph_index = model._decode_and_pool_batched_subcomplex_embeddings(
            node_embeddings=z_raw,
            subcomplex_vertices=subcomplex_vertices,
            num_available_subcomplexes=num_available_subcomplexes,
            node_ptr=batch.ptr,
        )
        loop_features, loop_graph_index = _loop_decode_and_pool_gcn(
            model,
            node_embeddings=z_raw,
            subcomplex_vertices=subcomplex_vertices,
            num_available_subcomplexes=num_available_subcomplexes,
            node_ptr=batch.ptr,
        )

    assert torch.equal(vectorized_graph_index, loop_graph_index)
    assert torch.allclose(vectorized_features, loop_features, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        model.subcomplex_decoder_head(vectorized_features),
        model.subcomplex_decoder_head(loop_features),
        atol=1e-5,
        rtol=1e-5,
    )


def test_forward_deterministic_selects_argmax_subcomplex():
    model = _build_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        actions, action_indices, _, _, _ = model(batch, deterministic=True)
    argmax_indices = torch.argmax(logits, dim=1)
    num_available = batch.num_available_subcomplexes.to(dtype=torch.long)
    candidate_start = torch.cumsum(num_available, dim=0) - num_available

    for graph_idx, action_idx in enumerate(argmax_indices.tolist()):
        expected_action = batch.subcomplex_vertices[candidate_start[graph_idx] + action_idx]
        assert torch.equal(actions[graph_idx], expected_action)
        assert int(action_indices[graph_idx].item()) == action_idx


def test_circuit_pool_forward_deterministic_selects_argmax_subcomplex():
    model = _build_circuit_pool_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        actions, action_indices, _, _, _ = model(batch, deterministic=True)
    argmax_indices = torch.argmax(logits, dim=1)
    num_available = batch.num_available_subcomplexes.to(dtype=torch.long)
    candidate_start = torch.cumsum(num_available, dim=0) - num_available

    for graph_idx, action_idx in enumerate(argmax_indices.tolist()):
        expected_action = batch.subcomplex_vertices[candidate_start[graph_idx] + action_idx]
        assert torch.equal(actions[graph_idx], expected_action)
        assert int(action_indices[graph_idx].item()) == action_idx


def test_gcn_forward_deterministic_selects_argmax_subcomplex():
    model = _build_gcn_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        actions, action_indices, _, _, _ = model(batch, deterministic=True)
    argmax_indices = torch.argmax(logits, dim=1)
    num_available = batch.num_available_subcomplexes.to(dtype=torch.long)
    candidate_start = torch.cumsum(num_available, dim=0) - num_available

    for graph_idx, action_idx in enumerate(argmax_indices.tolist()):
        expected_action = batch.subcomplex_vertices[candidate_start[graph_idx] + action_idx]
        assert torch.equal(actions[graph_idx], expected_action)
        assert int(action_indices[graph_idx].item()) == action_idx


def test_snn_simplex_forward_deterministic_selects_argmax_subcomplex():
    model = _build_snn_simplex_agent()
    batch = _build_snn_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        actions, action_indices, _, _, _ = model(batch, deterministic=True)
    argmax_indices = torch.argmax(logits, dim=1)
    num_available = batch.num_available_subcomplexes.to(dtype=torch.long)
    candidate_start = torch.cumsum(num_available, dim=0) - num_available

    for graph_idx, action_idx in enumerate(argmax_indices.tolist()):
        expected_action = batch.subcomplex_vertices[candidate_start[graph_idx] + action_idx]
        assert torch.equal(actions[graph_idx], expected_action)
        assert int(action_indices[graph_idx].item()) == action_idx


def test_get_log_prob_matches_distribution_for_selected_indices():
    model = _build_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        chosen_indices = torch.argmax(probs, dim=1)
        log_prob, entropy = model.get_log_prob(logits, chosen_indices)

    assert torch.allclose(log_prob, dist.log_prob(chosen_indices), atol=1e-6)
    assert torch.allclose(entropy, dist.entropy(), atol=1e-6)


def test_circuit_pool_get_log_prob_matches_distribution_for_selected_indices():
    model = _build_circuit_pool_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        chosen_indices = torch.argmax(probs, dim=1)
        log_prob, entropy = model.get_log_prob(logits, chosen_indices)

    assert torch.allclose(log_prob, dist.log_prob(chosen_indices), atol=1e-6)
    assert torch.allclose(entropy, dist.entropy(), atol=1e-6)


def test_gcn_get_log_prob_matches_distribution_for_selected_indices():
    model = _build_gcn_agent()
    batch = _build_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        chosen_indices = torch.argmax(probs, dim=1)
        log_prob, entropy = model.get_log_prob(logits, chosen_indices)

    assert torch.allclose(log_prob, dist.log_prob(chosen_indices), atol=1e-6)
    assert torch.allclose(entropy, dist.entropy(), atol=1e-6)


def test_snn_simplex_get_log_prob_matches_distribution_for_selected_indices():
    model = _build_snn_simplex_agent()
    batch = _build_snn_batch()

    with torch.no_grad():
        _, logits = model.get_value_and_logits(batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        chosen_indices = torch.argmax(probs, dim=1)
        log_prob, entropy = model.get_log_prob(logits, chosen_indices)

    assert torch.allclose(log_prob, dist.log_prob(chosen_indices), atol=1e-6)
    assert torch.allclose(entropy, dist.entropy(), atol=1e-6)


def test_snn_simplex_requires_simplex_topology_fields():
    model = _build_snn_simplex_agent()
    batch = _build_batch()

    with pytest.raises(AttributeError, match="simplex_vertices"):
        model.get_value_and_logits(batch)


def test_gcn_subcomplex_agent_contains_no_egnn_modules():
    model = _build_gcn_agent()

    assert not any(isinstance(module, EGNN) for module in model.modules())


def test_subcomplex_policy_factory_builds_egnn_and_gcn_agents():
    egnn_model = build_subcomplex_agent(
        model_type="egnn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        device="cpu",
    )
    circuit_pool_model = build_subcomplex_agent(
        model_type="egnn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        subcomplex_actor_type="circuit_pool",
        device="cpu",
    )
    gnn_model = build_subcomplex_agent(
        model_type="egnn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        subcomplex_actor_type="gnn",
        device="cpu",
    )
    gcn_model = build_subcomplex_agent(
        model_type="gcn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        device="cpu",
    )
    gcn_gnn_model = build_subcomplex_agent(
        model_type="gcn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        subcomplex_actor_type="gnn",
        device="cpu",
    )
    snn_simplex_model = build_subcomplex_agent(
        model_type="egnn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        subcomplex_actor_type="snn_simplex",
        device="cpu",
    )

    assert isinstance(egnn_model, EGNNSubcomplexAgent)
    assert isinstance(circuit_pool_model, EGNNSubcomplexAgent)
    assert circuit_pool_model.subcomplex_actor_type == "circuit_pool"
    assert isinstance(gnn_model, EGNNSubcomplexAgent)
    assert egnn_model.subcomplex_actor_type == "gnn"
    assert gnn_model.subcomplex_actor_type == "gnn"
    assert isinstance(gcn_model, GCNSubcomplexAgent)
    assert gcn_model.subcomplex_actor_type == "gnn"
    assert isinstance(gcn_gnn_model, GCNSubcomplexAgent)
    assert gcn_gnn_model.subcomplex_actor_type == "gnn"
    assert isinstance(snn_simplex_model, EGNNSubcomplexAgent)
    assert snn_simplex_model.subcomplex_actor_type == "snn_simplex"


def test_subcomplex_policy_factory_normalizes_default_actor_alias():
    model = build_subcomplex_agent(
        model_type="gcn",
        in_channels=3,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
        mlp_hidden_channel_list=[8],
        subcomplex_actor_type="default",
        device="cpu",
    )

    assert model.subcomplex_actor_type == "gnn"


def test_predictor_decode_shape_and_sigmoid_consistency():
    model = _build_predictor()
    z = torch.randn(5, 16)
    subcomplex_vertices = torch.tensor(
        [
            [0, 1, -1],
            [1, 2, 3],
            [0, 4, -1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        probs = model.decode(z, subcomplex_vertices, sigmoid=True)
        logits = model.decode(z, subcomplex_vertices, sigmoid=False)

    assert tuple(probs.shape) == (3,)
    assert tuple(logits.shape) == (3,)
    assert torch.allclose(probs, torch.sigmoid(logits), atol=1e-6)


def test_predictor_decode_invariant_to_vertex_order():
    model = _build_predictor()
    z = torch.randn(5, 16)
    subcomplex_a = torch.tensor([[0, 2, 4]], dtype=torch.long)
    subcomplex_b = torch.tensor([[4, 0, 2]], dtype=torch.long)

    with torch.no_grad():
        logit_a = model.decode(z, subcomplex_a, sigmoid=False)
        logit_b = model.decode(z, subcomplex_b, sigmoid=False)

    assert torch.allclose(logit_a, logit_b, atol=1e-6)


def test_predictor_recon_loss_matches_manual_bce():
    model = _build_predictor()
    z = torch.randn(6, 16)
    pos_subcomplex = torch.tensor([[0, 1, -1], [2, 3, 4]], dtype=torch.long)
    neg_subcomplex = torch.tensor([[1, 5, -1], [0, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        loss = model.recon_loss(z, pos_subcomplex, neg_subcomplex)
        pos_pred = model.decode(z, pos_subcomplex, sigmoid=True)
        neg_pred = model.decode(z, neg_subcomplex, sigmoid=True)
        expected = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()

    assert torch.allclose(loss, expected, atol=1e-6)


def test_predictor_test_returns_valid_metrics():
    model = _build_predictor()
    z = torch.randn(6, 16)
    pos_subcomplex = torch.tensor([[0, 1, -1], [2, 3, 4]], dtype=torch.long)
    neg_subcomplex = torch.tensor([[1, 5, -1], [0, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        auc, ap = model.test(z, pos_subcomplex, neg_subcomplex)

    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ap <= 1.0
