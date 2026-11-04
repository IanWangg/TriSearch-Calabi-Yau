"""GL(4,Z)-invariant two-face features and InvariantTwoFaceAgent."""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from core.cy_two_face_data import build_two_face_data, TwoFaceData
from core.two_face_invariant_features import (
    InvariantTwoFaceData, NUM_POINT_FEATURES, add_invariant_fields, facet_normals)
from models.invariant_two_face_agent import InvariantTwoFaceAgent

SHARD = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/probe_20261103/stageA/shards/sup_0000.pt")


class _Config:
    """Hashable stand-in for CYPointConfiguration (needs only labels and points)."""

    def __init__(self, labels, points):
        self.labels = tuple(labels)
        self.points = tuple(points)


def _cube_points():
    return [[1 if (label >> k) & 1 else -1 for k in range(4)] for label in range(16)]


def _cube_squares():
    """The 24 squares of the 4-cube [-1,1]^4 as cyclic 4-cycles of labels 0..15."""
    points = _cube_points()
    squares = []
    for i, j in combinations(range(4), 2):
        others = [k for k in range(4) if k not in (i, j)]
        for fixed in product((-1, 1), repeat=2):
            cycle = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                v = [0, 0, 0, 0]
                v[i], v[j], v[others[0]], v[others[1]] = a, b, fixed[0], fixed[1]
                cycle.append(points.index(v))
            squares.append(tuple(cycle))
    return squares


def _restrictions(faces):
    restrictions = []
    for a, b, c, d in faces:
        restrictions.append(((a, b, c, d), (tuple(sorted((a, b, c))), tuple(sorted((a, c, d))))))
    return tuple(restrictions)


def cube_data(order_seed=None, transform=None):
    """TwoFaceData of the 4-cube. order_seed permutes point labels and restriction order only
    (same triangulation, same action list), so the policy output must be unchanged."""
    points = _cube_points()
    labels, faces = list(range(16)), _cube_squares()
    actions = tuple(faces)
    if order_seed is not None:
        rng = np.random.default_rng(order_seed)
        labels = [int(v) for v in rng.permutation(16)]
        faces = [faces[k] for k in rng.permutation(len(faces))]
    config = _Config(labels, [points[label] for label in labels])
    data = build_two_face_data(config, _restrictions(faces), actions)
    if transform is not None:
        data.x = data.x @ torch.as_tensor(transform, dtype=data.x.dtype).T
    return data


def simplex_data():
    """Reflexive 4-simplex conv(e1..e4, -(e1+..+e4)): five points, ten triangular 2-faces, no actions."""
    points = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (-1, -1, -1, -1)]
    faces = list(combinations(range(5), 3))
    config = _Config(range(5), points)
    return build_two_face_data(config, tuple((face, (tuple(sorted(face)),)) for face in faces), ())


def signed_permutation(rng):
    U = np.zeros((4, 4), dtype=np.int64)
    U[np.arange(4), rng.permutation(4)] = rng.choice([-1, 1], size=4)
    return U


def shear(i, j, s):
    U = np.eye(4, dtype=np.int64)
    U[i, j] += s
    return U


def make_model(seed=0):
    torch.manual_seed(seed)
    return InvariantTwoFaceAgent(in_channels=4, out_channels=16, hidden_channels=16, num_layers=2,
                                 mp_layers=2, two_face_encoder="deep_sets").eval()


def outputs(model, data):
    with torch.no_grad():
        value, logits = model.get_value_and_logits(Batch.from_data_list([data]))
    return value.numpy(), logits.numpy()


def sorted_rows(array):
    return array[np.lexsort(array.T[::-1])]


def test_facets_of_cube_are_signed_coordinate_vectors():
    normals = facet_normals(np.asarray(_cube_points()))
    expected = sorted(tuple(sign * np.eye(4, dtype=np.int64)[i]) for i in range(4) for sign in (-1, 1))
    assert sorted(map(tuple, normals.tolist())) == expected


def test_facet_normals_reject_non_reflexive_polytope():
    with pytest.raises(ValueError):
        facet_normals(2 * np.asarray(_cube_points()))  # facet dual vectors are halves: not integral


def test_cube_point_invariants_match_hand_counts():
    data = add_invariant_fields(cube_data())
    assert isinstance(data, InvariantTwoFaceData) and data.num_points == 16
    features = data.point_x_inv.numpy()
    assert features.shape == (16, NUM_POINT_FEATURES)
    assert (features[:, 0] == 6).all()  # six squares through each vertex
    # <u, v> is -v_i or +v_i for the two facets on axis i, so each vertex pairs -1 with 4 facets and +1 with 4.
    np.testing.assert_array_equal(features[:, 1:6], np.tile([4, 0, 4, 0, 0], (16, 1)))
    assert (features[:, 6] >= 4).all()


def test_invariants_are_unchanged_by_gl4z_and_relabelling():
    base = add_invariant_fields(cube_data())
    U_list = [signed_permutation(np.random.default_rng(3)), shear(0, 2, 1), shear(3, 1, -1) @ shear(2, 0, 1)]
    for U in U_list:
        moved = add_invariant_fields(cube_data(transform=U))
        np.testing.assert_array_equal(moved.point_x_inv.numpy(), base.point_x_inv.numpy())
        np.testing.assert_array_equal(moved.face_x_inv.numpy(), base.face_x_inv.numpy())
    relabelled = add_invariant_fields(cube_data(order_seed=5))
    assert relabelled.num_points == 16
    np.testing.assert_array_equal(sorted_rows(relabelled.point_x_inv.numpy()), sorted_rows(base.point_x_inv.numpy()))
    np.testing.assert_array_equal(sorted_rows(relabelled.face_x_inv.numpy()), sorted_rows(base.face_x_inv.numpy()))


def test_point_graph_matches_triangle_edges_and_face_counts():
    data = cube_data()
    fields = add_invariant_fields(data)
    point = fields.point_index.numpy()
    triangles = data.triangle_vertices.numpy()
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, a), (a, c), (c, a), (b, c), (c, b)):
            edges.add((int(point[u]), int(point[v])))
    assert set(map(tuple, fields.point_edge_index.t().tolist())) == edges
    degree = np.bincount([u for u, _ in edges], minlength=16)
    np.testing.assert_array_equal(fields.point_x_inv.numpy()[:, 6], degree)
    assert fields.face_x_inv[:, 0].sum().item() == data.node_face.numel()
    assert fields.face_x_inv[:, 1].sum().item() == data.num_triangles
    # Square faces have no interior points: every node lies on the boundary of the face.
    assert (fields.face_x_inv[:, 2] == 0).all()


def test_pyg_batch_offsets_point_keys_and_matches_single_graphs():
    cube = add_invariant_fields(cube_data())
    simplex = add_invariant_fields(simplex_data())
    permuted = add_invariant_fields(cube_data(order_seed=9, transform=shear(1, 0, 1)))
    graphs = [cube, simplex, permuted]
    batch = Batch.from_data_list(graphs)
    assert batch.num_points.tolist() == [16, 5, 16]
    node_offsets = np.cumsum([0] + [g.node_face.numel() for g in graphs])
    point_offsets = [0, 16, 21]
    for g, (n0, n1), offset in zip(graphs, zip(node_offsets[:-1], node_offsets[1:]), point_offsets):
        assert torch.equal(batch.point_index[n0:n1], g.point_index + offset)
    edge_counts = [g.point_edge_index.size(1) for g in graphs]
    edge_slices = np.cumsum([0] + edge_counts)
    for g, (e0, e1), offset in zip(graphs, zip(edge_slices[:-1], edge_slices[1:]), point_offsets):
        assert torch.equal(batch.point_edge_index[:, e0:e1], g.point_edge_index + offset)
    assert batch.point_edge_index.max().item() < sum(point_offsets[1:]) + 16 + 5 - 21 + 21
    model = make_model()
    with torch.no_grad():
        value, _ = model.get_value_and_logits(batch)
    for k, g in enumerate(graphs):
        single_value, _ = outputs(model, g)
        np.testing.assert_allclose(value[k].item(), single_value[0], atol=1e-5)


def test_batch_round_trip_keeps_invariant_type():
    graphs = [add_invariant_fields(cube_data()), add_invariant_fields(simplex_data())]
    recovered = Batch.from_data_list(graphs).to_data_list()
    assert all(isinstance(r, InvariantTwoFaceData) for r in recovered)
    assert [r.num_points for r in recovered] == [16, 5]
    assert torch.equal(recovered[1].point_index, graphs[1].point_index)


def test_add_invariant_fields_does_not_mutate_input():
    data = cube_data()
    keys = set(data.keys())
    add_invariant_fields(data)
    assert set(data.keys()) == keys and not isinstance(data, InvariantTwoFaceData)


@pytest.mark.skipif(not SHARD.exists(), reason="probe shard not available")
def test_real_states_have_reflexive_facets_and_integer_invariants():
    shard = torch.load(SHARD, weights_only=False)
    for data in shard[:20]:
        fields = add_invariant_fields(data)
        points = np.rint(data.x.double().numpy()).astype(np.int64)
        facets = facet_normals(points)
        pairing = points @ facets.T
        assert pairing.min() >= -1
        assert ((pairing == -1).sum(axis=0) >= 4).all()
        assert fields.point_x_inv.shape == (fields.num_points, NUM_POINT_FEATURES)


def test_agent_is_invariant_to_signed_permutations_and_shears():
    base_value, base_logits = outputs(make_model(), add_invariant_fields(cube_data()))
    assert base_logits.shape == (1, 24) and np.isfinite(base_logits).all()
    rng = np.random.default_rng(11)
    transforms = [signed_permutation(rng) for _ in range(3)] + [shear(0, 1, 1), shear(2, 3, -1)]
    for U in transforms:
        value, logits = outputs(make_model(), add_invariant_fields(cube_data(transform=U)))
        assert np.abs(value - base_value).max() < 1e-5
        assert np.abs(logits - base_logits).max() < 1e-5


def test_agent_is_invariant_to_node_face_triangle_and_action_order():
    base_value, base_logits = outputs(make_model(), add_invariant_fields(cube_data()))
    for seed in (1, 2):
        value, logits = outputs(make_model(), add_invariant_fields(cube_data(order_seed=seed)))
        assert np.abs(value - base_value).max() < 1e-5
        assert np.abs(logits - base_logits).max() < 1e-5


def test_agent_never_reads_batch_x():
    model = make_model()
    data = add_invariant_fields(cube_data())
    value, logits = outputs(model, data)
    batch = Batch.from_data_list([data])
    batch.x = torch.full_like(batch.x, float("nan"))
    with torch.no_grad():
        nan_value, nan_logits = model.get_value_and_logits(batch)
    assert torch.isfinite(nan_value).all() and torch.isfinite(nan_logits).all()
    np.testing.assert_array_equal(nan_value.numpy(), value)
    np.testing.assert_array_equal(nan_logits.numpy(), logits)


def test_agent_rejects_data_without_invariant_fields():
    with pytest.raises(ValueError):
        make_model()(Batch.from_data_list([cube_data()]))


def test_agent_backward_reaches_point_mpnn_and_heads():
    model = make_model().train()
    batch = Batch.from_data_list([add_invariant_fields(cube_data())])
    value, logits = model.get_value_and_logits(batch)
    (value.sum() + logits.logsumexp(dim=-1).sum()).backward()
    assert model.point_egnn.layers[0].message[0].weight.grad.abs().sum() > 0
    assert model.rho[0].weight.grad.abs().sum() > 0
    assert model.actor_mlp[0].weight.grad.abs().sum() > 0
