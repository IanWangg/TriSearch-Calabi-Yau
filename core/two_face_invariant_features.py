"""GL(4,Z)-invariant side fields for TwoFaceData (CPU, numpy/torch).

add_invariant_fields(data) returns a new InvariantTwoFaceData with the original keys plus:
  point_index       (N,) long: merged lattice point of each node (nodes with equal x share an id,
                    ids assigned in order of first appearance).
  num_points        int: number of merged points M.
  point_edge_index  (2, E) long: both directions of the union of triangle edges between merged points.
  point_x_inv       (M, 8) float: per merged point, the integer invariants
                    [2-faces containing it,
                     #facets with <u_F, v> == -1, == 0, == 1, == 2, >= 3,
                     degree in the merged point graph,
                     distinct merged triangles containing it].
  face_x_inv        (num_faces, 3) float: [points in face, triangles in face,
                    points of the face lying in exactly one 2-face].

Facets come from the convex hull of the merged points. Each facet gets the primitive integer
u_F with <u_F, v> = -1 on the facet and <u_F, v> >= -1 for every merged point v. Any failure raises.
All features are computed from invariants only, so they are unchanged by GL(4,Z) acting on x and
by any relabelling of nodes, triangles or faces.
"""

from collections import OrderedDict

import numpy as np
import torch
from scipy.spatial import ConvexHull

from core.cy_two_face_data import TwoFaceData

POINT_FEATURE_NAMES = ("faces_containing", "pair_m1", "pair_0", "pair_1", "pair_2", "pair_ge3",
                       "degree", "triangles_containing")
FACE_FEATURE_NAMES = ("points", "triangles", "interior_points")
NUM_POINT_FEATURES = len(POINT_FEATURE_NAMES)
NUM_FACE_FEATURES = len(FACE_FEATURE_NAMES)
MIN_FACET_POINTS = 4


class InvariantTwoFaceData(TwoFaceData):
    """TwoFaceData whose merged-point keys are offset by num_points when batched."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("point_index", "point_edge_index"):
            return int(self.num_points)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "point_edge_index":
            return 1
        if key == "point_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def merge_points(coordinates):
    """Return (point_index, unique_points) with ids in order of first appearance."""
    ids, rows, index = {}, [], np.empty(len(coordinates), dtype=np.int64)
    for node, row in enumerate(coordinates):
        key = tuple(row.tolist())
        point = ids.get(key)
        if point is None:
            point = ids[key] = len(rows)
            rows.append(row)
        index[node] = point
    return index, np.asarray(rows, dtype=np.float64).reshape(-1, coordinates.shape[1])


def facet_normals(points):
    """Primitive integer u_F (rows, sorted) with <u_F, v> = -1 on facet F and >= -1 on all points."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4 or len(points) < 5:
        raise ValueError("facet_normals needs at least five 4-dimensional points.")
    if not np.array_equal(points, np.rint(points)):
        raise ValueError("Lattice coordinates must be integers.")
    integer_points = np.rint(points).astype(np.int64)
    hull = ConvexHull(integer_points.astype(np.float64))  # raises QhullError on degenerate input
    equations = hull.equations
    if not np.all(equations[:, -1] < -1e-9):
        raise ValueError("The origin is not in the interior of the point hull.")
    # n.x + d <= 0 inside with d < 0, so w = n / (-d) satisfies w.x <= 1 inside and w.x = 1 on the facet.
    dual = equations[:, :-1] / (-equations[:, -1:])
    groups = {}
    for row in dual:
        groups.setdefault(tuple(np.round(row, 6)), []).append(row)
    facets = []
    for rows in groups.values():
        w = np.mean(rows, axis=0)
        w_int = np.rint(w)
        if np.abs(w - w_int).max() > 1e-6:
            raise ValueError("A facet dual vector is not integral; the polytope is not reflexive.")
        u = -w_int.astype(np.int64)
        if np.gcd.reduce(np.abs(u)) != 1:
            raise ValueError("A facet dual vector is not primitive.")
        pairing = integer_points @ u
        if pairing.min() < -1:
            raise ValueError("A point pairs below -1 with a facet normal.")
        if int((pairing == -1).sum()) < MIN_FACET_POINTS:
            raise ValueError("A facet contains fewer than four points.")
        facets.append(tuple(int(v) for v in u))
    return np.array(sorted(facets), dtype=np.int64)


def invariant_fields(data):
    """Plain-dict form of the new keys for one TwoFaceData (not mutating data)."""
    coordinates = data.x.detach().cpu().double().numpy()
    node_face = data.node_face.detach().cpu().numpy().astype(np.int64)
    triangles = data.triangle_vertices.detach().cpu().numpy().astype(np.int64)
    triangle_face = data.triangle_face.detach().cpu().numpy().astype(np.int64)
    num_faces = int(data.num_faces)
    if num_faces < 1 or len(triangles) == 0:
        raise ValueError("Invariant fields need at least one face and one triangle.")

    point_index, points = merge_points(coordinates)
    num_points = len(points)
    facets = facet_normals(points)
    pairings = np.rint(points).astype(np.int64) @ facets.T  # (M, F)

    triangle_points = point_index[triangles]
    sorted_triangle_points = np.sort(triangle_points, axis=1)
    if (np.diff(sorted_triangle_points, axis=1) == 0).any():
        raise ValueError("A triangle repeats a merged point.")
    edges = np.sort(np.concatenate([triangle_points[:, [0, 1]], triangle_points[:, [1, 2]],
                                    triangle_points[:, [0, 2]]]), axis=1)
    edges = np.unique(edges, axis=0)
    point_edge_index = np.concatenate([edges, edges[:, ::-1]]).T.copy()
    degree = np.bincount(edges.ravel(), minlength=num_points)
    merged_triangles = np.unique(sorted_triangle_points, axis=0)
    triangle_count = np.bincount(merged_triangles.ravel(), minlength=num_points)

    point_face_pairs = np.unique(point_index * num_faces + node_face)
    faces_per_point = np.bincount(point_face_pairs // num_faces, minlength=num_points)
    pair_counts = [(pairings == value).sum(axis=1) for value in (-1, 0, 1, 2)]
    pair_counts.append((pairings >= 3).sum(axis=1))
    point_x_inv = np.stack([faces_per_point, *pair_counts, degree, triangle_count], axis=1).astype(np.float32)

    interior = (faces_per_point[point_index] == 1).astype(np.float64)
    face_x_inv = np.stack([
        np.bincount(node_face, minlength=num_faces),
        np.bincount(triangle_face, minlength=num_faces),
        np.bincount(node_face, weights=interior, minlength=num_faces),
    ], axis=1).astype(np.float32)
    return dict(
        point_index=torch.as_tensor(point_index, dtype=torch.long),
        num_points=num_points,
        point_edge_index=torch.as_tensor(point_edge_index, dtype=torch.long),
        point_x_inv=torch.as_tensor(point_x_inv),
        face_x_inv=torch.as_tensor(face_x_inv),
    )


def add_invariant_fields(data):
    """Return a new InvariantTwoFaceData with the invariant fields added; `data` is not modified."""
    out = InvariantTwoFaceData()
    for key, value in data.items():
        out[key] = value
    for key, value in invariant_fields(data).items():
        out[key] = value
    return out


# ----------------------------------------------------------------- policy observation path

INVARIANT_TOPOLOGY_CACHE_SIZE = 50_000
_invariant_topology_cache = OrderedDict()
_invariant_topology_stats = {"hits": 0, "misses": 0}


def invariant_topology_cache_stats():
    return dict(_invariant_topology_stats, entries=len(_invariant_topology_cache))


def clear_invariant_topology_cache():
    _invariant_topology_cache.clear()
    _invariant_topology_stats.update(hits=0, misses=0)


def create_invariant_two_face_data_from_state(state, actions=()):
    """Rollout/eval observation for two_face_invariant: create_two_face_data_from_state plus invariant fields.

    The invariant fields depend only on the topology, so they are cached per (configuration, restrictions).
    Actions do not change them. Returns an InvariantTwoFaceData; the base TwoFaceData is not mutated.
    """
    from core.cy_two_face_data import create_two_face_data_from_state
    from mdp.cy_state_record import two_face_restrictions

    base = create_two_face_data_from_state(state, actions)
    key = (state.configuration, two_face_restrictions(state.configuration, state.simplices))
    fields = _invariant_topology_cache.get(key)
    if fields is None:
        _invariant_topology_stats["misses"] += 1
        fields = invariant_fields(base)
        _invariant_topology_cache[key] = fields
        if len(_invariant_topology_cache) > INVARIANT_TOPOLOGY_CACHE_SIZE:
            _invariant_topology_cache.popitem(last=False)
    else:
        _invariant_topology_stats["hits"] += 1
        _invariant_topology_cache.move_to_end(key)
    out = InvariantTwoFaceData()
    for name, value in base.items():
        out[name] = value
    for name, value in fields.items():
        out[name] = value
    return out
