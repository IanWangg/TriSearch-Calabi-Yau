"""Geometry-free, face-separated observations for the two-face policy.

Only coordinates and combinatorial tensors are cached. Both caches are assigned
their share of the existing tensor budget by core.cy_data_utils.
"""

from dataclasses import dataclass
from itertools import combinations

import torch
from torch_geometric.data import Data

from core.cy_bounded_cache import BoundedLRU
from mdp.cy_state_record import two_face_restrictions


TWO_FACE_OBSERVATION_SCHEMA_VERSION = 1
TWO_FACE_TOPOLOGY_CACHE = BoundedLRU(0)
TWO_FACE_ACTION_CACHE = BoundedLRU(0)


class TwoFaceData(Data):
    observation_kind = "two_face"
    observation_schema_version = TWO_FACE_OBSERVATION_SCHEMA_VERSION

    @property
    def num_faces(self):
        # PyG's deprecated property counts mesh triangles via `face`; our faces
        # are ambient polygons, so retain the explicit scalar (or batch tensor).
        return self["num_faces"]

    @num_faces.setter
    def num_faces(self, value):
        self["num_faces"] = value

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "triangle_vertices", "added_triangle_vertices"):
            return self.num_nodes
        if key in ("node_face", "triangle_face", "action_face"):
            return int(self.num_faces)
        if key in ("triangle_edge_index", "removed_triangle_ids"):
            return int(self.num_triangles)
        if key == "subcomplex_vertices":
            return 0  # Backend labels, not packed node indices.
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "triangle_edge_index"):
            return 1
        if key in ("triangle_vertices", "added_triangle_vertices", "removed_triangle_ids",
                   "node_face", "triangle_face", "action_face", "subcomplex_vertices"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


@dataclass(frozen=True)
class _FaceTopology:
    fields: dict
    node_maps: tuple
    triangle_maps: tuple


def _index_tensor(rows, width):
    return torch.tensor(rows, dtype=torch.long, device="cpu").reshape(-1, width)


def _build_topology(configuration, restrictions):
    if len(configuration.labels) != len(configuration.points):
        raise ValueError("Point labels and coordinates must have the same length.")
    coordinates = dict(zip(configuration.labels, configuration.points))
    if len(coordinates) != len(configuration.labels):
        raise ValueError("Point configuration contains duplicate labels.")
    nodes, node_faces, triangles, triangle_faces = [], [], [], []
    point_edges, dual_edges, node_maps, triangle_maps = [], [], [], []
    for face_index, (labels, face_triangles) in enumerate(restrictions):
        if len(set(labels)) != len(labels) or any(label not in coordinates for label in labels):
            raise ValueError(f"Invalid ambient labels for face {labels}.")
        node_map = {label: len(nodes) + index for index, label in enumerate(labels)}
        nodes.extend(coordinates[label] for label in labels)
        node_faces.extend([face_index] * len(labels))
        triangle_map, edge_triangles = {}, {}
        for triangle in face_triangles:
            triangle_id = len(triangles)
            triangle_map[triangle] = triangle_id
            triangles.append(tuple(node_map[label] for label in triangle))
            triangle_faces.append(face_index)
            for edge in combinations(triangle, 2):
                edge_triangles.setdefault(edge, []).append(triangle_id)
        for (left, right), incident in sorted(edge_triangles.items()):
            point_edges.extend(((node_map[left], node_map[right]), (node_map[right], node_map[left])))
            if len(incident) > 2:
                raise ValueError(f"Non-manifold triangle edge in face {labels}.")
            if len(incident) == 2:
                dual_edges.extend((tuple(incident), tuple(reversed(incident))))
        node_maps.append(node_map)
        triangle_maps.append(triangle_map)
    fields = dict(
        x=torch.tensor(nodes, dtype=torch.float32, device="cpu"),
        edge_index=_index_tensor(point_edges, 2).t().contiguous(),
        node_face=torch.tensor(node_faces, dtype=torch.long, device="cpu"),
        triangle_vertices=_index_tensor(triangles, 3),
        triangle_face=torch.tensor(triangle_faces, dtype=torch.long, device="cpu"),
        triangle_edge_index=_index_tensor(dual_edges, 2).t().contiguous(),
        num_faces=len(restrictions), num_triangles=len(triangles),
    )
    return _FaceTopology(fields, tuple(node_maps), tuple(triangle_maps))


def _build_actions(topology, actions):
    faces, removed_ids, added_vertices = [], [], []
    for action in actions:
        if len(action) != 4 or len(set(action)) != 4:
            raise ValueError("two_face actions must contain exactly four distinct backend labels.")
        circuit = frozenset(action)
        matching = [index for index, node_map in enumerate(topology.node_maps)
                    if circuit.issubset(node_map)]
        if len(matching) != 1:
            raise ValueError(f"Circuit {action} must belong to exactly one ambient two-face.")
        face = matching[0]
        node_map, triangle_map = topology.node_maps[face], topology.triangle_maps[face]
        candidates = tuple(combinations(sorted(action), 3))
        removed = [triangle for triangle in candidates if triangle in triangle_map]
        added = [triangle for triangle in candidates if triangle not in triangle_map]
        if len(removed) != 2 or len(added) != 2:
            raise ValueError(f"Circuit {action} must replace exactly two current triangles.")
        faces.append(face)
        removed_ids.append([triangle_map[triangle] for triangle in removed])
        added_vertices.append([[node_map[label] for label in triangle] for triangle in added])
    return dict(
        action_face=torch.tensor(faces, dtype=torch.long, device="cpu"),
        removed_triangle_ids=_index_tensor(removed_ids, 2),
        added_triangle_vertices=torch.tensor(added_vertices, dtype=torch.long, device="cpu").reshape(-1, 2, 3),
        subcomplex_vertices=_index_tensor(actions, 4), num_available_subcomplexes=len(actions),
    )


def build_two_face_data(configuration, restrictions, actions=()):
    """Build from canonical restrictions and backend actions, preserving their order."""
    key = (configuration, restrictions, TWO_FACE_OBSERVATION_SCHEMA_VERSION)
    topology = TWO_FACE_TOPOLOGY_CACHE.get(key)
    if topology is None:
        if not restrictions:
            raise ValueError("two_face observation requires nonempty ambient face restrictions.")
        topology = _build_topology(configuration, restrictions)
        TWO_FACE_TOPOLOGY_CACHE[key] = topology
    actions = tuple(tuple(int(label) for label in action) for action in actions)
    action_key = (key, actions)
    action_fields = TWO_FACE_ACTION_CACHE.get(action_key)
    if action_fields is None:
        action_fields = _build_actions(topology, actions)
        TWO_FACE_ACTION_CACHE[action_key] = action_fields
    return TwoFaceData(**topology.fields, **action_fields)


def create_two_face_data_from_state(state, actions=()):
    configuration = getattr(state, "configuration", None)
    if configuration is None:
        raise ValueError("two_face observation requires managed geometry state metadata.")
    return build_two_face_data(configuration, two_face_restrictions(configuration, state.simplices), actions)
