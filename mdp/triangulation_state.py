try:
    from sage.all import *
    from sage.geometry.triangulation.point_configuration import PointConfiguration
    from sage.geometry.triangulation.element import Triangulation
    from sage.geometry.triangulation.base import Point
except ModuleNotFoundError:
    # Sage is optional for transition/state-diff logic used in training/testing.
    PointConfiguration = object
    Triangulation = object
    Point = object

try:
    from triangulumancer import Triangulation as TrimancerTriangulation
except ModuleNotFoundError:
    TrimancerTriangulation = object

from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Self, Tuple

try:
    from core.bistellar_flip_lookup import PointConfigurationFlipLookup
except ModuleNotFoundError:
    class PointConfigurationFlipLookup:
        pass

def find_edge(three_simplices) -> Tuple:
    a, b, c = three_simplices
    edge = set(a).intersection(set(b)).intersection(set(c))
    return tuple(sorted(edge))


def simplices_key(point_config_index: int, simplices: FrozenSet[Tuple[int]]) -> str:
    canonical_simplices = tuple(sorted(tuple(sorted(simplex)) for simplex in simplices))
    return f"{point_config_index}:{canonical_simplices}"

class TriangulationState:
    """Immutable Properties"""
    vertices: List[float]
    simplices: FrozenSet[Tuple[int]]
    edges: FrozenSet[Tuple[int]]
    point_config_index: int
    key: str

    """Mutable Properties for MCTS"""
    children: List[Self] = []
    available_remove_actions: Dict[Tuple[int], Tuple[FrozenSet]] = {} # represented by edges -> flips
    available_add_actions: Dict[Tuple[int], Tuple[FrozenSet]] = {} # represented by edges -> flips
    available_flip_pairs: Tuple[Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]], ...] = ()
    available_subcomplex_actions: Tuple[Tuple[int, ...], ...] = ()
    subcomplex_to_flips: Dict[Tuple[int, ...], Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]] = {}
    ambiguous_subcomplex_actions: FrozenSet[Tuple[int, ...]] = frozenset()
    value: List[float] = []
    visitation: int = 0
    _SHARED_SUBCOMPLEX_CACHE: Dict[
        str,
        Tuple[
            Tuple[Tuple[int, ...], ...],
            Dict[Tuple[int, ...], Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]],
            FrozenSet[Tuple[int, ...]],
        ],
    ] = {}

    def __init__(
        self, 
        data_dict = None,
        vertices: List[float] = None,
        point_config_index: int = None,
        simplices: FrozenSet[Tuple[int]] = None,
        edges: FrozenSet[Tuple[int]] = None,
        triangulation: TrimancerTriangulation = None,
    ):
        if data_dict is not None:
            self.vertices = data_dict["vertices"]
            self.simplices = frozenset(sorted([tuple(sorted(simplex)) for simplex in data_dict["tri_simplices"]]))
            self.edges = frozenset(sorted([tuple(sorted(edge)) for edge in data_dict["tri_edges"]]))
            self.point_config_index = data_dict["point_config_index"]
            self.key = simplices_key(self.point_config_index, self.simplices)  # stable unique key for each triangulation
        else:
            self.vertices = vertices
            self.simplices = frozenset(sorted(simplices))
            self.edges = frozenset(sorted(edges))
            self.point_config_index = point_config_index
            self.key = simplices_key(self.point_config_index, self.simplices)  # stable unique key for each triangulation

        # store the original triangulation object for triangulumancer
        self.triangulation = triangulation
        self.neighbours = None

        self.children = []
        self.available_remove_actions = {}
        self.available_add_actions = {}
        self.available_flip_pairs = tuple()
        self.available_subcomplex_actions = tuple()
        self.subcomplex_to_flips = {}
        self.ambiguous_subcomplex_actions = frozenset()
        self.value = []
        self.visitation = 0
        self.actions_ready = False
        self.subcomplex_actions_ready = False

        # Reuse previously-computed subcomplex actions when revisiting the same state key.
        self._load_cached_subcomplex_actions()

    def find_neightbours(self):
        assert self.triangulation is not None, "Triangulation object is not provided!"
        if self.neighbours is None:
            self.neighbours = self.triangulation.neighbors()

        return self.neighbours

    @staticmethod
    def _common_face_vertices(simplex_collection: Iterable[Tuple[int, ...]]) -> Tuple[int, ...]:
        simplex_list = list(simplex_collection)
        if not simplex_list:
            return tuple()
        common_vertices = set(simplex_list[0])
        for simplex in simplex_list[1:]:
            common_vertices.intersection_update(simplex)
        return tuple(sorted(common_vertices))

    @staticmethod
    def _edges_from_simplices(simplices: Iterable[Tuple[int, ...]]) -> FrozenSet[Tuple[int, int]]:
        all_edges = set()
        for simplex in simplices:
            simplex_tuple = tuple(sorted(simplex))
            for edge in combinations(simplex_tuple, 2):
                all_edges.add(edge)
        return frozenset(sorted(all_edges))

    def _register_edge_action(
        self,
        flip_from: FrozenSet[Tuple[int, ...]],
        flip_to: FrozenSet[Tuple[int, ...]],
    ) -> None:
        shared_face = self._common_face_vertices(
            flip_from if len(flip_from) >= len(flip_to) else flip_to
        )
        if len(shared_face) != 2:
            return
        if len(flip_from) >= len(flip_to):
            self.available_remove_actions[shared_face] = (flip_from, flip_to)
        else:
            self.available_add_actions[shared_face] = (flip_from, flip_to)

    def _find_available_actions_from_lookup(self, flip_lookup: PointConfigurationFlipLookup) -> None:
        candidate_flip_indices = set()
        for simplex in self.simplices:
            candidate_flip_indices.update(
                flip_lookup.simplex_to_flip_indices.get(tuple(sorted(simplex)), ())
            )

        available_flip_pairs = []
        for flip_index in sorted(candidate_flip_indices):
            left_raw, right_raw = flip_lookup.flip_pairs[flip_index]
            left = frozenset(left_raw)
            right = frozenset(right_raw)
            if left.issubset(self.simplices):
                available_flip_pairs.append((left, right))
                self._register_edge_action(left, right)
            elif right.issubset(self.simplices):
                available_flip_pairs.append((right, left))
                self._register_edge_action(right, left)

        self.available_flip_pairs = tuple(available_flip_pairs)

    def find_available_actions(self, flip_dict):
        # assert self.actions_ready == False, "Actions have already been computed!"

        if self.actions_ready:
            if not self.subcomplex_actions_ready:
                self.ensure_subcomplex_action_cache()
            return
        
        assert not self.available_add_actions, "Remove Actions have already been computed!"
        assert not self.available_remove_actions, "Add Actions have already been computed!"

        if isinstance(flip_dict, PointConfigurationFlipLookup):
            self._find_available_actions_from_lookup(flip_dict)
        else:
            available_flip_pairs = []
            for edge, flips in flip_dict.items():
                for sub_tri1, sub_tri2 in flips:
                    if sub_tri1.issubset(self.simplices): # can add this edge for 2 -> 3 flip
                        self.available_add_actions[edge] = (sub_tri1, sub_tri2)
                        available_flip_pairs.append((sub_tri1, sub_tri2))
                    elif sub_tri2.issubset(self.simplices): # can remove this edge for 3 -> 2 flip
                        self.available_remove_actions[edge] = (sub_tri2, sub_tri1)
                        available_flip_pairs.append((sub_tri2, sub_tri1))
            self.available_flip_pairs = tuple(available_flip_pairs)

        self.actions_ready = True
        self.ensure_subcomplex_action_cache()

    @staticmethod
    def _changed_vertices_from_flips(
        flip_from: Iterable[Tuple[int, ...]],
        flip_to: Iterable[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        changed_vertices = set()
        for simplex in flip_from:
            changed_vertices.update(simplex)
        for simplex in flip_to:
            changed_vertices.update(simplex)
        return tuple(sorted(changed_vertices))

    @staticmethod
    def _canonicalize_subcomplex_action(subcomplex_action: Iterable[int]) -> Tuple[int, ...]:
        try:
            action_values = [int(v) for v in subcomplex_action]
        except TypeError as exc:
            raise TypeError(
                f"Invalid subcomplex action {subcomplex_action}. Expected an iterable of integers."
            ) from exc

        canonical_vertices = tuple(sorted({v for v in action_values if v >= 0}))
        if not canonical_vertices:
            raise ValueError(f"Subcomplex action {action_values} is empty after removing padding.")
        return canonical_vertices

    def _build_subcomplex_action_cache(self):
        all_actions = list(self.available_flip_pairs)
        if not all_actions:
            all_actions = list(self.available_add_actions.values()) + list(self.available_remove_actions.values())
            if all_actions:
                self.available_flip_pairs = tuple(all_actions)
        if not all_actions:
            self.available_subcomplex_actions = tuple()
            self.subcomplex_to_flips = {}
            self.ambiguous_subcomplex_actions = frozenset()
            self.subcomplex_actions_ready = True
            self._SHARED_SUBCOMPLEX_CACHE[self.key] = (
                self.available_subcomplex_actions,
                dict(self.subcomplex_to_flips),
                self.ambiguous_subcomplex_actions,
            )
            return

        seen = set()
        subcomplex_actions = []
        subcomplex_to_flips = {}
        ambiguous_subcomplexes = set()

        for flip_from, flip_to in all_actions:
            subcomplex = self._changed_vertices_from_flips(flip_from, flip_to)

            if subcomplex not in seen:
                seen.add(subcomplex)
                subcomplex_actions.append(subcomplex)

            if subcomplex in subcomplex_to_flips:
                ambiguous_subcomplexes.add(subcomplex)
            else:
                subcomplex_to_flips[subcomplex] = (flip_from, flip_to)

        self.available_subcomplex_actions = tuple(subcomplex_actions)
        self.subcomplex_to_flips = subcomplex_to_flips
        self.ambiguous_subcomplex_actions = frozenset(ambiguous_subcomplexes)
        self.subcomplex_actions_ready = True
        self._SHARED_SUBCOMPLEX_CACHE[self.key] = (
            self.available_subcomplex_actions,
            dict(self.subcomplex_to_flips),
            self.ambiguous_subcomplex_actions,
        )

    def _load_cached_subcomplex_actions(self) -> bool:
        cached = self._SHARED_SUBCOMPLEX_CACHE.get(self.key)
        if cached is None:
            return False

        subcomplex_actions, subcomplex_to_flips, ambiguous_subcomplexes = cached
        self.available_subcomplex_actions = subcomplex_actions
        self.subcomplex_to_flips = dict(subcomplex_to_flips)
        self.ambiguous_subcomplex_actions = ambiguous_subcomplexes
        self.subcomplex_actions_ready = True
        return True

    def ensure_subcomplex_action_cache(self):
        if self.subcomplex_actions_ready:
            return
        if self._load_cached_subcomplex_actions():
            return
        self._build_subcomplex_action_cache()

    def get_available_subcomplex_actions(self) -> Tuple[Tuple[int, ...], ...]:
        self.ensure_subcomplex_action_cache()
        return self.available_subcomplex_actions

    def get_flips_from_subcomplex_action(
        self,
        subcomplex_action: Iterable[int],
    ) -> Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]:
        target_subcomplex = self._canonicalize_subcomplex_action(subcomplex_action)
        self.ensure_subcomplex_action_cache()

        if target_subcomplex in self.ambiguous_subcomplex_actions:
            raise ValueError(
                f"Subcomplex action {target_subcomplex} matches multiple available actions in state {self.key}."
            )

        if target_subcomplex not in self.subcomplex_to_flips:
            raise ValueError(
                f"Subcomplex action {target_subcomplex} does not match any available action in state {self.key}."
            )
        return self.subcomplex_to_flips[target_subcomplex]

    def transition_from_subcomplex_action(self, subcomplex_action: Iterable[int]):
        flip_from, flip_to = self.get_flips_from_subcomplex_action(subcomplex_action)
        return self.transition(flip_from, flip_to)

    def get_flips_from_actions(self, edge):
        edge_tuple = tuple(edge)
        if edge_tuple in self.available_remove_actions:
            return self.available_remove_actions[edge_tuple]
        elif edge_tuple in self.available_add_actions:
            return self.available_add_actions[edge_tuple]
        else:
            raise ValueError("Invalid actions! Not found in the state available action list")
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TriangulationState):
            return self.key == other.key
        elif isinstance(other, str):
            return self.key == other
        else:
            return False
    
    def __str__(self):
        return self.key
    
    def transition(self, flip_from: FrozenSet[Tuple[int]], flip_to: FrozenSet[Tuple[int]]) -> Tuple[FrozenSet[Tuple[int]], FrozenSet[Tuple[int]], str]:
        """
        Return the key after replacing the sub-triangulation (flip_from) with the other (flip_to)
        """
        # conduct the bistellar flips
        new_simplicies = (self.simplices - flip_from).union(flip_to)
        new_edges = self._edges_from_simplices(new_simplicies)
        # get the key for the next state for O(1) query
        new_key = simplices_key(self.point_config_index, new_simplicies)

        return new_simplicies, new_edges, new_key

    def find_changed_subcomplex(self, other: Self) -> FrozenSet[int]:
        """
        Find the changed subcomplex vertex set between two adjacent triangulations.

        For adjacent states A (self) and B (other), let:
            removed = A \\ B
            added = B \\ A
        where removed/added are sets of simplices.
        The changed subcomplex S is the vertex set contained in removed U added.

        Raises:
            ValueError: if the two states are not adjacent under a single 2<->3 flip.
        """
        if not isinstance(other, TriangulationState):
            raise TypeError("other must be a TriangulationState.")
        if self.point_config_index != other.point_config_index:
            raise ValueError("Triangulations belong to different point configurations.")

        removed_simplices = self.simplices - other.simplices
        added_simplices = other.simplices - self.simplices

        if not removed_simplices and not added_simplices:
            raise ValueError("Triangulations are identical; no changed subcomplex exists.")

        reference_simplices = removed_simplices or added_simplices
        simplex_size = len(next(iter(reference_simplices)))
        expected_changed_vertex_count = simplex_size + 1
        changed_vertices = set()
        for simplex in removed_simplices:
            changed_vertices.update(simplex)
        for simplex in added_simplices:
            changed_vertices.update(simplex)

        if len(changed_vertices) != expected_changed_vertex_count:
            raise ValueError(
                "Triangulations are not adjacent by a single bistellar flip "
                f"(changed vertex count {len(changed_vertices)} != expected {expected_changed_vertex_count})."
            )
        if len(removed_simplices) + len(added_simplices) != expected_changed_vertex_count:
            raise ValueError(
                "Triangulations are not adjacent by a single bistellar flip "
                f"(simplex differences sum to {len(removed_simplices) + len(added_simplices)}, "
                f"expected {expected_changed_vertex_count})."
            )
        return frozenset(changed_vertices)

    def set_value(self, value: float):
        # keep track of the value estimated for this state
        # add once each time we visit this state
        self.value.append(value)
