from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Self, Tuple

from mdp.triangulation_state import TriangulationState, simplices_key

try:
    from cytools.triangulation import Triangulation as CYToolsTriangulation
except ModuleNotFoundError:
    CYToolsTriangulation = object


def _normalize_simplex(simplex: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(vertex) for vertex in simplex))


def _normalize_simplices(simplices: Iterable[Iterable[int]]) -> FrozenSet[Tuple[int, ...]]:
    return frozenset(sorted(_normalize_simplex(simplex) for simplex in simplices))


def _normalize_edges(edges: Iterable[Iterable[int]]) -> FrozenSet[Tuple[int, ...]]:
    return frozenset(sorted(_normalize_simplex(edge) for edge in edges))


def _edges_from_simplices(simplices: Iterable[Tuple[int, ...]]) -> FrozenSet[Tuple[int, ...]]:
    all_edges = set()
    for simplex in simplices:
        all_edges.update(_normalize_simplex(edge) for edge in combinations(simplex, 2))
    return frozenset(sorted(all_edges))


def _normalize_vertices(points: Any) -> List[List[int]]:
    if points is None:
        raise ValueError("Expected a 2D array-like of vertices from cytools.")

    vertices: List[List[int]] = []
    for point in points:
        vertices.append([int(coord) for coord in point])

    if not vertices or not isinstance(vertices[0], list):
        raise ValueError("Expected a 2D array-like of vertices from cytools.")
    return vertices


def _safe_bool_method_call(obj: Any, method_name: str) -> bool:
    from core.cytools_config import REGULARITY_BACKEND
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return False
    try:
        if method_name == "is_regular":
            return bool(method(backend=REGULARITY_BACKEND))
        return bool(method())
    except Exception:
        return False


def _infer_vertices_from_cy_triangulation(triangulation: CYToolsTriangulation) -> List[List[int]]:
    points_getter = getattr(triangulation, "points", None)
    if points_getter is None or not callable(points_getter):
        raise ValueError("cy_triangulation does not expose points().")
    return _normalize_vertices(points_getter())


CYTransitionOutput = Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]], str]
NEIGHBOR_MODES = ("regular", "two_neighbors")


def _normalize_neighbor_mode(neighbor_mode: str) -> str:
    resolved_mode = str(neighbor_mode).strip().lower()
    if resolved_mode not in NEIGHBOR_MODES:
        raise ValueError(
            f"Unknown neighbor_mode '{neighbor_mode}'. "
            f"Expected one of: {', '.join(NEIGHBOR_MODES)}."
        )
    return resolved_mode


def _state_key_for_neighbor_mode(
    point_config_index: int,
    simplices: FrozenSet[Tuple[int, ...]],
    neighbor_mode: str,
) -> str:
    base_key = simplices_key(point_config_index, simplices)
    if neighbor_mode == "regular":
        return base_key
    return f"{neighbor_mode}|{base_key}"


def _face_restriction_map(triangulation: CYToolsTriangulation) -> Dict[
    Tuple[int, ...], FrozenSet[Tuple[int, ...]]
]:
    restriction_getter = getattr(triangulation, "restrict", None)
    if restriction_getter is None or not callable(restriction_getter):
        raise RuntimeError(
            "CYTools two_neighbors contract violation: triangulation does not expose restrict()."
        )

    face_restrictions: Dict[Tuple[int, ...], FrozenSet[Tuple[int, ...]]] = {}
    for face_triangulation in restriction_getter(as_poly=True):
        face_labels = tuple(sorted(int(label) for label in face_triangulation.labels))
        if face_labels in face_restrictions:
            raise RuntimeError(
                "CYTools two_neighbors contract violation: duplicate 2-face restriction "
                f"with labels {face_labels}."
            )
        face_restrictions[face_labels] = _normalize_simplices(
            face_triangulation.simplices()
        )
    return face_restrictions


def _two_neighbor_circuit(
    source: CYToolsTriangulation,
    destination: CYToolsTriangulation,
) -> Tuple[int, ...]:
    source_faces = _face_restriction_map(source)
    destination_faces = _face_restriction_map(destination)
    if source_faces.keys() != destination_faces.keys():
        raise RuntimeError(
            "CYTools two_neighbors contract violation: source and destination have "
            "different 2-face sets."
        )

    changed_faces = [
        face_labels
        for face_labels in source_faces
        if source_faces[face_labels] != destination_faces[face_labels]
    ]
    if len(changed_faces) != 1:
        raise RuntimeError(
            "CYTools two_neighbors contract violation: expected exactly one changed "
            f"2-face, found {len(changed_faces)}."
        )

    face_labels = changed_faces[0]
    removed = source_faces[face_labels] - destination_faces[face_labels]
    added = destination_faces[face_labels] - source_faces[face_labels]
    removed_vertices = set().union(*(set(simplex) for simplex in removed)) if removed else set()
    added_vertices = set().union(*(set(simplex) for simplex in added)) if added else set()
    if (
        len(removed) != 2
        or len(added) != 2
        or any(len(simplex) != 3 for simplex in removed.union(added))
        or removed_vertices != added_vertices
        or len(removed_vertices) != 4
    ):
        raise RuntimeError(
            "CYTools two_neighbors contract violation: the changed 2-face is not a "
            "2-to-2 diagonal flip on a four-vertex circuit "
            f"(face={face_labels}, removed={sorted(removed)}, added={sorted(added)})."
        )
    return tuple(sorted(removed_vertices))


class CYTriangulationState(TriangulationState):
    _SHARED_SUBCOMPLEX_CACHE: Dict[
        str,
        Tuple[
            Tuple[Tuple[int, ...], ...],
            Dict[Tuple[int, ...], Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]],
            FrozenSet[Tuple[int, ...]],
        ],
    ] = {}
    _SHARED_NEIGHBOUR_FLIP_CACHE: Dict[
        str,
        Tuple[Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]], ...],
    ] = {}
    _SHARED_SUBCOMPLEX_TRANSITION_CACHE: Dict[str, Dict[Tuple[int, ...], CYTransitionOutput]] = {}
    _SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE: Dict[str, Dict[Tuple[int, ...], CYToolsTriangulation]] = {}

    def __init__(
        self,
        data_dict: Optional[Dict[str, Any]] = None,
        vertices: Optional[List[List[int]]] = None,
        point_config_index: Optional[int] = None,
        simplices: Optional[Iterable[Iterable[int]]] = None,
        edges: Optional[Iterable[Iterable[int]]] = None,
        cy_triangulation: Optional[CYToolsTriangulation] = None,
        is_frst: Optional[bool] = None,
        neighbor_mode: str = "regular",
    ):
        self.neighbor_mode = _normalize_neighbor_mode(neighbor_mode)
        source_vertices = vertices
        source_simplices = simplices
        source_edges = edges
        source_point_config_index = point_config_index

        if data_dict is not None:
            source_vertices = data_dict.get("vertices", source_vertices)
            source_simplices = data_dict.get("tri_simplices", source_simplices)
            source_edges = data_dict.get("tri_edges", source_edges)
            source_point_config_index = data_dict.get("point_config_index", source_point_config_index)

        if source_simplices is None and cy_triangulation is not None:
            source_simplices = cy_triangulation.simplices()
        if source_simplices is None:
            raise ValueError("simplices must be provided (directly, via data_dict, or via cy_triangulation).")
        normalized_simplices = _normalize_simplices(source_simplices)

        if source_edges is None:
            normalized_edges = _edges_from_simplices(normalized_simplices)
        else:
            normalized_edges = _normalize_edges(source_edges)

        if source_vertices is None and cy_triangulation is not None:
            source_vertices = _infer_vertices_from_cy_triangulation(cy_triangulation)
        if source_vertices is None:
            raise ValueError("vertices must be provided (directly, via data_dict, or via cy_triangulation).")
        normalized_vertices = _normalize_vertices(source_vertices)

        super().__init__(
            data_dict=None,
            vertices=normalized_vertices,
            point_config_index=source_point_config_index,
            simplices=normalized_simplices,
            edges=normalized_edges,
            triangulation=None,
        )
        self.key = _state_key_for_neighbor_mode(
            self.point_config_index,
            self.simplices,
            self.neighbor_mode,
        )
        if self.neighbor_mode == "two_neighbors":
            self.available_subcomplex_actions = tuple()
            self.subcomplex_to_flips = {}
            self.ambiguous_subcomplex_actions = frozenset()
            self.subcomplex_actions_ready = False
            self._load_cached_subcomplex_actions()

        self.cy_triangulation = cy_triangulation
        if is_frst is not None:
            self.is_frst = bool(is_frst)
        else:
            self.is_frst = self._infer_is_frst()

        self._subcomplex_transition_cache: Dict[Tuple[int, ...], CYTransitionOutput] = {}
        self._subcomplex_neighbour_cache: Dict[Tuple[int, ...], CYToolsTriangulation] = {}
        self._load_cached_transition_cache()
        self._load_cached_neighbour_cache()

    def _load_cached_transition_cache(self) -> bool:
        cached = self._SHARED_SUBCOMPLEX_TRANSITION_CACHE.get(self.key)
        if cached is None:
            return False
        self._subcomplex_transition_cache = dict(cached)
        return True

    def _load_cached_neighbour_cache(self) -> bool:
        cached = self._SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE.get(self.key)
        if cached is None:
            return False
        self._subcomplex_neighbour_cache = dict(cached)
        return True

    def _infer_is_frst(self) -> bool:
        if self.cy_triangulation is None:
            return False

        return (
            _safe_bool_method_call(self.cy_triangulation, "is_fine")
            and _safe_bool_method_call(self.cy_triangulation, "is_star")
            and _safe_bool_method_call(self.cy_triangulation, "is_regular")
        )

    @property
    def reward(self) -> int:
        return 1 if self.is_frst else 0

    @property
    def terminal(self) -> bool:
        if len(self.simplices) <= 1:
            return True
        if self.is_frst:
            return True

        if self.actions_ready:
            return not (self.available_add_actions or self.available_remove_actions)

        if self.cy_triangulation is None:
            return False

        return len(self.find_neightbours()) == 0

    def find_neightbours(self):
        if self.cy_triangulation is None:
            raise AssertionError("cy_triangulation is not provided.")
        if self.neighbours is None:
            if self.neighbor_mode == "two_neighbors":
                neighbours = self.cy_triangulation.neighbor_triangulations(
                    two_neighbors=True
                )
            else:
                try:
                    neighbours = self.cy_triangulation.neighbor_triangulations(only_regular=True)
                except TypeError:
                    neighbours = self.cy_triangulation.neighbor_triangulations()
                    neighbours = [tri for tri in neighbours if _safe_bool_method_call(tri, "is_regular")]
            self.neighbours = list(neighbours)
        return self.neighbours

    def _compute_neighbor_flips(self) -> List[Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]]:
        cached_flips = self._SHARED_NEIGHBOUR_FLIP_CACHE.get(self.key)
        if cached_flips is not None:
            self._load_cached_subcomplex_actions()
            self._load_cached_transition_cache()
            self._load_cached_neighbour_cache()
            return list(cached_flips)

        flips: List[Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]]] = []
        seen_signatures = set()
        subcomplex_to_transition: Dict[Tuple[int, ...], CYTransitionOutput] = {}
        subcomplex_to_neighbour: Dict[Tuple[int, ...], CYToolsTriangulation] = {}
        subcomplex_to_flips: Dict[
            Tuple[int, ...],
            Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]]],
        ] = {}
        subcomplex_actions: List[Tuple[int, ...]] = []
        seen_subcomplexes: set[Tuple[int, ...]] = set()
        ambiguous_subcomplexes: set[Tuple[int, ...]] = set()

        for neighbour in self.find_neightbours():
            if self.neighbor_mode == "two_neighbors" and not (
                _safe_bool_method_call(neighbour, "is_fine")
                and _safe_bool_method_call(neighbour, "is_star")
                and _safe_bool_method_call(neighbour, "is_regular")
            ):
                raise RuntimeError(
                    "CYTools two_neighbors contract violation: destination "
                    "representative is not fine, star, and regular."
                )
            neighbour_simplices = _normalize_simplices(neighbour.simplices())
            flip_from = self.simplices - neighbour_simplices
            flip_to = neighbour_simplices - self.simplices

            # Skip invalid or degenerate transitions.
            if not flip_from or not flip_to:
                continue

            signature = (tuple(sorted(flip_from)), tuple(sorted(flip_to)))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            flips.append((flip_from, flip_to))

            if self.neighbor_mode == "two_neighbors":
                subcomplex = _two_neighbor_circuit(self.cy_triangulation, neighbour)
            else:
                subcomplex = self._changed_vertices_from_flips(flip_from, flip_to)
            if subcomplex not in seen_subcomplexes:
                seen_subcomplexes.add(subcomplex)
                subcomplex_actions.append(subcomplex)
            if subcomplex in subcomplex_to_transition:
                subcomplex_to_transition.pop(subcomplex, None)
                subcomplex_to_neighbour.pop(subcomplex, None)
                subcomplex_to_flips.pop(subcomplex, None)
                ambiguous_subcomplexes.add(subcomplex)
            elif subcomplex not in ambiguous_subcomplexes:
                next_edges = _edges_from_simplices(neighbour_simplices)
                next_key = _state_key_for_neighbor_mode(
                    self.point_config_index,
                    neighbour_simplices,
                    self.neighbor_mode,
                )
                subcomplex_to_transition[subcomplex] = (
                    neighbour_simplices,
                    next_edges,
                    next_key,
                )
                subcomplex_to_neighbour[subcomplex] = neighbour
                subcomplex_to_flips[subcomplex] = (flip_from, flip_to)

        if self.neighbor_mode == "two_neighbors":
            self.available_subcomplex_actions = tuple(subcomplex_actions)
            self.subcomplex_to_flips = dict(subcomplex_to_flips)
            self.ambiguous_subcomplex_actions = frozenset(ambiguous_subcomplexes)
            self.subcomplex_actions_ready = True
            self._SHARED_SUBCOMPLEX_CACHE[self.key] = (
                self.available_subcomplex_actions,
                dict(self.subcomplex_to_flips),
                self.ambiguous_subcomplex_actions,
            )

        self._SHARED_NEIGHBOUR_FLIP_CACHE[self.key] = tuple(flips)
        self._SHARED_SUBCOMPLEX_TRANSITION_CACHE[self.key] = dict(subcomplex_to_transition)
        self._SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE[self.key] = dict(subcomplex_to_neighbour)
        self._subcomplex_transition_cache = dict(subcomplex_to_transition)
        self._subcomplex_neighbour_cache = dict(subcomplex_to_neighbour)
        return flips

    def find_available_actions(self, flip_dict=None):
        if self.actions_ready:
            if not self.subcomplex_actions_ready:
                self.ensure_subcomplex_action_cache()
            return

        assert not self.available_add_actions, "Add actions have already been computed!"
        assert not self.available_remove_actions, "Remove actions have already been computed!"

        for action_idx, (flip_from, flip_to) in enumerate(self._compute_neighbor_flips()):
            action_key = (action_idx,)
            if len(flip_from) >= len(flip_to):
                self.available_remove_actions[action_key] = (flip_from, flip_to)
            else:
                self.available_add_actions[action_key] = (flip_from, flip_to)

        self.actions_ready = True
        self.ensure_subcomplex_action_cache()

    def _build_subcomplex_action_cache(self):
        all_actions = list(self.available_add_actions.values()) + list(self.available_remove_actions.values())
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

    def get_flips_from_actions(self, action):
        if isinstance(action, tuple):
            action_key = action
        elif isinstance(action, list):
            action_key = tuple(action)
        else:
            action_key = (int(action),)

        if action_key in self.available_remove_actions:
            return self.available_remove_actions[action_key]
        if action_key in self.available_add_actions:
            return self.available_add_actions[action_key]

        raise ValueError("Invalid action: not found in the state available action list.")

    def get_transition_output_from_subcomplex_action(
        self,
        subcomplex_action: Iterable[int],
    ) -> CYTransitionOutput:
        target_subcomplex = self._canonicalize_subcomplex_action(subcomplex_action)
        self.ensure_subcomplex_action_cache()

        if target_subcomplex in self.ambiguous_subcomplex_actions:
            raise ValueError(
                f"Subcomplex action {target_subcomplex} matches multiple available actions in state {self.key}."
            )

        cached_transition = self._subcomplex_transition_cache.get(target_subcomplex)
        if cached_transition is not None:
            return cached_transition

        flip_from, flip_to = self.get_flips_from_subcomplex_action(target_subcomplex)
        transition_output = self.transition(flip_from, flip_to)
        self._subcomplex_transition_cache[target_subcomplex] = transition_output
        if self.key not in self._SHARED_SUBCOMPLEX_TRANSITION_CACHE:
            self._SHARED_SUBCOMPLEX_TRANSITION_CACHE[self.key] = {}
        self._SHARED_SUBCOMPLEX_TRANSITION_CACHE[self.key][target_subcomplex] = transition_output
        return transition_output

    def get_next_cy_triangulation_from_subcomplex_action(
        self,
        subcomplex_action: Iterable[int],
    ) -> Optional[CYToolsTriangulation]:
        target_subcomplex = self._canonicalize_subcomplex_action(subcomplex_action)
        self.ensure_subcomplex_action_cache()

        if target_subcomplex in self.ambiguous_subcomplex_actions:
            return None

        neighbour = self._subcomplex_neighbour_cache.get(target_subcomplex)
        if neighbour is not None:
            return neighbour

        if self._load_cached_neighbour_cache():
            return self._subcomplex_neighbour_cache.get(target_subcomplex)
        return None

    def transition_from_subcomplex_action(self, subcomplex_action: Iterable[int]):
        return self.get_transition_output_from_subcomplex_action(subcomplex_action)

    def transition(
        self,
        flip_from: FrozenSet[Tuple[int, ...]],
        flip_to: FrozenSet[Tuple[int, ...]],
    ) -> Tuple[FrozenSet[Tuple[int, ...]], FrozenSet[Tuple[int, ...]], str]:
        new_simplices = (self.simplices - flip_from).union(flip_to)
        new_edges = _edges_from_simplices(new_simplices)
        new_key = _state_key_for_neighbor_mode(
            self.point_config_index,
            new_simplices,
            self.neighbor_mode,
        )
        return new_simplices, new_edges, new_key

    def find_changed_subcomplex(self, other: Self) -> FrozenSet[int]:
        if not isinstance(other, TriangulationState):
            raise TypeError("other must be a TriangulationState-compatible object.")
        if self.point_config_index != other.point_config_index:
            raise ValueError("Triangulations belong to different point configurations.")

        removed_simplices = self.simplices - other.simplices
        added_simplices = other.simplices - self.simplices

        if not removed_simplices and not added_simplices:
            raise ValueError("Triangulations are identical; no changed subcomplex exists.")
        if not removed_simplices or not added_simplices:
            raise ValueError("Triangulations are not adjacent under a valid CY flip.")

        changed_vertices = set()
        for simplex in removed_simplices:
            changed_vertices.update(simplex)
        for simplex in added_simplices:
            changed_vertices.update(simplex)
        return frozenset(changed_vertices)


def create_state_from_cy_triangulation(
    triangulation: CYToolsTriangulation,
    point_config_index: Optional[int] = None,
    add_origin: bool = True,
    neighbor_mode: str = "regular",
) -> CYTriangulationState:
    simplices = _normalize_simplices(triangulation.simplices())
    edges = _edges_from_simplices(simplices)
    vertices = _infer_vertices_from_cy_triangulation(triangulation)

    if add_origin:
        origin = [0] * len(vertices[0])
        if origin not in vertices:
            vertices = [origin] + vertices

    return CYTriangulationState(
        vertices=vertices,
        simplices=simplices,
        edges=edges,
        point_config_index=point_config_index,
        cy_triangulation=triangulation,
        neighbor_mode=neighbor_mode,
    )
