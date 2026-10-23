from __future__ import annotations

import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.cy_bounded_cache import BoundedLRU, retained_size
from core.cy_process_runtime import ManagedProcessPool
from core.cy_state_history import StateHistory
from mdp.cy_state_record import (
    CyStateRecord, CYPointConfiguration, evaluation_state_key,
    normalize_neighbor_mode as _normalize_neighbor_mode,
)
from mdp.cy_geometry_worker import execute_geometry_request

# An explicitly injected geometry factory remains supported for lightweight
# downstream tests. Production collection construction always uses workers.
Polytope = None

from mdp.cy_graph import (  # noqa: F401
    CanonicalAction,
    CanonicalSimplex,
    CanonicalSimplices,
    CYGraphNode,
    CYGraphTransition,
    CYRolloutCollection,
    CYStateExpansion,
    ExpandSummary,
    RandomRolloutStepResult,
    _sorted_simplices_tuple,
)


# ---------------------------------------------------------------------------
# Runtime state cache (was mdp/cy_cache.py)
# ---------------------------------------------------------------------------

@dataclass
class RuntimeStateCache:
    mode: str
    base_states: Dict[str, Any]
    max_hot_states: int
    hot_states: OrderedDict[str, Any]
    runtime_unique_keys: set[str]


def create_runtime_state_cache(
    *,
    mode: str,
    base_states: Mapping[str, Any],
    max_hot_states: int,
    max_bytes: int = 1024**3,
) -> RuntimeStateCache:
    return RuntimeStateCache(
        mode=str(mode),
        base_states=dict(base_states),
        max_hot_states=max(1, int(max_hot_states)),
        hot_states=BoundedLRU(max_bytes=max_bytes, max_entries=max_hot_states if mode == "lru" else None),
        runtime_unique_keys=set(),
    )


def get_state_from_runtime_cache(cache: RuntimeStateCache, key: str) -> Any | None:
    base_state = cache.base_states.get(key)
    if base_state is not None:
        return base_state

    hot_state = cache.hot_states.get(key)
    if hot_state is not None:
        cache.hot_states.move_to_end(key)
    return hot_state


def register_runtime_state(cache: RuntimeStateCache, state: Any) -> bool:
    key = state.key
    if key in cache.base_states:
        return False

    if hasattr(cache.runtime_unique_keys, "counts"):
        is_new_unique = cache.runtime_unique_keys.add(key)
    else:
        is_new_unique = key not in cache.runtime_unique_keys
        if is_new_unique:
            cache.runtime_unique_keys.add(key)

    if cache.mode == "none":
        return is_new_unique

    cache.hot_states[key] = state
    if cache.mode == "lru":
        while len(cache.hot_states) > cache.max_hot_states:
            cache.hot_states.popitem(last=False)
    return is_new_unique


def runtime_cache_total_unique_states(cache: RuntimeStateCache) -> int:
    return len(cache.base_states) + len(cache.runtime_unique_keys)


def runtime_cache_hot_size(cache: RuntimeStateCache) -> int:
    return len(cache.hot_states)


# ---------------------------------------------------------------------------
# Transition pool (was mdp/cy_transition_pool.py)
# ---------------------------------------------------------------------------

TransitionPool = ManagedProcessPool


def create_transition_pool(num_workers: int = 0, start_method: str = "spawn", **kwargs) -> TransitionPool:
    from mdp.cy_geometry_worker import clear_geometry_caches
    kwargs.setdefault("worker_reclaim", clear_geometry_caches)
    return TransitionPool(num_workers=num_workers, start_method=start_method, **kwargs)

def get_cy_shared_cache_sizes() -> Dict[str, int]:
    module = sys.modules.get("mdp.cy_triangulation_state")
    if module is None:
        return dict.fromkeys(("subcomplex", "neighbour_flip", "subcomplex_transition", "subcomplex_neighbour"), 0)
    CYTriangulationState = module.CYTriangulationState
    return {
        "subcomplex": len(CYTriangulationState._SHARED_SUBCOMPLEX_CACHE),
        "neighbour_flip": len(CYTriangulationState._SHARED_NEIGHBOUR_FLIP_CACHE),
        "subcomplex_transition": len(CYTriangulationState._SHARED_SUBCOMPLEX_TRANSITION_CACHE),
        "subcomplex_neighbour": len(CYTriangulationState._SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE),
    }


def prune_cy_shared_caches(
    *,
    keep_keys: Iterable[str] | None,
    max_entries: int | None,
) -> Dict[str, int]:
    module = sys.modules.get("mdp.cy_triangulation_state")
    if module is None:
        return get_cy_shared_cache_sizes()
    CYTriangulationState = module.CYTriangulationState
    max_entries_int = None if max_entries is None or int(max_entries) <= 0 else int(max_entries)
    keep_key_set = None if keep_keys is None else {str(key) for key in keep_keys}

    for cache_name in (
        "_SHARED_SUBCOMPLEX_CACHE",
        "_SHARED_NEIGHBOUR_FLIP_CACHE",
        "_SHARED_SUBCOMPLEX_TRANSITION_CACHE",
        "_SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE",
    ):
        cache_dict = getattr(CYTriangulationState, cache_name)
        if keep_key_set is not None:
            for key in list(cache_dict.keys()):
                if key not in keep_key_set:
                    cache_dict.pop(key, None)

        if max_entries_int is not None and len(cache_dict) > max_entries_int:
            overflow = len(cache_dict) - max_entries_int
            for key in list(cache_dict.keys())[:overflow]:
                cache_dict.pop(key, None)

    return get_cy_shared_cache_sizes()


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


def _is_target_triangulation_obj(triangulation: Any) -> bool:
    if triangulation is None:
        return False
    return _safe_bool_method_call(triangulation, "is_fine") and _safe_bool_method_call(triangulation, "is_regular")


def default_is_target_state(state: Any) -> bool:
    if hasattr(state, "is_target"):
        return bool(state.is_target)
    if hasattr(state, "is_frt"):
        return bool(state.is_frt)
    if hasattr(state, "is_frst") and bool(state.is_frst):
        return True
    return _is_target_triangulation_obj(getattr(state, "cy_triangulation", None))


def load_cy_sample_rows(dataset_path: str, max_rows: int | None = None) -> List[dict]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= int(max_rows):
                break

    if not rows:
        raise ValueError(f"No rows found in dataset: {path}")
    return rows


def _normalize_simplices_list(simplices: Iterable[Iterable[int]] | None) -> List[List[int]]:
    if simplices is None:
        return []
    return [[int(vertex) for vertex in simplex] for simplex in simplices]


def _iter_row_frst_simplices(row: Mapping[str, Any]) -> Iterable[List[List[int]]]:
    for frst_entry in row.get("frst_list", ()):
        frst_simplices = _normalize_simplices_list(frst_entry.get("simplices"))
        if frst_simplices:
            yield frst_simplices


def _iter_row_initial_simplices(row: Mapping[str, Any]) -> Iterable[List[List[int]]]:
    seen_simplices: set[CanonicalSimplices] = set()

    for frst_entry in row.get("frst_list", ()):
        for tri_entry in frst_entry.get("triangulation_list", ()):
            tri_simplices = _normalize_simplices_list(tri_entry.get("simplices"))
            canonical_simplices = _sorted_simplices_tuple(tri_simplices)
            if not canonical_simplices or canonical_simplices in seen_simplices:
                continue
            seen_simplices.add(canonical_simplices)
            yield [list(simplex) for simplex in canonical_simplices]

    for tri_entry in row.get("non_fine_triangulation_list", ()):
        tri_simplices = _normalize_simplices_list(
            tri_entry.get("simplices", tri_entry.get("signature"))
        )
        canonical_simplices = _sorted_simplices_tuple(tri_simplices)
        if not canonical_simplices or canonical_simplices in seen_simplices:
            continue
        seen_simplices.add(canonical_simplices)
        yield [list(simplex) for simplex in canonical_simplices]


def _build_cy_rollout_collection_inline(
    rows: Sequence[dict],
    *,
    include_points_interior_to_facets: bool,
    neighbor_mode: str = "regular",
) -> CYRolloutCollection:
    from mdp.cy_triangulation_state import CYTriangulationState
    if Polytope is None:
        raise ModuleNotFoundError(
            "cytools is required for CY rollout. Activate the 'sage' environment."
        )

    resolved_neighbor_mode = _normalize_neighbor_mode(neighbor_mode)
    if resolved_neighbor_mode == "two_neighbors" and include_points_interior_to_facets:
        raise ValueError(
            "neighbor_mode='two_neighbors' requires "
            "include_points_interior_to_facets=False."
        )

    base_states: Dict[str, CYTriangulationState] = {}
    initial_states_by_key: Dict[str, CYTriangulationState] = {}
    polytope_by_index: Dict[int, object] = {}
    vertices_by_polytope: Dict[int, List[List[int]]] = {}
    polytope_indices: List[int] = []

    for row in rows:
        polytope_index = int(row["polytope_index"])
        vertices = [[int(coord) for coord in point] for point in row["vertices"]]
        polytope = Polytope(vertices)
        polytope_by_index[polytope_index] = polytope
        vertices_by_polytope[polytope_index] = vertices
        polytope_indices.append(polytope_index)

        for frst_simplices in _iter_row_frst_simplices(row):
            frst_tri = polytope.triangulate(
                simplices=[list(simplex) for simplex in frst_simplices],
                include_points_interior_to_facets=include_points_interior_to_facets,
                check_input_simplices=False,
            )
            frst_state = CYTriangulationState(
                vertices=vertices,
                point_config_index=polytope_index,
                simplices=frst_simplices,
                cy_triangulation=frst_tri,
                is_frst=True if resolved_neighbor_mode == "regular" else None,
                neighbor_mode=resolved_neighbor_mode,
            )
            if resolved_neighbor_mode == "two_neighbors" and not frst_state.is_frst:
                raise ValueError(
                    "two_neighbors initial-state validation failed: dataset FRST entry "
                    f"for polytope {polytope_index} is not fine, star, and regular."
                )
            base_states.setdefault(frst_state.key, frst_state)
            if resolved_neighbor_mode == "two_neighbors":
                initial_states_by_key.setdefault(frst_state.key, frst_state)

        if resolved_neighbor_mode == "two_neighbors":
            continue
        for tri_simplices in _iter_row_initial_simplices(row):
            tri = polytope.triangulate(
                simplices=[list(simplex) for simplex in tri_simplices],
                include_points_interior_to_facets=include_points_interior_to_facets,
                check_input_simplices=False,
            )
            state = CYTriangulationState(
                vertices=vertices,
                point_config_index=polytope_index,
                simplices=tri_simplices,
                cy_triangulation=tri,
                neighbor_mode=resolved_neighbor_mode,
            )
            cached_state = base_states.setdefault(state.key, state)
            initial_states_by_key.setdefault(cached_state.key, cached_state)

    initial_states = list(initial_states_by_key.values())
    if not initial_states:
        if resolved_neighbor_mode == "two_neighbors":
            raise ValueError("No validated FRST initial states were found in the dataset.")
        raise ValueError("No non-fine initial states were found in the dataset.")

    return CYRolloutCollection(
        base_states=base_states,
        initial_states=initial_states,
        polytope_by_index=polytope_by_index,
        vertices_by_polytope=vertices_by_polytope,
        polytope_indices=sorted(set(polytope_indices)),
    )


def build_cy_rollout_collection(
    rows: Sequence[dict],
    *,
    include_points_interior_to_facets: bool,
    neighbor_mode: str = "regular",
    transition_pool: Any = None,
    two_face_state: bool = False,
    include_two_face_metadata: bool = False,
) -> CYRolloutCollection:
    mode = _normalize_neighbor_mode(neighbor_mode)
    if mode == "two_neighbors" and include_points_interior_to_facets:
        raise ValueError("neighbor_mode='two_neighbors' requires include_points_interior_to_facets=False.")
    if Polytope is not None:
        if two_face_state or include_two_face_metadata:
            raise ValueError("two_face_state / include_two_face_metadata requires managed geometry workers.")
        return _build_cy_rollout_collection_inline(rows, include_points_interior_to_facets=include_points_interior_to_facets, neighbor_mode=mode)
    owned = transition_pool is None
    pool = transition_pool or create_transition_pool(num_workers=1)
    base_states, initial_states, configurations = {}, {}, {}
    try:
        requests = ({"operation": "build_collection", "row": row,
                     "include_points_interior_to_facets": include_points_interior_to_facets,
                     "neighbor_mode": mode, "two_face_state": two_face_state,
                     "include_two_face_metadata": include_two_face_metadata} for row in rows)
        for result in pool.imap(execute_geometry_request, requests):
            configuration = result["configuration"]
            if configuration.index in configurations and configurations[configuration.index] != configuration:
                raise ValueError(f"Conflicting point configuration {configuration.index}.")
            configurations[configuration.index] = configuration
            for state in result["base_states"]:
                state.configuration = configuration
                base_states.setdefault(state.key, state)
            for key in result["initial_keys"]:
                initial_states.setdefault(key, base_states[key])
        if not initial_states:
            raise ValueError("No validated FRST initial states were found in the dataset." if mode == "two_neighbors" else "No non-fine initial states were found in the dataset.")
        return CYRolloutCollection(
            base_states=base_states, initial_states=list(initial_states.values()),
            polytope_by_index=configurations,
            vertices_by_polytope={i: c.input_vertices for i, c in configurations.items()},
            polytope_indices=sorted(configurations), transition_pool=pool,
            owns_transition_pool=owned,
        )
    except BaseException:
        if owned:
            pool.shutdown()
        raise


def _expand_cy_state_worker(payload: Tuple[Any, bool]) -> CYStateExpansion:
    state, objective_mode = payload
    simplices = _sorted_simplices_tuple(getattr(state, "simplices", ()))
    if not objective_mode and (default_is_target_state(state) or len(simplices) <= 1):
        return CYStateExpansion(
            key=str(state.key),
            point_config_index=int(state.point_config_index),
            simplices=simplices,
            candidate_actions=tuple(),
            ambiguous_actions=frozenset(),
            transitions=tuple(),
        )

    if not bool(getattr(state, "actions_ready", False)):
        state.find_available_actions()

    all_actions = tuple(tuple(int(v) for v in action) for action in state.get_available_subcomplex_actions())
    ambiguous_actions = frozenset(
        tuple(int(v) for v in action) for action in getattr(state, "ambiguous_subcomplex_actions", frozenset())
    )
    candidate_actions = tuple(action for action in all_actions if action not in ambiguous_actions)

    transitions: List[Tuple[CanonicalAction, CYGraphTransition]] = []
    for action in candidate_actions:
        next_simplices, _next_edges, next_key = state.get_transition_output_from_subcomplex_action(action)
        next_tri = None
        get_next_tri = getattr(state, "get_next_cy_triangulation_from_subcomplex_action", None)
        if callable(get_next_tri):
            next_tri = get_next_tri(action)
        transitions.append(
            (
                action,
                CYGraphTransition(
                    next_key=str(next_key),
                    next_simplices=_sorted_simplices_tuple(next_simplices),
                    next_is_target=_is_target_triangulation_obj(next_tri) if next_tri is not None else None,
                ),
            )
        )

    return CYStateExpansion(
        key=str(state.key),
        point_config_index=int(state.point_config_index),
        simplices=simplices,
        candidate_actions=candidate_actions,
        ambiguous_actions=ambiguous_actions,
        transitions=tuple(transitions),
    )


class CYRandomRolloutEngine:
    def __init__(
        self,
        *,
        collection: CYRolloutCollection | None = None,
        base_states: Mapping[str, Any] | None = None,
        initial_states: Sequence[Any] | None = None,
        polytope_by_index: Mapping[int, Any] | None = None,
        vertices_by_polytope: Mapping[int, List[List[int]]] | None = None,
        include_points_interior_to_facets: bool = True,
        state_cache_mode: str = "lru",
        max_hot_states: int = 100000,
        state_factory: Optional[Callable[[int, CanonicalSimplices], Any]] = None,
        is_target_state_fn: Optional[Callable[[Any], bool]] = None,
        reward_function: Optional[Callable[[Any, Any], float]] = None,
        neighbor_mode: str = "regular",
        cache_budget_bytes: int = 4 * 1024**3,
        history_path: str | None = None,
        transition_pool: Any = None,
        action_order: str = "canonical",
        two_face_state: bool = False,
    ):
        if type(two_face_state) is not bool:
            raise ValueError("two_face_state must be a boolean.")
        self.two_face_state = two_face_state
        if collection is not None:
            base_states = collection.base_states
            initial_states = collection.initial_states
            polytope_by_index = collection.polytope_by_index
            vertices_by_polytope = collection.vertices_by_polytope

        self.neighbor_mode = _normalize_neighbor_mode(neighbor_mode)
        if self.neighbor_mode == "two_neighbors" and include_points_interior_to_facets:
            raise ValueError(
                "neighbor_mode='two_neighbors' requires "
                "include_points_interior_to_facets=False."
            )
        self.base_states = dict(base_states or {})
        self.initial_states = list(initial_states or [])
        self.polytope_by_index = dict(polytope_by_index or {})
        self.vertices_by_polytope = dict(vertices_by_polytope or {})
        self.include_points_interior_to_facets = bool(include_points_interior_to_facets)
        self.state_cache = create_runtime_state_cache(
            mode=state_cache_mode,
            base_states=self.base_states,
            max_hot_states=max_hot_states,
            max_bytes=max(0, int(cache_budget_bytes)) // 4,
        )
        self.state_factory = state_factory or self._default_state_factory
        self.is_target_state_fn = is_target_state_fn or default_is_target_state
        self.reward_function = reward_function
        if action_order not in ("canonical", "native"):
            raise ValueError("action_order must be 'canonical' or 'native'.")
        self.action_order = action_order
        self.transition_pool = transition_pool or getattr(collection, "transition_pool", None)
        self._owned_collection = collection if getattr(collection, "owns_transition_pool", False) else None
        self._managed = any(isinstance(state, CyStateRecord) for state in self.base_states.values())
        history_cache_bytes = min(8 * 1024**2, max(0, int(cache_budget_bytes)) // 64)
        objective_cache_bytes = min(16 * 1024**2, max(0, int(cache_budget_bytes)) // 16)
        self.history = StateHistory(history_path, cache_bytes=history_cache_bytes) if self._managed else None
        if self.history is not None and len(self.history.keys("discovered")):
            self.history.close()
            self.history = None
            raise ValueError("Rollout history is run-local; use a fresh history_path for a new engine. "
                             "Existing checkpoints restore policy weights, not rollout history.")
        if self.history is not None:
            self.state_cache.runtime_unique_keys = self.history.keys("materialized")
        self._discovered_keys = self.history.keys("discovered") if self.history else set()
        self._expanded_keys = self.history.keys("expanded") if self.history else set()
        self._cumulative_edges = 0
        self._base_edges = 0
        self._polytope_totals = {}
        self._graph_max_bytes = (max(0, int(cache_budget_bytes)) * 3 // 4
                                 - history_cache_bytes - objective_cache_bytes)
        self._graph_max_entries = max(1, int(max_hot_states))
        self._graph_bytes = 0
        self._node_sizes = {}
        self._graph_evictions = 0
        self._objective_cache = BoundedLRU(max_bytes=objective_cache_bytes, max_entries=8192)
        self._active_keys = set()

        self.nodes_by_key: Dict[str, CYGraphNode] = OrderedDict()
        self.graph_by_polytope: Dict[int, Dict[str, CYGraphNode]] = {}
        for state in self.base_states.values():
            if isinstance(state, CyStateRecord):
                state.bind_objective_provider(self.objective_value)
            self._register_state_node(state)
        self.prune_runtime_caches()

    def _default_state_factory(self, point_config_index: int, simplices: CanonicalSimplices) -> Any:
        if point_config_index not in self.polytope_by_index:
            raise KeyError(f"Unknown polytope index {point_config_index}")

        polytope = self.polytope_by_index[point_config_index]
        if isinstance(polytope, CYPointConfiguration):
            state = CyStateRecord(polytope, frozenset(simplices), self.neighbor_mode)
            state.bind_objective_provider(self.objective_value)
            return state
        from mdp.cy_triangulation_state import CYTriangulationState
        triangulation = polytope.triangulate(
            simplices=[list(simplex) for simplex in simplices],
            include_points_interior_to_facets=self.include_points_interior_to_facets,
            check_input_simplices=False,
        )
        return CYTriangulationState(
            vertices=self.vertices_by_polytope[point_config_index],
            point_config_index=point_config_index,
            simplices=simplices,
            cy_triangulation=triangulation,
            neighbor_mode=self.neighbor_mode,
        )

    def _register_node(self, *, key: str, point_config_index: int, simplices: CanonicalSimplices) -> Tuple[CYGraphNode, bool]:
        node = self.nodes_by_key.get(key)
        if node is not None:
            self.nodes_by_key.move_to_end(key)
            return node, False

        node = CYGraphNode(
            key=str(key),
            point_config_index=int(point_config_index),
            simplices=simplices,
        )
        self.nodes_by_key[node.key] = node
        self.graph_by_polytope.setdefault(node.point_config_index, {})[node.key] = node
        if self.history is not None:
            is_new = self._discovered_keys.add(node.key)
        else:
            is_new = node.key not in self._discovered_keys
            self._discovered_keys.add(node.key)
        if is_new:
            self._polytope_totals.setdefault(node.point_config_index, {"nodes": 0, "edges": 0, "expanded_nodes": 0})["nodes"] += 1
        self._account_node(node)
        return node, is_new

    def _account_node(self, node):
        size = retained_size(node)
        self._graph_bytes += size - self._node_sizes.get(node.key, 0)
        self._node_sizes[node.key] = size

    def prune_runtime_caches(self, keep_keys=(), *, pressure=False):
        protected = set(keep_keys) | self._active_keys
        limit_bytes = 0 if pressure else self._graph_max_bytes
        limit_entries = 0 if pressure else self._graph_max_entries
        examined = 0
        while self.nodes_by_key and (self._graph_bytes > limit_bytes or len(self.nodes_by_key) > limit_entries):
            if examined >= len(self.nodes_by_key):
                break
            key = next(iter(self.nodes_by_key))
            if key in protected:
                self.nodes_by_key.move_to_end(key)
                examined += 1
                continue
            node = self.nodes_by_key.pop(key)
            self._graph_bytes -= self._node_sizes.pop(key, 0)
            group = self.graph_by_polytope[node.point_config_index]
            group.pop(key, None)
            if not group:
                self.graph_by_polytope.pop(node.point_config_index, None)
            self._graph_evictions += 1
        if pressure:
            self.state_cache.hot_states.clear()
            self._objective_cache.clear()

    def memory_stats(self):
        return {"resident_graph_nodes": len(self.nodes_by_key),
                "resident_graph_edges": sum(len(node.transitions) for node in self.nodes_by_key.values()),
                "resident_graph_bytes": self._graph_bytes,
                "graph_evictions": self._graph_evictions,
                "hot_state_bytes": self.state_cache.hot_states.bytes,
                "objective_cache_bytes": self._objective_cache.bytes}

    def release_active_states(self):
        self._active_keys.clear()
        self.prune_runtime_caches()

    def objective_value(self, state, name):
        if self.two_face_state and name != "max_kcup":
            raise ValueError("two_face_state currently supports only max_kcup.")
        if name in ("min_tri", "max_tri"):
            return float(len(state.simplices))
        cache_key = (name, evaluation_state_key(state, self.two_face_state))
        cached = self._objective_cache.get(cache_key)
        if cached is not None:
            return cached
        pool = self.transition_pool
        if pool is None:
            raise RuntimeError("Geometry objective requires a managed transition pool.")
        result = next(pool.imap(execute_geometry_request, [{"operation": "objective",
            "configuration": state.configuration, "state": state.to_payload(), "reward_name": name,
            "two_face_state": self.two_face_state}]))
        self._objective_cache[cache_key] = float(result)
        return float(result)

    def objective_values(self, states, name):
        """Yield an ordered logical batch using the existing bounded worker pool.

        The caller owns logical query accounting. In-flight duplicate requests
        share physical work only when objective caching is enabled. Consume or
        close this iterator before submitting any other geometry work.
        """
        states = list(states)
        if self.two_face_state and name != "max_kcup":
            raise ValueError("two_face_state currently supports only max_kcup.")
        if name in ("min_tri", "max_tri"):
            yield from (float(len(state.simplices)) for state in states)
            return
        requests, positions, cached_values, pending = [], [], {}, {}
        cache_enabled = self._objective_cache.max_bytes > 0 and self._objective_cache.max_entries != 0
        for index, state in enumerate(states):
            key = (name, evaluation_state_key(state, self.two_face_state))
            cached = self._objective_cache.get(key)
            if cached is not None:
                cached_values[index] = cached
                positions.append(None)
                continue
            if not cache_enabled or key not in pending:
                pending[key] = len(requests)
                requests.append({"operation": "objective", "configuration": state.configuration,
                                 "state": state.to_payload(), "reward_name": name,
                                 "two_face_state": self.two_face_state})
            positions.append(pending[key])
        if requests and self.transition_pool is None:
            raise RuntimeError("Geometry objective requires a managed transition pool.")
        outputs = (iter(self.transition_pool.imap(execute_geometry_request, requests, chunksize=1))
                   if requests else iter(()))
        resolved = {}
        try:
            for index, (state, position) in enumerate(zip(states, positions)):
                if position is None:
                    yield cached_values[index]
                else:
                    if position not in resolved:
                        resolved[position] = float(next(outputs))
                        self._objective_cache[(name, evaluation_state_key(state, self.two_face_state))] = resolved[position]
                    yield resolved[position]
        finally:
            close = getattr(outputs, "close", None)
            if close is not None:
                close()

    def close(self):
        for state in self.base_states.values():
            if isinstance(state, CyStateRecord):
                state.bind_objective_provider(None)
        self.state_cache.hot_states.clear()
        if self.history is not None:
            self.history.close()
            self.history = None
        if self._owned_collection is not None:
            self._owned_collection.close()
            self._owned_collection = None

    def _register_state_node(self, state: Any) -> Tuple[CYGraphNode, bool]:
        return self._register_node(
            key=str(state.key),
            point_config_index=int(state.point_config_index),
            simplices=_sorted_simplices_tuple(getattr(state, "simplices", ())),
        )

    def _store_expansion(self, expansion: CYStateExpansion) -> int:
        node, _ = self._register_node(
            key=expansion.key,
            point_config_index=expansion.point_config_index,
            simplices=expansion.simplices,
        )
        node.candidate_actions = expansion.candidate_actions
        node.ambiguous_actions = expansion.ambiguous_actions
        node.transitions = {action: transition for action, transition in expansion.transitions}
        node.expanded = True
        first_expansion = (self._expanded_keys.add(node.key) if self.history is not None
                           else node.key not in self._expanded_keys)
        if first_expansion:
            if self.history is None:
                self._expanded_keys.add(node.key)
            self._cumulative_edges += len(node.transitions)
            if node.key in self.base_states:
                self._base_edges += len(node.transitions)
            totals = self._polytope_totals.setdefault(node.point_config_index, {"nodes": 0, "edges": 0, "expanded_nodes": 0})
            totals["edges"] += len(node.transitions)
            totals["expanded_nodes"] += 1
        self._account_node(node)

        discovered = 0
        for transition in node.transitions.values():
            if self._managed:
                is_new = self._discovered_keys.add(transition.next_key)
                if is_new:
                    self._polytope_totals[expansion.point_config_index]["nodes"] += 1
            else:
                _, is_new = self._register_node(
                    key=transition.next_key,
                    point_config_index=expansion.point_config_index,
                    simplices=transition.simplices_from(expansion.simplices),
                )
            discovered += int(is_new)
        return discovered

    def get_state(self, key: str) -> Any | None:
        return get_state_from_runtime_cache(self.state_cache, key)

    def materialize_state(self, key: str) -> Any:
        cached = self.get_state(key)
        if cached is not None:
            return cached

        node = self.nodes_by_key.get(key)
        if node is None:
            raise KeyError(f"State key {key} is not present in the rollout graph.")

        state = self.state_factory(node.point_config_index, node.simplices)
        register_runtime_state(self.state_cache, state)
        return state

    def materialize_transition(self, source, transition):
        cached = self.get_state(transition.next_key)
        if cached is not None:
            return cached
        simplices = transition.simplices_from(source.simplices)
        state = self.state_factory(source.point_config_index, simplices)
        if isinstance(state, CyStateRecord):
            state.is_target = bool(transition.next_is_target)
            state.is_frst = bool(transition.next_is_frst)
            # Charge selected-state edges before admitting the materialized state.
            state.edges
        self._register_state_node(state)
        register_runtime_state(self.state_cache, state)
        return state

    def expand_states(
        self,
        states: Sequence[Any],
        *,
        use_multiprocessing: bool = False,
        transition_pool: Any = None,
        transition_mp_chunksize: int = 32,
        transition_mp_min_batch: int = 32,
    ) -> ExpandSummary:
        pool = transition_pool or self.transition_pool
        if pool is not None and hasattr(pool, "check_memory"):
            pool.check_memory()
        self._active_keys = {str(state.key) for state in states}
        unique_unexpanded: Dict[str, Any] = {}
        for state in states:
            self._register_state_node(state)
            if not self.nodes_by_key[str(state.key)].expanded:
                unique_unexpanded.setdefault(str(state.key), state)

        if not unique_unexpanded:
            self.prune_runtime_caches()
            return ExpandSummary(expanded_count=0, discovered_count=0, used_multiprocessing=False)

        pending_states = list(unique_unexpanded.values())
        if self._managed:
            if pool is None:
                raise RuntimeError("Managed states require a transition pool, including serial rollouts.")
            self.transition_pool = pool
            expansion_payloads = ({"operation": "expand", "configuration": state.configuration,
                "state": state.to_payload(), "objective_mode": self.reward_function is not None,
                "action_order": self.action_order}
                for state in pending_states)
            expansion_outputs = pool.imap(execute_geometry_request, expansion_payloads, chunksize=1)
            use_mp = len(getattr(pool, "worker_pids", ())) > 1
        else:
            expansion_payloads = ((state, self.reward_function is not None) for state in pending_states)
            use_mp = bool(use_multiprocessing) and pool is not None and len(pending_states) >= max(1, int(transition_mp_min_batch))
            if use_mp:
                # Infrastructure errors must not leave a pool alive and repeat its
                # in-flight work inside the trainer.
                mapper = getattr(pool, "imap", pool.map)
                expansion_outputs = mapper(_expand_cy_state_worker, expansion_payloads, chunksize=1)
            else:
                expansion_outputs = map(_expand_cy_state_worker, expansion_payloads)

        discovered = expanded_count = 0
        for expansion in expansion_outputs:
            discovered += self._store_expansion(expansion)
            expanded_count += 1
            self.prune_runtime_caches()
        return ExpandSummary(
            expanded_count=expanded_count,
            discovered_count=discovered,
            used_multiprocessing=use_mp,
        )

    def candidate_actions_for_states(
        self,
        states: Sequence[Any],
        **expand_kwargs: Any,
    ) -> Tuple[List[Tuple[CanonicalAction, ...]], ExpandSummary]:
        summary = self.expand_states(states, **expand_kwargs)
        action_lists = [self.nodes_by_key[str(state.key)].candidate_actions for state in states]
        return action_lists, summary

    def filter_actionable_initial_states(
        self,
        states: Optional[Sequence[Any]] = None,
        **expand_kwargs: Any,
    ) -> List[Any]:
        source_states = self.initial_states if states is None else list(states)
        actionable = []
        chunk_size = min(128, self._graph_max_entries)
        for start in range(0, len(source_states), chunk_size):
            chunk = source_states[start:start + chunk_size]
            action_lists, _summary = self.candidate_actions_for_states(chunk, **expand_kwargs)
            actionable.extend(state for state, actions in zip(chunk, action_lists) if actions)
        self._active_keys.clear()
        self.prune_runtime_caches()
        return actionable

    def sample_initial_states(
        self,
        num_states: int,
        *,
        rng: np.random.Generator,
        initial_state_pool: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        pool = self.initial_states if initial_state_pool is None else list(initial_state_pool)
        if not pool:
            raise ValueError("Cannot sample from an empty initial state pool.")
        indices = rng.integers(0, len(pool), size=int(num_states))
        return [pool[int(idx)] for idx in indices]

    def graph_node_count(self) -> int:
        return len(self._discovered_keys)

    def runtime_graph_node_count(self) -> int:
        return self.graph_node_count() - len(self.base_states)

    def graph_edge_count(self) -> int:
        return self._cumulative_edges

    def runtime_graph_edge_count(self) -> int:
        return self._cumulative_edges - self._base_edges

    def graph_stats_by_polytope(self) -> Dict[int, Dict[str, int]]:
        return {index: dict(values) for index, values in self._polytope_totals.items()}

    def compact_runtime_graph_to_base(self) -> Dict[str, int]:
        base_keys = set(self.base_states.keys())
        removed_nodes = 0
        removed_edges = 0

        for key, node in list(self.nodes_by_key.items()):
            if key in base_keys:
                continue
            removed_nodes += 1
            removed_edges += len(node.transitions)
            self.nodes_by_key.pop(key, None)
            self._graph_bytes -= self._node_sizes.pop(key, 0)
            poly_graph = self.graph_by_polytope.get(node.point_config_index)
            if poly_graph is not None:
                poly_graph.pop(key, None)
                if not poly_graph:
                    self.graph_by_polytope.pop(node.point_config_index, None)

        self.state_cache.hot_states.clear()
        # Discovery/visitation history deliberately survives graph compaction.
        for state in self.base_states.values():
            self._register_state_node(state)
        self.prune_runtime_caches()

        return {
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "remaining_nodes": len(self.nodes_by_key),
            "remaining_runtime_nodes": sum(key not in self.base_states for key in self.nodes_by_key),
            "remaining_edges": sum(len(node.transitions) for node in self.nodes_by_key.values()),
            "remaining_runtime_edges": sum(len(node.transitions) for key, node in self.nodes_by_key.items() if key not in self.base_states),
        }

    def rollout_step(
        self,
        states: Sequence[Any],
        *,
        rng: np.random.Generator,
        initial_state_pool: Sequence[Any],
        use_multiprocessing: bool = False,
        transition_pool: Any = None,
        transition_mp_chunksize: int = 32,
        transition_mp_min_batch: int = 32,
    ) -> RandomRolloutStepResult:
        current_states = list(states)
        action_lists, expand_summary = self.candidate_actions_for_states(
            current_states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )

        transitioned_states: List[Any] = []
        next_states: List[Any] = list(current_states)
        rewards = [0.0 for _ in current_states]
        dones = [False for _ in current_states]
        chosen_actions: List[Optional[CanonicalAction]] = []
        terminal_reasons = ["continue" for _ in current_states]
        frt_hits = 0
        collapsed_hits = 0
        dead_end_hits = 0

        unique_nonterminal_next_keys: Dict[str, None] = {}
        objective_mode = self.reward_function is not None
        for idx, (state, action_candidates) in enumerate(zip(current_states, action_lists)):
            if len(action_candidates) == 0:
                transitioned_states.append(state)
                dones[idx] = True
                terminal_reasons[idx] = "dead_end_current"
                chosen_actions.append(None)
                dead_end_hits += 1
                continue

            action_idx = int(rng.integers(0, len(action_candidates)))
            action = action_candidates[action_idx]
            chosen_actions.append(action)

            transition = self.nodes_by_key[str(state.key)].transitions[action]
            if not objective_mode and transition.next_is_target is True:
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
                transitioned_states.append(state)
                continue

            if not objective_mode and transition.num_next_simplices(state.simplices) <= 1:
                rewards[idx] = -1.0
                dones[idx] = True
                terminal_reasons[idx] = "single_simplex"
                collapsed_hits += 1
                transitioned_states.append(state)
                continue

            next_state = self.materialize_transition(state, transition)
            transitioned_states.append(next_state)
            next_states[idx] = next_state
            if objective_mode:
                rewards[idx] = float(self.reward_function(state, next_state))
            elif self.is_target_state_fn(next_state):
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
                continue
            unique_nonterminal_next_keys.setdefault(str(next_state.key), next_state)

        nonterminal_next_states = list(unique_nonterminal_next_keys.values())
        next_expand_summary = self.expand_states(
            nonterminal_next_states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )

        for idx, transitioned_state in enumerate(next_states):
            if dones[idx]:
                continue
            if len(self.nodes_by_key[str(transitioned_state.key)].candidate_actions) == 0:
                dones[idx] = True
                terminal_reasons[idx] = "dead_end_next"
                dead_end_hits += 1

        reset_indices = [idx for idx, done in enumerate(dones) if done]
        if reset_indices:
            reset_states = self.sample_initial_states(
                len(reset_indices),
                rng=rng,
                initial_state_pool=initial_state_pool,
            )
            for idx, reset_state in zip(reset_indices, reset_states):
                next_states[idx] = reset_state

        return RandomRolloutStepResult(
            input_states=current_states,
            transitioned_states=transitioned_states,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
            chosen_actions=chosen_actions,
            terminal_reasons=terminal_reasons,
            reset_count=len(reset_indices),
            frt_hits=frt_hits,
            collapsed_hits=collapsed_hits,
            dead_end_hits=dead_end_hits,
            expanded_states=expand_summary.expanded_count + next_expand_summary.expanded_count,
            discovered_states=expand_summary.discovered_count + next_expand_summary.discovered_count,
            used_multiprocessing=expand_summary.used_multiprocessing or next_expand_summary.used_multiprocessing,
            candidate_actions=action_lists,
        )


def get_rollout_memory_stats(engine: CYRandomRolloutEngine) -> Dict[str, int]:
    shared_sizes = get_cy_shared_cache_sizes()
    return {
        **engine.memory_stats(),
        "graph_nodes": engine.graph_node_count(),
        "runtime_graph_nodes": engine.runtime_graph_node_count(),
        "graph_edges": engine.graph_edge_count(),
        "runtime_graph_edges": engine.runtime_graph_edge_count(),
        "cached_states": runtime_cache_total_unique_states(engine.state_cache),
        "hot_cache": runtime_cache_hot_size(engine.state_cache),
        "shared_subcomplex": shared_sizes["subcomplex"],
        "shared_neighbour_flip": shared_sizes["neighbour_flip"],
        "shared_subcomplex_transition": shared_sizes["subcomplex_transition"],
        "shared_subcomplex_neighbour": shared_sizes["subcomplex_neighbour"],
    }


def maybe_compact_rollout_memory(
    engine: CYRandomRolloutEngine,
    *,
    graph_max_nodes: int | None,
    shared_cache_max_entries: int | None,
) -> Dict[str, Any]:
    before = get_rollout_memory_stats(engine)
    compacted_graph = False
    if graph_max_nodes is not None and int(graph_max_nodes) > 0:
        compacted_graph = sum(key not in engine.base_states for key in engine.nodes_by_key) > int(graph_max_nodes)
        if compacted_graph:
            engine.compact_runtime_graph_to_base()

    pruned_shared = False
    if shared_cache_max_entries is not None and int(shared_cache_max_entries) > 0:
        current_shared_sizes = get_cy_shared_cache_sizes()
        pruned_shared = any(size > int(shared_cache_max_entries) for size in current_shared_sizes.values())
        if pruned_shared:
            prune_cy_shared_caches(
                keep_keys=engine.base_states.keys(),
                max_entries=int(shared_cache_max_entries),
            )

    after = get_rollout_memory_stats(engine)
    return {
        "compacted_graph": compacted_graph,
        "pruned_shared": pruned_shared,
        "before": before,
        "after": after,
    }
