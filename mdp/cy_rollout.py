import json
import multiprocessing as mp
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    # Import cytools before CYTriangulationState. The reverse order can segfault
    # in the Sage/FLINT stack when Polytope(...) is constructed later.
    from cytools.polytope import Polytope
    from core.cytools_config import configure_cytools
    configure_cytools()
except ModuleNotFoundError:
    Polytope = None

from mdp.cy_triangulation_state import CYTriangulationState

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
) -> RuntimeStateCache:
    return RuntimeStateCache(
        mode=str(mode),
        base_states=dict(base_states),
        max_hot_states=max(1, int(max_hot_states)),
        hot_states=OrderedDict(),
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

    is_new_unique = key not in cache.runtime_unique_keys
    if is_new_unique:
        cache.runtime_unique_keys.add(key)

    if cache.mode == "none":
        return is_new_unique

    cache.hot_states[key] = state
    cache.hot_states.move_to_end(key)
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

class TransitionPool:
    def __init__(self, num_workers: int = 0, start_method: str = "spawn"):
        resolved_workers = num_workers if num_workers and num_workers > 0 else (mp.cpu_count() or 1)
        mp_context = mp.get_context(start_method)
        self._executor = ProcessPoolExecutor(max_workers=resolved_workers, mp_context=mp_context)

    def map(self, func, iterable, chunksize: int = 1):
        return list(self._executor.map(func, iterable, chunksize=chunksize))

    def shutdown(self):
        self._executor.shutdown(wait=True)


def create_transition_pool(num_workers: int = 0, start_method: str = "spawn") -> TransitionPool:
    return TransitionPool(num_workers=num_workers, start_method=start_method)

_CY_ROLLOUT_MP_DISABLED = False


def get_cy_shared_cache_sizes() -> Dict[str, int]:
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


def build_cy_rollout_collection(
    rows: Sequence[dict],
    *,
    include_points_interior_to_facets: bool,
) -> CYRolloutCollection:
    if Polytope is None:
        raise ModuleNotFoundError(
            "cytools is required for CY rollout. Activate the 'sage' environment."
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
                is_frst=True,
            )
            base_states.setdefault(frst_state.key, frst_state)

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
            )
            cached_state = base_states.setdefault(state.key, state)
            initial_states_by_key.setdefault(cached_state.key, cached_state)

    initial_states = list(initial_states_by_key.values())
    if not initial_states:
        raise ValueError("No non-fine initial states were found in the dataset.")

    return CYRolloutCollection(
        base_states=base_states,
        initial_states=initial_states,
        polytope_by_index=polytope_by_index,
        vertices_by_polytope=vertices_by_polytope,
        polytope_indices=sorted(set(polytope_indices)),
    )


def _expand_cy_state_worker(state: Any) -> CYStateExpansion:
    simplices = _sorted_simplices_tuple(getattr(state, "simplices", ()))
    if default_is_target_state(state) or len(simplices) <= 1:
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
    ):
        if collection is not None:
            base_states = collection.base_states
            initial_states = collection.initial_states
            polytope_by_index = collection.polytope_by_index
            vertices_by_polytope = collection.vertices_by_polytope

        self.base_states = dict(base_states or {})
        self.initial_states = list(initial_states or [])
        self.polytope_by_index = dict(polytope_by_index or {})
        self.vertices_by_polytope = dict(vertices_by_polytope or {})
        self.include_points_interior_to_facets = bool(include_points_interior_to_facets)
        self.state_cache = create_runtime_state_cache(
            mode=state_cache_mode,
            base_states=self.base_states,
            max_hot_states=max_hot_states,
        )
        self.state_factory = state_factory or self._default_state_factory
        self.is_target_state_fn = is_target_state_fn or default_is_target_state

        self.nodes_by_key: Dict[str, CYGraphNode] = {}
        self.graph_by_polytope: Dict[int, Dict[str, CYGraphNode]] = {}
        for state in self.base_states.values():
            self._register_state_node(state)

    def _default_state_factory(self, point_config_index: int, simplices: CanonicalSimplices) -> CYTriangulationState:
        if Polytope is None:
            raise ModuleNotFoundError(
                "cytools is required for CY rollout. Activate the 'sage' environment."
            )
        if point_config_index not in self.polytope_by_index:
            raise KeyError(f"Unknown polytope index {point_config_index}")

        polytope = self.polytope_by_index[point_config_index]
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
        )

    def _register_node(self, *, key: str, point_config_index: int, simplices: CanonicalSimplices) -> Tuple[CYGraphNode, bool]:
        node = self.nodes_by_key.get(key)
        if node is not None:
            return node, False

        node = CYGraphNode(
            key=str(key),
            point_config_index=int(point_config_index),
            simplices=simplices,
        )
        self.nodes_by_key[node.key] = node
        self.graph_by_polytope.setdefault(node.point_config_index, {})[node.key] = node
        return node, True

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

        discovered = 0
        for transition in node.transitions.values():
            _, is_new = self._register_node(
                key=transition.next_key,
                point_config_index=expansion.point_config_index,
                simplices=transition.next_simplices,
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

    def expand_states(
        self,
        states: Sequence[Any],
        *,
        use_multiprocessing: bool = False,
        transition_pool: Any = None,
        transition_mp_chunksize: int = 32,
        transition_mp_min_batch: int = 32,
    ) -> ExpandSummary:
        global _CY_ROLLOUT_MP_DISABLED
        unique_unexpanded: Dict[str, Any] = {}
        for state in states:
            self._register_state_node(state)
            if not self.nodes_by_key[str(state.key)].expanded:
                unique_unexpanded.setdefault(str(state.key), state)

        if not unique_unexpanded:
            return ExpandSummary(expanded_count=0, discovered_count=0, used_multiprocessing=False)

        pending_states = list(unique_unexpanded.values())
        use_mp = (
            bool(use_multiprocessing)
            and not _CY_ROLLOUT_MP_DISABLED
            and transition_pool is not None
            and len(pending_states) >= max(1, int(transition_mp_min_batch))
        )

        if use_mp:
            try:
                expansion_outputs = transition_pool.map(
                    _expand_cy_state_worker,
                    pending_states,
                    chunksize=max(1, int(transition_mp_chunksize)),
                )
            except Exception as exc:
                warnings.warn(
                    "CY rollout multiprocessing disabled after transition pool failure; "
                    "falling back to sequential expansion for the remainder of this process. "
                    f"Original error: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _CY_ROLLOUT_MP_DISABLED = True
                expansion_outputs = [_expand_cy_state_worker(state) for state in pending_states]
                use_mp = False
        else:
            expansion_outputs = [_expand_cy_state_worker(state) for state in pending_states]

        discovered = 0
        for expansion in expansion_outputs:
            discovered += self._store_expansion(expansion)
        return ExpandSummary(
            expanded_count=len(expansion_outputs),
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
        action_lists, _summary = self.candidate_actions_for_states(source_states, **expand_kwargs)
        return [state for state, actions in zip(source_states, action_lists) if len(actions) > 0]

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
        return len(self.nodes_by_key)

    def runtime_graph_node_count(self) -> int:
        return sum(int(key not in self.base_states) for key in self.nodes_by_key)

    def graph_edge_count(self) -> int:
        return sum(len(node.transitions) for node in self.nodes_by_key.values())

    def runtime_graph_edge_count(self) -> int:
        return sum(
            len(node.transitions)
            for key, node in self.nodes_by_key.items()
            if key not in self.base_states
        )

    def graph_stats_by_polytope(self) -> Dict[int, Dict[str, int]]:
        stats: Dict[int, Dict[str, int]] = {}
        for polytope_index, nodes in self.graph_by_polytope.items():
            stats[polytope_index] = {
                "nodes": len(nodes),
                "edges": sum(len(node.transitions) for node in nodes.values()),
                "expanded_nodes": sum(int(node.expanded) for node in nodes.values()),
            }
        return stats

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
            poly_graph = self.graph_by_polytope.get(node.point_config_index)
            if poly_graph is not None:
                poly_graph.pop(key, None)
                if not poly_graph:
                    self.graph_by_polytope.pop(node.point_config_index, None)

        self.state_cache.hot_states.clear()
        self.state_cache.runtime_unique_keys.clear()
        for state in self.base_states.values():
            self._register_state_node(state)

        return {
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "remaining_nodes": self.graph_node_count(),
            "remaining_runtime_nodes": self.runtime_graph_node_count(),
            "remaining_edges": self.graph_edge_count(),
            "remaining_runtime_edges": self.runtime_graph_edge_count(),
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
            if transition.next_is_target is True:
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
                transitioned_states.append(state)
                continue

            if len(transition.next_simplices) <= 1:
                rewards[idx] = -1.0
                dones[idx] = True
                terminal_reasons[idx] = "single_simplex"
                collapsed_hits += 1
                transitioned_states.append(state)
                continue

            next_state = self.materialize_state(transition.next_key)
            transitioned_states.append(next_state)
            next_states[idx] = next_state
            if self.is_target_state_fn(next_state):
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
                continue
            unique_nonterminal_next_keys.setdefault(str(next_state.key), None)

        nonterminal_next_states = [self.materialize_state(key) for key in unique_nonterminal_next_keys]
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
        )


def get_rollout_memory_stats(engine: CYRandomRolloutEngine) -> Dict[str, int]:
    shared_sizes = get_cy_shared_cache_sizes()
    return {
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
        compacted_graph = engine.runtime_graph_node_count() > int(graph_max_nodes)
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
