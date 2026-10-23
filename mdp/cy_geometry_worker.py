"""Managed-worker entry point for CYTools operations.

Importing this module is cheap. Geometry imports and objects are confined to
requests executed in the managed workers; only immutable descriptions leave.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from mdp.cy_graph import CYGraphTransition, CYStateExpansion
from mdp.cy_state_record import (
    BoundedCache,
    CYPointConfiguration,
    CyStateRecord,
    canonical_simplices,
    normalize_neighbor_mode,
    state_key,
    two_face_state_key,
)


_CONFIGURATIONS = BoundedCache(max_entries=4096, max_bytes=64 * 1024**2)
_POLYTOPES = BoundedCache(max_entries=2, max_bytes=256 * 1024**2)
_OBJECTIVES = BoundedCache(max_entries=8192, max_bytes=16 * 1024**2)


def configure_geometry_worker(cache_budget_bytes: int) -> None:
    """Set this worker's combined retained-cache allowance without imports."""
    allowance = int(cache_budget_bytes)
    if allowance < 0:
        raise ValueError("cache_budget_bytes must be non-negative.")
    polytope_bytes = allowance * 7 // 10
    configuration_bytes = allowance * 2 // 10
    _POLYTOPES.resize(max_bytes=polytope_bytes)
    _CONFIGURATIONS.resize(max_bytes=configuration_bytes)
    _OBJECTIVES.resize(max_bytes=allowance - polytope_bytes - configuration_bytes)


def clear_geometry_caches() -> None:
    """Pressure callback: release idle geometry and return allocator pages."""
    import ctypes
    import gc
    import sys

    _POLYTOPES.clear()
    _OBJECTIVES.clear()
    _CONFIGURATIONS.clear()
    legacy = sys.modules.get("mdp.cy_triangulation_state")
    state_type = getattr(legacy, "CYTriangulationState", None)
    if state_type is not None:
        for name in ("_SHARED_SUBCOMPLEX_CACHE", "_SHARED_NEIGHBOUR_FLIP_CACHE",
                     "_SHARED_SUBCOMPLEX_TRANSITION_CACHE", "_SHARED_SUBCOMPLEX_NEIGHBOUR_CACHE"):
            getattr(state_type, name).clear()
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


def _polytope_class():
    # Configure threads before CYTools imports NumPy/native libraries. Import
    # CYTools before any legacy state module that might initialize Sage/FLINT.
    from core.cytools_config import configure_cytools

    configure_cytools(worker=True)
    from cytools.polytope import Polytope

    return Polytope


def _register_configuration(configuration: CYPointConfiguration) -> None:
    current = _CONFIGURATIONS.get(configuration.index)
    if current is not None and current != configuration:
        _POLYTOPES.pop(configuration.index, None)
        _OBJECTIVES.clear()
    _CONFIGURATIONS[configuration.index] = configuration


def _get_polytope(configuration: CYPointConfiguration):
    _register_configuration(configuration)
    cached = _POLYTOPES.get(configuration.index)
    if cached is None or cached[0] != configuration:
        # CYTools' optimal-coordinate transform can depend on the constructor
        # inputs. Rebuilding from all lattice points with explicit labels can
        # alter the backend's neighbour order even when the geometry is equal.
        # Reproduce the original constructor, then verify its exact label map.
        polytope = _polytope_class()(configuration.input_vertices)
        points = tuple(tuple(int(coord) for coord in point) for point in polytope.points())
        labels = tuple(int(label) for label in polytope.labels)
        if points != configuration.points or labels != configuration.labels:
            raise RuntimeError("Reconstructed point configuration changed its points or label mapping.")
        _POLYTOPES[configuration.index] = (configuration, polytope)
        return polytope
    return cached[1]


def _truth(triangulation, name: str) -> bool:
    method = getattr(triangulation, name, None)
    if not callable(method):
        return False
    try:
        if name == "is_regular":
            from core.cytools_config import REGULARITY_BACKEND

            return bool(method(backend=REGULARITY_BACKEND))
        return bool(method())
    except Exception:
        return False


def _triangulate(polytope, configuration: CYPointConfiguration, simplices):
    return polytope.triangulate(
        simplices=[list(simplex) for simplex in simplices],
        include_points_interior_to_facets=configuration.include_points_interior_to_facets,
        check_input_simplices=False,
    )


def _initial_simplices(row):
    seen = set()
    for entry in row.get("frst_list", ()):
        for triangulation in entry.get("triangulation_list", ()):
            simplices = canonical_simplices(triangulation.get("simplices") or ())
            if simplices and simplices not in seen:
                seen.add(simplices)
                yield simplices
    for triangulation in row.get("non_fine_triangulation_list", ()):
        simplices = canonical_simplices(triangulation.get("simplices", triangulation.get("signature")) or ())
        if simplices and simplices not in seen:
            seen.add(simplices)
            yield simplices


def _build_collection(request):
    row = request["row"]
    mode = normalize_neighbor_mode(request.get("neighbor_mode", "regular"))
    interior = bool(request.get("include_points_interior_to_facets", True))
    if mode == "two_neighbors" and interior:
        raise ValueError("neighbor_mode='two_neighbors' requires include_points_interior_to_facets=False.")
    vertices = tuple(tuple(int(coord) for coord in point) for point in row["vertices"])
    polytope = _polytope_class()(vertices)
    configuration = CYPointConfiguration(
        index=int(row["polytope_index"]), input_vertices=vertices,
        points=tuple(tuple(int(coord) for coord in point) for point in polytope.points()),
        labels=tuple(int(label) for label in polytope.labels),
        include_points_interior_to_facets=interior,
        two_face_labels=(tuple(sorted(tuple(sorted(int(label) for label in face.labels))
                                      for face in polytope.faces(2)))
                         if (request.get("two_face_state", False)
                             or request.get("include_two_face_metadata", False)) else ()),
    )
    _register_configuration(configuration)
    states = {}
    initial_keys = {}
    for entry in row.get("frst_list", ()):
        simplices = canonical_simplices(entry.get("simplices") or ())
        if not simplices:
            continue
        triangulation = _triangulate(polytope, configuration, simplices)
        is_frst = mode == "regular" or (
            _truth(triangulation, "is_fine") and _truth(triangulation, "is_star")
            and _truth(triangulation, "is_regular")
        )
        if not is_frst:
            raise ValueError("two_neighbors initial-state validation failed: dataset FRST entry "
                             f"for polytope {configuration.index} is not fine, star, and regular.")
        state = CyStateRecord(configuration, frozenset(simplices), mode, True, True)
        states.setdefault(state.key, state)
        if mode == "two_neighbors":
            initial_keys.setdefault(state.key, None)
        del triangulation
    if mode == "regular":
        for simplices in _initial_simplices(row):
            triangulation = _triangulate(polytope, configuration, simplices)
            fine = _truth(triangulation, "is_fine")
            regular = fine and _truth(triangulation, "is_regular")
            state = CyStateRecord(configuration, frozenset(simplices), mode,
                                  regular and _truth(triangulation, "is_star"), regular)
            states.setdefault(state.key, state)
            initial_keys.setdefault(state.key, None)
            del triangulation
    _POLYTOPES[configuration.index] = (configuration, polytope)
    return {"configuration": configuration, "base_states": tuple(states.values()),
            "initial_keys": tuple(initial_keys)}


def _face_restrictions(triangulation):
    faces = {}
    for face in triangulation.restrict(as_poly=True):
        labels = tuple(sorted(int(label) for label in face.labels))
        if labels in faces:
            raise RuntimeError("CYTools two_neighbors contract violation: duplicate 2-face restriction.")
        faces[labels] = frozenset(canonical_simplices(face.simplices()))
    return faces


def _fallback_circuit(source_faces, destination):
    destination_faces = _face_restrictions(destination)
    if source_faces.keys() != destination_faces.keys():
        raise RuntimeError("CYTools two_neighbors contract violation: different 2-face sets.")
    changed = [face for face in source_faces if source_faces[face] != destination_faces[face]]
    if len(changed) != 1:
        raise RuntimeError("CYTools two_neighbors contract violation: expected exactly one changed 2-face.")
    face = changed[0]
    removed, added = source_faces[face] - destination_faces[face], destination_faces[face] - source_faces[face]
    left = {vertex for simplex in removed for vertex in simplex}
    right = {vertex for simplex in added for vertex in simplex}
    if len(removed) != 2 or len(added) != 2 or any(len(s) != 3 for s in removed | added) or left != right or len(left) != 4:
        raise RuntimeError("CYTools two_neighbors contract violation: invalid four-vertex diagonal flip.")
    return tuple(sorted(left))


def _expand(request, configuration, payload, polytope):
    index, canonical_source, mode, is_frst, is_target = payload
    action_order = request.get("action_order", "native")
    if action_order not in ("native", "canonical"):
        raise ValueError("action_order must be 'native' or 'canonical'.")
    source = frozenset(canonical_source)
    key = state_key(index, source, mode)
    if not request.get("objective_mode", False) and (is_target or len(source) <= 1):
        return CYStateExpansion(key, index, canonical_source, (), frozenset(), ())
    triangulation = _triangulate(polytope, configuration, canonical_source)
    method = triangulation.neighbor_triangulations
    tracked = mode == "two_neighbors" and "two_neighbors_track_flips" in inspect.signature(method).parameters
    if mode == "two_neighbors":
        neighbours = method(two_neighbors=True, two_neighbors_track_flips=True) if tracked else method(two_neighbors=True)
    else:
        try:
            neighbours = method(only_regular=True)
        except TypeError:
            neighbours = [tri for tri in method() if _truth(tri, "is_regular")]
    neighbours = list(neighbours)
    source_faces = _face_restrictions(triangulation) if mode == "two_neighbors" and not tracked else None
    seen_signatures = set()
    entries = []
    for position in range(len(neighbours)):
        item = neighbours[position]
        neighbours[position] = None
        destination = item[0] if tracked else item
        next_simplices = frozenset(canonical_simplices(destination.simplices()))
        removed, added = source - next_simplices, next_simplices - source
        if not removed or not added:
            continue
        signature = (removed, added)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        if mode == "two_neighbors":
            circuit = tuple(sorted(int(vertex) for vertex in item[2])) if tracked else _fallback_circuit(source_faces, destination)
            if len(circuit) != 4 or len(set(circuit)) != 4:
                raise RuntimeError("CYTools two_neighbors contract violation: invalid tracked circuit.")
            next_target = next_frst = True  # CYTools guarantees FRST representatives.
        else:
            circuit = tuple(sorted({vertex for simplex in removed | added for vertex in simplex}))
            next_target = _truth(destination, "is_fine")  # only_regular=True already established regularity.
            next_frst = next_target and _truth(destination, "is_star")
        transition = CYGraphTransition(
            next_key=state_key(index, next_simplices, mode),
            next_simplices=(), next_is_target=next_target,
            removed_simplices=canonical_simplices(removed), added_simplices=canonical_simplices(added),
            next_is_frst=next_frst,
        )
        entries.append((len(removed) >= len(added), circuit, transition))
    # Legacy regular states expose add actions before remove actions. Native
    # TOPCOM MarkedFlips order varies with process history, so canonical mode
    # explicitly sorts within those groups for worker-independent trajectories.
    if mode == "regular":
        entries.sort(key=(lambda entry: (entry[0], entry[1], entry[2].next_key))
                     if action_order == "canonical" else (lambda entry: entry[0]))
    elif action_order == "canonical":
        entries.sort(key=lambda entry: (entry[1], entry[2].next_key))
    actions = {}
    ambiguous = set()
    for _, circuit, transition in entries:
        if circuit in actions:
            ambiguous.add(circuit)
        else:
            actions[circuit] = transition
    candidates = tuple(action for action in actions if action not in ambiguous)
    return CYStateExpansion(key, index, canonical_source, candidates, frozenset(ambiguous),
                            tuple((action, actions[action]) for action in candidates))


def _objective(request, configuration, payload, polytope):
    index, simplices, mode, *_ = payload
    name = str(request["reward_name"])
    two_face_state = request.get("two_face_state", False)
    if two_face_state and name != "max_kcup":
        raise ValueError("two_face_state currently supports only max_kcup.")
    if name in ("min_tri", "max_tri"):
        return float(len(simplices))
    key = (two_face_state_key(configuration, simplices) if two_face_state
           else state_key(index, simplices, mode))
    # Registration entries can be evicted independently of scalar objectives.
    # Include the immutable geometry, because an index may be reused by another
    # collection sharing this worker pool after the old registration disappears.
    cache_key = (configuration, name, key)
    cached = _OBJECTIVES.get(cache_key)
    if cached is not None:
        return cached
    from reward_functions import get_objective, get_reward

    triangulation = _triangulate(polytope, configuration, simplices)
    state = SimpleNamespace(key=key, simplices=frozenset(simplices), cy_triangulation=triangulation)
    value = float(get_objective(name, reward=get_reward(name))(state))
    _OBJECTIVES[cache_key] = value
    return value


def execute_geometry_request(request):
    """Execute a compact dict request; return descriptions or objective scalars.

    ``build_collection`` takes one dataset row. ``expand``/``objective`` take
    ``state=record.to_payload()`` and optionally ``configuration``. Supplying the
    immutable configuration makes retries safe after worker replacement/cache
    eviction. Expansions accept ``action_order='native'|'canonical'``; native
    preserves the backend order, while canonical gives worker-independent action
    indices. ``register`` accepts ``configurations``; ``trim`` releases caches.
    """
    operation = request["operation"]
    if operation == "cache_stats":
        return {"polytopes": _POLYTOPES.stats(), "configurations": _CONFIGURATIONS.stats(),
                "objectives": _OBJECTIVES.stats()}
    if operation == "trim":
        clear_geometry_caches()
        return None
    if operation == "register":
        for configuration in request["configurations"]:
            _register_configuration(configuration)
        return None
    if operation == "build_collection":
        return _build_collection(request)
    if operation not in ("expand", "objective"):
        raise ValueError(f"Unknown geometry operation '{operation}'.")
    payload = request["state"]
    if isinstance(payload, CyStateRecord):
        payload = payload.to_payload()
    configuration = request.get("configuration")
    if configuration is None:
        configuration = _CONFIGURATIONS.get(payload[0])
    if configuration is None:
        raise KeyError(f"Point configuration {payload[0]} is not registered in this worker.")
    if configuration.index != payload[0]:
        raise ValueError("State and point configuration indices differ.")
    polytope = _get_polytope(configuration)
    try:
        return (_expand(request, configuration, payload, polytope) if operation == "expand"
                else _objective(request, configuration, payload, polytope))
    finally:
        # CYTools lazily populates face/cone caches. Re-estimate at every request,
        # and evict the polytope when those caches exceed the worker allowance.
        _POLYTOPES[configuration.index] = (configuration, polytope)
