from __future__ import annotations

import concurrent.futures
import json
import os

from core.cytools_config import REGULARITY_BACKEND
import signal
import threading
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.cy_data_utils import K3Record, load_k3_records

try:
    from cytools import fetch_polytopes
    from cytools.polytope import Polytope
    from core.cytools_config import configure_cytools
    configure_cytools()
except ModuleNotFoundError:
    Polytope = None
    fetch_polytopes = None


def _require_cytools():
    if Polytope is None:
        raise ModuleNotFoundError(
            "cytools is required for CY data generation. Activate the 'sage' environment."
        )


def _require_cytools_fetch():
    _require_cytools()
    if fetch_polytopes is None:
        raise ModuleNotFoundError(
            "cytools.fetch_polytopes is required for 4D CY data generation. Activate the 'sage' environment."
        )


def _build_n_lattice_polytope(record: K3Record):
    _require_cytools()
    m_vertices = np.asarray(record.m_vertices, dtype=np.int64)
    m_polytope = Polytope(m_vertices)
    n_polytope = m_polytope.dual_polytope()
    return m_polytope, n_polytope


def _normalize_point_matrix(points: Any) -> List[List[int]]:
    array = np.asarray(points, dtype=np.int64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return [[int(coord) for coord in row] for row in array.tolist()]


def _load_json_or_jsonl_records(path: str) -> List[Dict[str, Any]]:
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Polytope file not found: {resolved_path}")

    if resolved_path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))
        return records

    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("polytopes"), list):
            return [dict(item) for item in payload["polytopes"]]
        if isinstance(payload.get("polytope_specs"), list):
            return [dict(item) for item in payload["polytope_specs"]]
    raise ValueError(
        f"Unsupported polytope file format at {resolved_path}. "
        "Expected a JSON list, a JSON object with 'polytopes'/'polytope_specs', or JSONL."
    )


def _extract_polytope_points_from_record(record: Dict[str, Any]) -> List[List[int]]:
    for key in ("vertices", "n_points", "n_vertices"):
        if key in record and record[key] is not None:
            return _normalize_point_matrix(record[key])
    raise ValueError(
        "Polytope record is missing point data. Expected one of: "
        "'vertices', 'n_points', or 'n_vertices'."
    )


def _load_4d_n_lattice_polytope_specs_from_file(
    *,
    polytope_file: str,
    num_polytopes: Optional[int],
) -> List[Dict[str, Any]]:
    loaded_records = _load_json_or_jsonl_records(polytope_file)
    if num_polytopes is not None and int(num_polytopes) <= 0:
        raise ValueError("num_polytopes must be positive when provided.")

    limited_records = (
        loaded_records
        if num_polytopes is None
        else loaded_records[: int(num_polytopes)]
    )
    if num_polytopes is not None and len(loaded_records) < int(num_polytopes):
        warnings.warn(
            "Requested "
            f"{num_polytopes} fourfold polytopes from {polytope_file}, but the file contains only "
            f"{len(loaded_records)}. Proceeding with the available polytopes.",
            stacklevel=2,
        )

    specs: List[Dict[str, Any]] = []
    for loaded_index, record in enumerate(limited_records):
        spec = {
            "polytope_index": int(record.get("polytope_index", loaded_index)),
            "vertices": _extract_polytope_points_from_record(record),
            "polytope_source": "polytope_file",
        }
        if record.get("h11") is not None:
            spec["h11"] = int(record["h11"])
        if record.get("requested_num_vertices") is not None:
            spec["requested_num_vertices"] = int(record["requested_num_vertices"])
        elif record.get("num_vertices") is not None:
            spec["requested_num_vertices"] = int(record["num_vertices"])
        if record.get("favorable") is not None:
            spec["favorable"] = bool(record["favorable"])
        specs.append(spec)
    return specs


def _fetch_4d_n_lattice_polytope_specs(
    *,
    num_polytopes: int,
    h11: int,
    num_vertices: Optional[int],
    favorable: Optional[bool],
) -> List[Dict[str, Any]]:
    _require_cytools_fetch()
    if num_polytopes <= 0:
        raise ValueError("num_polytopes must be positive.")

    fetch_kwargs: Dict[str, Any] = {
        "h11": int(h11),
        "dim": 4,
        "lattice": "N",
        "limit": int(num_polytopes),
        "as_list": False,
    }
    if favorable is not None:
        fetch_kwargs["favorable"] = bool(favorable)
    if num_vertices is not None:
        fetch_kwargs["n_vertices"] = int(num_vertices)

    polytope_generator = fetch_polytopes(**fetch_kwargs)
    specs: List[Dict[str, Any]] = []
    for polytope_index, polytope in enumerate(polytope_generator):
        if polytope_index >= int(num_polytopes):
            break
        specs.append(
            {
                "polytope_index": int(polytope_index),
                "h11": int(h11),
                "requested_num_vertices": None
                if num_vertices is None
                else int(num_vertices),
                "vertices": _normalize_point_matrix(polytope.vertices()),
            }
        )
        if favorable is not None:
            specs[-1]["favorable"] = bool(favorable)

    if len(specs) < int(num_polytopes):
        warnings.warn(
            "Requested "
            f"{num_polytopes} fourfold polytopes, but fetch_polytopes returned only {len(specs)}. "
            "Proceeding with the available polytopes.",
            stacklevel=2,
        )
    return specs


def _normalize_polytope_indices(polytope_indices: Sequence[int]) -> List[int]:
    if len(polytope_indices) == 0:
        raise ValueError("polytope_indices must not be empty.")
    normalized = sorted({int(index) for index in polytope_indices})
    if normalized[0] < 0:
        raise ValueError("polytope_indices values must be non-negative.")
    return normalized


def _load_k3_records_by_indices(
    *,
    k3_path: Optional[str],
    polytope_indices: Sequence[int],
) -> List[K3Record]:
    normalized_indices = _normalize_polytope_indices(polytope_indices)
    records = load_k3_records(
        k3_path=k3_path,
        max_polytopes=int(max(normalized_indices)) + 1,
    )
    missing_indices = [
        int(index) for index in normalized_indices if int(index) >= len(records)
    ]
    if missing_indices:
        raise ValueError(
            "Requested k3 polytope_index values are out of range: "
            f"{missing_indices}. Available indices stop at {len(records) - 1}."
        )
    return [records[int(index)] for index in normalized_indices]


def _canonical_simplex(simplex) -> Tuple[int, ...]:
    return tuple(sorted(int(vertex) for vertex in simplex))


def triangulation_signature(triangulation) -> Tuple[Tuple[int, ...], ...]:
    simplices = triangulation.simplices(as_indices=True)
    simplices_array = np.asarray(simplices, dtype=np.int64)
    if simplices_array.ndim == 1:
        simplices_array = simplices_array.reshape(1, -1)
    canonical = tuple(sorted(_canonical_simplex(simplex) for simplex in simplices_array.tolist()))
    return canonical


def _is_regular(triangulation) -> bool:
    try:
        return bool(triangulation.is_regular(backend=REGULARITY_BACKEND))
    except TypeError:
        return bool(triangulation.is_regular())


def _is_frst(triangulation) -> bool:
    return bool(_is_regular(triangulation) and triangulation.is_fine() and triangulation.is_star())


def _is_fine_regular(triangulation) -> bool:
    # Fine + regular is sufficient for conversion to FRST by lowering origin height.
    return bool(_is_regular(triangulation) and triangulation.is_fine())


def _num_simplices(triangulation) -> int:
    simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64)
    if simplices.ndim == 1:
        simplices = simplices.reshape(1, -1)
    return int(simplices.shape[0])


def serialize_triangulation(triangulation) -> Dict[str, Any]:
    points = np.asarray(triangulation.points(), dtype=np.int64)
    poly_indices = np.asarray(triangulation.points(as_poly_indices=True), dtype=np.int64)
    simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64)

    if points.ndim == 1:
        points = points.reshape(1, -1)
    if simplices.ndim == 1:
        simplices = simplices.reshape(1, -1)

    signature = triangulation_signature(triangulation)

    return {
        "num_points": int(points.shape[0]),
        "point_indices": [int(v) for v in poly_indices.tolist()],
        "points": [[int(coord) for coord in row] for row in points.tolist()],
        "num_simplices": int(simplices.shape[0]),
        "simplices": [[int(vertex) for vertex in simplex] for simplex in simplices.tolist()],
        "signature": [[int(vertex) for vertex in simplex] for simplex in signature],
        "is_regular": bool(_is_regular(triangulation)),
        "is_fine": bool(triangulation.is_fine()),
        "is_star": bool(triangulation.is_star()),
        "is_frst": bool(_is_frst(triangulation)),
    }


def _sample_random_heights(
    num_points: int,
    rng: np.random.Generator,
    *,
    height_scale: float = 1.0,
) -> np.ndarray:
    heights = rng.normal(loc=0.0, scale=height_scale, size=num_points).astype(np.float64)
    # Break exact ties deterministically to avoid degenerate random height vectors.
    heights += np.linspace(0.0, 1e-9, num_points, dtype=np.float64)
    return heights


def generate_random_regular_triangulation(
    n_polytope,
    rng: np.random.Generator,
    *,
    triangulation_backend: str = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = True,
    max_retries: int = 8,
    height_scale: float = 1.0,
    triangulation_verbosity: int = 0,
) -> Tuple[Any, List[float]]:
    all_point_indices = np.asarray(n_polytope.points(as_indices=True), dtype=np.int64)
    num_points = int(all_point_indices.size)
    if num_points <= 0:
        raise ValueError("N-lattice polytope has no points to triangulate.")

    last_error: Optional[Exception] = None
    for _ in range(max_retries):
        heights = _sample_random_heights(num_points, rng, height_scale=height_scale)
        try:
            triangulation = n_polytope.triangulate(
                points=all_point_indices,
                heights=heights,
                make_star=make_star,
                include_points_interior_to_facets=include_points_interior_to_facets,
                backend=triangulation_backend,
                verbosity=triangulation_verbosity,
            )
        except Exception as exc:  # cytools backends can raise different exception types.
            last_error = exc
            continue

        if not _is_regular(triangulation):
            continue

        return triangulation, [float(value) for value in heights.tolist()]

    if last_error is not None:
        raise RuntimeError(
            f"Failed to generate random regular triangulation after {max_retries} retries."
        ) from last_error
    raise RuntimeError(
        f"Failed to generate random regular triangulation after {max_retries} retries."
    )


def _sample_uniform_random_heights(
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    heights = rng.random(num_points).astype(np.float64)
    # Break exact ties deterministically to avoid degenerate height vectors.
    heights += np.linspace(0.0, 1e-9, num_points, dtype=np.float64)
    return heights


def _triangulation_signature_in_polytope_index_space(triangulation) -> List[List[int]]:
    local_to_poly = np.asarray(triangulation.points(as_poly_indices=True), dtype=np.int64)
    signature = triangulation_signature(triangulation)
    return [
        [int(local_to_poly[int(vertex)]) for vertex in simplex]
        for simplex in signature
    ]


def serialize_random_height_non_fine_triangulation(
    triangulation,
    *,
    heights: Sequence[float],
) -> Dict[str, Any]:
    return {
        "heights": [float(value) for value in heights],
        "signature": _triangulation_signature_in_polytope_index_space(triangulation),
    }


def _generate_random_height_non_fine_triangulations(
    n_polytope,
    *,
    num_triangulations: int,
    max_tries: int,
    seed: int,
    triangulation_backend: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = None,
    triangulation_verbosity: int = 0,
) -> Dict[str, Any]:
    if num_triangulations <= 0:
        raise ValueError("num_triangulations must be positive.")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive.")

    points = np.asarray(n_polytope.points(), dtype=np.int64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    num_points = int(points.shape[0])
    if num_points <= 0:
        raise ValueError("N-lattice polytope has no points to triangulate.")

    rng = np.random.default_rng(seed)
    unique_by_signature: Dict[Tuple[Tuple[int, ...], ...], Dict[str, Any]] = {}
    fine_discard_count = 0
    duplicate_count = 0
    non_regular_count = 0
    error_count = 0
    last_error_name: Optional[str] = None
    attempt_count = 0

    while len(unique_by_signature) < num_triangulations and attempt_count < max_tries:
        attempt_count += 1
        heights = _sample_uniform_random_heights(num_points, rng)
        triangulate_kwargs: Dict[str, Any] = {
            "include_points_interior_to_facets": bool(include_points_interior_to_facets),
            "heights": heights,
            "check_heights": True,
            "verbosity": int(triangulation_verbosity),
        }
        if triangulation_backend is not None:
            triangulate_kwargs["backend"] = triangulation_backend
        if make_star is not None:
            triangulate_kwargs["make_star"] = bool(make_star)

        try:
            triangulation = n_polytope.triangulate(**triangulate_kwargs)
        except Exception as exc:
            error_count += 1
            last_error_name = type(exc).__name__
            continue

        if not _is_regular(triangulation):
            non_regular_count += 1
            continue
        if bool(triangulation.is_fine()):
            fine_discard_count += 1
            continue

        signature = triangulation_signature(triangulation)
        if signature in unique_by_signature:
            duplicate_count += 1
            continue

        unique_by_signature[signature] = {
            "triangulation": triangulation,
            "heights": [float(value) for value in heights.tolist()],
        }

    serialized = [
        serialize_random_height_non_fine_triangulation(
            unique_by_signature[signature]["triangulation"],
            heights=unique_by_signature[signature]["heights"],
        )
        for signature in sorted(unique_by_signature)
    ]
    return {
        "triangulations": serialized,
        "diagnostics": {
            "seed": int(seed),
            "requested_num_triangulations": int(num_triangulations),
            "max_tries": int(max_tries),
            "attempt_count": int(attempt_count),
            "generated_count": int(len(serialized)),
            "fine_discard_count": int(fine_discard_count),
            "duplicate_count": int(duplicate_count),
            "non_regular_count": int(non_regular_count),
            "error_count": int(error_count),
            "last_error": last_error_name,
            "target_reached": bool(len(serialized) >= num_triangulations),
            "height_sampling_distribution": "uniform_unit_interval",
        },
    }


def _get_regular_neighbors(
    current,
    *,
    neighbor_backend: Optional[str],
    sanity_notes: Optional[List[str]] = None,
) -> List[Any]:
    base_kwargs = {
        "only_regular": True,
        "only_fine": False,
        "only_star": False,
    }

    if neighbor_backend is None:
        neighbors = list(current.neighbor_triangulations(**base_kwargs))
    else:
        try:
            neighbors = list(
                current.neighbor_triangulations(
                    **base_kwargs,
                    backend=neighbor_backend,
                    verbose=False,
                )
            )
        except (TypeError, ValueError):
            # Some cytools versions/backends reject explicit backend values.
            neighbors = list(current.neighbor_triangulations(**base_kwargs))

    # TOPCOM may incorrectly return [] for non-trivial triangulations.
    if len(neighbors) == 0 and sanity_notes is not None and _num_simplices(current) != 1:
        sanity_notes.append(
            "neighbor_triangulations returned no regular neighbors on non-trivial triangulation."
        )
    return neighbors


def _upsert_fine_regular_seed(unique_frsts: Dict[Tuple[Tuple[int, ...], ...], Any], triangulation) -> None:
    if not _is_fine_regular(triangulation):
        return
    signature = triangulation_signature(triangulation)
    if signature not in unique_frsts:
        unique_frsts[signature] = triangulation


class _FairSamplingTimeoutError(TimeoutError):
    pass


def _raise_fair_sampling_timeout(_signum, _frame):
    raise _FairSamplingTimeoutError("random_triangulations_fair call timed out.")


def _call_with_optional_timeout(
    call_fn,
    *,
    timeout_seconds: Optional[float],
):
    if timeout_seconds is None:
        return call_fn()
    if timeout_seconds <= 0:
        raise ValueError("fair_call_timeout_seconds must be positive when provided.")

    # Signal-based timeout works on Unix main threads; otherwise run without timeout.
    if threading.current_thread() is not threading.main_thread():
        return call_fn()
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return call_fn()

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_fair_sampling_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return call_fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0 or old_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _generate_frst_seeds_with_random_triangulations_fair(
    n_polytope,
    *,
    target_count: int,
    seed: int,
    fair_backend: str = "cgal",
    fair_backend_fallback: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: bool = True,
    fair_max_retries: int = 20,
    fair_max_attempt_rounds: int = 4,
    fair_call_timeout_seconds: Optional[float] = None,
    fast_only: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if fair_max_attempt_rounds <= 0:
        raise ValueError("fair_max_attempt_rounds must be positive.")
    if fair_call_timeout_seconds is not None and fair_call_timeout_seconds <= 0:
        raise ValueError("fair_call_timeout_seconds must be positive when provided.")

    unique_frsts: Dict[Tuple[Tuple[int, ...], ...], Any] = {}
    seed_rng = np.random.default_rng(seed)
    failures: List[str] = []
    fair_rounds: List[Dict[str, Any]] = []
    fast_rounds: List[Dict[str, Any]] = []
    fallback_to_fast_used = bool(fast_only)

    if not fast_only:
        for _ in range(fair_max_attempt_rounds):
            if len(unique_frsts) >= target_count:
                break
            round_seed = int(seed_rng.integers(0, np.iinfo(np.uint32).max))
            needed = target_count - len(unique_frsts)

            fair_info = {
                "seed": round_seed,
                "requested": int(needed),
                "obtained_before_round": int(len(unique_frsts)),
                "obtained_after_round": int(len(unique_frsts)),
                "status": "ok",
            }
            timeout_hit = False
            try:
                candidates = _call_with_optional_timeout(
                    lambda: n_polytope.random_triangulations_fair(
                        N=needed,
                        as_list=True,
                        progress_bar=False,
                        seed=round_seed,
                        backend=fair_backend,
                        make_star=make_star,
                        include_points_interior_to_facets=include_points_interior_to_facets,
                        max_retries=fair_max_retries,
                    ),
                    timeout_seconds=fair_call_timeout_seconds,
                )
                fair_info["returned_count"] = int(len(candidates))
                for triangulation in candidates:
                    _upsert_fine_regular_seed(unique_frsts, triangulation)
                fair_info["obtained_after_round"] = int(len(unique_frsts))
                if len(candidates) < needed:
                    fair_info["status"] = "short_return"
            except _FairSamplingTimeoutError:
                fair_info["status"] = "timeout"
                fair_info["returned_count"] = 0
                fair_info["timeout_seconds"] = float(fair_call_timeout_seconds)
                failures.append(f"fair:{fair_backend}:timeout")
                timeout_hit = True
            except Exception as exc:
                fair_info["status"] = f"error:{type(exc).__name__}"
                fair_info["returned_count"] = 0
                failures.append(f"fair:{fair_backend}:{type(exc).__name__}")
            fair_rounds.append(fair_info)
            if timeout_hit:
                # Avoid repeated long stalls; immediately proceed to fast fallback.
                break

    fair_obtained_count = int(len(unique_frsts))

    # Fallback to random_triangulations_fast when fair sampling under-delivers.
    if len(unique_frsts) < target_count:
        fallback_to_fast_used = True
    for _ in range(fair_max_attempt_rounds):
        if len(unique_frsts) >= target_count:
            break
        round_seed = int(seed_rng.integers(0, np.iinfo(np.uint32).max))
        needed = target_count - len(unique_frsts)
        fast_backend = fair_backend_fallback if fair_backend_fallback is not None else fair_backend

        fast_info = {
            "seed": round_seed,
            "requested": int(needed),
            "backend": fast_backend,
            "obtained_before_round": int(len(unique_frsts)),
            "obtained_after_round": int(len(unique_frsts)),
            "status": "ok",
        }
        try:
            candidates = n_polytope.random_triangulations_fast(
                N=needed,
                as_list=True,
                progress_bar=False,
                seed=round_seed,
                backend=fast_backend,
                make_star=make_star,
                only_fine=True,
                include_points_interior_to_facets=include_points_interior_to_facets,
                max_retries=max(fair_max_retries, 100),
            )
            fast_info["returned_count"] = int(len(candidates))
            for triangulation in candidates:
                _upsert_fine_regular_seed(unique_frsts, triangulation)
            fast_info["obtained_after_round"] = int(len(unique_frsts))
            if len(candidates) < needed:
                fast_info["status"] = "short_return"
        except Exception as exc:
            fast_info["status"] = f"error:{type(exc).__name__}"
            fast_info["returned_count"] = 0
            failures.append(f"fast:{fast_backend}:{type(exc).__name__}")
        fast_rounds.append(fast_info)

    # Final deterministic fallback to guarantee at least one fine+regular seed.
    if len(unique_frsts) == 0:
        deterministic_backends: List[str] = []
        for backend in (fair_backend, fair_backend_fallback, "cgal", "qhull"):
            if backend is None:
                continue
            if backend not in deterministic_backends:
                deterministic_backends.append(backend)

        deterministic_configs = [
            (include_points_interior_to_facets, make_star),
            (False, make_star),
            (include_points_interior_to_facets, True),
            (False, True),
        ]

        for backend in deterministic_backends:
            for include_opt, star_opt in deterministic_configs:
                try:
                    fallback = n_polytope.triangulate(
                        include_points_interior_to_facets=include_opt,
                        make_star=star_opt,
                        backend=backend,
                        verbosity=0,
                    )
                except Exception as exc:
                    failures.append(f"deterministic:{backend}:{type(exc).__name__}")
                    continue

                if _is_fine_regular(fallback):
                    _upsert_fine_regular_seed(unique_frsts, fallback)
                    break
            if len(unique_frsts) > 0:
                break

        if len(unique_frsts) == 0:
            raise RuntimeError(
                "Failed to obtain any fine+regular seed from random_triangulations_fair, "
                "random_triangulations_fast, and deterministic fallback."
            )

    frst_list = list(unique_frsts.values())[:target_count]
    diagnostics = {
        "requested_frst_count": int(target_count),
        "obtained_frst_count": int(len(frst_list)),
        "obtained_from_fair_count": int(min(fair_obtained_count, len(frst_list))),
        "frst_seed_sampler": "fast" if fast_only else "fair_then_fast",
        "fast_only": bool(fast_only),
        "fallback_to_fast_used": bool(fallback_to_fast_used),
        "fair_backend": fair_backend,
        "fair_backend_fallback": fair_backend_fallback,
        "fair_max_retries": int(fair_max_retries),
        "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
        "fair_call_timeout_seconds": None
        if fair_call_timeout_seconds is None
        else float(fair_call_timeout_seconds),
        "fair_rounds": fair_rounds,
        "fast_rounds": fast_rounds,
        "failure_events": failures,
    }
    return frst_list, diagnostics


def _bfs_collect_non_fine_regular_states(
    source_triangulation,
    *,
    max_depth: int,
    max_nodes: int,
    neighbor_backend: Optional[str],
) -> Dict[str, Any]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")

    source_signature = triangulation_signature(source_triangulation)
    visited = {source_signature}
    queue = deque([(source_triangulation, 0)])
    expanded_count = 0
    truncated = False
    stop_reason = "exhausted_or_max_depth"
    sanity_notes: List[str] = []
    collected: Dict[Tuple[Tuple[int, ...], ...], Dict[str, Any]] = {}

    while queue:
        if expanded_count >= max_nodes:
            truncated = True
            stop_reason = "max_nodes_expanded"
            break

        current, depth = queue.popleft()
        expanded_count += 1
        if depth >= max_depth:
            continue

        neighbors = _get_regular_neighbors(
            current,
            neighbor_backend=neighbor_backend,
            sanity_notes=sanity_notes,
        )
        for neighbor in neighbors:
            signature = triangulation_signature(neighbor)
            if signature in visited:
                continue

            visited.add(signature)
            next_depth = depth + 1
            if not neighbor.is_fine():
                existing = collected.get(signature)
                if existing is None or next_depth < int(existing["distance"]):
                    collected[signature] = {
                        "triangulation": neighbor,
                        "distance": int(next_depth),
                    }

            if len(visited) >= max_nodes:
                truncated = True
                stop_reason = "max_nodes_visited"
                queue.clear()
                break
            queue.append((neighbor, next_depth))

        if truncated:
            break

    return {
        "collection": collected,
        "visited_count": int(len(visited)),
        "expanded_count": int(expanded_count),
        "truncated": bool(truncated),
        "stop_reason": stop_reason,
        "sanity_notes": sanity_notes,
    }


def _select_signatures_for_collection(
    signatures: List[Tuple[Tuple[int, ...], ...]],
    rng: np.random.Generator,
    *,
    max_count: int,
    collect_all: bool,
) -> List[Tuple[Tuple[int, ...], ...]]:
    if collect_all or len(signatures) <= max_count:
        return list(signatures)
    chosen = rng.choice(len(signatures), size=max_count, replace=False)
    return [signatures[int(idx)] for idx in chosen.tolist()]


def _sample_from_frst_via_bfs(
    source_frst,
    rng: np.random.Generator,
    *,
    max_depth: int,
    max_nodes: int,
    neighbor_backend: Optional[str],
) -> Dict[str, Any]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")

    source_signature = triangulation_signature(source_frst)
    visited = {source_signature}
    queue = deque([(source_frst, 0)])
    candidates: List[Tuple[Any, int]] = []
    expanded_count = 0
    truncated = False
    stop_reason = "exhausted_or_max_depth"

    while queue:
        if expanded_count >= max_nodes:
            truncated = True
            stop_reason = "max_nodes_expanded"
            break

        current, depth = queue.popleft()
        expanded_count += 1
        if depth >= max_depth:
            continue

        neighbors = _get_regular_neighbors(current, neighbor_backend=neighbor_backend)
        if len(neighbors) > 1:
            order = rng.permutation(len(neighbors))
            neighbors = [neighbors[int(i)] for i in order]

        for neighbor in neighbors:
            signature = triangulation_signature(neighbor)
            if signature in visited:
                continue

            visited.add(signature)
            candidate_depth = depth + 1
            candidates.append((neighbor, candidate_depth))

            if len(visited) >= max_nodes:
                truncated = True
                stop_reason = "max_nodes_visited"
                queue.clear()
                break

            queue.append((neighbor, candidate_depth))

        if truncated:
            break

    if len(candidates) == 0:
        sampled_triangulation = source_frst
        sampled_distance = 0
        found_nontrivial_sample = False
    else:
        sampled_idx = int(rng.integers(0, len(candidates)))
        sampled_triangulation, sampled_distance = candidates[sampled_idx]
        found_nontrivial_sample = True

    return {
        "sampled_triangulation": sampled_triangulation,
        "distance_from_source_frst": int(sampled_distance),
        "found_nontrivial_sample": bool(found_nontrivial_sample),
        "visited_count": int(len(visited)),
        "expanded_count": int(expanded_count),
        "truncated": bool(truncated),
        "stop_reason": stop_reason,
        "candidate_count": int(len(candidates)),
    }


def _normalize_collection_depths(
    collection_depths: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if collection_depths is None:
        return None
    normalized = sorted({int(depth) for depth in collection_depths})
    if len(normalized) == 0:
        raise ValueError("collection_depths must not be empty when provided.")
    if normalized[0] < 1:
        raise ValueError("collection_depths values must be >= 1.")
    return normalized


def _collect_non_frst_states_at_depths_from_frst(
    source_frst,
    *,
    collection_depths: Sequence[int],
    max_depth: int,
    max_nodes: int,
    neighbor_backend: Optional[str],
) -> Dict[str, Any]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")
    if len(collection_depths) == 0:
        raise ValueError("collection_depths must not be empty.")

    target_depths = set(int(depth) for depth in collection_depths)
    queue = deque([(source_frst, 0)])
    visited = {triangulation_signature(source_frst)}
    candidates: List[Tuple[Any, int]] = []
    expanded_count = 0
    truncated = False
    stop_reason = "exhausted_or_max_depth"

    while queue:
        if expanded_count >= max_nodes:
            truncated = True
            stop_reason = "max_nodes_expanded"
            break

        current, depth = queue.popleft()
        expanded_count += 1
        if depth >= max_depth:
            continue

        neighbors = _get_regular_neighbors(current, neighbor_backend=neighbor_backend)
        for neighbor in neighbors:
            signature = triangulation_signature(neighbor)
            if signature in visited:
                continue

            visited.add(signature)
            candidate_depth = depth + 1
            if candidate_depth in target_depths and not _is_frst(neighbor):
                candidates.append((neighbor, candidate_depth))

            if len(visited) >= max_nodes:
                truncated = True
                stop_reason = "max_nodes_visited"
                queue.clear()
                break

            queue.append((neighbor, candidate_depth))

        if truncated:
            break

    return {
        "candidates": candidates,
        "visited_count": int(len(visited)),
        "expanded_count": int(expanded_count),
        "truncated": bool(truncated),
        "stop_reason": stop_reason,
        "candidate_count": int(len(candidates)),
    }


def find_nearest_frst_bfs(
    start_triangulation,
    *,
    max_depth: int = 4,
    max_nodes: int = 300,
    neighbor_backend: Optional[str] = None,
) -> Dict[str, Any]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")

    start_signature = triangulation_signature(start_triangulation)
    if _is_frst(start_triangulation):
        return {
            "found": True,
            "distance": 0,
            "nearest_frst": serialize_triangulation(start_triangulation),
            "visited_count": 1,
            "expanded_count": 0,
            "truncated": False,
            "stop_reason": "start_is_frst",
        }

    queue = deque([(start_triangulation, 0)])
    visited = {start_signature}
    expanded_count = 0

    while queue:
        if expanded_count >= max_nodes:
            return {
                "found": False,
                "distance": None,
                "nearest_frst": None,
                "visited_count": int(len(visited)),
                "expanded_count": int(expanded_count),
                "truncated": True,
                "stop_reason": "max_nodes_expanded",
            }

        current, depth = queue.popleft()
        expanded_count += 1
        if depth >= max_depth:
            continue

        neighbors = _get_regular_neighbors(current, neighbor_backend=neighbor_backend)

        for neighbor in neighbors:
            signature = triangulation_signature(neighbor)
            if signature in visited:
                continue

            visited.add(signature)
            if _is_frst(neighbor):
                return {
                    "found": True,
                    "distance": int(depth + 1),
                    "nearest_frst": serialize_triangulation(neighbor),
                    "visited_count": int(len(visited)),
                    "expanded_count": int(expanded_count),
                    "truncated": False,
                    "stop_reason": "found",
                }

            if len(visited) >= max_nodes:
                return {
                    "found": False,
                    "distance": None,
                    "nearest_frst": None,
                    "visited_count": int(len(visited)),
                    "expanded_count": int(expanded_count),
                    "truncated": True,
                    "stop_reason": "max_nodes_visited",
                }

            queue.append((neighbor, depth + 1))

    return {
        "found": False,
        "distance": None,
        "nearest_frst": None,
        "visited_count": int(len(visited)),
        "expanded_count": int(expanded_count),
        "truncated": False,
        "stop_reason": "exhausted_or_max_depth",
    }


def _serialize_polytope(record: K3Record, m_polytope, n_polytope) -> Dict[str, Any]:
    m_points = np.asarray(m_polytope.points(), dtype=np.int64)
    n_points = np.asarray(n_polytope.points(), dtype=np.int64)
    if m_points.ndim == 1:
        m_points = m_points.reshape(1, -1)
    if n_points.ndim == 1:
        n_points = n_points.reshape(1, -1)

    return {
        "polytope_index": int(record.record_index),
        "source_header": record.header,
        "ambient_dim": int(record.ambient_dim),
        "num_m_vertices": int(len(record.m_vertices)),
        "m_vertices": [[int(coord) for coord in vertex] for vertex in record.m_vertices],
        "num_m_points": int(m_points.shape[0]),
        "num_n_points": int(n_points.shape[0]),
        "n_points": [[int(coord) for coord in point] for point in n_points.tolist()],
        "lattice_space": "N",
    }


def _serialize_fetched_n_polytope(
    polytope_spec: Dict[str, Any],
    n_polytope: Any,
) -> Dict[str, Any]:
    n_points = np.asarray(n_polytope.points(), dtype=np.int64)
    n_vertices = np.asarray(n_polytope.vertices(), dtype=np.int64)
    if n_points.ndim == 1:
        n_points = n_points.reshape(1, -1)
    if n_vertices.ndim == 1:
        n_vertices = n_vertices.reshape(1, -1)

    entry = {
        "polytope_index": int(polytope_spec["polytope_index"]),
        "ambient_dim": int(n_polytope.ambient_dim()),
        "num_n_vertices": int(n_vertices.shape[0]),
        "n_vertices": [[int(coord) for coord in vertex] for vertex in n_vertices.tolist()],
        "num_n_points": int(n_points.shape[0]),
        "n_points": [[int(coord) for coord in point] for point in n_points.tolist()],
        "lattice_space": "N",
        "polytope_source": str(polytope_spec.get("polytope_source", "fetch_polytopes")),
        "requested_num_vertices": None
        if polytope_spec.get("requested_num_vertices") is None
        else int(polytope_spec["requested_num_vertices"]),
    }
    if polytope_spec.get("h11") is not None:
        entry["h11"] = int(polytope_spec["h11"])
    if "favorable" in polytope_spec:
        entry["favorable"] = bool(polytope_spec["favorable"])
    return entry


def _resolve_num_workers(num_workers: Optional[int], *, num_polytopes: int) -> int:
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, num_polytopes))
    resolved = int(num_workers)
    if resolved <= 0:
        raise ValueError("num_workers must be positive when provided.")
    return max(1, min(resolved, num_polytopes))


def _collect_samples_for_n_polytope(
    *,
    polytope_index: int,
    polytope_entry: Dict[str, Any],
    n_polytope: Any,
    polytope_seed: int,
    frst_target: int,
    num_triangulations_per_frst: int,
    fair_backend: str,
    fair_backend_fallback: Optional[str],
    fair_max_retries: int,
    fair_max_attempt_rounds: int,
    fair_call_timeout_seconds: Optional[float],
    fast: bool,
    include_points_interior_to_facets: bool,
    make_star: bool,
    neighbor_backend: Optional[str],
    max_depth: int,
    max_nodes: int,
    collection_depths: Optional[Sequence[int]],
    collection_all: bool,
    random_flip: bool,
) -> Dict[str, Any]:
    effective_collection_depths = None if random_flip else collection_depths
    target_depths = (
        None
        if effective_collection_depths is None
        else set(int(depth) for depth in effective_collection_depths)
    )
    rng = np.random.default_rng(polytope_seed)
    frst_seed_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    frst_seeds, frst_diag = _generate_frst_seeds_with_random_triangulations_fair(
        n_polytope,
        target_count=frst_target,
        seed=frst_seed_seed,
        fair_backend=fair_backend,
        fair_backend_fallback=fair_backend_fallback,
        include_points_interior_to_facets=include_points_interior_to_facets,
        make_star=make_star,
        fair_max_retries=fair_max_retries,
        fair_max_attempt_rounds=fair_max_attempt_rounds,
        fair_call_timeout_seconds=fair_call_timeout_seconds,
        fast_only=fast,
    )

    polytope_entry["frst_generation"] = {
        "seed": frst_seed_seed,
        "diagnostics": frst_diag,
    }
    polytope_entry["frst_seeds"] = [serialize_triangulation(frst) for frst in frst_seeds]

    collected_states: Dict[Tuple[Tuple[int, ...], ...], Dict[str, Any]] = {}
    source_bfs_reports: List[Dict[str, Any]] = []
    candidate_pool_size_total = 0
    selected_count_total = 0

    if random_flip:
        flip_depth = int(max_depth)
        max_attempts_per_frst = int(max(1, max_nodes))
        for source_frst_index, source_frst in enumerate(frst_seeds):
            selected_for_source = 0
            non_fine_candidate_count = 0
            duplicate_count = 0
            error_count = 0
            attempt_count = 0
            last_error_name: Optional[str] = None

            while (
                selected_for_source < num_triangulations_per_frst
                and attempt_count < max_attempts_per_frst
            ):
                attempt_count += 1
                try:
                    flipped = source_frst.random_flips(
                        N=flip_depth,
                        only_regular=True,
                    )
                except Exception as exc:
                    error_count += 1
                    last_error_name = type(exc).__name__
                    continue

                if bool(flipped.is_fine()):
                    continue

                non_fine_candidate_count += 1
                signature = triangulation_signature(flipped)
                existing = collected_states.get(signature)
                if existing is not None:
                    duplicate_count += 1
                    continue

                collected_states[signature] = {
                    "triangulation": flipped,
                    "distance": int(flip_depth),
                    "source_frst_index": int(source_frst_index),
                    "source_frst": source_frst,
                    "visited_count": 0,
                    "expanded_count": 0,
                    "truncated": False,
                    "stop_reason": "random_flip",
                    "candidate_count": int(non_fine_candidate_count),
                }
                selected_for_source += 1

            candidate_pool_size_total += int(non_fine_candidate_count)
            selected_count_total += int(selected_for_source)
            truncated = bool(selected_for_source < num_triangulations_per_frst)
            stop_reason = (
                "random_flip_collected_target"
                if not truncated
                else "random_flip_max_attempts"
            )
            sanity_notes: List[str] = []
            if last_error_name is not None:
                sanity_notes.append(f"random_flips_error:{last_error_name}")
            source_bfs_reports.append(
                {
                    "source_frst_index": int(source_frst_index),
                    "visited_count": 0,
                    "expanded_count": 0,
                    "truncated": bool(truncated),
                    "stop_reason": stop_reason,
                    "candidate_count": int(non_fine_candidate_count),
                    "selected_count": int(selected_for_source),
                    "sanity_notes": sanity_notes,
                    "attempt_count": int(attempt_count),
                    "duplicate_count": int(duplicate_count),
                    "error_count": int(error_count),
                    "random_flip_depth": int(flip_depth),
                    "max_attempts": int(max_attempts_per_frst),
                }
            )
    else:
        for source_frst_index, source_frst in enumerate(frst_seeds):
            bfs_result = _bfs_collect_non_fine_regular_states(
                source_frst,
                max_depth=max_depth,
                max_nodes=max_nodes,
                neighbor_backend=neighbor_backend,
            )
            full_collection = bfs_result["collection"]

            candidate_signatures: List[Tuple[Tuple[int, ...], ...]] = []
            for signature, state_entry in full_collection.items():
                distance = int(state_entry["distance"])
                if target_depths is None or distance in target_depths:
                    candidate_signatures.append(signature)

            selected_signatures = _select_signatures_for_collection(
                candidate_signatures,
                rng,
                max_count=num_triangulations_per_frst,
                collect_all=collection_all,
            )

            candidate_pool_size_total += int(len(candidate_signatures))
            selected_count_total += int(len(selected_signatures))
            source_bfs_reports.append(
                {
                    "source_frst_index": int(source_frst_index),
                    "visited_count": int(bfs_result["visited_count"]),
                    "expanded_count": int(bfs_result["expanded_count"]),
                    "truncated": bool(bfs_result["truncated"]),
                    "stop_reason": str(bfs_result["stop_reason"]),
                    "candidate_count": int(len(candidate_signatures)),
                    "selected_count": int(len(selected_signatures)),
                    "sanity_notes": list(bfs_result["sanity_notes"]),
                }
            )

            for signature in selected_signatures:
                state_entry = full_collection[signature]
                distance = int(state_entry["distance"])
                existing = collected_states.get(signature)
                if existing is None or distance < int(existing["distance"]):
                    collected_states[signature] = {
                        "triangulation": state_entry["triangulation"],
                        "distance": int(distance),
                        "source_frst_index": int(source_frst_index),
                        "source_frst": source_frst,
                        "visited_count": int(bfs_result["visited_count"]),
                        "expanded_count": int(bfs_result["expanded_count"]),
                        "truncated": bool(bfs_result["truncated"]),
                        "stop_reason": str(bfs_result["stop_reason"]),
                        "candidate_count": int(len(candidate_signatures)),
                    }

    ordered_signatures = sorted(
        collected_states.keys(),
        key=lambda signature: (int(collected_states[signature]["distance"]), signature),
    )
    samples: List[Dict[str, Any]] = []
    for sample_index, signature in enumerate(ordered_signatures):
        selected_state = collected_states[signature]
        sample_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        samples.append(
            {
                "sample_index": int(sample_index),
                "sample_seed": sample_seed,
                "source_frst_index": int(selected_state["source_frst_index"]),
                "source_frst_seed": frst_seed_seed,
                "heights": None,
                "height_generation_method": "not_available_from_random_triangulations_fair",
                "generated_triangulation": serialize_triangulation(selected_state["triangulation"]),
                "distance_to_nearest_frst": int(selected_state["distance"]),
                "nearest_frst": serialize_triangulation(selected_state["source_frst"]),
                "bfs_search": {
                    "found": True,
                    "visited_count": int(selected_state["visited_count"]),
                    "expanded_count": int(selected_state["expanded_count"]),
                    "truncated": bool(selected_state["truncated"]),
                    "stop_reason": str(selected_state["stop_reason"]),
                    "candidate_count": int(selected_state["candidate_count"]),
                },
            }
        )

    polytope_entry["depth_collection"] = {
        "enabled": True,
        "strategy": "random_flip" if random_flip else "bfs_select",
        "random_flip": bool(random_flip),
        "random_flip_depth": int(max_depth) if random_flip else None,
        "collection_depths": None
        if effective_collection_depths is None
        else [int(depth) for depth in effective_collection_depths],
        "collection_depths_ignored": bool(random_flip),
        "collection_all": bool(collection_all),
        "candidate_pool_size": int(candidate_pool_size_total),
        "selected_count": int(selected_count_total),
        "unique_collected_count": int(len(samples)),
        "source_bfs_reports": source_bfs_reports,
    }
    polytope_entry["samples"] = samples

    truncated_count = sum(
        1 for report in source_bfs_reports if bool(report["truncated"])
    )
    return {
        "polytope_index": int(polytope_index),
        "polytope_entry": polytope_entry,
        "num_samples": int(len(samples)),
        "frst_found_count": int(len(samples)),
        "truncated_search_count": int(truncated_count),
    }


def _process_polytope_collection_job(job: Dict[str, Any]) -> Dict[str, Any]:
    record: K3Record = job["record"]
    m_polytope, n_polytope = _build_n_lattice_polytope(record)
    polytope_entry = _serialize_polytope(record, m_polytope, n_polytope)

    fair_call_timeout_seconds = job.get("fair_call_timeout_seconds")
    if fair_call_timeout_seconds is not None:
        fair_call_timeout_seconds = float(fair_call_timeout_seconds)

    return _collect_samples_for_n_polytope(
        polytope_index=int(record.record_index),
        polytope_entry=polytope_entry,
        n_polytope=n_polytope,
        polytope_seed=int(job["polytope_seed"]),
        frst_target=int(job["frst_target"]),
        num_triangulations_per_frst=int(job["num_triangulations_per_frst"]),
        fair_backend=str(job["fair_backend"]),
        fair_backend_fallback=job["fair_backend_fallback"],
        fair_max_retries=int(job["fair_max_retries"]),
        fair_max_attempt_rounds=int(job["fair_max_attempt_rounds"]),
        fair_call_timeout_seconds=fair_call_timeout_seconds,
        fast=bool(job.get("fast", False)),
        include_points_interior_to_facets=bool(job["include_points_interior_to_facets"]),
        make_star=bool(job["make_star"]),
        neighbor_backend=job["neighbor_backend"],
        max_depth=int(job["max_depth"]),
        max_nodes=int(job["max_nodes"]),
        collection_depths=job["collection_depths"],
        collection_all=bool(job["collection_all"]),
        random_flip=bool(job.get("random_flip", False)),
    )


def _process_fetched_polytope_collection_job(job: Dict[str, Any]) -> Dict[str, Any]:
    polytope_spec = dict(job["polytope_spec"])
    _require_cytools()
    n_polytope = Polytope(np.asarray(polytope_spec["vertices"], dtype=np.int64))
    polytope_entry = _serialize_fetched_n_polytope(polytope_spec, n_polytope)

    fair_call_timeout_seconds = job.get("fair_call_timeout_seconds")
    if fair_call_timeout_seconds is not None:
        fair_call_timeout_seconds = float(fair_call_timeout_seconds)

    return _collect_samples_for_n_polytope(
        polytope_index=int(polytope_spec["polytope_index"]),
        polytope_entry=polytope_entry,
        n_polytope=n_polytope,
        polytope_seed=int(job["polytope_seed"]),
        frst_target=int(job["frst_target"]),
        num_triangulations_per_frst=int(job["num_triangulations_per_frst"]),
        fair_backend=str(job["fair_backend"]),
        fair_backend_fallback=job["fair_backend_fallback"],
        fair_max_retries=int(job["fair_max_retries"]),
        fair_max_attempt_rounds=int(job["fair_max_attempt_rounds"]),
        fair_call_timeout_seconds=fair_call_timeout_seconds,
        fast=bool(job.get("fast", False)),
        include_points_interior_to_facets=bool(job["include_points_interior_to_facets"]),
        make_star=bool(job["make_star"]),
        neighbor_backend=job["neighbor_backend"],
        max_depth=int(job["max_depth"]),
        max_nodes=int(job["max_nodes"]),
        collection_depths=job["collection_depths"],
        collection_all=bool(job["collection_all"]),
        random_flip=bool(job.get("random_flip", False)),
    )


def _process_fetched_polytope_random_height_eval_job(job: Dict[str, Any]) -> Dict[str, Any]:
    polytope_spec = dict(job["polytope_spec"])
    _require_cytools()
    n_polytope = Polytope(np.asarray(polytope_spec["vertices"], dtype=np.int64))
    polytope_entry = _serialize_fetched_n_polytope(polytope_spec, n_polytope)

    generation_result = _generate_random_height_non_fine_triangulations(
        n_polytope,
        num_triangulations=int(job["num_triangulations"]),
        max_tries=int(job["max_tries"]),
        seed=int(job["polytope_seed"]),
        triangulation_backend=job.get("triangulation_backend"),
        include_points_interior_to_facets=bool(job["include_points_interior_to_facets"]),
        make_star=job.get("make_star"),
        triangulation_verbosity=int(job["triangulation_verbosity"]),
    )
    polytope_entry["random_height_generation"] = generation_result["diagnostics"]
    polytope_entry["non_fine_triangulations"] = generation_result["triangulations"]

    generated_count = int(len(generation_result["triangulations"]))
    requested_count = int(job["num_triangulations"])
    return {
        "polytope_index": int(polytope_spec["polytope_index"]),
        "polytope_entry": polytope_entry,
        "num_samples": generated_count,
        "underfilled_polytope_count": int(generated_count < requested_count),
    }


def _process_k3_polytope_random_height_eval_job(job: Dict[str, Any]) -> Dict[str, Any]:
    record: K3Record = job["record"]
    m_polytope, n_polytope = _build_n_lattice_polytope(record)
    polytope_entry = _serialize_polytope(record, m_polytope, n_polytope)

    generation_result = _generate_random_height_non_fine_triangulations(
        n_polytope,
        num_triangulations=int(job["num_triangulations"]),
        max_tries=int(job["max_tries"]),
        seed=int(job["polytope_seed"]),
        triangulation_backend=job.get("triangulation_backend"),
        include_points_interior_to_facets=bool(job["include_points_interior_to_facets"]),
        make_star=job.get("make_star"),
        triangulation_verbosity=int(job["triangulation_verbosity"]),
    )
    polytope_entry["random_height_generation"] = generation_result["diagnostics"]
    polytope_entry["non_fine_triangulations"] = generation_result["triangulations"]

    generated_count = int(len(generation_result["triangulations"]))
    requested_count = int(job["num_triangulations"])
    return {
        "polytope_index": int(record.record_index),
        "polytope_entry": polytope_entry,
        "num_samples": generated_count,
        "underfilled_polytope_count": int(generated_count < requested_count),
    }


def generate_cy_reflexive_dataset(
    *,
    num_polytopes: int,
    num_triangulations_per_frst: Optional[int] = None,
    triangulations_per_polytope: Optional[int] = None,
    seed: int = 0,
    k3_path: Optional[str] = None,
    triangulation_backend: str = "qhull",
    neighbor_backend: Optional[str] = None,
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = True,
    max_retries_per_triangulation: int = 8,
    height_scale: float = 1.0,
    frsts_per_polytope: Optional[int] = None,
    fair_backend: str = "cgal",
    fair_backend_fallback: Optional[str] = "qhull",
    fair_max_retries: int = 20,
    fair_max_attempt_rounds: int = 4,
    fair_call_timeout_seconds: Optional[float] = None,
    fast: bool = False,
    bfs_max_depth: int = 4,
    bfs_max_nodes: int = 300,
    max_depth: Optional[int] = None,
    max_node: Optional[int] = None,
    collection_depths: Optional[Sequence[int]] = None,
    collection_all: bool = False,
    random_flip: bool = False,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    if num_polytopes <= 0:
        raise ValueError("num_polytopes must be positive.")
    if num_triangulations_per_frst is None and triangulations_per_polytope is None:
        raise ValueError(
            "Either num_triangulations_per_frst or triangulations_per_polytope must be provided."
        )
    if num_triangulations_per_frst is not None and triangulations_per_polytope is not None:
        if int(num_triangulations_per_frst) != int(triangulations_per_polytope):
            raise ValueError(
                "num_triangulations_per_frst and triangulations_per_polytope must match when both are provided."
            )
    resolved_num_triangulations_per_frst = int(
        triangulations_per_polytope
        if num_triangulations_per_frst is None
        else num_triangulations_per_frst
    )
    if resolved_num_triangulations_per_frst <= 0:
        raise ValueError("num_triangulations_per_frst must be positive.")
    if frsts_per_polytope is not None and frsts_per_polytope <= 0:
        raise ValueError("frsts_per_polytope must be positive when provided.")
    if fair_call_timeout_seconds is not None and fair_call_timeout_seconds <= 0:
        raise ValueError("fair_call_timeout_seconds must be positive when provided.")

    resolved_bfs_max_depth = int(bfs_max_depth if max_depth is None else max_depth)
    resolved_bfs_max_nodes = int(bfs_max_nodes if max_node is None else max_node)
    if resolved_bfs_max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if resolved_bfs_max_nodes <= 0:
        raise ValueError("max_node/max_nodes must be positive.")
    if random_flip:
        normalized_collection_depths = None
        effective_bfs_max_depth = resolved_bfs_max_depth
    else:
        normalized_collection_depths = _normalize_collection_depths(collection_depths)
        if normalized_collection_depths is None:
            effective_bfs_max_depth = resolved_bfs_max_depth
        else:
            effective_bfs_max_depth = max(resolved_bfs_max_depth, max(normalized_collection_depths))

    records = load_k3_records(k3_path=k3_path, max_polytopes=num_polytopes)
    if len(records) < num_polytopes:
        raise ValueError(
            f"Requested {num_polytopes} polytopes, but only found {len(records)} entries."
        )

    resolved_num_workers = _resolve_num_workers(num_workers, num_polytopes=num_polytopes)
    root_rng = np.random.default_rng(seed)
    frst_target = (
        int(resolved_num_triangulations_per_frst)
        if frsts_per_polytope is None
        else int(frsts_per_polytope)
    )

    jobs: List[Dict[str, Any]] = []
    for record in records:
        polytope_seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
        jobs.append(
            {
                "record": record,
                "polytope_seed": polytope_seed,
                "frst_target": frst_target,
                "num_triangulations_per_frst": int(resolved_num_triangulations_per_frst),
                "fair_backend": fair_backend,
                "fair_backend_fallback": fair_backend_fallback,
                "fair_max_retries": int(fair_max_retries),
                "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
                "fair_call_timeout_seconds": None
                if fair_call_timeout_seconds is None
                else float(fair_call_timeout_seconds),
                "fast": bool(fast),
                "include_points_interior_to_facets": bool(include_points_interior_to_facets),
                "make_star": bool(make_star) if make_star is not None else True,
                "neighbor_backend": neighbor_backend,
                "max_depth": int(effective_bfs_max_depth),
                "max_nodes": int(resolved_bfs_max_nodes),
                "collection_depths": None
                if random_flip or normalized_collection_depths is None
                else [int(depth) for depth in normalized_collection_depths],
                "collection_all": bool(collection_all),
                "random_flip": bool(random_flip),
            }
        )

    if resolved_num_workers == 1:
        worker_outputs = [_process_polytope_collection_job(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_num_workers) as executor:
            futures = [executor.submit(_process_polytope_collection_job, job) for job in jobs]
            worker_outputs = [future.result() for future in futures]
    worker_outputs.sort(key=lambda item: int(item["polytope_index"]))

    dataset: Dict[str, Any] = {
        "metadata": {
            "seed": int(seed),
            "k3_path": str(k3_path) if k3_path is not None else None,
            "num_polytopes": int(num_polytopes),
            "num_triangulations_per_frst": int(resolved_num_triangulations_per_frst),
            "triangulations_per_polytope": int(resolved_num_triangulations_per_frst),
            "triangulation_backend": triangulation_backend,
            "neighbor_backend": neighbor_backend,
            "include_points_interior_to_facets": bool(include_points_interior_to_facets),
            "make_star": None if make_star is None else bool(make_star),
            "generation_mode": "frst_random_flip_collection"
            if random_flip
            else "frst_bfs_state_collection",
            "random_flip": bool(random_flip),
            "frsts_per_polytope": None if frsts_per_polytope is None else int(frsts_per_polytope),
            "fair_backend": fair_backend,
            "fair_backend_fallback": fair_backend_fallback,
            "fair_max_retries": int(fair_max_retries),
            "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
            "fair_call_timeout_seconds": None
            if fair_call_timeout_seconds is None
            else float(fair_call_timeout_seconds),
            "fast": bool(fast),
            "frst_seed_sampler": "fast" if fast else "fair_then_fast",
            "bfs_max_depth": int(resolved_bfs_max_depth),
            "bfs_max_nodes": int(resolved_bfs_max_nodes),
            "effective_bfs_max_depth": int(effective_bfs_max_depth),
            "collection_depths": None
            if random_flip or normalized_collection_depths is None
            else [int(depth) for depth in normalized_collection_depths],
            "collection_all": bool(collection_all),
            "collection_unit_k_per_frst": int(resolved_num_triangulations_per_frst),
            "num_workers": int(resolved_num_workers),
            "triangulation_verbosity": int(triangulation_verbosity),
            "lattice_space": "N",
            "legacy_random_height_fields_unused": {
                "max_retries_per_triangulation": int(max_retries_per_triangulation),
                "height_scale": float(height_scale),
            },
        },
        "polytopes": [],
    }

    total_samples = int(sum(int(item["num_samples"]) for item in worker_outputs))
    found_frst_count = int(sum(int(item["frst_found_count"]) for item in worker_outputs))
    truncated_search_count = int(
        sum(int(item["truncated_search_count"]) for item in worker_outputs)
    )
    dataset["polytopes"] = [item["polytope_entry"] for item in worker_outputs]
    dataset["metadata"]["num_samples"] = int(total_samples)
    dataset["metadata"]["frst_found_count"] = int(found_frst_count)
    dataset["metadata"]["frst_not_found_count"] = int(total_samples - found_frst_count)
    dataset["metadata"]["truncated_search_count"] = int(truncated_search_count)
    return dataset


def _build_polytope_jsonl_rows(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _simplices_in_polytope_vertex_space(triangulation_record: Dict[str, Any]) -> List[List[int]]:
        # `simplices` are in local triangulation-point indices; convert them to
        # polytope-vertex indices so they align with `vertices_n`.
        local_to_poly = [int(index) for index in triangulation_record["point_indices"]]
        simplices = triangulation_record["simplices"]
        return [
            [int(local_to_poly[int(vertex)]) for vertex in simplex]
            for simplex in simplices
        ]

    rows: List[Dict[str, Any]] = []
    for polytope_entry in dataset["polytopes"]:
        frst_seeds = list(polytope_entry.get("frst_seeds", []))
        frst_items: List[Dict[str, Any]] = []
        for frst_index, frst in enumerate(frst_seeds):
            frst_items.append(
                {
                    "frst_index": int(frst_index),
                    "simplices": _simplices_in_polytope_vertex_space(frst),
                    "triangulation_list": [],
                }
            )

        polytope_samples = list(polytope_entry.get("samples", []))
        total_triangulations = 0
        for sample in polytope_samples:
            tri_item = {
                "distance": int(sample["distance_to_nearest_frst"]),
                "simplices": _simplices_in_polytope_vertex_space(sample["generated_triangulation"]),
            }
            frst_index = int(sample["source_frst_index"])
            if 0 <= frst_index < len(frst_items):
                frst_items[frst_index]["triangulation_list"].append(tri_item)
                total_triangulations += 1

        row = {
            "polytope_index": int(polytope_entry["polytope_index"]),
            "vertices": polytope_entry["n_points"],
            "frst_list": frst_items,
            "non_fine_triangulation_count": int(total_triangulations),
        }
        if "h11" in polytope_entry:
            row["h11"] = int(polytope_entry["h11"])
        if "favorable" in polytope_entry:
            row["favorable"] = bool(polytope_entry["favorable"])
        rows.append(row)
    return rows


def _build_random_height_eval_polytope_rows(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for polytope_entry in dataset["polytopes"]:
        triangulation_list = [
            {
                "heights": [float(value) for value in triangulation["heights"]],
                "signature": [
                    [int(vertex) for vertex in simplex]
                    for simplex in triangulation["signature"]
                ],
            }
            for triangulation in polytope_entry.get("non_fine_triangulations", [])
        ]
        row = {
            "polytope_index": int(polytope_entry["polytope_index"]),
            "vertices": polytope_entry["n_points"],
            "non_fine_triangulation_list": triangulation_list,
            "non_fine_triangulation_count": int(len(triangulation_list)),
        }
        if "h11" in polytope_entry:
            row["h11"] = int(polytope_entry["h11"])
        if "favorable" in polytope_entry:
            row["favorable"] = bool(polytope_entry["favorable"])
        rows.append(row)
    return rows


def _build_compact_dataset_for_disk(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": dataset.get("metadata", {}),
        "polytopes": _build_polytope_jsonl_rows(dataset),
    }


def _build_random_height_eval_dataset_for_disk(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": dataset.get("metadata", {}),
        "polytopes": _build_random_height_eval_polytope_rows(dataset),
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _build_generation_metadata(
    *,
    seed: int,
    k3_path: Optional[str],
    num_polytopes: int,
    resolved_num_triangulations_per_frst: int,
    triangulation_backend: str,
    neighbor_backend: Optional[str],
    include_points_interior_to_facets: bool,
    make_star: Optional[bool],
    frsts_per_polytope: Optional[int],
    fair_backend: str,
    fair_backend_fallback: Optional[str],
    fair_max_retries: int,
    fair_max_attempt_rounds: int,
    fair_call_timeout_seconds: Optional[float],
    fast: bool,
    resolved_bfs_max_depth: int,
    resolved_bfs_max_nodes: int,
    effective_bfs_max_depth: int,
    normalized_collection_depths: Optional[List[int]],
    collection_all: bool,
    random_flip: bool,
    triangulation_verbosity: int,
    max_retries_per_triangulation: int,
    height_scale: float,
    resolved_num_workers: int,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "k3_path": str(k3_path) if k3_path is not None else None,
        "num_polytopes": int(num_polytopes),
        "num_triangulations_per_frst": int(resolved_num_triangulations_per_frst),
        "triangulations_per_polytope": int(resolved_num_triangulations_per_frst),
        "triangulation_backend": triangulation_backend,
        "neighbor_backend": neighbor_backend,
        "include_points_interior_to_facets": bool(include_points_interior_to_facets),
        "make_star": None if make_star is None else bool(make_star),
        "generation_mode": "frst_random_flip_collection"
        if random_flip
        else "frst_bfs_state_collection",
        "random_flip": bool(random_flip),
        "frsts_per_polytope": None if frsts_per_polytope is None else int(frsts_per_polytope),
        "fair_backend": fair_backend,
        "fair_backend_fallback": fair_backend_fallback,
        "fair_max_retries": int(fair_max_retries),
        "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
        "fair_call_timeout_seconds": None
        if fair_call_timeout_seconds is None
        else float(fair_call_timeout_seconds),
        "fast": bool(fast),
        "frst_seed_sampler": "fast" if fast else "fair_then_fast",
        "bfs_max_depth": int(resolved_bfs_max_depth),
        "bfs_max_nodes": int(resolved_bfs_max_nodes),
        "effective_bfs_max_depth": int(effective_bfs_max_depth),
        "collection_depths": None
        if random_flip or normalized_collection_depths is None
        else [int(depth) for depth in normalized_collection_depths],
        "collection_all": bool(collection_all),
        "collection_unit_k_per_frst": int(resolved_num_triangulations_per_frst),
        "num_workers": int(resolved_num_workers),
        "triangulation_verbosity": int(triangulation_verbosity),
        "lattice_space": "N",
        "legacy_random_height_fields_unused": {
            "max_retries_per_triangulation": int(max_retries_per_triangulation),
            "height_scale": float(height_scale),
        },
    }


def _run_incremental_collection_jobs(
    *,
    jobs: Sequence[Dict[str, Any]],
    num_polytopes: int,
    resolved_num_workers: int,
    process_job_fn: Any,
    job_index_fn: Any,
    metadata: Dict[str, Any],
    output_dir: str,
    output_name: str,
    compact_output: bool,
    resume: bool,
    checkpoint_path: Optional[str],
    log_every: int,
) -> Dict[str, Any]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_jsonl_path = output_root / f"{output_name}.samples.jsonl"
    dataset_json_path = output_root / f"{output_name}.json"
    checkpoint_file = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path is not None
        else output_root / f"{output_name}.checkpoint.json"
    )

    completed_indices: set[int] = set()
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    total_samples = 0
    found_frst_count = 0
    truncated_search_count = 0
    truncated_by_polytope: Dict[int, int] = {}

    if resume and checkpoint_file.exists():
        with checkpoint_file.open("r", encoding="utf-8") as handle:
            checkpoint_data = json.load(handle)
        progress = checkpoint_data.get("progress", {})
        completed_indices = {
            int(index)
            for index in progress.get("completed_polytope_indices", [])
            if 0 <= int(index) < num_polytopes
        }
        total_samples = int(progress.get("num_samples", 0))
        found_frst_count = int(progress.get("frst_found_count", total_samples))
        truncated_search_count = int(progress.get("truncated_search_count", 0))
        truncated_by_polytope = {
            int(index): int(value)
            for index, value in progress.get("truncated_by_polytope", {}).items()
            if 0 <= int(index) < num_polytopes
        }
        if samples_jsonl_path.exists():
            with samples_jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_by_index[int(row["polytope_index"])] = row
        # Trust existing .samples.jsonl rows as durable completion state to avoid
        # duplicate appends after crashes between row write and checkpoint write.
        completed_indices = set(rows_by_index.keys())
        total_samples = int(
            sum(
                int(row.get("non_fine_triangulation_count", 0))
                for row in rows_by_index.values()
            )
        )
        found_frst_count = int(total_samples)
        truncated_search_count = int(
            sum(int(truncated_by_polytope.get(index, 0)) for index in completed_indices)
        )
    else:
        if samples_jsonl_path.exists():
            samples_jsonl_path.unlink()
        if dataset_json_path.exists():
            dataset_json_path.unlink()
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    pending_jobs = [
        job for job in jobs if int(job_index_fn(job)) not in completed_indices
    ]
    completed_before_start = int(len(completed_indices))

    start_time = time.monotonic()
    total_target = len(jobs)
    last_logged_done = -1

    def _save_checkpoint() -> None:
        progress_payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "progress": {
                "completed_polytope_count": int(len(completed_indices)),
                "completed_polytope_indices": sorted(int(index) for index in completed_indices),
                "num_samples": int(total_samples),
                "frst_found_count": int(found_frst_count),
                "truncated_search_count": int(truncated_search_count),
                "truncated_by_polytope": {
                    str(index): int(value)
                    for index, value in sorted(truncated_by_polytope.items())
                },
            },
            "metadata": metadata,
        }
        _write_json_atomic(checkpoint_file, progress_payload)

    def _log_progress(force: bool = False) -> None:
        nonlocal last_logged_done
        done = len(completed_indices)
        if force and done == last_logged_done:
            return
        if not force and done % int(log_every) != 0:
            return
        elapsed = max(1e-6, time.monotonic() - start_time)
        completed_now = max(0, done - completed_before_start)
        rate = completed_now / elapsed
        remaining = max(0, total_target - done)
        eta_seconds = int(remaining / rate) if rate > 0 else -1
        eta_str = f"{eta_seconds}s" if eta_seconds >= 0 else "unknown"
        print(
            f"[progress] completed={done}/{total_target} "
            f"samples={total_samples} truncated={truncated_search_count} "
            f"rate={rate:.2f} poly/s eta={eta_str}"
        )
        last_logged_done = done

    def _handle_result(
        result: Dict[str, Any],
        *,
        handle,
    ) -> None:
        nonlocal total_samples, found_frst_count, truncated_search_count
        polytope_index = int(result["polytope_index"])
        if polytope_index in completed_indices:
            return
        compact_row = _build_polytope_jsonl_rows({"polytopes": [result["polytope_entry"]]})[0]
        rows_by_index[polytope_index] = compact_row
        handle.write(json.dumps(compact_row))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

        completed_indices.add(polytope_index)
        truncated_by_polytope[polytope_index] = int(result["truncated_search_count"])
        total_samples += int(result["num_samples"])
        found_frst_count += int(result["frst_found_count"])
        truncated_search_count += int(result["truncated_search_count"])
        _save_checkpoint()
        _log_progress()

    print(
        f"[start] total_polytopes={total_target} pending={len(pending_jobs)} "
        f"resume={resume} workers={resolved_num_workers}"
    )
    with samples_jsonl_path.open("a", encoding="utf-8") as samples_handle:
        if len(pending_jobs) == 0:
            pass
        elif resolved_num_workers == 1:
            for job in pending_jobs:
                result = process_job_fn(job)
                _handle_result(result, handle=samples_handle)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_num_workers) as executor:
                futures = [
                    executor.submit(process_job_fn, job)
                    for job in pending_jobs
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    _handle_result(result, handle=samples_handle)

    _log_progress(force=True)
    sorted_polytopes = [
        rows_by_index[index] for index in sorted(rows_by_index)
    ]
    metadata["num_samples"] = int(total_samples)
    metadata["frst_found_count"] = int(found_frst_count)
    metadata["frst_not_found_count"] = int(total_samples - found_frst_count)
    metadata["truncated_search_count"] = int(truncated_search_count)

    if not compact_output:
        compact_dataset = {
            "metadata": metadata,
            "polytopes": sorted_polytopes,
        }
        _write_json_atomic(dataset_json_path, compact_dataset)

    paths = {
        "samples_jsonl": str(samples_jsonl_path),
        "checkpoint_json": str(checkpoint_file),
    }
    if not compact_output:
        paths["dataset_json"] = str(dataset_json_path)

    summary = {
        "num_polytopes": int(num_polytopes),
        "completed_polytope_count": int(len(completed_indices)),
        "num_samples": int(total_samples),
        "frst_found_count": int(found_frst_count),
        "frst_not_found_count": int(total_samples - found_frst_count),
        "truncated_search_count": int(truncated_search_count),
        "per_polytope": [
            {
                "polytope_index": int(poly["polytope_index"]),
                "num_samples": int(poly["non_fine_triangulation_count"]),
            }
            for poly in sorted_polytopes
        ],
    }
    return {
        "paths": paths,
        "summary": summary,
        "metadata": metadata,
    }


def generate_and_save_cy_reflexive_dataset_incremental(
    *,
    num_polytopes: int,
    num_triangulations_per_frst: Optional[int] = None,
    triangulations_per_polytope: Optional[int] = None,
    seed: int = 0,
    k3_path: Optional[str] = None,
    triangulation_backend: str = "qhull",
    neighbor_backend: Optional[str] = None,
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = True,
    max_retries_per_triangulation: int = 8,
    height_scale: float = 1.0,
    frsts_per_polytope: Optional[int] = None,
    fair_backend: str = "cgal",
    fair_backend_fallback: Optional[str] = "qhull",
    fair_max_retries: int = 20,
    fair_max_attempt_rounds: int = 4,
    fair_call_timeout_seconds: Optional[float] = None,
    fast: bool = False,
    bfs_max_depth: int = 4,
    bfs_max_nodes: int = 300,
    max_depth: Optional[int] = None,
    max_node: Optional[int] = None,
    collection_depths: Optional[Sequence[int]] = None,
    collection_all: bool = False,
    random_flip: bool = False,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
    output_dir: str = "data/cy/output",
    output_name: str = "cy_reflexive_dataset",
    compact_output: bool = False,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    log_every: int = 10,
) -> Dict[str, Any]:
    if num_polytopes <= 0:
        raise ValueError("num_polytopes must be positive.")
    if num_triangulations_per_frst is None and triangulations_per_polytope is None:
        raise ValueError(
            "Either num_triangulations_per_frst or triangulations_per_polytope must be provided."
        )
    if num_triangulations_per_frst is not None and triangulations_per_polytope is not None:
        if int(num_triangulations_per_frst) != int(triangulations_per_polytope):
            raise ValueError(
                "num_triangulations_per_frst and triangulations_per_polytope must match when both are provided."
            )
    resolved_num_triangulations_per_frst = int(
        triangulations_per_polytope
        if num_triangulations_per_frst is None
        else num_triangulations_per_frst
    )
    if resolved_num_triangulations_per_frst <= 0:
        raise ValueError("num_triangulations_per_frst must be positive.")
    if frsts_per_polytope is not None and frsts_per_polytope <= 0:
        raise ValueError("frsts_per_polytope must be positive when provided.")
    if fair_call_timeout_seconds is not None and fair_call_timeout_seconds <= 0:
        raise ValueError("fair_call_timeout_seconds must be positive when provided.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    resolved_bfs_max_depth = int(bfs_max_depth if max_depth is None else max_depth)
    resolved_bfs_max_nodes = int(bfs_max_nodes if max_node is None else max_node)
    if resolved_bfs_max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if resolved_bfs_max_nodes <= 0:
        raise ValueError("max_node/max_nodes must be positive.")
    if random_flip:
        normalized_collection_depths = None
        effective_bfs_max_depth = resolved_bfs_max_depth
    else:
        normalized_collection_depths = _normalize_collection_depths(collection_depths)
        if normalized_collection_depths is None:
            effective_bfs_max_depth = resolved_bfs_max_depth
        else:
            effective_bfs_max_depth = max(resolved_bfs_max_depth, max(normalized_collection_depths))

    records = load_k3_records(k3_path=k3_path, max_polytopes=num_polytopes)
    if len(records) < num_polytopes:
        raise ValueError(
            f"Requested {num_polytopes} polytopes, but only found {len(records)} entries."
        )

    resolved_num_workers = _resolve_num_workers(num_workers, num_polytopes=num_polytopes)
    root_rng = np.random.default_rng(seed)
    frst_target = (
        int(resolved_num_triangulations_per_frst)
        if frsts_per_polytope is None
        else int(frsts_per_polytope)
    )
    jobs: List[Dict[str, Any]] = []
    for record in records:
        polytope_seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
        jobs.append(
            {
                "record": record,
                "polytope_seed": polytope_seed,
                "frst_target": frst_target,
                "num_triangulations_per_frst": int(resolved_num_triangulations_per_frst),
                "fair_backend": fair_backend,
                "fair_backend_fallback": fair_backend_fallback,
                "fair_max_retries": int(fair_max_retries),
                "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
                "fair_call_timeout_seconds": None
                if fair_call_timeout_seconds is None
                else float(fair_call_timeout_seconds),
                "fast": bool(fast),
                "include_points_interior_to_facets": bool(include_points_interior_to_facets),
                "make_star": bool(make_star) if make_star is not None else True,
                "neighbor_backend": neighbor_backend,
                "max_depth": int(effective_bfs_max_depth),
                "max_nodes": int(resolved_bfs_max_nodes),
                "collection_depths": None
                if random_flip or normalized_collection_depths is None
                else [int(depth) for depth in normalized_collection_depths],
                "collection_all": bool(collection_all),
                "random_flip": bool(random_flip),
            }
        )

    metadata = _build_generation_metadata(
        seed=seed,
        k3_path=k3_path,
        num_polytopes=num_polytopes,
        resolved_num_triangulations_per_frst=resolved_num_triangulations_per_frst,
        triangulation_backend=triangulation_backend,
        neighbor_backend=neighbor_backend,
        include_points_interior_to_facets=include_points_interior_to_facets,
        make_star=make_star,
        frsts_per_polytope=frsts_per_polytope,
        fair_backend=fair_backend,
        fair_backend_fallback=fair_backend_fallback,
        fair_max_retries=fair_max_retries,
        fair_max_attempt_rounds=fair_max_attempt_rounds,
        fair_call_timeout_seconds=fair_call_timeout_seconds,
        fast=fast,
        resolved_bfs_max_depth=resolved_bfs_max_depth,
        resolved_bfs_max_nodes=resolved_bfs_max_nodes,
        effective_bfs_max_depth=effective_bfs_max_depth,
        normalized_collection_depths=normalized_collection_depths,
        collection_all=collection_all,
        random_flip=random_flip,
        triangulation_verbosity=triangulation_verbosity,
        max_retries_per_triangulation=max_retries_per_triangulation,
        height_scale=height_scale,
        resolved_num_workers=resolved_num_workers,
    )
    return _run_incremental_collection_jobs(
        jobs=jobs,
        num_polytopes=num_polytopes,
        resolved_num_workers=resolved_num_workers,
        process_job_fn=_process_polytope_collection_job,
        job_index_fn=lambda job: int(job["record"].record_index),
        metadata=metadata,
        output_dir=output_dir,
        output_name=output_name,
        compact_output=compact_output,
        resume=resume,
        checkpoint_path=checkpoint_path,
        log_every=log_every,
    )


def generate_and_save_cy_4d_reflexive_dataset_incremental(
    *,
    num_polytopes: Optional[int] = None,
    h11: Optional[int] = None,
    num_vertices: Optional[int] = None,
    favorable: Optional[bool] = None,
    num_triangulations_per_frst: Optional[int] = None,
    triangulations_per_polytope: Optional[int] = None,
    polytope_file: Optional[str] = None,
    seed: int = 0,
    triangulation_backend: str = "qhull",
    neighbor_backend: Optional[str] = None,
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = True,
    max_retries_per_triangulation: int = 8,
    height_scale: float = 1.0,
    frsts_per_polytope: Optional[int] = None,
    fair_backend: str = "cgal",
    fair_backend_fallback: Optional[str] = "qhull",
    fair_max_retries: int = 20,
    fair_max_attempt_rounds: int = 4,
    fair_call_timeout_seconds: Optional[float] = None,
    fast: bool = False,
    bfs_max_depth: int = 4,
    bfs_max_nodes: int = 300,
    max_depth: Optional[int] = None,
    max_node: Optional[int] = None,
    collection_depths: Optional[Sequence[int]] = None,
    collection_all: bool = False,
    random_flip: bool = False,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
    output_dir: str = "data/cy/output",
    output_name: str = "cy_reflexive_dataset_4d",
    compact_output: bool = False,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    log_every: int = 10,
) -> Dict[str, Any]:
    if num_triangulations_per_frst is None and triangulations_per_polytope is None:
        raise ValueError(
            "Either num_triangulations_per_frst or triangulations_per_polytope must be provided."
        )
    if num_triangulations_per_frst is not None and triangulations_per_polytope is not None:
        if int(num_triangulations_per_frst) != int(triangulations_per_polytope):
            raise ValueError(
                "num_triangulations_per_frst and triangulations_per_polytope must match when both are provided."
            )
    resolved_num_triangulations_per_frst = int(
        triangulations_per_polytope
        if num_triangulations_per_frst is None
        else num_triangulations_per_frst
    )
    if resolved_num_triangulations_per_frst <= 0:
        raise ValueError("num_triangulations_per_frst must be positive.")
    if frsts_per_polytope is not None and frsts_per_polytope <= 0:
        raise ValueError("frsts_per_polytope must be positive when provided.")
    if num_vertices is not None and int(num_vertices) <= 0:
        raise ValueError("num_vertices must be positive when provided.")
    if fair_call_timeout_seconds is not None and fair_call_timeout_seconds <= 0:
        raise ValueError("fair_call_timeout_seconds must be positive when provided.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    resolved_bfs_max_depth = int(bfs_max_depth if max_depth is None else max_depth)
    resolved_bfs_max_nodes = int(bfs_max_nodes if max_node is None else max_node)
    if resolved_bfs_max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if resolved_bfs_max_nodes <= 0:
        raise ValueError("max_node/max_nodes must be positive.")
    if random_flip:
        normalized_collection_depths = None
        effective_bfs_max_depth = resolved_bfs_max_depth
    else:
        normalized_collection_depths = _normalize_collection_depths(collection_depths)
        if normalized_collection_depths is None:
            effective_bfs_max_depth = resolved_bfs_max_depth
        else:
            effective_bfs_max_depth = max(resolved_bfs_max_depth, max(normalized_collection_depths))

    if polytope_file is None:
        if num_polytopes is None or int(num_polytopes) <= 0:
            raise ValueError("num_polytopes must be positive when polytope_file is not provided.")
        if h11 is None:
            raise ValueError("h11 must be provided when polytope_file is not provided.")
        polytope_specs = _fetch_4d_n_lattice_polytope_specs(
            num_polytopes=int(num_polytopes),
            h11=int(h11),
            num_vertices=num_vertices,
            favorable=favorable,
        )
    else:
        polytope_specs = _load_4d_n_lattice_polytope_specs_from_file(
            polytope_file=polytope_file,
            num_polytopes=None if num_polytopes is None else int(num_polytopes),
        )
    actual_num_polytopes = int(len(polytope_specs))
    resolved_num_workers = _resolve_num_workers(
        num_workers,
        num_polytopes=max(1, actual_num_polytopes),
    )
    root_rng = np.random.default_rng(seed)
    frst_target = (
        int(resolved_num_triangulations_per_frst)
        if frsts_per_polytope is None
        else int(frsts_per_polytope)
    )
    jobs: List[Dict[str, Any]] = []
    for polytope_spec in polytope_specs:
        polytope_seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
        jobs.append(
            {
                "polytope_spec": polytope_spec,
                "polytope_seed": polytope_seed,
                "frst_target": frst_target,
                "num_triangulations_per_frst": int(resolved_num_triangulations_per_frst),
                "fair_backend": fair_backend,
                "fair_backend_fallback": fair_backend_fallback,
                "fair_max_retries": int(fair_max_retries),
                "fair_max_attempt_rounds": int(fair_max_attempt_rounds),
                "fair_call_timeout_seconds": None
                if fair_call_timeout_seconds is None
                else float(fair_call_timeout_seconds),
                "fast": bool(fast),
                "include_points_interior_to_facets": bool(include_points_interior_to_facets),
                "make_star": bool(make_star) if make_star is not None else True,
                "neighbor_backend": neighbor_backend,
                "max_depth": int(effective_bfs_max_depth),
                "max_nodes": int(resolved_bfs_max_nodes),
                "collection_depths": None
                if random_flip or normalized_collection_depths is None
                else [int(depth) for depth in normalized_collection_depths],
                "collection_all": bool(collection_all),
                "random_flip": bool(random_flip),
            }
        )

    metadata = _build_generation_metadata(
        seed=seed,
        k3_path=None,
        num_polytopes=actual_num_polytopes,
        resolved_num_triangulations_per_frst=resolved_num_triangulations_per_frst,
        triangulation_backend=triangulation_backend,
        neighbor_backend=neighbor_backend,
        include_points_interior_to_facets=include_points_interior_to_facets,
        make_star=make_star,
        frsts_per_polytope=frsts_per_polytope,
        fair_backend=fair_backend,
        fair_backend_fallback=fair_backend_fallback,
        fair_max_retries=fair_max_retries,
        fair_max_attempt_rounds=fair_max_attempt_rounds,
        fair_call_timeout_seconds=fair_call_timeout_seconds,
        fast=fast,
        resolved_bfs_max_depth=resolved_bfs_max_depth,
        resolved_bfs_max_nodes=resolved_bfs_max_nodes,
        effective_bfs_max_depth=effective_bfs_max_depth,
        normalized_collection_depths=normalized_collection_depths,
        collection_all=collection_all,
        random_flip=random_flip,
        triangulation_verbosity=triangulation_verbosity,
        max_retries_per_triangulation=max_retries_per_triangulation,
        height_scale=height_scale,
        resolved_num_workers=resolved_num_workers,
    )
    metadata.update(
        {
            "dataset_dimension": 4,
            "polytope_source": "cytools.fetch_polytopes"
            if polytope_file is None
            else "polytope_file",
            "polytope_file": None if polytope_file is None else str(Path(polytope_file).expanduser()),
            "requested_num_polytopes": actual_num_polytopes
            if num_polytopes is None
            else int(num_polytopes),
            "h11": None if h11 is None else int(h11),
            "num_vertices": None if num_vertices is None else int(num_vertices),
            "favorable": None if favorable is None else bool(favorable),
        }
    )

    return _run_incremental_collection_jobs(
        jobs=jobs,
        num_polytopes=actual_num_polytopes,
        resolved_num_workers=resolved_num_workers,
        process_job_fn=_process_fetched_polytope_collection_job,
        job_index_fn=lambda job: int(job["polytope_spec"]["polytope_index"]),
        metadata=metadata,
        output_dir=output_dir,
        output_name=output_name,
        compact_output=compact_output,
        resume=resume,
        checkpoint_path=checkpoint_path,
        log_every=log_every,
    )


def generate_cy_4d_random_height_eval_dataset(
    *,
    num_polytopes: int,
    h11: int,
    num_vertices: Optional[int] = None,
    favorable: Optional[bool] = None,
    num_triangulations: int,
    max_tries: int,
    seed: int = 0,
    triangulation_backend: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = None,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    if num_polytopes <= 0:
        raise ValueError("num_polytopes must be positive.")
    if num_triangulations <= 0:
        raise ValueError("num_triangulations must be positive.")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive.")
    if num_vertices is not None and int(num_vertices) <= 0:
        raise ValueError("num_vertices must be positive when provided.")

    polytope_specs = _fetch_4d_n_lattice_polytope_specs(
        num_polytopes=num_polytopes,
        h11=h11,
        num_vertices=num_vertices,
        favorable=favorable,
    )
    actual_num_polytopes = int(len(polytope_specs))
    resolved_num_workers = _resolve_num_workers(
        num_workers,
        num_polytopes=max(1, actual_num_polytopes),
    )

    root_rng = np.random.default_rng(seed)
    jobs: List[Dict[str, Any]] = []
    for polytope_spec in polytope_specs:
        jobs.append(
            {
                "polytope_spec": polytope_spec,
                "polytope_seed": int(root_rng.integers(0, np.iinfo(np.uint32).max)),
                "num_triangulations": int(num_triangulations),
                "max_tries": int(max_tries),
                "triangulation_backend": triangulation_backend,
                "include_points_interior_to_facets": bool(include_points_interior_to_facets),
                "make_star": None if make_star is None else bool(make_star),
                "triangulation_verbosity": int(triangulation_verbosity),
            }
        )

    if resolved_num_workers == 1:
        worker_outputs = [
            _process_fetched_polytope_random_height_eval_job(job)
            for job in jobs
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_num_workers) as executor:
            futures = [
                executor.submit(_process_fetched_polytope_random_height_eval_job, job)
                for job in jobs
            ]
            worker_outputs = [future.result() for future in futures]
    worker_outputs.sort(key=lambda item: int(item["polytope_index"]))

    total_samples = int(sum(int(item["num_samples"]) for item in worker_outputs))
    underfilled_polytope_count = int(
        sum(int(item["underfilled_polytope_count"]) for item in worker_outputs)
    )
    dataset = {
        "metadata": {
            "seed": int(seed),
            "num_polytopes": int(actual_num_polytopes),
            "requested_num_polytopes": int(num_polytopes),
            "dataset_dimension": 4,
            "polytope_source": "cytools.fetch_polytopes",
            "lattice_space": "N",
            "generation_mode": "random_height_non_fine_regular",
            "height_sampling_distribution": "uniform_unit_interval",
            "h11": int(h11),
            "num_vertices": None if num_vertices is None else int(num_vertices),
            "favorable": None if favorable is None else bool(favorable),
            "num_triangulations_per_polytope": int(num_triangulations),
            "max_tries": int(max_tries),
            "triangulation_backend": triangulation_backend,
            "include_points_interior_to_facets": bool(include_points_interior_to_facets),
            "make_star": None if make_star is None else bool(make_star),
            "triangulation_verbosity": int(triangulation_verbosity),
            "num_workers": int(resolved_num_workers),
            "num_samples": int(total_samples),
            "underfilled_polytope_count": int(underfilled_polytope_count),
        },
        "polytopes": [item["polytope_entry"] for item in worker_outputs],
    }
    return dataset


def save_cy_4d_random_height_eval_dataset(
    dataset: Dict[str, Any],
    *,
    output_dir: str,
    output_name: str = "cy_4d_random_height_eval_dataset",
    compact_output: bool = False,
) -> Dict[str, str]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_json_path = output_root / f"{output_name}.json"
    samples_jsonl_path = output_root / f"{output_name}.samples.jsonl"

    if not compact_output:
        compact_dataset = _build_random_height_eval_dataset_for_disk(dataset)
        with dataset_json_path.open("w", encoding="utf-8") as handle:
            json.dump(compact_dataset, handle, indent=2)

    rows = _build_random_height_eval_polytope_rows(dataset)
    with samples_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = {
        "samples_jsonl": str(samples_jsonl_path),
    }
    if not compact_output:
        paths["dataset_json"] = str(dataset_json_path)
    return paths


def generate_and_save_cy_4d_random_height_eval_dataset(
    *,
    num_polytopes: int,
    h11: int,
    num_vertices: Optional[int] = None,
    favorable: Optional[bool] = None,
    num_triangulations: int,
    max_tries: int,
    seed: int = 0,
    triangulation_backend: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = None,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
    output_dir: str = "data/cy/output_eval",
    output_name: str = "cy_4d_random_height_eval_dataset",
    compact_output: bool = False,
) -> Dict[str, Any]:
    dataset = generate_cy_4d_random_height_eval_dataset(
        num_polytopes=num_polytopes,
        h11=h11,
        num_vertices=num_vertices,
        favorable=favorable,
        num_triangulations=num_triangulations,
        max_tries=max_tries,
        seed=seed,
        triangulation_backend=triangulation_backend,
        include_points_interior_to_facets=include_points_interior_to_facets,
        make_star=make_star,
        triangulation_verbosity=triangulation_verbosity,
        num_workers=num_workers,
    )
    paths = save_cy_4d_random_height_eval_dataset(
        dataset,
        output_dir=output_dir,
        output_name=output_name,
        compact_output=compact_output,
    )
    rows = _build_random_height_eval_polytope_rows(dataset)
    summary = {
        "num_polytopes": int(dataset["metadata"]["num_polytopes"]),
        "num_samples": int(dataset["metadata"]["num_samples"]),
        "underfilled_polytope_count": int(dataset["metadata"]["underfilled_polytope_count"]),
        "per_polytope": [
            {
                "polytope_index": int(row["polytope_index"]),
                "num_samples": int(row["non_fine_triangulation_count"]),
            }
            for row in rows
        ],
    }
    return {
        "paths": paths,
        "summary": summary,
        "metadata": dataset["metadata"],
        "dataset": dataset,
    }


def generate_cy_k3_random_height_eval_dataset(
    *,
    polytope_indices: Sequence[int],
    seed: int = 0,
    k3_path: Optional[str] = None,
    num_triangulations: int,
    max_tries: int,
    triangulation_backend: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = None,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    normalized_indices = _normalize_polytope_indices(polytope_indices)
    if num_triangulations <= 0:
        raise ValueError("num_triangulations must be positive.")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive.")

    records = _load_k3_records_by_indices(
        k3_path=k3_path,
        polytope_indices=normalized_indices,
    )
    resolved_num_workers = _resolve_num_workers(
        num_workers,
        num_polytopes=max(1, len(records)),
    )

    root_rng = np.random.default_rng(seed)
    jobs: List[Dict[str, Any]] = []
    for record in records:
        jobs.append(
            {
                "record": record,
                "polytope_seed": int(root_rng.integers(0, np.iinfo(np.uint32).max)),
                "num_triangulations": int(num_triangulations),
                "max_tries": int(max_tries),
                "triangulation_backend": triangulation_backend,
                "include_points_interior_to_facets": bool(include_points_interior_to_facets),
                "make_star": None if make_star is None else bool(make_star),
                "triangulation_verbosity": int(triangulation_verbosity),
            }
        )

    if resolved_num_workers == 1:
        worker_outputs = [
            _process_k3_polytope_random_height_eval_job(job)
            for job in jobs
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_num_workers) as executor:
            futures = [
                executor.submit(_process_k3_polytope_random_height_eval_job, job)
                for job in jobs
            ]
            worker_outputs = [future.result() for future in futures]
    worker_outputs.sort(key=lambda item: int(item["polytope_index"]))

    total_samples = int(sum(int(item["num_samples"]) for item in worker_outputs))
    underfilled_polytope_count = int(
        sum(int(item["underfilled_polytope_count"]) for item in worker_outputs)
    )
    return {
        "metadata": {
            "seed": int(seed),
            "k3_path": str(k3_path) if k3_path is not None else None,
            "num_polytopes": int(len(records)),
            "polytope_indices": [int(index) for index in normalized_indices],
            "dataset_dimension": 3,
            "polytope_source": "k3.txt",
            "lattice_space": "N",
            "generation_mode": "random_height_non_fine_regular",
            "height_sampling_distribution": "uniform_unit_interval",
            "num_triangulations_per_polytope": int(num_triangulations),
            "max_tries": int(max_tries),
            "triangulation_backend": triangulation_backend,
            "include_points_interior_to_facets": bool(include_points_interior_to_facets),
            "make_star": None if make_star is None else bool(make_star),
            "triangulation_verbosity": int(triangulation_verbosity),
            "num_workers": int(resolved_num_workers),
            "num_samples": int(total_samples),
            "underfilled_polytope_count": int(underfilled_polytope_count),
        },
        "polytopes": [item["polytope_entry"] for item in worker_outputs],
    }


def save_cy_k3_random_height_eval_dataset(
    dataset: Dict[str, Any],
    *,
    output_dir: str,
    output_name: str = "cy_k3_random_height_eval_dataset",
    compact_output: bool = False,
) -> Dict[str, str]:
    return save_cy_4d_random_height_eval_dataset(
        dataset,
        output_dir=output_dir,
        output_name=output_name,
        compact_output=compact_output,
    )


def generate_and_save_cy_k3_random_height_eval_dataset(
    *,
    polytope_indices: Sequence[int],
    seed: int = 0,
    k3_path: Optional[str] = None,
    num_triangulations: int,
    max_tries: int,
    triangulation_backend: Optional[str] = "qhull",
    include_points_interior_to_facets: bool = True,
    make_star: Optional[bool] = None,
    triangulation_verbosity: int = 0,
    num_workers: Optional[int] = None,
    output_dir: str = "data/cy/output_eval",
    output_name: str = "cy_k3_random_height_eval_dataset",
    compact_output: bool = False,
) -> Dict[str, Any]:
    dataset = generate_cy_k3_random_height_eval_dataset(
        polytope_indices=polytope_indices,
        seed=seed,
        k3_path=k3_path,
        num_triangulations=num_triangulations,
        max_tries=max_tries,
        triangulation_backend=triangulation_backend,
        include_points_interior_to_facets=include_points_interior_to_facets,
        make_star=make_star,
        triangulation_verbosity=triangulation_verbosity,
        num_workers=num_workers,
    )
    paths = save_cy_k3_random_height_eval_dataset(
        dataset,
        output_dir=output_dir,
        output_name=output_name,
        compact_output=compact_output,
    )
    rows = _build_random_height_eval_polytope_rows(dataset)
    summary = {
        "num_polytopes": int(dataset["metadata"]["num_polytopes"]),
        "num_samples": int(dataset["metadata"]["num_samples"]),
        "underfilled_polytope_count": int(dataset["metadata"]["underfilled_polytope_count"]),
        "per_polytope": [
            {
                "polytope_index": int(row["polytope_index"]),
                "num_samples": int(row["non_fine_triangulation_count"]),
            }
            for row in rows
        ],
    }
    return {
        "paths": paths,
        "summary": summary,
        "metadata": dataset["metadata"],
        "dataset": dataset,
    }


def save_cy_dataset(
    dataset: Dict[str, Any],
    *,
    output_dir: str,
    output_name: str = "cy_reflexive_dataset",
    compact_output: bool = False,
) -> Dict[str, str]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_json_path = output_root / f"{output_name}.json"
    samples_jsonl_path = output_root / f"{output_name}.samples.jsonl"

    if not compact_output:
        compact_dataset = _build_compact_dataset_for_disk(dataset)
        with dataset_json_path.open("w", encoding="utf-8") as handle:
            json.dump(compact_dataset, handle, indent=2)

    flat_samples = _build_polytope_jsonl_rows(dataset)
    with samples_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in flat_samples:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = {
        "samples_jsonl": str(samples_jsonl_path),
    }
    if not compact_output:
        paths["dataset_json"] = str(dataset_json_path)
    return paths


def summarize_cy_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dataset.get("metadata", {})
    per_polytope = []
    for poly in dataset.get("polytopes", []):
        samples = poly.get("samples", [])
        found_count = sum(1 for sample in samples if sample["bfs_search"]["found"])
        per_polytope.append(
            {
                "polytope_index": int(poly["polytope_index"]),
                "num_samples": int(len(samples)),
                "found_frst_count": int(found_count),
                "not_found_frst_count": int(len(samples) - found_count),
            }
        )

    return {
        "num_polytopes": int(metadata.get("num_polytopes", 0)),
        "num_samples": int(metadata.get("num_samples", 0)),
        "frst_found_count": int(metadata.get("frst_found_count", 0)),
        "frst_not_found_count": int(metadata.get("frst_not_found_count", 0)),
        "truncated_search_count": int(metadata.get("truncated_search_count", 0)),
        "per_polytope": per_polytope,
    }
