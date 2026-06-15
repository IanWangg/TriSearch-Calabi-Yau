from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import Batch, Data

from mdp.cy_triangulation_state import CYTriangulationState, create_state_from_cy_triangulation

from core.snn_simplex_topology import (
    SNNSimplexTopologyTensors,
    add_snn_simplex_topology_to_data,
    build_snn_simplex_topology_tensors,
    simplex_vertices_tensor_from_simplices,
)
from core.training_types import CYDatasetSplit
from core.vertex_preprocessing import VertexPreprocessor

try:
    from cytools.polytope import Polytope
except ModuleNotFoundError:
    Polytope = None


@dataclass(frozen=True)
class K3Record:
    record_index: int
    header: str
    ambient_dim: int
    num_matrix_columns: int
    m_vertices: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class _CachedCYGraphTensors:
    x_cpu: torch.Tensor
    edge_index_cpu: torch.Tensor


_CYGraphCacheKey = Tuple[str, int, int, int, int]
_CYSubcomplexCacheKey = Tuple[_CYGraphCacheKey, Tuple[Tuple[int, ...], ...], int]

_CY_GRAPH_TENSOR_CACHE: Dict[_CYGraphCacheKey, _CachedCYGraphTensors] = {}
_CY_SUBCOMPLEX_TENSOR_CACHE: Dict[_CYSubcomplexCacheKey, torch.Tensor] = {}
_CY_SIMPLEX_TOPOLOGY_CACHE: Dict[_CYGraphCacheKey, torch.Tensor] = {}
_CY_SNN_SIMPLEX_TOPOLOGY_CACHE: Dict[_CYSubcomplexCacheKey, SNNSimplexTopologyTensors] = {}


def _cy_state_key(state: CYTriangulationState) -> str:
    key = getattr(state, "key", None)
    if key is not None:
        return str(key)
    simplices = tuple(sorted(tuple(sorted(simplex)) for simplex in getattr(state, "simplices", ())))
    return "%s:%s" % (int(getattr(state, "point_config_index", -1)), simplices)


def _cy_data_cache_key(state: CYTriangulationState) -> _CYGraphCacheKey:
    return (
        _cy_state_key(state),
        int(getattr(state, "point_config_index", -1)),
        len(getattr(state, "vertices", ())),
        len(getattr(state, "edges", ())),
        len(getattr(state, "simplices", ())),
    )


def _get_cached_cy_graph_tensors(state: CYTriangulationState) -> _CachedCYGraphTensors:
    key = _cy_data_cache_key(state)
    cached = _CY_GRAPH_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    cached = _CachedCYGraphTensors(
        x_cpu=torch.tensor(state.vertices, dtype=torch.float, device="cpu"),
        edge_index_cpu=pyg.utils.to_undirected(_edge_keys_to_index_tensor(state.edges)).cpu(),
    )
    _CY_GRAPH_TENSOR_CACHE[key] = cached
    return cached


def _normalize_cy_subcomplex_actions(actions: Sequence[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(vertex) for vertex in action) for action in actions)


def _get_cached_cy_subcomplex_tensor(
    *,
    state: CYTriangulationState,
    actions: Sequence[Tuple[int, ...]],
    final_width: int,
) -> torch.Tensor:
    normalized_actions = _normalize_cy_subcomplex_actions(actions)
    cache_key = (_cy_data_cache_key(state), normalized_actions, int(final_width))
    cached = _CY_SUBCOMPLEX_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    subcomplex_tensor = torch.full(
        (len(normalized_actions), int(final_width)),
        -1,
        dtype=torch.long,
        device="cpu",
    )
    for action_idx, action in enumerate(normalized_actions):
        if len(action) > int(final_width):
            raise ValueError(
                f"Subcomplex action at index {action_idx} has width {len(action)} > {final_width}."
            )
        if len(action) == 0:
            continue
        subcomplex_tensor[action_idx, : len(action)] = torch.tensor(action, dtype=torch.long, device="cpu")

    _CY_SUBCOMPLEX_TENSOR_CACHE[cache_key] = subcomplex_tensor
    return subcomplex_tensor


def _simplex_vertices_tensor_from_simplices(
    simplices: Iterable[Iterable[int]],
) -> torch.Tensor:
    return simplex_vertices_tensor_from_simplices(simplices)


def _get_cached_cy_simplex_topology_tensor(state: CYTriangulationState) -> torch.Tensor:
    cache_key = _cy_data_cache_key(state)
    cached = _CY_SIMPLEX_TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = _simplex_vertices_tensor_from_simplices(state.simplices)
    _CY_SIMPLEX_TOPOLOGY_CACHE[cache_key] = cached
    return cached


def _get_cached_cy_snn_simplex_topology_tensors(
    *,
    state: CYTriangulationState,
    actions: Sequence[Tuple[int, ...]],
    final_width: int,
    simplex_vertices: torch.Tensor,
    subcomplex_vertices: torch.Tensor,
) -> SNNSimplexTopologyTensors:
    normalized_actions = _normalize_cy_subcomplex_actions(actions)
    cache_key = (_cy_data_cache_key(state), normalized_actions, int(final_width))
    cached = _CY_SNN_SIMPLEX_TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = build_snn_simplex_topology_tensors(
        simplex_vertices=simplex_vertices,
        subcomplex_vertices=subcomplex_vertices,
    )
    _CY_SNN_SIMPLEX_TOPOLOGY_CACHE[cache_key] = cached
    return cached


def get_cy_data_tensor_cache_sizes() -> Dict[str, int]:
    return {
        "graph": len(_CY_GRAPH_TENSOR_CACHE),
        "subcomplex": len(_CY_SUBCOMPLEX_TENSOR_CACHE),
        "simplex_topology": len(_CY_SIMPLEX_TOPOLOGY_CACHE),
        "snn_simplex_topology": len(_CY_SNN_SIMPLEX_TOPOLOGY_CACHE),
    }


def prune_cy_data_tensor_caches(
    *,
    keep_keys: Iterable[str] | None,
    max_entries: int | None,
) -> Dict[str, int]:
    max_entries_int = None if max_entries is None or int(max_entries) <= 0 else int(max_entries)
    keep_key_set = None if keep_keys is None else {str(key) for key in keep_keys}

    if keep_key_set is not None:
        for key in list(_CY_GRAPH_TENSOR_CACHE.keys()):
            if key[0] not in keep_key_set:
                _CY_GRAPH_TENSOR_CACHE.pop(key, None)
        for cache_key in list(_CY_SUBCOMPLEX_TENSOR_CACHE.keys()):
            if cache_key[0][0] not in keep_key_set:
                _CY_SUBCOMPLEX_TENSOR_CACHE.pop(cache_key, None)
        for key in list(_CY_SIMPLEX_TOPOLOGY_CACHE.keys()):
            if key[0] not in keep_key_set:
                _CY_SIMPLEX_TOPOLOGY_CACHE.pop(key, None)
        for cache_key in list(_CY_SNN_SIMPLEX_TOPOLOGY_CACHE.keys()):
            if cache_key[0][0] not in keep_key_set:
                _CY_SNN_SIMPLEX_TOPOLOGY_CACHE.pop(cache_key, None)

    if max_entries_int is not None:
        if len(_CY_GRAPH_TENSOR_CACHE) > max_entries_int:
            overflow = len(_CY_GRAPH_TENSOR_CACHE) - max_entries_int
            for key in list(_CY_GRAPH_TENSOR_CACHE.keys())[:overflow]:
                _CY_GRAPH_TENSOR_CACHE.pop(key, None)
        if len(_CY_SUBCOMPLEX_TENSOR_CACHE) > max_entries_int:
            overflow = len(_CY_SUBCOMPLEX_TENSOR_CACHE) - max_entries_int
            for key in list(_CY_SUBCOMPLEX_TENSOR_CACHE.keys())[:overflow]:
                _CY_SUBCOMPLEX_TENSOR_CACHE.pop(key, None)
        if len(_CY_SIMPLEX_TOPOLOGY_CACHE) > max_entries_int:
            overflow = len(_CY_SIMPLEX_TOPOLOGY_CACHE) - max_entries_int
            for key in list(_CY_SIMPLEX_TOPOLOGY_CACHE.keys())[:overflow]:
                _CY_SIMPLEX_TOPOLOGY_CACHE.pop(key, None)
        if len(_CY_SNN_SIMPLEX_TOPOLOGY_CACHE) > max_entries_int:
            overflow = len(_CY_SNN_SIMPLEX_TOPOLOGY_CACHE) - max_entries_int
            for key in list(_CY_SNN_SIMPLEX_TOPOLOGY_CACHE.keys())[:overflow]:
                _CY_SNN_SIMPLEX_TOPOLOGY_CACHE.pop(key, None)

    return get_cy_data_tensor_cache_sizes()


def resolve_default_k3_path(k3_path: Optional[str] = None) -> Path:
    if k3_path is not None:
        resolved = Path(k3_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"k3 data file not found at: {resolved}")
        return resolved

    candidates = (Path("data/k3.txt"), Path("cy_data/k3.txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate k3.txt. Expected one of: data/k3.txt or cy_data/k3.txt."
    )


def _parse_k3_header(header_line: str) -> Tuple[int, int]:
    tokens = header_line.strip().split()
    if len(tokens) < 2:
        raise ValueError(f"Invalid k3 header line: '{header_line}'")

    try:
        ambient_dim = int(tokens[0])
        num_matrix_columns = int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric fields in k3 header line: '{header_line}'") from exc

    if ambient_dim <= 0 or num_matrix_columns <= 0:
        raise ValueError(f"Invalid non-positive dimensions in k3 header line: '{header_line}'")

    return ambient_dim, num_matrix_columns


def load_k3_records(
    k3_path: Optional[str] = None,
    *,
    max_polytopes: Optional[int] = None,
) -> List[K3Record]:
    resolved_path = resolve_default_k3_path(k3_path)

    if max_polytopes is not None and max_polytopes <= 0:
        raise ValueError("max_polytopes must be positive when provided.")

    records: List[K3Record] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    line_index = 0
    while line_index < len(lines):
        header = lines[line_index].strip()
        line_index += 1

        if not header:
            continue

        ambient_dim, num_matrix_columns = _parse_k3_header(header)

        matrix_rows: List[List[int]] = []
        for _ in range(ambient_dim):
            while line_index < len(lines) and not lines[line_index].strip():
                line_index += 1
            if line_index >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading matrix rows for record {len(records)}."
                )

            row_tokens = lines[line_index].strip().split()
            line_index += 1
            if len(row_tokens) != num_matrix_columns:
                raise ValueError(
                    "Invalid k3 matrix row width for record "
                    f"{len(records)}: expected {num_matrix_columns}, got {len(row_tokens)}."
                )

            try:
                matrix_rows.append([int(token) for token in row_tokens])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer matrix row in record {len(records)}: {row_tokens}"
                ) from exc

        matrix = np.asarray(matrix_rows, dtype=np.int64)
        m_vertices = tuple(tuple(int(v) for v in column) for column in matrix.T.tolist())

        records.append(
            K3Record(
                record_index=len(records),
                header=header,
                ambient_dim=ambient_dim,
                num_matrix_columns=num_matrix_columns,
                m_vertices=m_vertices,
            )
        )

        if max_polytopes is not None and len(records) >= max_polytopes:
            break

    return records


def triangulate_n_lattice_polytope(
    k3_record: K3Record,
    *,
    include_points_interior_to_facets: bool = True,
) -> Dict[str, Any]:
    if Polytope is None:
        raise ModuleNotFoundError(
            "cytools is required for CY triangulation loading. Activate the 'sage' environment."
        )

    m_vertices = np.asarray(k3_record.m_vertices, dtype=np.int64)
    m_polytope = Polytope(m_vertices)
    n_polytope = m_polytope.dual_polytope()

    # CY triangulation must always be performed in N-lattice space.
    n_triangulation = n_polytope.triangulate(
        include_points_interior_to_facets=include_points_interior_to_facets
    )

    return {
        "record_index": k3_record.record_index,
        "header": k3_record.header,
        "lattice_space": "N",
        "m_polytope": m_polytope,
        "n_polytope": n_polytope,
        "triangulation": n_triangulation,
    }


def load_cy_3d_dataset(
    k3_path: Optional[str] = None,
    *,
    max_polytopes: Optional[int] = None,
    include_points_interior_to_facets: bool = True,
    precompute_actions: bool = True,
) -> List[Dict[str, Any]]:
    records = load_k3_records(k3_path=k3_path, max_polytopes=max_polytopes)
    dataset_entries: List[Dict[str, Any]] = []

    for record in records:
        tri_info = triangulate_n_lattice_polytope(
            record,
            include_points_interior_to_facets=include_points_interior_to_facets,
        )
        state = create_state_from_cy_triangulation(
            tri_info["triangulation"],
            point_config_index=record.record_index,
            add_origin=False,
        )
        if precompute_actions:
            state.find_available_actions()

        dataset_entries.append(
            {
                "record": record,
                "state": state,
                "lattice_space": tri_info["lattice_space"],
                "m_polytope": tri_info["m_polytope"],
                "n_polytope": tri_info["n_polytope"],
                "triangulation": tri_info["triangulation"],
            }
        )

    return dataset_entries


def load_cy_3d_states(
    k3_path: Optional[str] = None,
    *,
    max_polytopes: Optional[int] = None,
    include_points_interior_to_facets: bool = True,
    precompute_actions: bool = True,
) -> List[CYTriangulationState]:
    entries = load_cy_3d_dataset(
        k3_path=k3_path,
        max_polytopes=max_polytopes,
        include_points_interior_to_facets=include_points_interior_to_facets,
        precompute_actions=precompute_actions,
    )
    return [entry["state"] for entry in entries]


def _edge_keys_to_index_tensor(edge_keys: Iterable[Tuple[int, int]]) -> torch.Tensor:
    edge_list = list(edge_keys)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def _get_cy_subcomplex_actions(
    state: CYTriangulationState,
    *,
    ensure_actions_ready: bool,
) -> List[Tuple[int, ...]]:
    if ensure_actions_ready and not state.actions_ready:
        state.find_available_actions()
    return list(state.get_available_subcomplex_actions())


def _infer_min_width_from_state(state: CYTriangulationState) -> int:
    if len(state.simplices) == 0:
        return 0
    simplex_dim_plus_one = len(next(iter(state.simplices)))
    return simplex_dim_plus_one + 1


def create_data_from_cy_state_with_subcomplex(
    state: CYTriangulationState,
    *,
    subcomplex_width: Optional[int] = None,
    ensure_actions_ready: bool = True,
    subcomplex_actions: Optional[Sequence[Tuple[int, ...]]] = None,
    vertex_preprocessor: VertexPreprocessor | None = None,
    include_simplex_topology: bool = False,
) -> Data:
    if subcomplex_actions is None:
        actions = _get_cy_subcomplex_actions(state, ensure_actions_ready=ensure_actions_ready)
    else:
        actions = list(subcomplex_actions)

    inferred_width = max((len(action) for action in actions), default=0)
    inferred_width = max(inferred_width, _infer_min_width_from_state(state))
    final_width = inferred_width if subcomplex_width is None else int(subcomplex_width)
    if final_width < inferred_width:
        raise ValueError(
            f"Requested subcomplex_width={final_width} is smaller than required width={inferred_width}."
        )

    cached_graph = _get_cached_cy_graph_tensors(state)
    vertices_tensor = cached_graph.x_cpu
    if vertex_preprocessor is not None:
        vertices_tensor = vertex_preprocessor.transform_vertices(
            point_config_index=int(state.point_config_index),
            vertices=vertices_tensor,
        )

    subcomplex_vertices = _get_cached_cy_subcomplex_tensor(
        state=state,
        actions=actions,
        final_width=final_width,
    )
    data = Data(
        x=vertices_tensor,
        edge_index=cached_graph.edge_index_cpu,
        subcomplex_vertices=subcomplex_vertices,
        num_available_subcomplexes=len(actions),
    )
    if include_simplex_topology:
        simplex_vertices = _get_cached_cy_simplex_topology_tensor(state)
        data.simplex_vertices = simplex_vertices
        data.num_top_simplices = int(simplex_vertices.size(0))
        add_snn_simplex_topology_to_data(
            data,
            _get_cached_cy_snn_simplex_topology_tensors(
                state=state,
                actions=actions,
                final_width=final_width,
                simplex_vertices=simplex_vertices,
                subcomplex_vertices=subcomplex_vertices,
            ),
        )
    data.edge_attr = None
    data.num_edges = data.edge_index.size(1)
    return data


def create_cy_batch_from_states_with_subcomplex(
    states: Sequence[CYTriangulationState],
    *,
    ensure_actions_ready: bool = True,
    vertex_preprocessor: VertexPreprocessor | None = None,
    include_simplex_topology: bool = False,
) -> Batch:
    if len(states) == 0:
        raise ValueError("states must be non-empty.")

    action_lists = [
        _get_cy_subcomplex_actions(state, ensure_actions_ready=ensure_actions_ready) for state in states
    ]
    width_candidates = [max((len(action) for action in actions), default=0) for actions in action_lists]
    width_candidates.extend(_infer_min_width_from_state(state) for state in states)
    batch_width = max(width_candidates)

    data_list = [
        create_data_from_cy_state_with_subcomplex(
            state,
            subcomplex_width=batch_width,
            ensure_actions_ready=False,
            subcomplex_actions=actions,
            vertex_preprocessor=vertex_preprocessor,
            include_simplex_topology=include_simplex_topology,
        )
        for state, actions in zip(states, action_lists)
    ]
    return Batch.from_data_list(data_list)


def polytope_vertex_count(row: dict) -> int:
    return len(row.get("vertices", ()))


def split_rows_by_vertex_count(
    rows: Sequence[dict],
    *,
    num_eval_polytopes: int,
) -> CYDatasetSplit:
    vertex_count_by_polytope: Dict[int, int] = {}
    for row in rows:
        polytope_index = int(row["polytope_index"])
        vertex_count = polytope_vertex_count(row)
        cached_count = vertex_count_by_polytope.get(polytope_index)
        if cached_count is not None and cached_count != vertex_count:
            raise ValueError(
                f"Inconsistent vertex count for polytope_index={polytope_index}: "
                f"{cached_count} vs {vertex_count}."
            )
        vertex_count_by_polytope[polytope_index] = vertex_count

    sorted_polytopes = sorted(
        vertex_count_by_polytope,
        key=lambda polytope_index: (-vertex_count_by_polytope[polytope_index], polytope_index),
    )
    if len(sorted_polytopes) < 2:
        raise ValueError("Need at least two distinct polytopes to build disjoint train/eval splits.")

    resolved_num_eval = max(1, min(int(num_eval_polytopes), len(sorted_polytopes) - 1))
    eval_polytope_indices = list(sorted_polytopes[:resolved_num_eval])
    eval_polytope_set = set(eval_polytope_indices)
    train_polytope_indices = [index for index in sorted_polytopes if index not in eval_polytope_set]
    train_polytope_set = set(train_polytope_indices)

    train_rows = [row for row in rows if int(row["polytope_index"]) in train_polytope_set]
    eval_rows = [row for row in rows if int(row["polytope_index"]) in eval_polytope_set]
    return CYDatasetSplit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_polytope_indices=train_polytope_indices,
        eval_polytope_indices=eval_polytope_indices,
    )


def mean_vertex_count(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([polytope_vertex_count(row) for row in rows]))


def infer_dataset_coordinate_dim(rows: Sequence[dict]) -> int:
    inferred_dim: int | None = None
    for row_index, row in enumerate(rows):
        for vertex_index, vertex in enumerate(row.get("vertices", ())):
            vertex_dim = len(vertex)
            if inferred_dim is None:
                inferred_dim = vertex_dim
                continue
            if vertex_dim != inferred_dim:
                raise ValueError(
                    "Inconsistent dataset vertex dimensions: "
                    f"expected {inferred_dim}, got {vertex_dim} "
                    f"at row {row_index}, vertex {vertex_index}."
                )
    if inferred_dim is None:
        raise ValueError("Unable to infer dataset coordinate dimension from empty vertex lists.")
    return inferred_dim


def resolve_policy_in_channels(
    rows: Sequence[dict],
    requested_in_channels: int | None,
) -> int:
    dataset_coordinate_dim = infer_dataset_coordinate_dim(rows)
    if requested_in_channels is None:
        return dataset_coordinate_dim

    resolved_in_channels = int(requested_in_channels)
    if resolved_in_channels != dataset_coordinate_dim:
        raise ValueError(
            "--in_channels must match the dataset vertex coordinate dimension: "
            f"got {resolved_in_channels}, expected {dataset_coordinate_dim}."
        )
    return resolved_in_channels
