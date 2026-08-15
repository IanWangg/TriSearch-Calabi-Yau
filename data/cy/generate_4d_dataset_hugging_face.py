#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from data.cy.generate_4d_dataset import _build_parser as _build_cytools_parser
from data.cy.pipeline import generate_and_save_cy_4d_reflexive_dataset_incremental


HUGGING_FACE_DATASET_REPO = "calabi-yau-data/polytopes-4d"
HUGGING_FACE_DATASET_REVISION = "main"
HUGGING_FACE_POLYTOPE_SOURCE = f"hugging_face:{HUGGING_FACE_DATASET_REPO}"
HUGGING_FACE_MAX_RETRIES = 5
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_CANONICAL_VERTEX_COUNTS = tuple(range(5, 34)) + (36,)

# The repository stores one Parquet file per vertex count.
_PARQUET_FILE_PATTERN = re.compile(r"^polytopes-4d-(\d+)-vertices\.parquet$")
_SOURCE_COLUMNS = (
    "vertices",
    "vertex_count",
    "facet_count",
    "point_count",
    "dual_point_count",
    "h11",
    "h12",
    "euler_characteristic",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_cytools_parser()
    parser.description = (
        "Generate a TriSearch-compatible 4D CY dataset from "
        f"Hugging Face dataset {HUGGING_FACE_DATASET_REPO}. Source vertices "
        "are interpreted in the N lattice, filtered by source h12 for the "
        "requested CY h11, and passed through the same FRST/non-fine "
        "triangulation pipeline as generate_4d_dataset.py."
    )

    for action in parser._actions:
        if action.dest == "polytope_file":
            action.help = (
                "Optional local JSON/JSONL N-lattice polytope file. When "
                "provided, bypass the Hugging Face source and reuse the local records."
            )
        elif action.dest == "num_polytopes":
            action.help = (
                "Number of matching 4D reflexive N-lattice polytopes to read "
                "from Hugging Face, or an optional cap with --polytope-file."
            )
        elif action.dest == "h11":
            action.help = (
                "Required CY h11 value. For source vertices used in the N lattice, "
                "this is the Hugging Face table's h12 field."
            )
        elif action.dest == "num_vertices":
            action.help = (
                "Optional N-lattice vertex-count filter; selects the matching "
                "vertex-partitioned Parquet file."
            )
        elif action.dest == "favorable":
            action.help = (
                "Optional favorability filter, evaluated in the N lattice with "
                "CYTools because the Hugging Face table has no favorability field."
            )

    source_group = parser.add_argument_group("Hugging Face source")
    source_group.add_argument(
        "--hf-repo-id",
        "--hf_repo_id",
        dest="hf_repo_id",
        type=str,
        default=HUGGING_FACE_DATASET_REPO,
        help="Hugging Face dataset repository ID.",
    )
    source_group.add_argument(
        "--hf-revision",
        "--hf_revision",
        dest="hf_revision",
        type=str,
        default=HUGGING_FACE_DATASET_REVISION,
        help="Dataset branch, tag, or commit. The resolved commit is recorded in output metadata.",
    )
    source_group.add_argument(
        "--hf-cache-dir",
        "--hf_cache_dir",
        dest="hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    return parser


def _resolve_hugging_face_snapshot(
    *,
    repo_id: str,
    revision: str,
) -> Tuple[List[str], str]:
    if (
        repo_id == HUGGING_FACE_DATASET_REPO
        and _COMMIT_SHA_PATTERN.fullmatch(str(revision)) is not None
    ):
        filenames = [
            f"polytopes-4d-{vertex_count:02d}-vertices.parquet"
            for vertex_count in _CANONICAL_VERTEX_COUNTS
        ]
        return filenames, str(revision)

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for Hugging Face 4D data generation. "
            "Activate the 'sage' environment."
        ) from exc

    dataset_info = None
    last_error: Optional[Exception] = None
    for attempt in range(HUGGING_FACE_MAX_RETRIES):
        try:
            dataset_info = HfApi().dataset_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=False,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < HUGGING_FACE_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))
    if dataset_info is None:
        raise RuntimeError(
            f"Could not inspect Hugging Face dataset {repo_id!r} at revision {revision!r} "
            f"after {HUGGING_FACE_MAX_RETRIES} attempts."
        ) from last_error

    filenames = [sibling.rfilename for sibling in dataset_info.siblings]
    resolved_revision = str(dataset_info.sha or revision)
    return filenames, resolved_revision


def _select_parquet_files(
    filenames: Sequence[str],
    *,
    num_vertices: Optional[int],
) -> List[str]:
    files_by_vertex_count: Dict[int, str] = {}
    for filename in filenames:
        match = _PARQUET_FILE_PATTERN.fullmatch(str(filename))
        if match is None:
            continue
        files_by_vertex_count[int(match.group(1))] = str(filename)

    if num_vertices is not None:
        selected = files_by_vertex_count.get(int(num_vertices))
        if selected is None:
            available = sorted(files_by_vertex_count)
            raise ValueError(
                f"Hugging Face dataset has no 4D polytope file with {num_vertices} vertices. "
                f"Available vertex counts: {available}."
            )
        return [selected]

    return [files_by_vertex_count[count] for count in sorted(files_by_vertex_count)]


def _load_hugging_face_parquet_rows(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    h11: int,
    cache_dir: Optional[str],
) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_url
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets and huggingface_hub are required for Hugging Face 4D data generation. "
            "Activate the 'sage' environment."
        ) from exc

    parquet_url = hf_hub_url(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    load_kwargs: Dict[str, Any] = {
        "data_files": {"full": [parquet_url]},
        "split": "full",
        "streaming": True,
        "columns": list(_SOURCE_COLUMNS),
        # The source row's vertices are used as N-lattice vertices. Under this
        # convention source h12 is the CY hypersurface h11 (mirror to source h11).
        "filters": [("h12", "=", int(h11))],
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    return load_dataset("parquet", **load_kwargs)


def _load_hugging_face_parquet_rows_with_retry(
    **load_kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(HUGGING_FACE_MAX_RETRIES):
        try:
            yield from _load_hugging_face_parquet_rows(**load_kwargs)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < HUGGING_FACE_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(
        f"Failed to stream {load_kwargs['filename']!r} from Hugging Face after "
        f"{HUGGING_FACE_MAX_RETRIES} attempts."
    ) from last_error


def _normalize_source_vertices(row: Dict[str, Any], *, filename: str) -> List[List[int]]:
    vertices = np.asarray(row.get("vertices"), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 4:
        raise ValueError(
            f"Invalid 4D vertices in Hugging Face source file {filename}: "
            f"received shape {vertices.shape}."
        )
    source_vertex_count = int(row.get("vertex_count", vertices.shape[0]))
    if vertices.shape[0] != source_vertex_count:
        raise ValueError(
            f"Hugging Face source vertex_count={source_vertex_count} does not match "
            f"the {vertices.shape[0]} vertex rows in {filename}."
        )
    return [[int(coordinate) for coordinate in vertex] for vertex in vertices.tolist()]


def load_hugging_face_4d_n_lattice_polytope_specs(
    *,
    num_polytopes: int,
    h11: int,
    num_vertices: Optional[int] = None,
    favorable: Optional[bool] = None,
    repo_id: str = HUGGING_FACE_DATASET_REPO,
    revision: str = HUGGING_FACE_DATASET_REVISION,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if int(num_polytopes) <= 0:
        raise ValueError("num_polytopes must be positive.")
    if int(h11) < 0:
        raise ValueError("h11 must be non-negative.")
    if num_vertices is not None and int(num_vertices) <= 0:
        raise ValueError("num_vertices must be positive when provided.")

    try:
        from cytools.polytope import Polytope
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cytools is required to validate Hugging Face polytopes in the N lattice. "
            "Activate the 'sage' environment."
        ) from exc

    filenames, resolved_revision = _resolve_hugging_face_snapshot(
        repo_id=repo_id,
        revision=revision,
    )
    parquet_files = _select_parquet_files(
        filenames,
        num_vertices=num_vertices,
    )
    if not parquet_files:
        raise ValueError(f"No vertex-partitioned Parquet files found in {repo_id!r}.")

    specs: List[Dict[str, Any]] = []
    scanned_matching_rows = 0
    rejected_favorability_rows = 0
    for filename in parquet_files:
        try:
            source_rows = _load_hugging_face_parquet_rows_with_retry(
                repo_id=repo_id,
                revision=resolved_revision,
                filename=filename,
                h11=int(h11),
                cache_dir=cache_dir,
            )
            for filtered_row_index, source_row in enumerate(source_rows):
                # Keep this defensive check even though Parquet predicate pushdown
                # requests h12 == h11.
                if int(source_row["h12"]) != int(h11):
                    continue
                scanned_matching_rows += 1
                vertices = _normalize_source_vertices(source_row, filename=filename)
                n_polytope = Polytope(np.asarray(vertices, dtype=np.int64))
                if int(n_polytope.ambient_dim()) != 4 or not bool(n_polytope.is_reflexive()):
                    raise ValueError(
                        f"Hugging Face source row {filtered_row_index} in {filename} "
                        "is not a 4D reflexive polytope."
                    )

                cytools_h11 = int(n_polytope.h11(lattice="N"))
                if cytools_h11 != int(h11):
                    raise ValueError(
                        "Hugging Face/CYTools Hodge mapping mismatch for source row "
                        f"{filtered_row_index} in {filename}: source h12={source_row['h12']}, "
                        f"but CYTools h11(N)={cytools_h11}."
                    )

                actual_favorable: Optional[bool] = None
                if favorable is not None:
                    actual_favorable = bool(n_polytope.is_favorable(lattice="N"))
                    if actual_favorable != bool(favorable):
                        rejected_favorability_rows += 1
                        continue

                source_metadata = {
                    "dataset_repo": str(repo_id),
                    "dataset_revision": str(resolved_revision),
                    "source_file": str(filename),
                    "filtered_row_index": int(filtered_row_index),
                    "source_h11": int(source_row["h11"]),
                    "source_h12": int(source_row["h12"]),
                    "source_vertex_count": int(source_row["vertex_count"]),
                    "source_facet_count": int(source_row["facet_count"]),
                    "source_point_count": int(source_row["point_count"]),
                    "source_dual_point_count": int(source_row["dual_point_count"]),
                    "source_euler_characteristic": int(source_row["euler_characteristic"]),
                }
                spec: Dict[str, Any] = {
                    "polytope_index": int(len(specs)),
                    "h11": int(h11),
                    "requested_num_vertices": None
                    if num_vertices is None
                    else int(num_vertices),
                    "vertices": vertices,
                    "polytope_source": HUGGING_FACE_POLYTOPE_SOURCE
                    if repo_id == HUGGING_FACE_DATASET_REPO
                    else f"hugging_face:{repo_id}",
                    "source_metadata": source_metadata,
                }
                if actual_favorable is not None:
                    spec["favorable"] = bool(actual_favorable)
                specs.append(spec)
                if len(specs) >= int(num_polytopes):
                    break
        except (ModuleNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed while streaming {filename!r} from Hugging Face dataset {repo_id!r}."
            ) from exc

        if len(specs) >= int(num_polytopes):
            break

    if not specs:
        favorable_text = "any" if favorable is None else str(bool(favorable))
        raise ValueError(
            "No matching 4D N-lattice polytopes were found in the Hugging Face dataset "
            f"for h11={h11}, num_vertices={num_vertices}, favorable={favorable_text}."
        )
    if len(specs) < int(num_polytopes):
        warnings.warn(
            f"Requested {num_polytopes} Hugging Face polytopes, but found only {len(specs)} "
            f"after h11/vertex/favorability filtering. Proceeding with available polytopes.",
            stacklevel=2,
        )

    source_metadata = {
        "polytope_source": f"hugging_face:{repo_id}",
        "dataset_repo": str(repo_id),
        "requested_revision": str(revision),
        "resolved_revision": str(resolved_revision),
        "source_lattice_interpretation": "N",
        "requested_h11": int(h11),
        "source_hodge_filter": {"column": "h12", "value": int(h11)},
        "requested_num_vertices": None
        if num_vertices is None
        else int(num_vertices),
        "source_parquet_files": [str(filename) for filename in parquet_files],
        "scanned_matching_rows": int(scanned_matching_rows),
        "rejected_favorability_rows": int(rejected_favorability_rows),
    }
    if repo_id == HUGGING_FACE_DATASET_REPO:
        source_metadata["dataset_license"] = "cc-by-sa-4.0"
    return specs, source_metadata


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    polytope_specs = None
    polytope_source_metadata = None
    if args.polytope_file is None:
        if args.num_polytopes is None or int(args.num_polytopes) <= 0:
            parser.error("--num-polytopes must be positive when --polytope-file is not provided.")
        if args.h11 is None:
            parser.error("--h11 is required when --polytope-file is not provided.")
        polytope_specs, polytope_source_metadata = (
            load_hugging_face_4d_n_lattice_polytope_specs(
                num_polytopes=int(args.num_polytopes),
                h11=int(args.h11),
                num_vertices=args.num_vertices,
                favorable=args.favorable,
                repo_id=args.hf_repo_id,
                revision=args.hf_revision,
                cache_dir=args.hf_cache_dir,
            )
        )

    run_result = generate_and_save_cy_4d_reflexive_dataset_incremental(
        num_polytopes=args.num_polytopes,
        h11=args.h11,
        num_vertices=args.num_vertices,
        favorable=args.favorable,
        num_triangulations_per_frst=args.num_triangulations_per_frst,
        polytope_file=args.polytope_file,
        seed=args.seed,
        triangulation_backend=args.triangulation_backend,
        neighbor_backend=args.neighbor_backend,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        make_star=args.make_star,
        max_retries_per_triangulation=args.max_retries_per_triangulation,
        height_scale=args.height_scale,
        frsts_per_polytope=args.frsts_per_polytope,
        fair_backend=args.fair_backend,
        fair_backend_fallback=args.fair_backend_fallback,
        fair_max_retries=args.fair_max_retries,
        fair_max_attempt_rounds=args.fair_max_attempt_rounds,
        fair_call_timeout_seconds=args.fair_call_timeout_seconds,
        fast=args.fast,
        bfs_max_depth=args.bfs_max_depth,
        bfs_max_nodes=args.bfs_max_nodes,
        collection_depths=args.collection_depths,
        collection_all=args.collection_all,
        random_flip=args.random_flip,
        triangulation_verbosity=args.triangulation_verbosity,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_name=args.output_name,
        compact_output=args.compact_output,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        log_every=args.log_every,
        polytope_specs=polytope_specs,
        polytope_source_metadata=polytope_source_metadata,
    )
    paths = run_result["paths"]
    summary = run_result["summary"]

    print("Hugging Face 4D CY dataset generation complete.")
    print("Saved files:")
    for key, value in paths.items():
        print(f"  - {key}: {value}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
