"""Strict evaluation counts around the existing training data loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

from eval.config import EvaluationSpec


def load_eval_polytopes(spec: EvaluationSpec, *, accept_polytope=None) -> tuple[list[dict], dict]:
    if spec.polytope_file is not None:
        if accept_polytope is not None:
            raise ValueError("Local evaluation inputs cannot be replaced through eligibility filtering.")
        from data.cy.pipeline import Polytope, _load_4d_n_lattice_polytope_specs_from_file

        path = Path(spec.polytope_file).expanduser().resolve()
        polytopes = _load_4d_n_lattice_polytope_specs_from_file(
            polytope_file=str(path), num_polytopes=spec.num_polytopes,
        )
        metadata = {"source": "polytope_file", "polytope_file": str(path),
                    "polytope_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "lattice_space": "N", "selection": "first_n_in_file_order"}
        for row in polytopes:
            polytope = Polytope(row["vertices"])
            if polytope.dim() != 4 or not polytope.is_reflexive():
                raise ValueError("Local evaluation requires a 4D reflexive polytope.")
            actual_h11 = int(polytope.h11(lattice="N"))
            if actual_h11 != spec.h11 or row.get("h11", actual_h11) != actual_h11:
                raise ValueError(f"Local polytope {row['polytope_index']} has CY h11={actual_h11}, "
                                 f"requested h11={spec.h11}, file h11={row.get('h11')}.")
            actual_vertices = len(polytope.vertices())
            actual_favorable = bool(polytope.is_favorable(lattice="N"))
            if spec.num_vertices is not None and actual_vertices != spec.num_vertices:
                raise ValueError(f"Local polytope has {actual_vertices} vertices; expected {spec.num_vertices}.")
            if spec.favorable is not None and actual_favorable != spec.favorable:
                raise ValueError("Local polytope does not match the requested favorable setting.")
            row.update(h11=actual_h11, favorable=actual_favorable,
                       source_metadata={**metadata, "cytools_h11": actual_h11,
                                        "cytools_h21": int(polytope.h21(lattice="N")),
                                        "num_vertices": actual_vertices})
    else:
        from data.cy.generate_4d_dataset_hugging_face import load_hugging_face_4d_n_lattice_polytope_specs

        Path(spec.hf_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
        polytopes, metadata = load_hugging_face_4d_n_lattice_polytope_specs(
            num_polytopes=spec.num_polytopes, h11=spec.h11,
            num_vertices=spec.num_vertices, favorable=spec.favorable,
            revision=spec.hf_revision, cache_dir=spec.hf_cache_dir,
            **({"accept_polytope": accept_polytope} if accept_polytope is not None else {}),
        )
        metadata = {**metadata, "selection": (
            "first_n_with_enough_distinct_frst_in_source_order" if accept_polytope is not None
            else "first_n_matching_in_source_order")}
    if len(polytopes) != spec.num_polytopes:
        raise ValueError(f"Requested {spec.num_polytopes} polytopes, obtained {len(polytopes)}.")
    signatures = {tuple(sorted(tuple(point) for point in row["vertices"])) for row in polytopes}
    indices = {row["polytope_index"] for row in polytopes}
    if len(signatures) != len(polytopes) or len(indices) != len(polytopes):
        raise ValueError("Source returned duplicate polytopes; refusing to duplicate evaluation inputs.")
    if any(row["h11"] != spec.h11 for row in polytopes):
        raise ValueError("Loaded polytopes do not match the requested CY h11.")
    return polytopes, metadata
