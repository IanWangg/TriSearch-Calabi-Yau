"""Generate shared FRST starts once, then save and reload them offline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from eval.config import EVAL_ROOT, EvaluationSpec, derive_seed
from eval.data.loader import load_eval_polytopes
from mdp.cy_state_record import canonical_simplices


@dataclass
class EvaluationSetup:
    rows: list[dict]
    metadata: dict
    path: Path | None = None

    @property
    def setup_id(self) -> str:
        payload = json.dumps({"rows": self.rows, "metadata": self.metadata}, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self, spec: EvaluationSpec | None = None) -> None:
        parameters = self.metadata["parameters"]
        if spec is not None and parameters != spec.setup_parameters():
            raise ValueError("Saved setup parameters do not match this evaluation specification.")
        if len(self.rows) != parameters["num_polytopes"]:
            raise ValueError("Setup polytope count does not match its specification.")
        indices, configurations = set(), set()
        for row in self.rows:
            index = row["polytope_index"]
            vertices = tuple(sorted(tuple(point) for point in row["vertices"]))
            if index in indices or vertices in configurations:
                raise ValueError("Setup contains duplicate polytopes.")
            indices.add(index)
            configurations.add(vertices)
            if row["h11"] != parameters["h11"]:
                raise ValueError(f"Polytope {index} has an unexpected h11.")
            starts = [canonical_simplices(entry["simplices"]) for entry in row["frst_list"]]
            if (len(starts) != parameters["num_starts"] or len(set(starts)) != len(starts)
                    or any(not simplices for simplices in starts)):
                raise ValueError(f"Polytope {index} must have exactly {parameters['num_starts']} distinct FRST starts.")


def save_eval_setup(setup: EvaluationSetup, path: str | Path) -> Path:
    from data.cy.pipeline import _write_json_atomic

    setup.validate()
    path = Path(path).expanduser().resolve()
    manifest = path / "setup.json"
    if manifest.exists():
        existing = load_eval_setup(path)
        if existing.setup_id != setup.setup_id:
            raise FileExistsError(f"A different setup already exists at {path}.")
        setup.path = path
        return path
    path.mkdir(parents=True, exist_ok=True)
    samples = path / "samples.jsonl"
    temporary = samples.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in setup.rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(samples)
    _write_json_atomic(manifest, {"format_version": 1, "setup_id": setup.setup_id,
                                  "metadata": setup.metadata})
    setup.path = path
    return path


def load_eval_setup(path: str | Path) -> EvaluationSetup:
    from mdp.cy_rollout import load_cy_sample_rows

    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / "setup.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported evaluation setup format_version.")
    setup = EvaluationSetup(load_cy_sample_rows(str(path / "samples.jsonl")), manifest["metadata"], path)
    if setup.setup_id != manifest["setup_id"]:
        raise ValueError("Evaluation setup checksum mismatch.")
    setup.validate()
    return setup


def prepare_eval_setup(spec: EvaluationSpec, *, output_dir: str | Path | None = None) -> EvaluationSetup:
    # These are the training pipeline's sampler and label-preserving serializers.
    from data.cy.pipeline import (
        Polytope, _build_polytope_jsonl_rows, _generate_frst_seeds_with_random_triangulations_fair,
        _serialize_fetched_n_polytope, serialize_triangulation,
    )

    entries, diagnostics, skipped = [], [], []
    seen_vertices = set()

    def sample_candidate(polytope_spec, polytope=None):
        index = polytope_spec["polytope_index"]
        signature = tuple(sorted(tuple(vertex) for vertex in polytope_spec["vertices"]))
        if signature in seen_vertices:
            raise ValueError(f"Duplicate source polytope at candidate {index}.")
        seen_vertices.add(signature)
        seed = derive_seed(spec.seed, "frst", index)
        if polytope is None:
            polytope = Polytope(polytope_spec["vertices"])
        try:
            starts, diagnostic = _generate_frst_seeds_with_random_triangulations_fair(
                polytope, target_count=spec.num_starts, seed=seed, fast_only=True,
                include_points_interior_to_facets=False, make_star=True,
            )
            serialized = [serialize_triangulation(start) for start in starts]
            if any(not start["is_frst"] for start in serialized):
                raise ValueError("Sampler returned a non-FRST triangulation.")
            keys = [canonical_simplices(start.simplices()) for start in starts]
            if len(set(keys)) != len(keys):
                raise ValueError("Sampler did not return distinct FRST starts.")
            if len(serialized) < spec.num_starts and spec.skip_insufficient_starts:
                skipped.append({"polytope_spec": polytope_spec, "seed": seed,
                                "reason": "insufficient_distinct_frst_starts",
                                "requested_num_starts": spec.num_starts, "obtained_num_starts": len(serialized),
                                "sampling_diagnostics": diagnostic})
                print(f"Skipped candidate {index}: {len(serialized)}/{spec.num_starts} distinct FRST starts", flush=True)
                return False
            if len(serialized) != spec.num_starts:
                raise ValueError(f"Requested {spec.num_starts} FRST starts, obtained {len(serialized)}; "
                                 f"sampler diagnostics: {diagnostic}")
        except Exception as exc:
            raise RuntimeError(f"FRST setup failed for polytope {index}, seed={seed}: {exc}") from exc
        entry = _serialize_fetched_n_polytope(polytope_spec, polytope)
        entry["frst_seeds"] = serialized
        entries.append(entry)
        diagnostics.append({"polytope_index": index, "seed": seed, **diagnostic})
        print(f"Accepted candidate {index}: {len(serialized)} starts ({len(entries)}/{spec.num_polytopes} polytopes)", flush=True)
        return True

    if spec.skip_insufficient_starts:
        polytopes, source_metadata = load_eval_polytopes(spec, accept_polytope=sample_candidate)
    else:
        polytopes, source_metadata = load_eval_polytopes(spec)
        for polytope_spec in polytopes:
            sample_candidate(polytope_spec)
    setup = EvaluationSetup(
        rows=_build_polytope_jsonl_rows({"polytopes": entries}),
        metadata={"parameters": spec.setup_parameters(), "source_metadata": source_metadata,
                  "polytope_specs": polytopes, "sampling_diagnostics": diagnostics,
                  **({"skipped_polytopes": skipped} if spec.skip_insufficient_starts else {})},
    )
    setup.validate(spec)
    save_eval_setup(setup, output_dir if output_dir is not None else EVAL_ROOT / "data" / "setups" / setup.setup_id)
    return setup
