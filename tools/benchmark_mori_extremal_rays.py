"""Compare cold Mori-cap objectives with CYTools and reusable MOSEK simplex LPs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import gcd
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _signature(rays):
    result = set()
    for ray in rays:
        row = tuple(int(value) for value in ray)
        divisor = gcd(*row)
        if divisor:
            result.add(tuple(value // divisor for value in row))
    return sorted(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default="data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl")
    parser.add_argument("--max_rows", type=int, default=2)
    parser.add_argument("--frst_per_polytope", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu_id", type=int)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    if min(args.max_rows, args.frst_per_polytope, args.repeats) < 1:
        parser.error("Row, FRST and repeat counts must be positive.")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output_dir must be inside this repository.")
    output.mkdir(parents=True, exist_ok=False)
    if args.cpu_id is not None:
        os.sched_setaffinity(0, {args.cpu_id})
    from core.cytools_config import configure_cytools

    configure_cytools(worker=True)
    from cytools import Polytope
    import numpy as np
    from core.mori_extremal_rays import mosek_extremal_rays

    # Warm up process-owned solver environments. Every timed call still uses a
    # new triangulation, CY, cone, and LP task, with no scalar objective cache.
    mosek_extremal_rays([[1, 0], [0, 1]])
    rows = []
    dataset = Path(args.dataset_path)
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == args.max_rows:
                    break
    if len(rows) != args.max_rows:
        raise ValueError("Dataset has fewer rows than requested.")
    report = dict(arguments=vars(args), created_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), mosek_version=version("Mosek"),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                 ("core/mori_extremal_rays.py", "tools/benchmark_mori_extremal_rays.py")},
                  cache="fresh geometry and task per call; solver imports/environment warmed",
                  results=[])
    (output / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "measurements.jsonl").open("w", encoding="utf-8") as measurements:
        for row in rows:
            polytope = Polytope(row["vertices"])
            entries = row["frst_list"][:args.frst_per_polytope]
            if len(entries) != args.frst_per_polytope:
                raise ValueError("Polytope has fewer FRSTs than requested.")
            for frst_index, entry in enumerate(entries):
                for repeat in range(args.repeats):
                    records, signatures = {}, {}
                    for backend in ("mosek_dual", "mosek_primal", "cytools"):
                        triangulation = polytope.triangulate(simplices=entry["simplices"],
                            include_points_interior_to_facets=False, check_input_simplices=False)
                        started = time.perf_counter()
                        cy = triangulation.get_cy()
                        cap = cy.mori_cone_cap(in_basis=True)
                        cap_sec = time.perf_counter() - started
                        stats = {}
                        started = time.perf_counter()
                        rays = (cap.extremal_rays() if backend == "cytools" else
                                mosek_extremal_rays(cap.rays(), diagnostics=stats,
                                    optimizer="dual_simplex" if backend == "mosek_dual" else "primal_simplex"))
                        extremal_sec = time.perf_counter() - started
                        record = dict(polytope_index=row["polytope_index"], frst_index=frst_index,
                            repeat=repeat, backend=backend, generating_rays=len(cap.rays()),
                            dimension=int(np.asarray(cap.rays()).shape[1]), extremal_rays=len(rays),
                            cap_sec=cap_sec, extremal_sec=extremal_sec,
                            objective_sec=cap_sec + extremal_sec, diagnostics=stats)
                        records[backend], signatures[backend] = record, _signature(rays)
                        measurements.write(json.dumps(record, allow_nan=False) + "\n")
                        measurements.flush()
                        print(json.dumps(record, allow_nan=False), flush=True)
                    if not signatures["cytools"] == signatures["mosek_dual"] == signatures["mosek_primal"]:
                        (output / "mismatch.json").write_text(json.dumps(signatures, indent=2) + "\n")
                        raise AssertionError("Normalized extremal-ray sets differ between backends.")
                    report["results"].append(dict(polytope_index=row["polytope_index"],
                        frst_index=frst_index, repeat=repeat, ray_sets_match=True,
                        **{backend: dict(extremal_speedup=records["cytools"]["extremal_sec"] / record["extremal_sec"],
                                        objective_speedup=records["cytools"]["objective_sec"] / record["objective_sec"])
                           for backend, record in records.items() if backend != "cytools"}))
    report["all_ray_sets_match"] = True
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["results"]), flush=True)


if __name__ == "__main__":
    main()
