"""Time Mori cap construction and extremal-ray extraction on fresh CY objects.

Run from the repository root with the sage Python interpreter. CYTools and BLAS
thread settings match the training workers; native LP solvers may create their
own thread pools. Timings exclude imports and data loading.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = (
    REPO_ROOT / "data/cy/output4d_hugging_face"
    / "cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl"
)


def timed_call(function):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    value = function()
    return value, time.perf_counter() - wall_start, time.process_time() - cpu_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--row_indices", type=int, nargs="+", default=[0, 500, 999])
    parser.add_argument("--frst_index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tmp/mori_cap_timing_results.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.frst_index < 0:
        parser.error("repeats must be positive and frst_index must be nonnegative")
    rows = [json.loads(line) for line in args.dataset_path.read_text().splitlines() if line.strip()]
    for index in args.row_indices:
        if index < 0 or index >= len(rows) or args.frst_index >= len(rows[index]["frst_list"]):
            parser.error(f"Invalid row_index/frst_index: {index}/{args.frst_index}")

    from core.cytools_config import configure_cytools

    configure_cytools(worker=True)
    from cytools import Polytope, config

    report = {
        "dataset_path": str(args.dataset_path.resolve()),
        "row_indices": args.row_indices,
        "frst_index": args.frst_index,
        "repeats": args.repeats,
        "cytools_threads": config.n_threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "extremal_rays_method": "lp (CYTools default, scipy HiGHS)",
        "fresh_geometry_for_each_sample": True,
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for row_index in args.row_indices:
        row = rows[row_index]
        simplices = row["frst_list"][args.frst_index]["simplices"]
        for repetition in range(args.repeats):
            sample = dict(row_index=row_index, polytope_index=row["polytope_index"],
                          frst_index=args.frst_index, repetition=repetition,
                          num_input_vertices=len(row["vertices"]), num_simplices=len(simplices))
            report["samples"].append(sample)
            print(json.dumps({"event": "sample_start", **sample}), flush=True)
            polytope, sample["polytope_sec"], _ = timed_call(lambda: Polytope(row["vertices"]))
            triangulation, sample["triangulate_sec"], _ = timed_call(lambda: polytope.triangulate(
                simplices=simplices, include_points_interior_to_facets=False, check_input_simplices=False))
            cy, sample["get_cy_sec"], _ = timed_call(triangulation.get_cy)
            cap, sample["mori_cap_wall_sec"], sample["mori_cap_cpu_sec"] = timed_call(
                lambda: cy.mori_cone_cap(in_basis=True))
            sample["num_generating_rays"] = len(cap.rays())
            save()
            print(json.dumps({"event": "mori_cap_complete", **sample}), flush=True)
            rays, sample["extremal_wall_sec"], sample["extremal_cpu_sec"] = timed_call(cap.extremal_rays)
            sample["num_extremal_rays"] = len(rays)
            sample["objective_wall_sec"] = sample["mori_cap_wall_sec"] + sample["extremal_wall_sec"]
            sample["extremal_fraction"] = sample["extremal_wall_sec"] / sample["objective_wall_sec"]
            save()
            print(json.dumps({"event": "sample_complete", **sample}), flush=True)
            del rays, cap, cy, triangulation, polytope
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
