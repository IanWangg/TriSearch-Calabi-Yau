#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from data.cy.pipeline import generate_and_save_cy_k3_random_height_eval_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate K3 evaluation data by sampling random heights on selected "
            "N-lattice polytopes from k3.txt and keeping unique non-fine regular triangulations."
        )
    )
    parser.add_argument(
        "--polytope-index",
        "--polytope_index",
        "--polytope-indices",
        "--polytope_indices",
        dest="polytope_indices",
        type=int,
        nargs="+",
        required=True,
        help="One or more K3 polytope indices to generate evaluation data for.",
    )
    parser.add_argument(
        "--k3-path",
        type=str,
        default="./cy_data/k3.txt",
        help="Path to k3.txt. Defaults to data/k3.txt then cy_data/k3.txt.",
    )
    parser.add_argument(
        "--num-triangulations",
        "--num_triangulations",
        "--num-triangulations-per-polytope",
        "--num_triangulations_per_polytope",
        "--num-tri",
        "--num_tri",
        "--k",
        dest="num_triangulations",
        type=int,
        required=True,
        help="Maximum number of unique non-fine triangulations to keep per polytope.",
    )
    parser.add_argument(
        "--max-tries",
        "--max_tries",
        dest="max_tries",
        type=int,
        default=100,
        help="Maximum number of random-height triangulation attempts per polytope.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    parser.add_argument("--output-dir", type=str, default="data/cy/output_eval", help="Directory for dataset outputs.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="cy_k3_random_height_eval_dataset",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--compact-output",
        "--compact_output",
        dest="compact_output",
        action="store_true",
        help="Save only <output-name>.samples.jsonl and skip the dataset JSON.",
    )
    parser.add_argument("--triangulation-backend", type=str, default="qhull", help="cytools triangulation backend.")
    parser.add_argument(
        "--include-points-interior-to-facets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include points interior to facets when triangulating.",
    )
    parser.add_argument(
        "--make-star",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional make_star override passed to cytools.triangulate.",
    )
    parser.add_argument("--triangulation-verbosity", type=int, default=0, help="cytools triangulation verbosity.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for per-polytope generation. Defaults to min(cpu_count, num-polytopes).",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    run_result = generate_and_save_cy_k3_random_height_eval_dataset(
        polytope_indices=args.polytope_indices,
        seed=args.seed,
        k3_path=args.k3_path,
        num_triangulations=args.num_triangulations,
        max_tries=args.max_tries,
        triangulation_backend=args.triangulation_backend,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        make_star=args.make_star,
        triangulation_verbosity=args.triangulation_verbosity,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_name=args.output_name,
        compact_output=args.compact_output,
    )
    print("K3 evaluation dataset generation complete.")
    print("Saved files:")
    for key, value in run_result["paths"].items():
        print(f"  - {key}: {value}")
    print("Summary:")
    print(json.dumps(run_result["summary"], indent=2))


if __name__ == "__main__":
    main()
