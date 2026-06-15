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

from data.cy.pipeline import generate_and_save_cy_4d_random_height_eval_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluation data for 4D N-lattice reflexive polytopes by sampling "
            "random heights and keeping unique non-fine regular triangulations."
        )
    )
    parser.add_argument("--num-polytopes", type=int, required=True, help="Number of 4D reflexive polytopes to fetch.")
    parser.add_argument("--h11", type=int, required=True, help="Required h11 value for fetched fourfold polytopes.")
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=None,
        help="Optional filter on the number of vertices for fetched fourfold polytopes.",
    )
    parser.add_argument(
        "--favorable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to filter favorable fourfold polytopes. When omitted, no favorability filter is applied.",
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
        "--output-name",
        "--output_name",
        type=str,
        default="cy_4d_random_height_eval_dataset",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--compact-output",
        "--compact_output",
        dest="compact_output",
        action="store_true",
        help="Save only <output-name>.samples.jsonl and skip the dataset JSON.",
    )
    parser.add_argument("--triangulation-backend", type=str, default="cgal", help="cytools triangulation backend.")
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

    run_result = generate_and_save_cy_4d_random_height_eval_dataset(
        num_polytopes=args.num_polytopes,
        h11=args.h11,
        num_vertices=args.num_vertices,
        favorable=args.favorable,
        num_triangulations=args.num_triangulations,
        max_tries=args.max_tries,
        seed=args.seed,
        triangulation_backend=args.triangulation_backend,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        make_star=args.make_star,
        triangulation_verbosity=args.triangulation_verbosity,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_name=args.output_name,
        compact_output=args.compact_output,
    )
    print("4D evaluation dataset generation complete.")
    print("Saved files:")
    for key, value in run_result["paths"].items():
        print(f"  - {key}: {value}")
    print("Summary:")
    print(json.dumps(run_result["summary"], indent=2))


if __name__ == "__main__":
    main()
