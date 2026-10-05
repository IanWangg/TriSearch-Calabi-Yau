"""Replot a completed CY evaluation or parallel benchmark without geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--additional_run_dirs", nargs="+", default=[],
                        help="Combine disjoint algorithms from completed runs with the same setup and budget.")
    parser.add_argument("--output_dir")
    args = parser.parse_args(argv)
    from eval.results.plotting import plot_evaluation

    if args.additional_run_dirs and args.output_dir is None:
        parser.error("--additional_run_dirs requires --output_dir to preserve existing comparison plots.")
    print(f"Plots: {plot_evaluation(args.run_dir, args.output_dir, additional_run_dirs=args.additional_run_dirs)}")


if __name__ == "__main__":
    main()
