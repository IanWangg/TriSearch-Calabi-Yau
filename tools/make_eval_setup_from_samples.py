"""Turn a dataset split (.samples.jsonl) into an offline evaluation setup.

The evaluation then uses the FRST starts stored in the dataset file, so every
method and checkpoint starts from the same states. Nothing is resampled.

Example:
  python tools/make_eval_setup_from_samples.py \
    --samples data/cy/favorable_v1/dev/h11_15.samples.jsonl \
    --output_dir eval/data/setups/favorable_v1_dev_h11_15
  python scripts/eval_cy.py --setup_path eval/data/setups/favorable_v1_dev_h11_15 \
    --polytope_file data/cy/favorable_v1/dev/h11_15.samples.jsonl \
    --num_polytopes 30 --h11 15 --num_starts 1 --objective_budget 256 ...

The script prints the exact selection arguments that eval_cy.py must receive.
One setup holds one h11, because EvaluationSetup checks a single h11.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.config import EvaluationSpec  # noqa: E402
from eval.setup import EvaluationSetup, save_eval_setup  # noqa: E402
from mdp.cy_rollout import load_cy_sample_rows  # noqa: E402


def build_setup(samples: Path, *, seed: int) -> tuple[EvaluationSetup, EvaluationSpec]:
    rows = load_cy_sample_rows(str(samples))
    h11_values = {int(row["h11"]) for row in rows}
    if len(h11_values) != 1:
        raise ValueError(f"Expected one h11 in {samples}, found {sorted(h11_values)}.")
    start_counts = Counter(len(row["frst_list"]) for row in rows)
    if len(start_counts) != 1:
        raise ValueError(f"Every polytope needs the same number of starts; found {dict(start_counts)}.")
    num_starts = next(iter(start_counts))
    rows = sorted(rows, key=lambda row: int(row["polytope_index"]))
    spec = EvaluationSpec(num_polytopes=len(rows), h11=h11_values.pop(), num_starts=num_starts,
                          objective_budget=1, seed=seed, polytope_file=str(samples))
    setup = EvaluationSetup(rows=rows, metadata={
        "parameters": spec.setup_parameters(),
        "source_metadata": {"source": "dataset_samples_file", "path": str(samples.resolve()),
                            "selection": "all_rows_with_stored_starts"},
        "polytope_specs": [{"polytope_index": row["polytope_index"], "vertices": row["vertices"]} for row in rows],
        "sampling_diagnostics": [],
    })
    setup.validate(spec)
    return setup, spec


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0,
                        help="Must match the --seed given to eval_cy.py (it is part of the setup parameters).")
    args = parser.parse_args(argv)
    setup, spec = build_setup(args.samples.expanduser().resolve(), seed=args.seed)
    path = save_eval_setup(setup, args.output_dir)
    print(f"Setup: {path}")
    print(f"setup_id: {setup.setup_id}")
    print("eval_cy.py selection arguments: "
          f"--setup_path {path} --polytope_file {spec.polytope_file} --num_polytopes {spec.num_polytopes} "
          f"--h11 {spec.h11} --num_starts {spec.num_starts} --seed {spec.seed}")


if __name__ == "__main__":
    main()
