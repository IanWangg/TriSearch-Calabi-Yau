"""Run one algorithm on one setup shard (one or more polytopes, shared starts).

Writes the standard evaluation result directory plus, for codebook algorithms,
face_mutation_candidates.jsonl with one line per child (feasibility, face, flip distance, dlnV).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORKTREE, ALGORITHMS, base_spec_kwargs  # noqa: E402

sys.path.insert(0, str(WORKTREE))
from eval.algorithm import get_algorithm  # noqa: E402
from eval.config import EvaluationSpec  # noqa: E402
from eval.pipeline import run_evaluation  # noqa: E402
from eval.setup import load_eval_setup  # noqa: E402

CODEBOOK = {"cyopt_ga", "cyopt_ga_nocross", "face_mutation_hc", "face_mutation_befs"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--setup-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()

    setup = load_eval_setup(args.setup_dir)
    parameters = setup.metadata["parameters"]
    spec = EvaluationSpec(num_polytopes=parameters["num_polytopes"], num_starts=parameters["num_starts"],
                          objective_budget=args.budget, seed=parameters["seed"], algorithms=(args.algorithm,),
                          **base_spec_kwargs())
    created = []

    def factory():
        algorithm = get_algorithm(args.algorithm)
        created.append(algorithm)
        return algorithm

    started = time.perf_counter()
    result = run_evaluation(spec, setup=setup, output_dir=args.output_dir,
                            algorithm_factories={args.algorithm: factory})
    wall = time.perf_counter() - started
    if args.algorithm in CODEBOOK:
        with (Path(args.output_dir) / "face_mutation_candidates.jsonl").open("w", encoding="utf-8") as stream:
            for algorithm in created:
                for record in getattr(algorithm, "candidate_log", []):
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
    queries = sum(rollout.objective_queries for rollout in result.rollouts)
    print(json.dumps({"algorithm": args.algorithm, "output_dir": str(result.output_dir),
                      "wall_sec": wall, "objective_queries": queries,
                      "rollouts": [dict(polytope=r.polytope_index, start=r.start_index,
                                        best=r.best_objective, initial=r.initial_objective,
                                        queries=r.objective_queries, reason=r.termination_reason)
                                   for r in result.rollouts]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
