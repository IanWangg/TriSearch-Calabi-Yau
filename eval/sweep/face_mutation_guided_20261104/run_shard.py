"""Run one guided or random-control algorithm on one B1 setup shard (one polytope, three shared starts).

Writes the standard evaluation result directory plus face_mutation_candidates.jsonl (one line per kept child:
feasibility, face, flip distance, dlnV, score and score rank) and scorer_timing.jsonl (one line per expansion).
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--setup-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--aggregation", default="mean2", choices=("mean2", "sum"))
    args = parser.parse_args()

    setup = load_eval_setup(args.setup_dir)
    parameters = setup.metadata["parameters"]
    spec = EvaluationSpec(num_polytopes=parameters["num_polytopes"], num_starts=parameters["num_starts"],
                          objective_budget=args.budget, seed=parameters["seed"], algorithms=(args.algorithm,),
                          **base_spec_kwargs())
    created = []

    def factory():
        algorithm = get_algorithm(args.algorithm, face_score_aggregation=args.aggregation)
        created.append(algorithm)
        return algorithm

    started = time.perf_counter()
    result = run_evaluation(spec, setup=setup, output_dir=args.output_dir,
                            algorithm_factories={args.algorithm: factory})
    wall = time.perf_counter() - started
    out = Path(args.output_dir)
    with (out / "face_mutation_candidates.jsonl").open("w", encoding="utf-8") as stream:
        for algorithm in created:
            for record in algorithm.candidate_log:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    with (out / "scorer_timing.jsonl").open("w", encoding="utf-8") as stream:
        for algorithm in created:
            for record in algorithm.scorer_log:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    queries = sum(rollout.objective_queries for rollout in result.rollouts)
    print(json.dumps({"algorithm": args.algorithm, "aggregation": args.aggregation,
                      "output_dir": str(result.output_dir), "wall_sec": wall, "objective_queries": queries,
                      "rollouts": [dict(polytope=r.polytope_index, start=r.start_index,
                                        best=r.best_objective, initial=r.initial_objective,
                                        queries=r.objective_queries, reason=r.termination_reason)
                                   for r in result.rollouts]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
