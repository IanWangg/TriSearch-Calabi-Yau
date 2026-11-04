"""Create shared start setups: one 10-polytope setup per seed, then one shard per polytope.

Usage: make_setups.py --seed S --num-polytopes N --num-starts K [--shards]
A shard is the same row(s) as the full setup, saved as its own num_polytopes=1 setup.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import WORKTREE, base_spec_kwargs, setup_dir, shard_setup_dir  # noqa: E402

sys.path.insert(0, str(WORKTREE))
from eval.config import EvaluationSpec  # noqa: E402
from eval.setup import EvaluationSetup, prepare_eval_setup, save_eval_setup, load_eval_setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-polytopes", type=int, required=True)
    parser.add_argument("--num-starts", type=int, required=True)
    parser.add_argument("--shards", action="store_true")
    args = parser.parse_args()
    spec = EvaluationSpec(num_polytopes=args.num_polytopes, num_starts=args.num_starts, objective_budget=1,
                          seed=args.seed, algorithms=("best_first",), **base_spec_kwargs())
    full_dir = setup_dir(args.seed, args.num_polytopes, args.num_starts)
    if (full_dir / "setup.json").exists():
        setup = load_eval_setup(full_dir)
    else:
        setup = prepare_eval_setup(spec, output_dir=full_dir)
    setup.validate(spec)
    print(f"full setup {setup.setup_id} at {full_dir}", flush=True)
    if not args.shards:
        return
    for row in setup.rows:
        index = row["polytope_index"]
        shard_spec = EvaluationSpec(num_polytopes=1, num_starts=args.num_starts, objective_budget=1,
                                    seed=args.seed, algorithms=("best_first",), **base_spec_kwargs())
        metadata = {**setup.metadata, "parameters": shard_spec.setup_parameters(),
                    "shard_of_setup_id": setup.setup_id}
        shard = EvaluationSetup(rows=[row], metadata=metadata)
        shard.validate(shard_spec)
        path = save_eval_setup(shard, shard_setup_dir(args.seed, index, args.num_starts))
        print(f"shard polytope {index} setup {shard.setup_id} at {path}", flush=True)


if __name__ == "__main__":
    main()
