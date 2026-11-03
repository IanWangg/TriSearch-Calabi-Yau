"""Run one RL BeFS inference variant on a saved setup, with a shared Kcup cache.

The cache is an SQLite file keyed by (objective, full FRST state key). It only
saves physical work: every lookup still counts as one logical query, so budgets
are unchanged. Seed it from earlier result directories with --seed_cache_from.

Example:
  python tools/tune_rl_search.py --setup_path runs/fv1_dev_eval/setup_dev_h11_15 \
    --polytope_file data/cy/favorable_v1/dev/h11_15.samples.jsonl --h11 15 \
    --checkpoint runs/<run>/checkpoints/200.pth --budget 256 --k 4 --value_discount 0.9 \
    --output_dir runs/fv1_infer_tune/<name> --cache runs/fv1_infer_tune/kcup_cache.sqlite3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class KcupCache:
    def __init__(self, path):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, timeout=120)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS objective (name TEXT, key_hash TEXT, value REAL, PRIMARY KEY (name, key_hash))")
        self.connection.commit()
        self.hits = self.misses = 0

    @staticmethod
    def digest(key):
        return hashlib.sha256(key.encode()).hexdigest()

    def get_many(self, name, keys):
        hashes = [self.digest(key) for key in keys]
        found = {}
        for start in range(0, len(hashes), 500):
            chunk = hashes[start:start + 500]
            rows = self.connection.execute(
                f"SELECT key_hash, value FROM objective WHERE name=? AND key_hash IN ({','.join('?' * len(chunk))})",
                [name, *chunk]).fetchall()
            found.update(rows)
        return [found.get(h) for h in hashes]

    def put_many(self, name, items):
        self.connection.executemany("INSERT OR IGNORE INTO objective VALUES (?, ?, ?)",
                                    [(name, self.digest(key), float(value)) for key, value in items])
        self.connection.commit()

    def seed_from(self, result_dir, name="max_kcup"):
        items = []
        with open(Path(result_dir) / "queries.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("status") == "ok":
                    items.append((row["state_key"], row["objective"]))
        self.put_many(name, items)
        return len(items)


def install_cache(cache):
    """Wrap engine.objective_values; order and logical accounting are preserved."""
    from mdp.cy_rollout import CYRandomRolloutEngine

    original = CYRandomRolloutEngine.objective_values

    def objective_values(self, states, name):
        states = list(states)
        if name != "max_kcup" or self.two_face_state:
            yield from original(self, states, name)
            return
        cached = cache.get_many(name, [state.key for state in states])
        missing = [state for state, value in zip(states, cached) if value is None]
        cache.hits += len(states) - len(missing)
        cache.misses += len(missing)
        computed = list(original(self, missing, name)) if missing else []
        if computed:
            cache.put_many(name, [(state.key, value) for state, value in zip(missing, computed)])
        values = iter(computed)
        for value in cached:
            yield value if value is not None else next(values)

    CYRandomRolloutEngine.objective_values = objective_values


def short(key):
    return hashlib.sha1(key.encode()).hexdigest()[:16] if isinstance(key, str) else key


class Observer:
    """Compact proposal/score log for offline analysis (keys are hashed)."""

    def __init__(self, path):
        self.stream = open(path, "w")

    def __call__(self, event):
        kind = event.get("event")
        if kind == "proposal":
            row = dict(event="proposal", polytope_index=event["polytope_index"], start_index=event["start_index"],
                       parent=short(event["parent_key"]), parent_query_index=event["parent_query_index"],
                       depth=event["depth"],
                       candidates=[[short(c["search_key"]), int(c["seen"]), int(c["selected"]), c["log_probability"]]
                                   for c in event["candidates"]])
        elif kind == "candidate_score":
            row = dict(event="score", polytope_index=event["polytope_index"], start_index=event["start_index"],
                       parent=short(event["parent_key"]), state=short(event["search_key"]),
                       query_index=event["query_index"], objective=event["objective"],
                       critic=event["critic"], score=event["score"])
        else:
            return
        self.stream.write(json.dumps(row) + "\n")

    def close(self):
        self.stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup_path", required=True)
    parser.add_argument("--polytope_file", required=True)
    parser.add_argument("--h11", type=int, required=True)
    parser.add_argument("--num_polytopes", type=int, default=30)
    parser.add_argument("--num_starts", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--k", type=int, default=4, help="Proposals per expansion; -1 = all unseen neighbors.")
    parser.add_argument("--value_discount", type=float, default=0.9)
    parser.add_argument("--proposal_mode", choices=("actor", "random", "canonical"), default="actor")
    parser.add_argument("--logp_coef", type=float, default=0.0)
    parser.add_argument("--widen_penalty", type=float)
    parser.add_argument("--proposal_seed", type=int, help="Own RNG seed for random proposals (repeats).")
    parser.add_argument("--plain_befs", action="store_true", help="Run plain best_first instead (ignores RL options).")
    parser.add_argument("--gpu_index", type=int, default=4)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache", help="Shared SQLite Kcup cache path.")
    parser.add_argument("--seed_cache_from", nargs="*", default=[], help="Result directories to load into the cache.")
    parser.add_argument("--observe", action="store_true", help="Write search_log.jsonl with actor/critic scores.")
    args = parser.parse_args(argv)

    from eval.config import EvaluationSpec
    from eval.setup import load_eval_setup
    from eval.pipeline import run_evaluation
    from eval.algorithm.rl import RLTunedBestFirst

    cache = None
    if args.cache:
        cache = KcupCache(args.cache)
        for directory in args.seed_cache_from:
            print(f"cache seeded with {cache.seed_from(directory)} rows from {directory}", flush=True)
        install_cache(cache)

    algorithm = "best_first" if args.plain_befs else "rl_value_best_first"
    spec = EvaluationSpec(
        num_polytopes=args.num_polytopes, h11=args.h11, num_starts=args.num_starts, objective_budget=args.budget,
        seed=args.seed, algorithms=(algorithm,), polytope_file=args.polytope_file,
        policy_proposal_count=None if args.plain_befs else args.k,
        value_discount=args.value_discount, policy_checkpoint=args.checkpoint,
        subcomplex_actor_type="two_face_deep_sets", gpu_index=args.gpu_index, force_cpu=args.force_cpu,
        transition_num_workers=args.workers,
    )
    setup = load_eval_setup(args.setup_path)
    setup.validate(spec)
    factories = {}
    variant = dict(algorithm=algorithm, checkpoint=str(Path(args.checkpoint).resolve()), budget=args.budget)
    if not args.plain_befs:
        def factory():
            return RLTunedBestFirst(args.k, args.value_discount, proposal_mode=args.proposal_mode,
                                    logp_coef=args.logp_coef, widen_penalty=args.widen_penalty,
                                    proposal_seed=args.proposal_seed)
        factories[algorithm] = factory
        variant.update(factory().describe())
    output_dir = Path(args.output_dir).resolve()
    observer = None
    if args.observe:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        observer = Observer(str(output_dir) + ".search_log.jsonl")
    started = time.time()
    try:
        result = run_evaluation(spec, setup, output_dir=output_dir, algorithm_factories=factories,
                                search_observer=observer)
    finally:
        if observer is not None:
            observer.close()
    variant.update(wall_sec=time.time() - started,
                   cache_hits=cache.hits if cache else None, cache_misses=cache.misses if cache else None)
    (output_dir / "variant.json").write_text(json.dumps(variant, indent=2))
    print(json.dumps(variant), flush=True)
    for rollout in result.rollouts:
        print(f"polytope={rollout.polytope_index} start={rollout.start_index} queries={rollout.objective_queries} "
              f"expansions={rollout.expansion_count} best={rollout.best_objective:.8g}", flush=True)


if __name__ == "__main__":
    main()
