#!/usr/bin/env python3
"""Summarize face-mutation study outputs. Reads result directories only; no geometry is run.

Usage:
  analyze.py [--results DIR ...] [--out DIR] [--budget B] [--bootstrap N] [--grid-step S]
Default results: every directory under R/runs/face_mutation_20261104/results (or the smoke directory
given explicitly). A unit is one (algorithm, seed, polytope, start); CIs are percentile bootstrap
over units. Outputs in --out: curves.csv, final_table.csv, feasibility.csv, jump_stats.csv,
summary.json and, if matplotlib is available, best_lnV_vs_queries.png.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/face_mutation_20261104/results")
FACE_ALGORITHMS = ("face_mutation_hc", "face_mutation_befs")
ORDER = ("cyopt_ga", "cyopt_ga_nocross", "face_mutation_hc", "face_mutation_befs", "best_first")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(directories):
    """Return units keyed by (algorithm, seed, polytope, start) and per-algorithm candidate records."""
    units, candidates = {}, defaultdict(list)
    for directory in directories:
        directory = Path(directory)
        if not (directory / "rollouts.jsonl").exists() or not (directory / "config.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        seed = config["spec"]["seed"]
        rollouts = read_jsonl(directory / "rollouts.jsonl")
        if not rollouts:
            continue
        queries = read_jsonl(directory / "queries.jsonl")
        objective_by_state = {}
        per_unit_queries = defaultdict(list)
        for event in queries:
            if event.get("status") != "ok":
                continue
            objective_by_state.setdefault(event["state_key"], event["objective"])
            per_unit_queries[(event["algorithm"], event["polytope_index"], event["start_index"])].append(event)
        for rollout in rollouts:
            key = (rollout["algorithm"], seed, rollout["polytope_index"], rollout["start_index"])
            ok = sorted(per_unit_queries[(rollout["algorithm"], rollout["polytope_index"], rollout["start_index"])],
                        key=lambda event: event["query_index"])
            units[key] = dict(
                initial=rollout["initial_objective"], final=rollout["best_objective"],
                queries_used=rollout["objective_queries"], reason=rollout["termination_reason"],
                query_index=np.array([event["query_index"] for event in ok], dtype=int),
                objective=np.array([event["objective"] for event in ok], dtype=float),
                best=np.array([event["best_objective"] for event in ok], dtype=float),
                source_objective=[(objective_by_state.get(event.get("source_key"))) for event in ok],
                source_keys=[event.get("source_key") for event in ok],
                directory=str(directory),
            )
            if rollout["algorithm"] == "best_first":
                # Flip children: parent objective is the objective of the source state, if queried.
                units[key]["flip_dlnv"] = [
                    (math.log(obj) - math.log(src)) if (obj and src and obj > 0 and src > 0) else None
                    for obj, src in zip(units[key]["objective"], units[key]["source_objective"])
                ]
        for algorithm in FACE_ALGORITHMS + ("cyopt_ga", "cyopt_ga_nocross"):
            path = directory / "face_mutation_candidates.jsonl"
            if algorithm in {r["algorithm"] for r in rollouts} and path.exists():
                for record in read_jsonl(path):
                    candidates[algorithm].append(dict(record, seed=seed))
    return units, candidates


def curve_at(unit, grid, goal_sign=1.0):
    """Best objective known after q queries (q=0 is the free start); returns ln V on the grid."""
    values = np.full(len(grid), unit["initial"], dtype=float)
    if len(unit["query_index"]):
        positions = np.searchsorted(unit["query_index"], grid, side="right") - 1
        valid = positions >= 0
        values[valid] = unit["best"][positions[valid]]
    return np.log(np.maximum(values, 1e-300))


def bootstrap_ci(matrix, rng, draws):
    """Mean and percentile 95% CI over rows (units), column by column."""
    n = matrix.shape[0]
    idx = rng.integers(0, n, size=(draws, n))
    means = matrix[idx].mean(axis=1)
    return matrix.mean(axis=0), np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def describe(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return dict(count=0)
    return dict(count=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)),
                min=float(values.min()), max=float(values.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="*", default=None)
    parser.add_argument("--out", default=str(HERE / "analysis"))
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--grid-step", type=int, default=25)
    args = parser.parse_args()
    if args.results:
        directories = []
        for item in args.results:
            directories.extend(glob.glob(str(Path(item) / "*")) if Path(item, "rollouts.jsonl").exists() is False
                               else [item])
    else:
        directories = sorted(glob.glob(str(DEFAULT_RESULTS / "*")))
    units, candidates = load(directories)
    if not units:
        raise SystemExit("No completed rollouts found.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    algorithms = [a for a in ORDER if any(key[0] == a for key in units)]
    grid = np.arange(0, args.budget + 1, args.grid_step)
    summary = dict(budget=args.budget, units={a: sum(1 for k in units if k[0] == a) for a in algorithms},
                   directories=len(directories))

    curves, final_rows = [], []
    for algorithm in algorithms:
        keys = sorted(k for k in units if k[0] == algorithm)
        matrix = np.array([curve_at(units[k], grid) for k in keys])
        mean, low, high = bootstrap_ci(matrix, rng, args.bootstrap)
        for q, m, lo, hi in zip(grid, mean, low, high):
            curves.append(dict(algorithm=algorithm, queries=int(q), mean_lnV=m, ci_low=lo, ci_high=hi))
        finals = np.array([math.log(max(units[k]["final"], 1e-300)) for k in keys])
        _, flo, fhi = bootstrap_ci(finals.reshape(-1, 1), rng, args.bootstrap)
        final_rows.append(dict(algorithm=algorithm, units=len(keys), mean_lnV=float(finals.mean()),
                               ci_low=float(flo[0]), ci_high=float(fhi[0]),
                               median_lnV=float(np.median(finals)),
                               mean_queries_used=float(np.mean([units[k]["queries_used"] for k in keys])),
                               budget_stops=sum(units[k]["reason"] == "budget_exhausted" for k in keys)))
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (out / "final_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    summary["final"] = final_rows

    feasibility = []
    for algorithm in FACE_ALGORITHMS:
        records = candidates.get(algorithm, [])
        if not records:
            continue
        feasible = sum(1 for r in records if r["feasible"])
        per_unit = defaultdict(list)
        for r in records:
            per_unit[(r["seed"], r["polytope_index"], r["start_hash"])].append(r["feasible"])
        rates = np.array([np.mean(v) for v in per_unit.values()])
        _, lo, hi = bootstrap_ci(rates.reshape(-1, 1), rng, args.bootstrap)
        feasibility.append(dict(algorithm=algorithm, children=len(records), feasible=feasible,
                                feasible_rate=feasible / len(records), unit_rate_mean=float(rates.mean()),
                                unit_rate_ci_low=float(lo[0]), unit_rate_ci_high=float(hi[0])))
    if feasibility:
        with (out / "feasibility.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(feasibility[0]))
            writer.writeheader()
            writer.writerows(feasibility)
    summary["feasibility"] = feasibility

    jump_rows, jump_summary = [], {}
    for algorithm in FACE_ALGORITHMS:
        records = candidates.get(algorithm, [])
        if records:
            feasible_records = [r for r in records if r["feasible"]]
            jump_summary[f"{algorithm}_flip_distance"] = describe([r["flip_distance"] for r in feasible_records])
            jump_summary[f"{algorithm}_dlnV"] = describe([r["dlnV"] for r in feasible_records])
            jump_summary[f"{algorithm}_face_symdiff"] = describe([r["face_symdiff"] for r in feasible_records])
            for r in feasible_records:
                jump_rows.append(dict(kind=algorithm, flip_distance=r["flip_distance"], dlnV=r["dlnV"]))
    if "best_first" in algorithms:
        flip_dlnv, flip_count = [], 0
        for key in units:
            if key[0] == "best_first":
                flip_dlnv.extend(units[key].get("flip_dlnv", []))
                flip_count += len(units[key]["objective"])
        jump_summary["best_first_flip_distance"] = dict(value=1, note="single flip by definition")
        jump_summary["best_first_dlnV"] = describe(flip_dlnv)
        jump_summary["best_first_children_with_parent_value"] = int(sum(v is not None for v in flip_dlnv))
        jump_summary["best_first_children"] = flip_count
        for value in flip_dlnv:
            if value is not None:
                jump_rows.append(dict(kind="best_first", flip_distance=1, dlnV=value))
    if jump_rows:
        with (out / "jump_stats.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["kind", "flip_distance", "dlnV"])
            writer.writeheader()
            writer.writerows(jump_rows)
    summary["jumps"] = jump_summary

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4.5))
        for algorithm in algorithms:
            rows = [r for r in curves if r["algorithm"] == algorithm]
            x = [r["queries"] for r in rows]
            axis.plot(x, [r["mean_lnV"] for r in rows], label=algorithm)
            axis.fill_between(x, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], alpha=0.2)
        axis.set_xlabel("logical objective queries")
        axis.set_ylabel("best ln V so far (mean, 95% bootstrap CI)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out / "best_lnV_vs_queries.png", dpi=150)
        plt.close(figure)
    except ImportError:
        summary["plot"] = "matplotlib unavailable"

    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    for row in final_rows:
        print(f"{row['algorithm']:20s} units={row['units']:3d} mean lnV={row['mean_lnV']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] median={row['median_lnV']:.3f} "
              f"budget_stops={row['budget_stops']}")
    for row in feasibility:
        print(f"{row['algorithm']:20s} feasibility {row['feasible']}/{row['children']} = {row['feasible_rate']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
