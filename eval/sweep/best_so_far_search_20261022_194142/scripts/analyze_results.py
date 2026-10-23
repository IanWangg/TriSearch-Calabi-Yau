"""Validate paired search logs and summarize volumes within the query budget."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from eval.results.plotting import read_comparison


def analyze(study, name, output_root):
    data = read_comparison(study / "results" / name)
    output = output_root / name
    output.mkdir(parents=True, exist_ok=False)
    pairs = [(p, s) for p in data.polytopes for s in range(data.spec["num_starts"])]
    algorithms = data.spec["algorithms"]
    budget = data.spec["objective_budget"]
    values = {a: np.array([data.curves[a, p, s][budget] for p, s in pairs]) for a in algorithms}
    records = []
    for index, (p, s) in enumerate(pairs):
        records.append(dict(polytope_index=p, start_index=s,
                            initial_evaluation_state_key=data.rollouts[algorithms[0], p, s].get("initial_evaluation_state_key"),
                            **{a: float(values[a][index]) for a in algorithms}))
    with (output / "paired_volumes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    statistics = {}
    for algorithm, endpoints in values.items():
        rolls = [data.rollouts[algorithm, p, s] for p, s in pairs]
        statistics[algorithm] = dict(mean=float(endpoints.mean()), median=float(np.median(endpoints)),
            std=float(endpoints.std(ddof=1)), minimum=float(endpoints.min()), maximum=float(endpoints.max()),
            termination_reasons=dict(Counter(row["termination_reason"] for row in rolls)),
            objective_queries_mean=float(np.mean([row["objective_queries"] for row in rolls])),
            unique_initial_two_face_states=len({row.get("initial_evaluation_state_key",
                                                        row["initial_state_key"]) for row in rolls}))
    comparisons = []
    for algorithm in algorithms:
        if not algorithm.startswith("rl_"):
            continue
        for baseline in ("best_first", "beam_search"):
            left, right = values[algorithm], values[baseline]
            ties = np.isclose(left, right, rtol=1e-6, atol=0)
            wins = (left > right) & ~ties
            losses = (left < right) & ~ties
            per_polytope = []
            for p in data.polytopes:
                indices = [i for i, pair in enumerate(pairs) if pair[0] == p]
                per_polytope.append(dict(polytope_index=p, mean=float(left[indices].mean()),
                    baseline_mean=float(right[indices].mean()), wins=int(wins[indices].sum()),
                    ties=int(ties[indices].sum()), losses=int(losses[indices].sum())))
            comparisons.append(dict(algorithm=algorithm, baseline=baseline, wins=int(wins.sum()),
                ties=int(ties.sum()), losses=int(losses.sum()), count=len(pairs),
                win_fraction=float(wins.mean()), ratio_of_means=float(left.mean() / right.mean()),
                mean_paired_ratio=float(np.mean(left / right)), median_paired_ratio=float(np.median(left / right)),
                geometric_mean_paired_ratio=float(np.exp(np.mean(np.log(left / right)))),
                per_polytope=per_polytope))
    equivalent_starts = {}
    for index, (p, start) in enumerate(pairs):
        row = data.rollouts[algorithms[0], p, start]
        equivalent_starts.setdefault(row.get("initial_evaluation_state_key", row["initial_state_key"]), []).append(index)
    class_comparisons = []
    for comparison in comparisons:
        algorithm, baseline = comparison["algorithm"], comparison["baseline"]
        left = np.array([values[algorithm][indices].mean() for indices in equivalent_starts.values()])
        right = np.array([values[baseline][indices].mean() for indices in equivalent_starts.values()])
        ties = np.isclose(left, right, rtol=1e-6, atol=0)
        class_comparisons.append(dict(algorithm=algorithm, baseline=baseline, count=len(left),
            wins=int(((left > right) & ~ties).sum()), ties=int(ties.sum()),
            losses=int(((left < right) & ~ties).sum()), ratio_of_means=float(left.mean() / right.mean())))
    anytime = []
    for cutoff in (1, 4, 8, 16, 32, 64, 128):
        if cutoff > budget:
            continue
        at_cutoff = {a: np.array([data.curves[a, p, s][cutoff] for p, s in pairs]) for a in algorithms}
        for algorithm in algorithms:
            if not algorithm.startswith("rl_"):
                continue
            for baseline in ("best_first", "beam_search"):
                left, right = at_cutoff[algorithm], at_cutoff[baseline]
                ties = np.isclose(left, right, rtol=1e-6, atol=0)
                anytime.append(dict(budget=cutoff, algorithm=algorithm, baseline=baseline,
                    mean=float(left.mean()), baseline_mean=float(right.mean()),
                    wins=int(((left > right) & ~ties).sum()), ties=int(ties.sum()),
                    losses=int(((left < right) & ~ties).sum()), count=len(pairs)))
    with (output / "anytime_comparisons.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(anytime[0]))
        writer.writeheader()
        writer.writerows(anytime)
    result = dict(checkpoint_sha256=data.policy_checkpoint_sha256, setup_id=data.setup_id,
        spec=data.spec, endpoint_definition="best volume at query_index <= objective_budget; overshoot excluded",
        pairing="same initial full FRST and objective, checked by read_comparison",
        tie_relative_tolerance=1e-6, statistics=statistics, comparisons=comparisons, anytime=anytime,
        two_face_class_averaged_comparisons=class_comparisons)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(dataset=name, statistics=statistics, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["h11_15", "h11_30"])
    parser.add_argument("--current_root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    study = args.current_root / "eval/sweep/best_so_far_search_20261022_194142"
    analyze(study, args.dataset, args.output_dir)
