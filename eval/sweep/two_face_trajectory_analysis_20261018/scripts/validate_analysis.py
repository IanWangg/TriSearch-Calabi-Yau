"""Cross-check derived analysis against independent saved evidence."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]


def main(source):
    with (source / "trajectories.jsonl").open() as stream:
        trajectories = [json.loads(line) for line in stream]
    metrics = pd.read_csv(source / "trajectory_metrics.csv")
    legacy = pd.read_csv(ROOT / "eval/results/runs/h11_50_two_face_checkpoint_sweep_20261012_000945/plots_recorded/endpoints.csv")
    derived = metrics[metrics.h11 == 50].copy()
    derived["absolute_source_run"] = derived.source_run.map(lambda p: str(ROOT / p))
    merged = derived.merge(legacy, left_on=["absolute_source_run", "polytope_index"],
                           right_on=["source_run", "polytope_index"], validate="one_to_one")
    assert len(merged) == len(derived) == len(legacy) == 440
    assert np.allclose(merged.endpoint_volume, merged.best_volume_at_1000_queries, rtol=1e-12, atol=0)
    cases, conflicts, roots = [], {}, defaultdict(dict)
    for path in source.glob("*_graph.json"):
        graph = json.loads(path.read_text())
        pair = graph["h11"], graph["group"], graph["polytope_index"]
        for node_id, n in enumerate(graph["nodes"]):
            if n["max_volume"] / n["min_volume"] > 1.000001:
                conflicts[pair + (node_id,)] = n
    initial_by_pair = defaultdict(list)
    for t in trajectories:
        pair = t["h11"], t["group"], t["polytope_index"]
        initial_by_pair[pair].append(t["queries"][0][3])
        for q, n, parent, volume, depth, ex in t["queries"]:
            if pair + (n,) in conflicts:
                cases.append(dict(h11=t["h11"], group=t["group"], polytope_index=t["polytope_index"],
                    trajectory_id=t["trajectory_id"], source_run=t["source_run"], query_index=q,
                    two_face_sha256=conflicts[pair + (n,)]["digest"], volume=volume,
                    is_current_best=math.isclose(volume, t["curve"][q], rel_tol=1e-12),
                    is_endpoint_best=math.isclose(volume, t["curve"][-1], rel_tol=1e-12)))
            if q and ex == 1 and t["algorithm"] != "cyopt_ga":
                roots[pair][n] = volume
    for values in initial_by_pair.values():
        assert max(values) / min(values) <= 1.000001
    degree_samples = {}
    for h11, study in [(30, "h11_30_two_face_checkpoint_sweep_20261011_202310"),
                       (50, "h11_50_two_face_checkpoint_sweep_20261012_000945")]:
        for group in ["favorable", "non_favorable"]:
            path = ROOT / "eval/results/runs" / study / "diagnostics" / f"node_degree_samples_{group}.jsonl"
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["sample_index"] == 0:
                        degree_samples[h11, group, row["polytope_index"]] = row["two_face_degree"]
    assert len(roots) == len(degree_samples) == 20
    assert all(len(values) == degree_samples[pair] for pair, values in roots.items())
    expected = []
    for t in trajectories:
        if t["checkpoint"] is None or t["proposal_count"] == -1:
            continue
        pair = t["h11"], t["group"], t["polytope_index"]
        values = list(roots[pair].values())
        degree = len(values)
        num_best = sum(v >= max(values) * (1 - 1e-6) for v in values)
        selected = [q for q in t["queries"] if q[0] and q[5] == 1]
        k = len(selected)
        chance = 1 - (math.comb(degree - num_best, k) / math.comb(degree, k) if degree - num_best >= k else 0)
        expected.append(dict(trajectory_id=t["trajectory_id"], h11=t["h11"], group=t["group"],
            polytope_index=t["polytope_index"], checkpoint=t["checkpoint"], algorithm=t["algorithm"],
            mode=t["mode"], proposal_count=t["proposal_count"], initial_degree=degree,
            uniform_without_replacement_best_hit_probability=chance,
            actual_best_hit=any(q[3] >= max(values) * (1 - 1e-6) for q in selected)))
    pd.DataFrame(expected).to_csv(source / "initial_uniform_reference.csv", index=False)
    report = dict(existing_h11_50_endpoints_matched=440, initial_pairing_metric_tolerance=1e-6,
        independent_initial_degree_matches=20, metric_conflict_nodes=len(conflicts),
        conflicting_observations=len(cases), current_best_conflicting_observations=sum(r["is_current_best"] for r in cases),
        endpoint_best_conflicting_observations=sum(r["is_endpoint_best"] for r in cases),
        metric_conflict_cases=cases,
        limitation="Recorded best impact only; does not establish counterfactual search priorities with a different solver.")
    (source / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "metric_conflict_cases"})
    print(pd.DataFrame(expected).query("h11==50 and algorithm=='rl_value_best_first'").groupby(
        ["checkpoint", "mode", "proposal_count"]).agg(actual_hits=("actual_best_hit", "sum"),
        uniform_expected_hits=("uniform_without_replacement_best_hit_probability", "sum")).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    main(parser.parse_args().source_dir)
