"""Compare existing ordinary BeFS and checkpoint_600 results offline."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval/data/cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.results.plotting import (
    COMPARISON_FIELDS, ComparisonData, _check_pairs, _write_csv,
    comparison_statistics, read_comparison,
)

notes = json.loads((STUDY / "experiment_notes.json").read_text())
assert notes["status"] == "complete"
baseline_run = "eval/results/runs/h11_50_metric_value_policy_top_4_20260927_044127/algorithms/best_first"
baseline = read_comparison(ROOT / baseline_run)
assert baseline.setup_id == notes["setup_id"]
assert baseline.spec["algorithms"] == ["best_first"]
baseline_config = json.loads((ROOT / baseline_run / "config.json").read_text())
assert "policy" not in baseline_config
assert baseline_config["neighbor_mode"] == "two_neighbors"
assert baseline_config["budget_unit"] == "logical_objective_query"
assert baseline_config["budget_tail"] == "complete_parent_expansion"

queries, last_indices = {}, {}
for line in (ROOT / baseline_run / "queries.jsonl").open():
    row = json.loads(line)
    key = row["polytope_index"], row["start_index"]
    assert row["status"] == "ok"
    assert row["query_index"] == last_indices.get(key, -1) + 1
    state = hashlib.sha256(row["state_key"].encode()).digest()
    known = queries.setdefault(key, {})
    assert state not in known
    known[state] = (row["depth"], row["objective"], row["query_index"])
    last_indices[key] = row["query_index"]
counts, expanded = {}, {}
for line in (ROOT / baseline_run / "expansions.jsonl").open():
    row = json.loads(line)
    key = row["polytope_index"], row["start_index"]
    state = hashlib.sha256(row["state_key"].encode()).digest()
    known = expanded.setdefault(key, set())
    assert row["status"] == "ok" and state not in known
    assert row["expansion_index"] == len(known) + 1
    assert row["queries_before"] == counts.get(key, 0) < notes["settings"]["objective_budget"]
    assert 0 <= row["round_queries"] <= row["candidate_count"]
    assert row["objective_queries"] == row["queries_before"] + row["round_queries"]
    depth, objective, index = queries[key][state]
    assert depth == row["depth"] and index <= row["queries_before"]
    assert math.isclose(objective, row["objective"], rel_tol=1e-9)
    known.add(state)
    counts[key] = row["objective_queries"]
for (_, polytope, start), row in baseline.rollouts.items():
    assert counts[polytope, start] == last_indices[polytope, start] == row["objective_queries"]
    assert len(expanded[polytope, start]) == row["expansion_count"]

curves, rollouts = dict(baseline.curves), dict(baseline.rollouts)
sources = {"best_first": baseline_run}
for tag, job in notes["jobs"].items():
    data = read_comparison(ROOT / job["run"])
    assert data.setup_id == baseline.setup_id
    assert data.policy_checkpoint_sha256 == job["checkpoint_sha256"]
    for field in COMPARISON_FIELDS:
        assert data.spec.get(field) == baseline.spec.get(field), field
    for (algorithm, polytope, start), row in data.rollouts.items():
        curves[tag, polytope, start] = data.curves[algorithm, polytope, start]
        rollouts[tag, polytope, start] = {**row, "algorithm": tag}
    sources[tag] = job["run"]
tags = list(sources)
combined = ComparisonData({**baseline.spec, "algorithms": tags}, baseline.setup_id, rollouts, curves)
_check_pairs(combined)
assert len(rollouts) == 25
_, overall = comparison_statistics(combined)
output = STUDY / "plots/baseline_comparison"
output.mkdir(exist_ok=True)
budget = baseline.spec["objective_budget"]

endpoints = [dict(job=tag, polytope_index=polytope, start_index=start,
                  initial_objective=row["initial_objective"],
                  best_at_budget=float(curves[tag, polytope, start][-1]),
                  best_complete_search=row["best_objective"], objective_queries=row["objective_queries"],
                  budget_overshoot=row["budget_overshoot"])
             for (tag, polytope, start), row in sorted(rollouts.items())]
paired = []
for tag in tags[1:]:
    for polytope in combined.polytopes:
        base, rl = float(curves["best_first", polytope, 0][-1]), float(curves[tag, polytope, 0][-1])
        paired.append(dict(job=tag, polytope_index=polytope, baseline_befs_best=base, rl_best=rl,
                           rl_to_baseline_ratio=rl / base,
                           result="tie" if math.isclose(base, rl, rel_tol=1e-9)
                           else "rl_win" if rl > base else "baseline_win"))
summary = []
for tag in tags:
    pairs = [row for row in paired if row["job"] == tag]
    summary.append(dict(job=tag, median_best_at_budget=float(overall[tag]["median"][-1]),
                        rl_wins=sum(row["result"] == "rl_win" for row in pairs),
                        ties=sum(row["result"] == "tie" for row in pairs),
                        baseline_wins=sum(row["result"] == "baseline_win" for row in pairs)))
_write_csv(output / "endpoints.csv", endpoints)
_write_csv(output / "paired_vs_befs.csv", paired)
_write_csv(output / "algorithm_summary.csv", summary)

styles = {"best_first": ("Ordinary BeFS", "#111111", "-")}
for tag in tags[1:]:
    styles[tag] = (("Entropy 0.01" if "entropy_coef" in tag else "Parallel env 512")
                   + (" / RL Beam" if "beam_search" in tag else " / RL BeFS"),
                   "#1f77b4" if "entropy_coef" in tag else "#d62728",
                   "-" if "beam_search" in tag else "--")
figure, axes = plt.subplots(3, 2, figsize=(13, 11))
for ax, polytope in zip(axes.flat, combined.polytopes):
    for tag in tags:
        label, color, linestyle = styles[tag]
        ax.step(np.arange(budget + 1), curves[tag, polytope, 0], where="post",
                label=label, color=color, linestyle=linestyle, linewidth=2 if tag == "best_first" else 1.5)
    ax.set(title=f"Polytope {polytope}", xlabel="Logical objective queries", ylabel="Best CY volume")
    ax.set_yscale("log", base=10)
    ax.set_xlim(0, budget)
    ax.grid(alpha=.2)
axes.flat[-1].set_visible(False)
figure.suptitle("Ordinary BeFS vs RL checkpoint iteration 600 | h11=50 | shared starts")
figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
figure.tight_layout(rect=(0, .06, 1, .95))
for extension in ("png", "pdf"):
    figure.savefig(output / f"best_volume_by_polytope.{extension}", dpi=180, bbox_inches="tight")
plt.close(figure)

provenance = dict(
    format_version=2, status="passed", setup_id=baseline.setup_id, settings=notes["settings"],
    source_runs=sources, checkpoint_sha256_by_job={tag: job["checkpoint_sha256"] for tag, job in notes["jobs"].items()},
    baseline_source_sha256={name: hashlib.sha256((ROOT / baseline_run / name).read_bytes()).hexdigest()
                           for name in ("config.json", "summary.json", "queries.jsonl", "expansions.jsonl", "rollouts.jsonl")},
    num_paired_rollouts=25, shared_initial_keys_and_objectives_verified=True,
    baseline_query_deduplication_and_parent_budget_accounting_verified=True,
    endpoint_rule="best volume among query_index <= 2000; completed expansion overshoot excluded",
    plotted_metric="raw_best_cy_volume", yscale="log10_with_raw_volume_values",
    baseline_score="absolute volume", baseline_proposals="all unvisited neighbors",
    rl_score="ln(volume) + 0.9 * critic", rl_proposals="actor top 4 unvisited neighbors",
    algorithms_ignored=["beam_search"], geometry_rerun=False, algorithm_summary=summary,
)
(output / "plot_config.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
