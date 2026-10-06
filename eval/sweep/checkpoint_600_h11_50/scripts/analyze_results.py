"""Validate and compare the four checkpoint_600 runs without rerunning geometry."""

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

from eval.config import derive_seed
from eval.results.plotting import (
    ComparisonData, _check_pairs, _write_csv, comparison_statistics, read_comparison,
)

notes = json.loads((STUDY / "experiment_notes.json").read_text())
output = STUDY / "plots"
output.mkdir(exist_ok=True)
curves, rollouts, audits = {}, {}, []
model_config = None
common_spec = None

for tag, job in notes["jobs"].items():
    run = ROOT / job["run"]
    data = read_comparison(run)
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    algorithm = job["algorithm"]
    assert data.setup_id == notes["setup_id"]
    assert data.policy_checkpoint_sha256 == job["checkpoint_sha256"]
    assert hashlib.sha256((ROOT / job["checkpoint"]).read_bytes()).hexdigest() == job["checkpoint_sha256"]
    assert data.spec["algorithms"] == [algorithm]
    for field, expected in notes["settings"].items():
        assert data.spec[field] == expected, (tag, field)
    assert config["policy"]["value_score_definitions"][algorithm] == "ln_objective_plus_discounted_value"
    assert config["policy"]["resolved_policy_proposal_counts"][algorithm] == 4
    if model_config is None:
        model_config = config["policy"]["model"]
        common_spec = data.spec
    assert config["policy"]["model"] == model_config
    for (_, polytope, start), row in data.rollouts.items():
        assert row["seed"] == derive_seed(0, "rollout", algorithm, polytope, start)
        assert row["transition_count"] == 0
        assert row["budget_overshoot"] <= 3
        rollouts[tag, polytope, start] = {**row, "algorithm": tag, "source_algorithm": algorithm}
        curves[tag, polytope, start] = data.curves[algorithm, polytope, start]

    queries, last_indices = {}, {}
    for line in (run / "queries.jsonl").open():
        row = json.loads(line)
        key = row["polytope_index"], row["start_index"]
        assert row["status"] == "ok"
        assert row["query_index"] == last_indices.get(key, -1) + 1
        state = hashlib.sha256(row["state_key"].encode()).digest()
        known = queries.setdefault(key, {})
        assert state not in known, (tag, key, "duplicate queried state")
        known[state] = (row["depth"], row["objective"], row["query_index"])
        last_indices[key] = row["query_index"]
    counts, expanded = {}, {}
    for line in (run / "expansions.jsonl").open():
        row = json.loads(line)
        key = row["polytope_index"], row["start_index"]
        state = hashlib.sha256(row["state_key"].encode()).digest()
        known = expanded.setdefault(key, set())
        assert row["status"] == "ok" and state not in known
        assert row["expansion_index"] == len(known) + 1
        assert row["queries_before"] == counts.get(key, 0) < 2000
        assert 0 <= row["round_queries"] <= min(4, row["candidate_count"])
        assert row["objective_queries"] == row["queries_before"] + row["round_queries"]
        depth, objective, index = queries[key][state]
        assert depth == row["depth"] and index <= row["queries_before"]
        assert math.isclose(objective, row["objective"], rel_tol=1e-9)
        counts[key] = row["objective_queries"]
        known.add(state)
    for (_, polytope, start), row in data.rollouts.items():
        key = polytope, start
        assert counts[key] == last_indices[key] == row["objective_queries"]
        assert len(expanded[key]) == row["expansion_count"]
    stats = summary["policy_stats"][algorithm]
    assert stats["value_states"] == summary["objective_queries"]
    assert stats["action_batches"] > 0
    assert config["policy"]["device"] == "cuda:0" and stats["cuda_peak_allocated_bytes"] > 0
    audits.append(dict(job=tag, run=job["run"], checkpoint_sha256=job["checkpoint_sha256"],
                       num_rollouts=summary["num_rollouts"], objective_queries=summary["objective_queries"],
                       expansion_count=summary["expansion_count"], budget_overshoot=summary["budget_overshoot"],
                       device=config["policy"]["device"], wall_sec=stats["wall_sec"]))

setup = ROOT / notes["setup_path"]
for name, expected in notes["setup_sha256"].items():
    assert hashlib.sha256((setup / name).read_bytes()).hexdigest() == expected
for name, expected in notes.get("source_sha256", {}).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
assert sorted(job["physical_gpu_index"] for job in notes["jobs"].values()) == [3, 4, 5, 6]
tags = list(notes["jobs"])
combined = ComparisonData({**common_spec, "algorithms": tags}, notes["setup_id"], rollouts, curves)
_check_pairs(combined)
assert len(rollouts) == 20
per_polytope, overall = comparison_statistics(combined)
budget = notes["settings"]["objective_budget"]
rows = [{"job": tag, "model": notes["jobs"][tag]["model"],
         "algorithm": row["source_algorithm"], "polytope_index": polytope, "start_index": start,
         "initial_objective": row["initial_objective"], "best_at_budget": float(curves[tag, polytope, start][-1]),
         "best_complete_search": row["best_objective"], "objective_queries": row["objective_queries"],
         "budget_overshoot": row["budget_overshoot"], "expansion_count": row["expansion_count"],
         "termination_reason": row["termination_reason"]}
        for (tag, polytope, start), row in sorted(rollouts.items())]
_write_csv(output / "paired_endpoints.csv", rows)
summary_rows = [dict(job=tag, model=notes["jobs"][tag]["model"], algorithm=notes["jobs"][tag]["algorithm"],
                     median_best_at_budget=float(overall[tag]["median"][-1]),
                     q25_best_at_budget=float(overall[tag]["q25"][-1]),
                     q75_best_at_budget=float(overall[tag]["q75"][-1]),
                     **{field: sum(row[field] for row in rows if row["job"] == tag)
                        for field in ("objective_queries", "budget_overshoot", "expansion_count")})
                for tag in tags]
_write_csv(output / "algorithm_summary.csv", summary_rows)
paired = []
for algorithm in ("rl_value_beam_search", "rl_value_best_first"):
    entropy, parallel = "entropy_coef_0_01_" + algorithm, "num_states_512_" + algorithm
    for polytope in combined.polytopes:
        first, second = float(curves[entropy, polytope, 0][-1]), float(curves[parallel, polytope, 0][-1])
        paired.append(dict(algorithm=algorithm, polytope_index=polytope,
                           entropy_best=first, parallel_env_best=second, parallel_to_entropy_ratio=second / first,
                           result="tie" if math.isclose(first, second, rel_tol=1e-9)
                           else "parallel_env_win" if second > first else "entropy_win"))
_write_csv(output / "checkpoint_comparison.csv", paired)
plot_config = dict(
    format_version=2, setup_id=notes["setup_id"], objective_name="max_kcup",
    settings=notes["settings"], source_runs={tag: job["run"] for tag, job in notes["jobs"].items()},
    checkpoint_sha256_by_job={tag: job["checkpoint_sha256"] for tag, job in notes["jobs"].items()},
    model=model_config, budget_endpoint="best among query_index <= objective_budget",
    plotted_metric="raw_best_cy_volume", yscale="log10_with_raw_volume_values",
    per_polytope_statistics="one shared initial state; individual best-volume trajectory",
    overall_statistics="median and inclusive q25/q75 across the five paired polytopes",
    search_score="ln(volume_child) + 0.9 * critic(child)",
)
(output / "plot_config.json").write_text(json.dumps(plot_config, indent=2) + "\n")

styles = {tag: ("Entropy 0.01" if "entropy_coef" in tag else "Parallel env 512",
                "#1f77b4" if "entropy_coef" in tag else "#d62728",
                "-" if "beam_search" in tag else "--") for tag in tags}
figure, axes = plt.subplots(3, 2, figsize=(13, 11))
for ax, polytope in zip(axes.flat, combined.polytopes):
    for tag in tags:
        label, color, linestyle = styles[tag]
        label += " / Beam" if "beam_search" in tag else " / BeFS"
        ax.step(np.arange(budget + 1), curves[tag, polytope, 0], where="post",
                label=label, color=color, linestyle=linestyle)
    ax.set(title=f"Polytope {polytope}", xlabel="Logical objective queries", ylabel="Best CY volume")
    ax.set_yscale("log", base=10)
    ax.set_xlim(0, budget)
    ax.grid(alpha=.2)
axes.flat[-1].set_visible(False)
figure.suptitle("Checkpoint iteration 600 | h11=50 | one shared start per polytope")
figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
figure.tight_layout(rect=(0, .06, 1, .95))
for extension in ("png", "pdf"):
    figure.savefig(output / f"best_volume_by_polytope.{extension}", dpi=180, bbox_inches="tight")
plt.close(figure)
result = dict(status="passed", setup_id=notes["setup_id"], settings=notes["settings"],
              endpoint_rule="best objective at query_index <= 2000; overshoot excluded",
              shared_starts_and_objectives_verified=True, checkpoint_and_setup_hashes_verified=True,
              search_and_geometry_source_hashes_verified=True,
              query_deduplication_and_parent_budget_accounting_verified=True,
              source_runs=audits, algorithm_summary=summary_rows, checkpoint_comparison=paired)
(output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
notes["status"] = "complete"
for job in notes["jobs"].values():
    job["status"] = "complete"
(STUDY / "experiment_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
report = ["\n## Completed results\n", "Budget endpoints exclude queries above 2000.\n",
          "| Polytope | Entropy Beam | Entropy BeFS | Parallel env Beam | Parallel env BeFS |",
          "| --- | ---: | ---: | ---: | ---: |"]
for polytope in combined.polytopes:
    report.append("| " + str(polytope) + " | " + " | ".join(
        f"{curves[tag, polytope, 0][-1]:.8g}" for tag in tags) + " |")
report.extend(["\n[Paired endpoints](plots/paired_endpoints.csv) · "
               "[Checkpoint comparison](plots/checkpoint_comparison.csv) · "
               "[Search curves](plots/best_volume_by_polytope.png) · "
               "[Validation](plots/validation.json)\n"])
readme = STUDY / "README.md"
readme.write_text(readme.read_text().split("\n## Completed results\n")[0] + "\n".join(report))
print(json.dumps(result, indent=2))
