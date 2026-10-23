"""Offline paired comparison of frozen current and historical CY results.

No policy inference, geometry, resampling, or raw-result rewriting occurs.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np

from eval.results.plotting import read_comparison
from eval.setup import load_eval_setup
from eval.research_paths import resolve_archived_path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prefix_trace(path, polytopes, budget):
    traces = {p: [] for p in polytopes}
    with (path / "queries.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            p = event["polytope_index"]
            if p in traces and event["start_index"] == 0 and event["query_index"] <= budget:
                traces[p].append((event["state_key"], event["objective"]))
    return traces


def render(output, config, sources, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval/data/cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    styles = {
        "current_rollout": ("#d62728", "-"),
        "current_befs": ("#ff7f0e", "-"),
        "befs": ("#1f77b4", "--"),
        "ga": ("#e377c2", "--"),
        "old_600_befs": ("#9467bd", "-."),
        "old_400_befs": ("#8c564b", "--"),
        "old_600_beam": ("#7f7f7f", ":"),
    }
    x = np.arange(config["budget"] + 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for ax, p in zip(axes.flat, config["polytopes"]):
        for source in sources:
            if not source["primary"]:
                continue
            color, style = styles[source["name"]]
            ax.plot(x, curves[source["name"]][p], label=source["label"], color=color, linestyle=style)
        ax.set(title=f"Polytope {p}", xlabel="Logical objective queries", ylabel="Best CY volume")
        ax.set_yscale("log")
        ax.set_xlim(0, config["budget"])
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(alpha=.2)
    axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("h11=30 | 5 paired FRST starts (one per polytope) | same 128-query prefix")
    fig.tight_layout(rect=(0, .10, 1, .96))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"comparison_by_polytope.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for source in sources:
        if not source["primary"]:
            continue
        rows = np.stack([curves[source["name"]][p] for p in config["polytopes"]])
        q25, median, q75 = np.quantile(rows, [.25, .5, .75], axis=0)
        color, style = styles[source["name"]]
        ax.plot(x, median, label=source["label"], color=color, linestyle=style)
        ax.fill_between(x, q25, q75, alpha=.10, color=color)
    ax.set(xlabel="Logical objective queries", ylabel="Best CY volume",
           title="h11=30 | Median and IQR over 5 paired starts")
    ax.set_yscale("log")
    ax.set_xlim(0, config["budget"])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"comparison_overall.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "plot_config.json", dict(format_version=2, budget=config["budget"],
        statistics="Raw best volume, per-polytope curves and overall median/IQR; no rerun",
        sources=sources, curves_clipped_to_budget=True))


def main(args):
    config = json.loads(args.config.read_text())
    for source in config["sources"]:
        source["path"] = str(resolve_archived_path(source["path"], historical_root=args.historical_root,
                                                   current_root=args.current_root))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    (output / "plots").mkdir()
    budget, polytopes = config["budget"], config["polytopes"]
    curves, metadata, reference, setups = {}, {}, {}, {}
    objective_values = {}
    for source in config["sources"]:
        name, path = source["name"], Path(source["path"])
        print(f"Validating {name}: {path}", flush=True)
        data = read_comparison(path)
        raw = json.loads((path / "config.json").read_text())
        assert len(data.spec["algorithms"]) == 1
        algorithm = data.spec["algorithms"][0]
        assert data.spec["h11"] == 30 and data.spec["reward_function"] == "max_kcup"
        assert data.spec["two_face_state"] and data.spec["seed"] == 0
        assert data.spec["objective_budget"] >= budget
        assert sorted(data.polytopes) == polytopes
        setup_path = resolve_archived_path(raw["setup_path"], historical_root=args.historical_root,
                                           current_root=args.current_root)
        if setup_path not in setups:
            setups[setup_path] = load_eval_setup(setup_path)
        setup = setups[setup_path]
        rows = {row["polytope_index"]: row for row in setup.rows}
        curves[name] = {}
        for p in polytopes:
            row = data.rollouts[algorithm, p, 0]
            current = dict(initial_state_key=row["initial_state_key"],
                           initial_evaluation_state_key=row["initial_evaluation_state_key"],
                           initial_objective=row["initial_objective"], vertices=rows[p]["vertices"],
                           simplices=rows[p]["frst_list"][0]["simplices"])
            if p not in reference:
                reference[p] = current
            else:
                expected = reference[p]
                for field in ("vertices", "simplices", "initial_state_key", "initial_evaluation_state_key"):
                    assert current[field] == expected[field], (name, p, field)
                assert math.isclose(current["initial_objective"], expected["initial_objective"], rel_tol=1e-9)
            curves[name][p] = data.curves[algorithm, p, 0][:budget + 1].copy()
        for key, value in data.evaluation_objectives.items():
            previous = objective_values.setdefault(key, value)
            assert math.isclose(previous, value, rel_tol=1e-6), (name, key, previous, value)
        metadata[name] = dict(source=source, spec=data.spec, setup_id=data.setup_id,
            checkpoint_sha256=data.policy_checkpoint_sha256, policy=raw.get("policy"),
            environment=raw.get("environment"),
            source_hashes={f: digest(path / f) for f in ("config.json", "summary.json", "queries.jsonl", "rollouts.jsonl")})
    assert metadata["current_rollout"]["checkpoint_sha256"] == metadata["current_befs"]["checkpoint_sha256"]
    assert metadata["old_600_befs"]["checkpoint_sha256"] == metadata["old_600_beam"]["checkpoint_sha256"]
    primary = [source["name"] for source in config["sources"] if source["primary"]]
    endpoints = {name: np.array([curves[name][p][-1] for p in polytopes]) for name in curves}
    stats = {}
    for name, values in endpoints.items():
        stats[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
            std=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
            largest_polytope_fraction_of_total=float(values.max() / values.sum()),
            mean_ratio_to_befs=float(values.mean() / endpoints["befs"].mean()),
            mean_ratio_to_ga=float(values.mean() / endpoints["ga"].mean()))
    comparisons = []
    methods = [name for name in endpoints if name != "historical_befs"]
    for left_name in methods:
        for right_name in methods:
            if left_name == right_name:
                continue
            left, right = endpoints[left_name], endpoints[right_name]
            ties = np.isclose(left, right, rtol=config["tie_relative_tolerance"], atol=0)
            comparisons.append(dict(method=left_name, baseline=right_name, wins=int(((left > right) & ~ties).sum()),
                ties=int(ties.sum()), losses=int(((left < right) & ~ties).sum()), count=len(polytopes),
                ratio_of_means=float(left.mean() / right.mean()), mean_paired_ratio=float(np.mean(left / right)),
                median_paired_ratio=float(np.median(left / right))))
    records = [dict(polytope_index=p, start_index=0, initial_volume=reference[p]["initial_objective"],
                    **{name: float(curves[name][p][-1]) for name in curves}) for p in polytopes]
    write_csv(output / "tables/paired_endpoints.csv", records)
    write_csv(output / "tables/method_summary.csv", [dict(method=name, **values) for name, values in stats.items()])
    write_csv(output / "tables/pairwise_comparisons.csv", comparisons)
    anytime = [dict(method=name, polytope_index=p, query_index=q, best_volume=float(values[q]))
               for name, by_polytope in curves.items() for p, values in by_polytope.items()
               for q in range(budget + 1)]
    write_csv(output / "tables/budget_curves.csv", anytime)
    old = next(Path(s["path"]) for s in config["sources"] if s["name"] == "historical_befs")
    new = next(Path(s["path"]) for s in config["sources"] if s["name"] == "befs")
    old_trace, new_trace = prefix_trace(old, polytopes, budget), prefix_trace(new, polytopes, budget)
    audit = dict(endpoints_match=bool(np.allclose(endpoints["befs"], endpoints["historical_befs"], rtol=1e-6)),
        prefix_states_match={str(p): [key for key, _ in old_trace[p]] == [key for key, _ in new_trace[p]]
                             for p in polytopes})
    summary = dict(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config, pairing_verified=True, source_metadata=metadata, statistics=stats,
        comparisons=comparisons, historical_befs_audit=audit,
        initial_two_face_classes=len({v["initial_evaluation_state_key"] for v in reference.values()}),
        endpoint_definition="best objective at query_index <= 128; historical suffix and budget overshoot excluded",
        analysis_script_sha256=digest(Path(__file__)))
    render(output / "plots", config, config["sources"], curves)
    write_json(output / "summary.json", summary)
    print(json.dumps(dict(statistics=stats, comparisons=comparisons, historical_befs_audit=audit), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/comparison.json")
    parser.add_argument("--historical_root", type=Path, default=ROOT)
    parser.add_argument("--current_root", type=Path, default=ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    main(parser.parse_args())
