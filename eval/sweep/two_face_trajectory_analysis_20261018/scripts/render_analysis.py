"""Render saved trajectory diagnostics without rereading raw geometry logs."""

import argparse
from collections import deque
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(source, output):
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(source / "trajectory_metrics.csv")
    paired = pd.read_csv(source / "paired_comparisons.csv")
    initial = pd.read_csv(source / "initial_actor_selection.csv")
    with (source / "trajectories.jsonl").open() as stream:
        trajectories = [json.loads(line) for line in stream]
    by_id = {t["trajectory_id"]: t for t in trajectories}
    for frame in [metrics, paired, initial]:
        frame["checkpoint"] = frame.checkpoint.fillna(0).astype(int)
        frame["proposal_count"] = frame.proposal_count.fillna(0).astype(int)
    keys = ["h11", "group", "checkpoint", "algorithm", "mode", "proposal_count"]
    grouped = metrics.groupby(keys, dropna=False)
    summary = grouped.agg(
        trajectories=("trajectory_id", "size"), median_endpoint_volume=("endpoint_volume", "median"),
        median_max_depth=("max_depth", "median"), median_best_depth=("best_depth", "median"),
        median_expansions=("expansions", "median"), median_best_query=("best_query", "median"),
        median_best_path_drawdown_dex=("best_path_max_drawdown_dex", "median"),
        median_downhill_query_fraction=("downhill_query_fraction", "median"),
        median_frontier_inversion_fraction=("frontier_inversion_fraction", "median"),
        median_frontier_regret_max_dex=("frontier_regret_max_dex", "median"),
        median_gain_after_500_dex=("gain_after_500_dex", "median"),
        median_observed_pool_gap_dex=("observed_pool_gap_dex", "median"))
    summary.to_csv(output / "configuration_summary.csv")
    paired["win"] = paired.outcome == "win"
    paired["tie"] = paired.outcome == "tie"
    for grouping, name in [(keys + ["comparison"], "paired_summary_by_group"),
                           ([k for k in keys if k != "group"] + ["comparison"], "paired_summary_all_polytopes")]:
        paired.groupby(grouping, dropna=False).agg(
            pairs=("trajectory_id", "size"), wins=("win", "sum"), ties=("tie", "sum"),
            median_ratio=("endpoint_ratio", "median"), geometric_mean_ratio=("log10_endpoint_ratio", lambda v: 10 ** v.mean()),
            median_jaccard=("state_jaccard", "median")).to_csv(output / f"{name}.csv")
    initial.groupby(keys, dropna=False).agg(
        pairs=("trajectory_id", "size"), best_neighbor_hits=("best_root_neighbor_hit", "sum"),
        median_selected_percentile=("selected_mean_percentile", "median"),
        median_initial_best_gap_dex=("root_best_gap_dex", "median")).to_csv(output / "initial_actor_summary.csv")

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def label(row):
        if not row.checkpoint:
            return row.algorithm
        algo = "BeFS" if row.algorithm == "rl_value_best_first" else "Beam"
        proposal = "all" if row.proposal_count == -1 else f"{row['mode'][0]}{row.proposal_count}"
        return f"{row.checkpoint} {algo} {proposal}"

    for h11 in (30, 50):
        part = paired[(paired.h11 == h11) & (paired.comparison == "ordinary_befs")].copy()
        part["label"] = part.apply(label, axis=1)
        part["polytope"] = part.group.map({"non_favorable": "N", "favorable": "F"}) + part.polytope_index.astype(str)
        table = part.pivot(index="label", columns="polytope", values="log10_endpoint_ratio")
        table = table.reindex(sorted(table.columns, key=lambda c: (c[0] != "N", int(c[1:]))), axis=1)
        fig, ax = plt.subplots(figsize=(11, max(6, .25 * len(table))))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#d0d0d0")
        im = ax.imshow(np.ma.masked_invalid(table.values), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table)), table.index, fontsize=8)
        ax.set_title(f"h11={h11}: log10(RL best volume / ordinary BeFS), q<=1000\nN: non_favorable; F: favorable; gray: unfinished configuration")
        fig.colorbar(im, ax=ax, label="log10 paired endpoint ratio; color clipped at +/-3")
        save(fig, f"h11_{h11}_paired_endpoint_heatmap")

        fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharex=True)
        selected = [(0, "best_first", "not_applicable", 0, "BeFS", "#555555"),
                    (0, "cyopt_ga", "not_applicable", 0, "GA", "#b03593"),
                    (400, "rl_value_best_first", "stochastic", 8, "400 BeFS s8", "#0072b2"),
                    (600, "rl_value_best_first", "none", -1, "600 BeFS all", "#009e73"),
                    (800, "rl_value_best_first", "none", -1, "800 BeFS all", "#d55e00")]
        for row_index, group in enumerate(["non_favorable", "favorable"]):
            indices = sorted(metrics[(metrics.h11 == h11) & (metrics.group == group)].polytope_index.unique())
            for col, index in enumerate(indices):
                ax = axes[row_index, col]
                for checkpoint, algorithm, mode, count, name, color in selected:
                    match = metrics[(metrics.h11 == h11) & (metrics.group == group) & (metrics.polytope_index == index)
                                    & (metrics.checkpoint == checkpoint) & (metrics.algorithm == algorithm)
                                    & (metrics["mode"] == mode) & (metrics.proposal_count == count)]
                    if len(match):
                        t = by_id[int(match.iloc[0].trajectory_id)]
                        ax.plot(t["curve"], label=name, color=color, linewidth=1.4)
                ax.set_yscale("log")
                ax.set_title(f"{'N' if group == 'non_favorable' else 'F'}{index}")
                ax.grid(alpha=.2)
                ax.set_xlabel("Objective queries")
                ax.set_ylabel("Best CY volume")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, 1.03))
        fig.suptitle(f"h11={h11}: selected recorded searches (absent lines mean unfinished runs)", y=1.07)
        fig.tight_layout()
        save(fig, f"h11_{h11}_selected_volume_curves")

    h50 = metrics[metrics.h11 == 50]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for algorithm, marker, color, name in [("rl_value_beam_search", "o", "#0072b2", "RL Beam"),
                                           ("rl_value_best_first", "^", "#d55e00", "RL BeFS")]:
        d = h50[h50.algorithm == algorithm]
        axes[0].scatter(d.max_depth, d.observed_pool_gap_dex, s=18, alpha=.5, marker=marker, color=color, label=name)
        axes[1].scatter(d.best_path_max_drawdown_dex, d.observed_pool_gap_dex, s=18, alpha=.5, marker=marker, color=color)
        if algorithm == "rl_value_best_first":
            axes[2].scatter(d.frontier_inversion_fraction, d.observed_pool_gap_dex, s=18, alpha=.5, color=color)
    for ax in axes:
        ax.set_ylabel("Gap to best pooled observed volume (dex)")
        ax.grid(alpha=.2)
    axes[0].set_xlabel("Maximum discovery depth")
    axes[0].legend()
    axes[1].set_xlabel("Maximum drawdown on best ancestry path (dex)")
    axes[2].set_xlabel("BeFS fraction of nonmax-volume frontier expansions")
    fig.suptitle("h11=50: descriptive search diagnostics; pooled best is not a global optimum")
    fig.tight_layout()
    save(fig, "h11_50_search_diagnostics")

    # Common root provides a paired view of immediate actor selection quality.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, h11 in zip(axes, [30, 50]):
        d = initial[(initial.h11 == h11) & (initial.algorithm == "rl_value_best_first") & (initial.proposal_count > 0)]
        for mode, marker in [("greedy", "o"), ("stochastic", "s")]:
            for cp, color in [(400, "#0072b2"), (600, "#009e73"), (800, "#d55e00")]:
                a = d[(d["mode"] == mode) & (d.checkpoint == cp)]
                agg = a.groupby("proposal_count").best_root_neighbor_hit.mean()
                ax.plot(agg.index, agg.values, marker=marker, color=color,
                        linestyle="-" if mode == "greedy" else "--", label=f"{cp} {mode}")
        ax.set_title(f"h11={h11} (available completed cohorts)")
        ax.set_xticks([4, 8, 12])
        ax.set_xlabel("Actor proposals per expansion")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Fraction selecting a best initial neighbor")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "initial_actor_selection")

    # Both cases use the same fixed configuration: one gain and one failure.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    case_rows = []
    for case, (group, index) in enumerate([("favorable", 0), ("non_favorable", 2)]):
        graph = json.loads((source / f"h11_50_{group}_polytope_{index}_graph.json").read_text())
        adjacency = [[] for _ in graph["nodes"]]
        for a, b in graph["edges"]:
            adjacency[a].append(b)
            adjacency[b].append(a)
        distance = {graph["root"]: 0}
        queue = deque([graph["root"]])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if b not in distance:
                    distance[b] = distance[a] + 1
                    queue.append(b)
        ax = axes[case, 0]
        cloud = ax.hexbin([distance[n] for n in distance],
                         [np.log10(graph["nodes"][n]["max_volume"]) for n in distance],
                         gridsize=55, bins="log", mincnt=1, cmap="Greys", linewidths=0)
        fig.colorbar(cloud, ax=ax, label="Observed state count")
        for algorithm, checkpoint, mode, count, color, name in [
                ("rl_value_best_first", 400, "stochastic", 8, "#0072b2", "400 BeFS s8"),
                ("best_first", 0, "not_applicable", 0, "#d55e00", "ordinary BeFS")]:
            match = metrics[(metrics.h11 == 50) & (metrics.group == group) & (metrics.polytope_index == index)
                            & (metrics.checkpoint == checkpoint) & (metrics.algorithm == algorithm)
                            & (metrics["mode"] == mode) & (metrics.proposal_count == count)].iloc[0]
            t = by_id[int(match.trajectory_id)]
            node_to_query = {q[1]: q for q in t["queries"]}
            current = max(t["queries"], key=lambda q: q[3])
            path = []
            while True:
                path.append(current)
                if current[0] == 0:
                    break
                current = node_to_query[current[2]]
            path.reverse()
            ax.plot([distance[q[1]] for q in path], np.log10([q[3] for q in path]),
                    color=color, marker=".", markersize=3, linewidth=1.2, label=name)
            axes[case, 1].plot(t["curve"], color=color, label=name)
            for q in path:
                case_rows.append(dict(group=group, polytope_index=index, trajectory_id=t["trajectory_id"],
                                      method=name, query_index=q[0], node_id=q[1], parent_id=q[2], volume=q[3],
                                      discovery_depth=q[4], observed_graph_distance=distance[q[1]]))
        ax.set_title(f"h11=50 {group} P{index}: observed graph and best ancestry")
        ax.set_xlabel("Distance from start in the observed union graph")
        ax.set_ylabel("log10(CY volume)")
        ax.legend(fontsize=8)
        axes[case, 1].set_yscale("log")
        axes[case, 1].set_xlabel("Objective queries")
        axes[case, 1].set_ylabel("Best CY volume")
        axes[case, 1].grid(alpha=.2)
        axes[case, 1].legend(fontsize=8)
    fig.suptitle("Observed graph case studies: fixed checkpoint 400 / BeFS / stochastic k=8\nLines trace first-discovery ancestry, not chronological frontier jumps or guaranteed shortest paths")
    fig.tight_layout()
    save(fig, "observed_graph_case_studies")
    pd.DataFrame(case_rows).to_csv(output / "graph_case_paths.csv", index=False)

    write_config = dict(format_version=2, comparison_budget=1000, source=str(source.resolve()),
        volume_curves="raw best CY volume, log10 axis; one saved start per polytope",
        ratios="per-polytope endpoint ratios before aggregation; not ratios of cohort medians",
        missing_runs="excluded; visible as missing heatmap cells; pair counts always retained",
        graph="only observed queried flip edges; GA adds objective observations, not edges",
        diagnostic_gap="log10(pooled best observed volume / endpoint); not a certified optimality gap")
    (output / "plot_config.json").write_text(json.dumps(write_config, indent=2) + "\n")
    print(f"Saved figures and summary tables to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_dir, args.output_dir)
