"""Offline, query-aligned max_kcup comparisons; never rerun geometry to plot."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np


ALGORITHM_STYLES = {
    "random": ("Random", "#7f7f7f", "--"),
    "greedy": ("Greedy", "#bc8f00", "--"),
    "best_first": ("BeFS", "#1f77b4", "--"),
    "beam_search": ("Beam", "#17becf", "--"),
    "cyopt_ga": ("GA (cyopt)", "#e377c2", "-."),
    "rl_stochastic_policy": ("RL stochastic", "#d62728", "-"),
    "rl_policy_beam_search": ("RL policy beam", "#9467bd", "-"),
    "rl_value_beam_search": ("RL value beam", "#2ca02c", "-"),
    "rl_value_best_first": ("RL value BeFS", "#ff7f0e", "-"),
    "rl_metric_value_beam_search": ("RL metric + value beam", "#006d2c", "-."),
    "rl_metric_value_best_first": ("RL metric + value BeFS", "#54278f", "-."),
}
EARLY_STOP_REASONS = {"no_neighbors", "no_unvisited_neighbors", "frontier_exhausted",
                      "dna_space_singleton", "no_feasible_offspring"}

COMPARISON_FIELDS = ("num_polytopes", "num_starts", "h11", "objective_budget", "seed",
                     "reward_function", "beam_width", "policy_proposal_count", "value_discount")


@dataclass
class ComparisonData:
    spec: dict
    setup_id: str
    rollouts: dict[tuple[str, int, int], dict]
    curves: dict[tuple[str, int, int], np.ndarray]
    policy_checkpoint_sha256: str | None = None

    @property
    def polytopes(self):
        return sorted({key[1] for key in self.rollouts})


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _key(row):
    return row["algorithm"], row["polytope_index"], row["start_index"]


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Expected finite positive max_kcup volume, got {value!r}.")
    return float(value)


def _check_pairs(data):
    spec = data.spec
    expected = {(algorithm, polytope, start) for algorithm in spec["algorithms"]
                for polytope in data.polytopes for start in range(spec["num_starts"])}
    if len(data.polytopes) != spec["num_polytopes"] or set(data.rollouts) != expected:
        raise ValueError("Incomplete algorithm/polytope/start combinations.")
    for polytope in data.polytopes:
        initial_keys = set()
        for start in range(spec["num_starts"]):
            paired = [data.rollouts[algorithm, polytope, start] for algorithm in spec["algorithms"]]
            first = paired[0]
            if any(row["initial_state_key"] != first["initial_state_key"] or not math.isclose(
                    row["initial_objective"], first["initial_objective"], rel_tol=1e-9) for row in paired):
                raise ValueError(f"Unpaired initial states/objectives at polytope={polytope}, start={start}.")
            initial_keys.add(first["initial_state_key"])
        if len(initial_keys) != spec["num_starts"]:
            raise ValueError(f"Duplicate initial states at polytope={polytope}.")


def read_comparison(run_dir: str | Path) -> ComparisonData:
    """Accept a complete v1/v2 run or a complete parallel benchmark manifest."""
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "benchmark.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Parallel benchmark is incomplete or has an unsupported format.")
        data = ComparisonData(manifest["spec"], manifest["setup_id"], {}, {})
        if set(manifest["jobs"]) != set(data.spec["algorithms"]):
            raise ValueError("Parallel benchmark is missing algorithms.")
        for algorithm, job in manifest["jobs"].items():
            if job["status"] != "complete":
                raise ValueError(f"Incomplete parallel job: {algorithm}.")
            child = read_comparison(run_dir / job["run_dir"])
            if child.setup_id != data.setup_id or child.spec["algorithms"] != [algorithm]:
                raise ValueError(f"Mismatched setup or algorithm in parallel job: {algorithm}.")
            if child.spec.get("skip_insufficient_starts", False) != data.spec.get("skip_insufficient_starts", False):
                raise ValueError(f"Mismatched start selection policy in parallel job: {algorithm}.")
            # Resources may differ; the comparison's scientific settings may not.
            for field in COMPARISON_FIELDS:
                if child.spec.get(field) != data.spec.get(field):
                    raise ValueError(f"Mismatched {field} in parallel job: {algorithm}.")
            if algorithm == "cyopt_ga":
                for field in ("ga_population_size", "ga_mutation_rate", "ga_elitism", "ga_max_stalled_generations",
                              "ga_face_max_points", "ga_face_samples"):
                    if child.spec.get(field) != data.spec.get(field):
                        raise ValueError(f"Mismatched {field} in parallel job: {algorithm}.")
            checkpoint = child.policy_checkpoint_sha256
            if checkpoint is not None:
                if data.policy_checkpoint_sha256 not in (None, checkpoint):
                    raise ValueError("Parallel jobs used different policy checkpoints.")
                data.policy_checkpoint_sha256 = checkpoint
            data.rollouts.update(child.rollouts)
            data.curves.update(child.curves)
        _check_pairs(data)
        return data

    if (run_dir / "failure.json").exists() or not (run_dir / "summary.json").is_file():
        raise ValueError(f"Refusing failed or incomplete evaluation: {run_dir}.")
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    version = config.get("format_version")
    if version not in (1, 2) or summary.get("format_version", version) != version:
        raise ValueError("Unsupported or inconsistent evaluation result format_version.")
    if summary.get("status") != "complete" or summary.get("setup_id") != config["setup_id"]:
        raise ValueError("Evaluation summary is incomplete or has a mismatched setup.")
    spec = config["spec"]
    if spec["reward_function"] != "max_kcup":
        raise ValueError("Volume comparisons currently require reward_function=max_kcup.")
    budget = spec["objective_budget"]
    data = ComparisonData(spec, config["setup_id"], {}, {},
                          config.get("policy", {}).get("checkpoint_sha256"))
    for row in _read_jsonl(run_dir / "rollouts.jsonl"):
        key = _key(row)
        if key in data.rollouts:
            raise ValueError(f"Duplicate rollout: {key}.")
        if row["objective_name"] != "max_kcup" or row["objective_goal"] != "max" or row["objective_budget"] != budget:
            raise ValueError(f"Inconsistent rollout objective/budget: {key}.")
        count = row["objective_queries"]
        if type(count) is not int or count < 0 or row["budget_overshoot"] != max(0, count - budget):
            raise ValueError(f"Invalid query count/overshoot: {key}.")
        reason = row["termination_reason"]
        if reason not in EARLY_STOP_REASONS | {"budget_exhausted"} or (count < budget and reason not in EARLY_STOP_REASONS):
            raise ValueError(f"Invalid early termination: {key}.")
        _positive(row["initial_objective"])
        _positive(row["best_objective"])
        data.rollouts[key] = row
        data.curves[key] = np.empty(budget + 1, dtype=float)
    _check_pairs(data)
    if summary["num_rollouts"] != len(data.rollouts):
        raise ValueError("Summary rollout count differs from logs.")
    counts = dict.fromkeys(data.rollouts, -1)
    best = dict.fromkeys(data.rollouts, -math.inf)
    best_keys = {}
    for event in _read_jsonl(run_dir / "queries.jsonl"):
        key = _key(event)
        q = event["query_index"]
        if key not in counts or type(q) is not int or q != counts[key] + 1:
            raise ValueError(f"Unknown rollout or nonconsecutive query index: {key}, q={q}.")
        if event["status"] != "ok" or event["is_initial"] != (q == 0):
            raise ValueError(f"Failed or invalid query: {key}, q={q}.")
        value = _positive(event["objective"])
        if value > best[key]:
            best[key], best_keys[key] = value, event["state_key"]
        if not math.isclose(_positive(event["best_objective"]), best[key], rel_tol=1e-9):
            raise ValueError(f"Inconsistent cumulative best: {key}, q={q}.")
        row = data.rollouts[key]
        if q == 0 and (event["state_key"] != row["initial_state_key"] or not math.isclose(
                value, row["initial_objective"], rel_tol=1e-9)):
            raise ValueError(f"Initial query differs from rollout: {key}.")
        if event["best_state_key"] != best_keys[key]:
            raise ValueError(f"Inconsistent best state key: {key}, q={q}.")
        if q <= budget:
            data.curves[key][q] = best[key]
        counts[key] = q
    for key, row in data.rollouts.items():
        if counts[key] != row["objective_queries"] or counts[key] < 0:
            raise ValueError(f"Missing queries: {key}.")
        if not math.isclose(best[key], row["best_objective"], rel_tol=1e-9) or best_keys[key] != row["best_state_key"]:
            raise ValueError(f"Rollout best differs from queries: {key}.")
        if counts[key] < budget:
            data.curves[key][counts[key] + 1:] = best[key]
    for field in ("objective_queries", "budget_overshoot", "transition_count", "expansion_count"):
        if field in summary and summary[field] != sum(row[field] for row in data.rollouts.values()):
            raise ValueError(f"Summary {field} differs from logs.")
    return data


def combine_comparisons(run_dirs) -> ComparisonData:
    """Merge completed, disjoint algorithms after verifying the paired experiment."""
    inputs = [read_comparison(path) for path in run_dirs]
    if not inputs:
        raise ValueError("At least one completed run is required.")
    first = inputs[0]
    combined = ComparisonData({**first.spec, "algorithms": []}, first.setup_id, {}, {})
    policy_spec = None
    for child in inputs:
        if child.setup_id != first.setup_id:
            raise ValueError("Cannot combine runs with different setup_id values.")
        for field in COMPARISON_FIELDS:
            if child.spec.get(field) != first.spec.get(field):
                raise ValueError(f"Cannot combine runs with different {field} values.")
        if child.spec.get("skip_insufficient_starts", False) != first.spec.get("skip_insufficient_starts", False):
            raise ValueError("Cannot combine runs with different start selection policies.")
        if set(combined.spec["algorithms"]) & set(child.spec["algorithms"]):
            raise ValueError("Cannot combine duplicate algorithms; select disjoint completed runs.")
        checkpoint = child.policy_checkpoint_sha256
        if checkpoint is not None:
            if combined.policy_checkpoint_sha256 not in (None, checkpoint):
                raise ValueError("Cannot combine runs with different policy checkpoints.")
            if policy_spec is not None:
                for field in ("subcomplex_actor_type", "in_channels", "out_channels", "hidden_channels", "num_layers"):
                    if policy_spec.get(field) != child.spec.get(field):
                        raise ValueError(f"Cannot combine runs with different policy {field} values.")
            combined.policy_checkpoint_sha256, policy_spec = checkpoint, child.spec
        combined.spec["algorithms"].extend(child.spec["algorithms"])
        if "cyopt_ga" in child.spec["algorithms"]:
            combined.spec.update({key: value for key, value in child.spec.items() if key.startswith("ga_")})
        combined.rollouts.update(child.rollouts)
        combined.curves.update(child.curves)
    _check_pairs(combined)
    return combined


def comparison_statistics(data: ComparisonData):
    """Median/IQR of raw best volumes, within each polytope and across all starts."""
    per_polytope, overall = {}, {}
    for algorithm in data.spec["algorithms"]:
        all_values = []
        for polytope in data.polytopes:
            values = np.stack([data.curves[algorithm, polytope, start]
                               for start in range(data.spec["num_starts"])])
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
            per_polytope[algorithm, polytope] = dict(q25=q25, median=median, q75=q75)
            all_values.append(values)
        # Each polytope has the same required number of starts.
        q25, median, q75 = np.quantile(np.concatenate(all_values), [0.25, 0.5, 0.75], axis=0)
        overall[algorithm] = dict(q25=q25, median=median, q75=q75)
    return per_polytope, overall


def _write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def plot_evaluation(run_dir: str | Path, output_dir: str | Path | None = None, *, additional_run_dirs=()) -> Path:
    """Generate deterministic derived artifacts; original evaluation logs are read only."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dirs = [run_dir, *(Path(path).expanduser().resolve() for path in additional_run_dirs)]
    if len(run_dirs) > 1 and output_dir is None:
        raise ValueError("Combined comparisons require a separate output_dir.")
    data = combine_comparisons(run_dirs)
    per_polytope, overall = comparison_statistics(data)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data/cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    spec = data.spec
    algorithms = spec["algorithms"]
    budget = spec["objective_budget"]
    queries = np.arange(budget + 1)
    styles = {name: ALGORITHM_STYLES.get(name, (name.replace("_", " "), "#333333", "-")) for name in algorithms}
    columns = min(2, len(data.polytopes))
    polytope_rows = math.ceil(len(data.polytopes) / columns)
    title = f"CY h11={spec['h11']} | {len(data.polytopes)} polytopes | {spec['num_starts']} starts each"

    def save(figure, name):
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        height = 3.6 * polytope_rows + 1.2
        legend_columns = min(3 if columns == 1 else 4, len(algorithms))
        legend_height = 0.25 * math.ceil(len(algorithms) / legend_columns) + 0.2
        figure, axes = plt.subplots(polytope_rows, columns, figsize=(9 if columns == 1 else 13, height), squeeze=False)
        for ax, polytope in zip(axes.flat, data.polytopes):
            for algorithm in algorithms:
                stats = per_polytope[algorithm, polytope]
                label, color, linestyle = styles[algorithm]
                ax.step(queries, stats["median"], where="post", label=label, color=color, ls=linestyle,
                        marker="o" if budget == 0 else None)
                ax.fill_between(queries, stats["q25"], stats["q75"], step="post", color=color, alpha=0.12)
            ax.set(title=f"Polytope {polytope}", xlabel="Logical objective queries", ylabel="Best CY volume (log10 scale)")
            ax.set_yscale("log", base=10)
            lower = min(per_polytope[name, polytope]["q25"].min() for name in algorithms)
            upper = max(per_polytope[name, polytope]["q75"].max() for name in algorithms)
            # Do not magnify solver noise when all plotted volumes are nearly equal.
            if upper / lower < 1.01:
                ax.set_ylim(lower / 1.05, upper * 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if budget == 0:
                ax.set_xticks([0])
            else:
                ax.set_xlim(0, budget)
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(data.polytopes):]:
            ax.set_visible(False)
        figure.suptitle(title + "\nMedian and interquartile range across starts")
        figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=legend_columns, frameon=False)
        figure.tight_layout(rect=(0, legend_height / height, 1, 1 - 0.65 / height))
        save(figure, "best_volume_by_polytope")

        figure, ax = plt.subplots(figsize=(9, 5.5))
        for algorithm in algorithms:
            stats = overall[algorithm]
            label, color, linestyle = styles[algorithm]
            ax.step(queries, stats["median"], where="post", label=label, color=color, ls=linestyle,
                    marker="o" if budget == 0 else None)
            ax.fill_between(queries, stats["q25"], stats["q75"],
                            step="post", color=color, alpha=0.12)
        ax.set(title=title + "\nMedian and interquartile range across all starts",
               xlabel="Logical objective queries", ylabel="Best CY volume (log10 scale)")
        ax.set_yscale("log", base=10)
        lower = min(overall[name]["q25"].min() for name in algorithms)
        upper = max(overall[name]["q75"].max() for name in algorithms)
        if upper / lower < 1.01:
            ax.set_ylim(lower / 1.05, upper * 1.05)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if budget == 0:
            ax.set_xticks([0])
        else:
            ax.set_xlim(0, budget)
        ax.legend(frameon=False, fontsize=9)
        figure.tight_layout()
        save(figure, "best_volume_overall")

        height = 3.8 * polytope_rows + 0.7
        figure, axes = plt.subplots(polytope_rows, columns, figsize=(9 if columns == 1 else 14, height), squeeze=False)
        for ax, polytope in zip(axes.flat, data.polytopes):
            values = [[data.curves[algorithm, polytope, start][-1] for start in range(spec["num_starts"])]
                      for algorithm in algorithms]
            boxes = ax.boxplot(values, orientation="horizontal", patch_artist=True, showfliers=False)
            for index, (algorithm, box, points) in enumerate(zip(algorithms, boxes["boxes"], values), start=1):
                box.set_facecolor(styles[algorithm][1])
                box.set_alpha(0.35)
                ax.scatter(points, index + np.linspace(-0.12, 0.12, len(points)), s=12,
                           color=styles[algorithm][1], zorder=3)
            ax.set_yticks(range(1, len(algorithms) + 1), [styles[name][0] for name in algorithms])
            ax.invert_yaxis()
            ax.set(title=f"Polytope {polytope}", xlabel=f"Best CY volume within {budget} queries", xscale="log")
            lower, upper = np.min(values), np.max(values)
            if upper / lower < 1.01:
                ax.set_xlim(lower / 1.05, upper * 1.05)
            ax.grid(axis="x", alpha=0.2)
        for ax in list(axes.flat)[len(data.polytopes):]:
            ax.set_visible(False)
        figure.suptitle(title + "\nBudget endpoint distribution; dots represent individual starts")
        figure.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
        save(figure, "budget_endpoint_distribution")

    def rollout_rows():
        for key in sorted(data.rollouts):
            row = data.rollouts[key]
            endpoint = float(data.curves[key][-1])
            yield {**{name: row[name] for name in (
                "algorithm", "polytope_index", "start_index", "seed", "initial_objective",
                "objective_budget", "objective_queries", "budget_overshoot", "transition_count", "termination_reason")},
                "expansion_count": row.get("expansion_count", ""), "best_at_budget": endpoint,
                "best_complete_search": row["best_objective"]}

    _write_csv(output_dir / "benchmark_summary.csv", rollout_rows())
    _write_csv(output_dir / "polytope_curves.csv", (
        dict(algorithm=algorithm, polytope_index=polytope, query_index=q,
             **{name: float(values[q]) for name, values in stats.items()})
        for (algorithm, polytope), stats in per_polytope.items() for q in queries))
    _write_csv(output_dir / "overall_curves.csv", (
        dict(algorithm=algorithm, query_index=q,
             **{name: float(values[q]) for name, values in stats.items()})
        for algorithm, stats in overall.items() for q in queries))
    _write_csv(output_dir / "algorithm_summary.csv", (
        dict(algorithm=algorithm, num_rollouts=sum(key[0] == algorithm for key in data.rollouts),
             median_best_at_budget=float(stats["median"][-1]),
             q25_best_at_budget=float(stats["q25"][-1]), q75_best_at_budget=float(stats["q75"][-1]),
             objective_queries=sum(row["objective_queries"] for key, row in data.rollouts.items() if key[0] == algorithm),
             budget_overshoot=sum(row["budget_overshoot"] for key, row in data.rollouts.items() if key[0] == algorithm),
             early_stops=sum(row["objective_queries"] < budget for key, row in data.rollouts.items() if key[0] == algorithm))
        for algorithm, stats in overall.items()))
    (output_dir / "plot_config.json").write_text(json.dumps({
        "format_version": 2, "run_dir": str(run_dir), "run_dirs": [str(path) for path in run_dirs],
        "setup_id": data.setup_id, "spec": spec,
        "policy_checkpoint_sha256": data.policy_checkpoint_sha256,
        "budget_endpoint": "best among query_index <= objective_budget",
        "early_stop": "carry forward only successfully terminated rollouts",
        "polytope_curve": "median and 25th/75th percentiles across starts",
        "overall_curve": "median and 25th/75th percentiles of raw best volume across all starts; equal start counts per polytope",
        "volume_axis": "base 10 logarithmic scale; tick values are raw CY volumes",
        "near_constant_volume_axes": "5% padding if displayed max/min < 1.01; underlying values unchanged",
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_dir
