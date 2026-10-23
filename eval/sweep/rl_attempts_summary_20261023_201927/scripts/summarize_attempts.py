"""Summarize existing CY RL experiments without running geometry or policies."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT
CURRENT = ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
from eval.results.plotting import read_comparison
from eval.research_paths import resolve_archived_path

OUT = Path(__file__).resolve().parents[1]
SOURCES = []
METRICS = []
ENDPOINTS = []
TOL = 1e-6
NATIVE_CACHE = {}


def csv_rows(path):
    path = resolve_archived_path(path, historical_root=MAIN, current_root=CURRENT)
    SOURCES.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        evidence="Previously validated derived endpoint table"))
    return list(csv.DictReader(path.open()))


def native(path, budget):
    path = resolve_archived_path(path, historical_root=MAIN, current_root=CURRENT)
    if str(path) not in NATIVE_CACHE:
        print("Validating native run", path, flush=True)
        NATIVE_CACHE[str(path)] = read_comparison(path)
        SOURCES.append(dict(path=str(path), evidence="Native read_comparison validation",
            config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest()))
    data = NATIVE_CACHE[str(path)]
    assert data.spec["objective_budget"] >= budget
    name = data.spec["algorithms"][0]
    assert len(data.spec["algorithms"]) == 1
    return {(p, s): (float(values[budget]), data.rollouts[name, p, s]["initial_objective"])
            for (a, p, s), values in data.curves.items()}


def group_from_rows(rows, method_field="algorithm", value_field="best_at_budget", suffix=None):
    values, initials = defaultdict(dict), defaultdict(dict)
    for row in rows:
        name = row[method_field] + (suffix(row) if suffix else "")
        key = (int(row["polytope_index"]), int(row["start_index"]))
        assert key not in values[name], (name, key)
        values[name][key] = float(row[value_field])
        if row.get("initial_objective"):
            initials[name][key] = float(row["initial_objective"])
    return dict(values), dict(initials)


def register(group, values, *, h11, budget, initials=None, notes="", befs="best_first", ga="cyopt_ga"):
    for method, cases in values.items():
        for case, value in cases.items():
            assert math.isfinite(value) and value > 0
            ENDPOINTS.append(dict(benchmark=group, h11=h11, budget=budget, method=method,
                                  case=json.dumps(case), best_volume=value))
        if method in (befs, ga, "historical_befs"):
            continue
        for baseline, baseline_name in ((befs, "BeFS"), (ga, "GA")):
            if baseline not in values:
                continue
            shared = sorted(set(cases) & set(values[baseline]))
            assert len(shared) == len(cases), (group, method, baseline)
            if initials and method in initials and baseline in initials:
                for key in shared:
                    assert math.isclose(initials[method][key], initials[baseline][key], rel_tol=1e-9)
            left = np.array([cases[k] for k in shared])
            right = np.array([values[baseline][k] for k in shared])
            ties = np.isclose(left, right, rtol=TOL, atol=0)
            METRICS.append(dict(benchmark=group, h11=h11, budget=budget, method=method,
                baseline=baseline_name, pairs=len(shared), wins=int(((left > right) & ~ties).sum()),
                ties=int(ties.sum()), losses=int(((left < right) & ~ties).sum()),
                mean_volume=float(left.mean()), baseline_mean_volume=float(right.mean()),
                ratio_of_means=float(left.mean()/right.mean()),
                median_paired_ratio=float(np.median(left/right)),
                geometric_mean_paired_ratio=float(np.exp(np.log(left/right).mean())), notes=notes))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(args):
    global MAIN, CURRENT, OUT
    MAIN, CURRENT, OUT = (path.resolve() for path in (args.historical_root, args.current_root, args.output_dir))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "tables").mkdir()
    previous = args.history_dir.resolve() if args.history_dir else CURRENT / "eval/sweep/best_so_far_history_comparison_20261022_211226"
    table = csv_rows(previous / "tables/paired_endpoints.csv")
    fields = [k for k in table[0] if k not in ("polytope_index", "start_index", "initial_volume")]
    values = {name: {(int(r["polytope_index"]), 0): float(r[name]) for r in table} for name in fields}
    initials = {name: {(int(r["polytope_index"]), 0): float(r["initial_volume"]) for r in table} for name in fields}
    current = CURRENT / "eval/sweep/best_so_far_search_20261022_194142"
    for name, algorithm in (("current_policy_beam", "rl_policy_beam_search"),
                            ("current_value_beam", "rl_value_beam_search")):
        raw = native(current / "results/h11_30/algorithms" / algorithm, 128)
        values[name] = {key: item[0] for key, item in raw.items() if key[1] == 0}
        initials[name] = {key: item[1] for key, item in raw.items() if key[1] == 0}
    register("h30_current_and_legacy_common5_q128", values, h11=30, budget=128, initials=initials,
        befs="befs", ga="ga", notes="Five identical full-FRST starts; current checkpoint 598; two-face identity; legacy 1000-query prefixes. Old encoder includes triangle_dual; new does not.")

    search_analysis = args.search_analysis_dir.resolve() if args.search_analysis_dir else current / "analysis"
    full25 = csv_rows(search_analysis / "h11_30/paired_volumes.csv")
    methods = [k for k in full25[0] if k not in ("polytope_index", "start_index", "initial_evaluation_state_key")]
    vals = {n: {(int(r["polytope_index"]), int(r["start_index"])): float(r[n]) for r in full25} for n in methods}
    register("h30_current_full25_q128", vals, h11=30, budget=128,
        notes="25 FRST starts but 13 two-face classes; GA unavailable for all 25 with this exact two-face benchmark.")

    for h11, folder in ((23, "user_polytope_h11_23_comparison_20260926"),
                        (30, "h11_30_comparison_20260926")):
        rows = csv_rows(MAIN / "eval/results/runs" / folder / "benchmark_summary.csv")
        vals, init = group_from_rows(rows)
        register(f"h{h11}_snn_original_and_metric_q" + ("1000" if h11 == 23 else "2000"),
            vals, h11=h11, budget=1000 if h11==23 else 2000, initials=init,
            notes="Full-FRST SNN; old rl_value uses step reward+value, rl_metric_value uses ln(volume)+value. h23: one polytope, ten FRSTs in one initial two-face class.")

    # Count-bonus checkpoint: same first start per each of the five h30 polytopes.
    count_rows = csv_rows(MAIN / "eval/results/runs/h11_30_checkpoint_600_value_search_20260928_120548/plots/endpoint_summary.csv")
    count_vals, count_init = group_from_rows(count_rows)
    base_rows = [r for r in csv_rows(MAIN / "eval/results/runs/h11_30_comparison_20260926/benchmark_summary.csv")
                 if r["start_index"] == "0" and r["algorithm"] in ("cyopt_ga","rl_metric_value_beam_search","rl_metric_value_best_first")]
    base_vals, base_init = group_from_rows(base_rows)
    count_vals.update(base_vals); count_init.update(base_init)
    register("h30_snn_count_bonus600_common5_q2000", count_vals, h11=30, budget=2000, initials=count_init,
        notes="rl_value entries use count_bonus=.01 checkpoint600 and metric scoring. Baseline SNN latest has different checkpoint history; not an isolated count-bonus ablation.")

    coeff = csv_rows(MAIN / "eval/sweep/value_coefficient_h11_30/plots/endpoints.csv")
    vals, _ = group_from_rows(coeff, suffix=lambda r: ("_alpha_" + r["coefficient"]) if r["coefficient"] else "")
    ga_rows = [r for r in csv_rows(MAIN / "eval/results/runs/h11_30_comparison_20260926/benchmark_summary.csv") if r["algorithm"]=="cyopt_ga"]
    ga_vals, _ = group_from_rows(ga_rows); vals.update(ga_vals)
    register("h30_snn_value_coefficient_q2000", vals, h11=30, budget=2000,
        notes="Actor top4 fixed even at alpha=0; coefficients tuned on the same five polytopes, not independent validation.")

    # h50 SNN checkpoints: trim all archived runs to q1000, the completed GA budget.
    native_config = Path(__file__).resolve().parents[1] / "configs/h50_native_runs.json"
    selected = json.loads(native_config.read_text())
    vals, init = {}, {}
    for name, path in selected.items():
        raw = native(path, 1000)
        vals[name] = {k:v[0] for k,v in raw.items()}
        init[name] = {k:v[1] for k,v in raw.items()}
    register("h50_snn_models_common5_q1000", vals, h11=50, budget=1000, initials=init,
        notes="Same five starts; all runs truncated to1000 because GA archive budget=1000. RL/BeFS full-FRST; GA uses DNA, unaffected by visited-class dedup. Do not mix with native2000 wins.")

    seven = csv_rows(MAIN / "eval/sweep/h11_50_two_face_seven_curves_20261006_233702/plots/endpoint_table.csv")
    rows = {r["method"]:r for r in seven}
    for rep in ("full_FRST", "two_face"):
        mapping={"beam":f"RL Beam (512 env) / {rep}", "befs_rl":f"RL BeFS (512 env) / {rep}",
                 "best_first":f"BeFS / {rep}", "cyopt_ga":"GA (cyopt)"}
        vals = {n:{(p,0):float(rows[label][f"polytope_{p}"]) for p in range(5)} for n,label in mapping.items()}
        register(f"h50_snn512_second_cohort_{rep}_q2000", vals, h11=50, budget=2000,
            notes="Different five-polytopes cohort from h50_snn_models; same checkpoint512 and starting states across full/two-face. Historical diagnostic validated native pairing.")

    trajectory_analysis = args.trajectory_analysis_dir.resolve() if args.trajectory_analysis_dir else MAIN / "eval/sweep/two_face_trajectory_analysis_20261018/results/analysis_20261018_01"
    tf = csv_rows(trajectory_analysis / "trajectory_metrics.csv")
    for h11 in (30,50):
        vals, init = defaultdict(dict), defaultdict(dict)
        for r in tf:
            if int(r["h11"]) != h11:
                continue
            name = r["algorithm"] if r["algorithm"] in ("best_first", "cyopt_ga") else f"ckpt{r['checkpoint']}_{r['algorithm']}_{r['mode']}_{r['proposal_count']}"
            key = (r["group"], int(r["polytope_index"]), 0)
            assert key not in vals[name], (name,key)
            vals[name][key]=float(r["endpoint_volume"]);init[name][key]=float(r["initial_volume"])
        assert "best_first" in vals and "cyopt_ga" in vals
        register(f"h{h11}_legacy_two_face_sweep_q1000",dict(vals),h11=h11,budget=1000,initials=dict(init),
            notes="Old triangle-dual+DeepSets architecture. Partial h30 configurations keep actual n=5 rather than silently pooling with n=10. Five known cross-run numerical exceptions were audited as non-endpoint bests; not repaired.")

    write_csv(OUT / "tables/algorithm_gaps.csv", METRICS)
    write_csv(OUT / "tables/all_endpoints.csv", ENDPOINTS)
    write_json = lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
    write_json(OUT / "summary.json",dict(status="complete",created_at_utc=datetime.now(timezone.utc).isoformat(),
        metrics=METRICS, sources=SOURCES, native_runs_revalidated=list(NATIVE_CACHE),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_geometry_queries=0,new_policy_inferences=0,
        definitions=dict(ratio_of_means="mean(method volume)/mean(baseline volume)",
                         median_paired_ratio="median(method_i volume / baseline_i volume)",
                         ties_relative_tolerance=TOL),
        architecture_audit=dict(legacy_two_face=dict(parameters=204482,triangle_dual=True,episode_context=False),
                                current_two_face=dict(parameters=163074,triangle_dual=False,episode_context=True)),
        limitations=["Benchmarks, query budgets, state identities and checkpoint families must remain separate.",
                     "Selection of strong historical settings is retrospective tuning; no per-case oracle checkpoint selection.",
                     "Evaluated best_so_far checkpoint has 598 total training iterations; ongoing training is outside this frozen comparison.",
                     "Raw means may be dominated by one polytope. Same polytope reused across configs is not an independent observation."]))
    print("Completed",len(METRICS),"paired algorithm-baseline summaries from",len(ENDPOINTS),"endpoint entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_root", type=Path, default=ROOT)
    parser.add_argument("--current_root", type=Path, default=ROOT)
    parser.add_argument("--history_dir", type=Path, help="Regenerated paired history tables.")
    parser.add_argument("--search_analysis_dir", type=Path, help="Regenerated search analysis containing h11_30/.")
    parser.add_argument("--trajectory_analysis_dir", type=Path, help="Regenerated trajectory metrics directory.")
    parser.add_argument("--output_dir", type=Path, required=True)
    main(parser.parse_args())
