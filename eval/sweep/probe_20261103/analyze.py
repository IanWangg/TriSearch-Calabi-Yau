"""Analysis for the value-generalization and supervised-upper-bound probe (SPEC.md, 2026-11-03).

Reads runs/probe_20261103/stageA, stageB and probe/ (written by tools/probe_collect.py and
probe_train.py) and prints markdown tables with 95% bootstrap CIs over polytopes
(B = 1000 resamples of polytopes; per-polytope quantities are resampled as units).

Writes runs/probe_20261103/analysis_tables.md and analysis.json.

Example:
  python eval/sweep/probe_20261103/analyze.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

MAIN_ROOT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
RL_CKPTS = ("G", "R", "G50")
RL_LABEL = {"G": "G (g=0.99, it200)", "R": "R (g=0.95, it200)", "G50": "G50 (g=0.99, it50)"}
RL_TRAINING_LOG = MAIN_ROOT / "runs/fv1_mix_g099_s0_20261103_0604/iteration_metrics.jsonl"
PROBE_VARIANTS = ("plain_s0", "plain_s1", "ctx_s0", "ctx_s1")
B = 1000
NAN = float("nan")


# ---------------------------------------------------------------- basic statistics

def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return NAN
    return float(spearmanr(a, b).correlation)


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def explained_variance(target, prediction):
    if len(target) < 3 or np.var(target) == 0:
        return NAN
    return float(1 - np.var(target - prediction) / np.var(target))


def partial_spearman(x, y, z):
    """Rank-residual partial Spearman: correlate ranks of x and y after regressing out ranks of z."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])

    def residual(r):
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        return r - design @ coef

    return pearson(residual(rx), residual(ry))


def nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else NAN


def concat(groups):
    return {key: np.concatenate([group[key] for group in groups]) for key in groups[0]} if groups else {}


def bootstrap(groups, stat, seed):
    """Point estimate and 95% percentile CI, resampling polytopes (groups) with replacement."""
    if not groups:
        return NAN, (NAN, NAN)
    point = stat(concat(groups))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        index = rng.integers(0, len(groups), len(groups))
        draws.append(stat(concat([groups[i] for i in index])))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    interval = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if len(draws) else (NAN, NAN)
    return point, interval


def cell(point_and_ci):
    point, (lo, hi) = point_and_ci
    if not math.isfinite(point):
        return "n/a"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- per-state features

def features(V, logp, dlnv, y1, lnV=NAN, y2=NAN, mc_best=NAN, mc_boot=NAN, child_V=None, child_y1=None):
    """Per-state scalars used by the statistics. Actor ranking needs >= 2 children."""
    dlnv = np.asarray(dlnv, dtype=float)
    n = len(dlnv)
    row = dict(n=n, V=V, lnV=lnV, y1=y1, y2=NAN if y2 is None else y2,
               mc_best=mc_best, mc_boot=mc_boot,
               a_r1=NAN, a_r4=NAN, a_rand4=NAN, a_rho=NAN,
               c_hitA=NAN, c_hitB=NAN, c_hitC=NAN, c_rhoA=NAN, c_rhoB=NAN, c_rhoC=NAN, c_rand=NAN)
    if n >= 2:
        best = int(np.argmax(dlnv))
        order = np.argsort(-np.asarray(logp, dtype=float), kind="stable")
        rank = int(np.where(order == best)[0][0])
        row.update(a_r1=float(rank < 1), a_r4=float(rank < 4), a_rand4=min(4, n) / n,
                   a_rho=spearman(logp, dlnv))
    if child_y1 is not None and child_V is not None:
        child_y1 = np.array([NAN if v is None else v for v in child_y1], dtype=float)
        valid = np.isfinite(child_y1)
        k = int(valid.sum())
        if k >= 1:
            target = dlnv[valid] + child_y1[valid]
            child_V = np.asarray(child_V, dtype=float)[valid]
            scores = {"A": dlnv[valid] + GAMMA * child_V, "B": dlnv[valid], "C": child_V}
            for name, score in scores.items():
                hit = target[int(np.argmax(score))] >= target.max() - 1e-12
                row[f"c_hit{name}"] = float(hit)
                row[f"c_rho{name}"] = spearman(score, target) if k >= 2 else NAN
            row["c_rand"] = 1.0 / k
    return row


GAMMA = 0.99
STATE_KEYS = ("n", "V", "lnV", "y1", "y2", "mc_best", "mc_boot", "a_r1", "a_r4", "a_rand4", "a_rho",
              "c_hitA", "c_hitB", "c_hitC", "c_rhoA", "c_rhoB", "c_rhoC", "c_rand")
LABELS = ("y1", "y2", "mc_best")


def make_groups(rows, h11=None):
    """One dict of arrays per polytope. Within-polytope Spearman is stored as length-1 arrays."""
    by_polytope = defaultdict(list)
    for row in rows:
        if h11 is None or row["h11"] == h11:
            by_polytope[(row["h11"], row["polytope_index"])].append(row)
    groups = []
    for key in sorted(by_polytope):
        members = by_polytope[key]
        group = {name: np.array([m[name] for m in members], dtype=float) for name in STATE_KEYS}
        V = group["V"]
        for label in LABELS:
            y = group[label]
            mask = np.isfinite(V) & np.isfinite(y)
            value = spearman(V[mask], y[mask]) if mask.sum() >= 5 else NAN
            group[f"w_{label}"] = np.array([value], dtype=float)
        groups.append(group)
    return groups


def mask_pair(d, a, b):
    m = np.isfinite(d[a]) & np.isfinite(d[b])
    return d[a][m], d[b][m]


# ---------------------------------------------------------------- loading

def load_rl(run_dir, ckpts=RL_CKPTS):
    """Rows per (checkpoint, set) built from stage A records and stage B labels."""
    stage_b = {}
    for set_name in ("crit", "dev"):
        path = run_dir / "stageB" / f"{set_name}.jsonl"
        if path.exists():
            for line in path.open():
                record = json.loads(line)
                stage_b[record["state_id"]] = record
    rows = {(ck, s): [] for ck in ckpts for s in ("crit", "dev")}
    for set_name in ("crit", "dev"):
        path = run_dir / "stageA" / f"records_{set_name}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            record = json.loads(line)
            labels = stage_b.get(record["state_id"], {})
            mc = labels.get("mc")
            child_y1 = labels.get("child_y1")
            for ck in ckpts:
                out = record["ckpt"][ck]
                logp = np.asarray(out["logp"], dtype=float)
                dlnv = np.array([v - record["lnV"] for _, v in record["children"]], dtype=float)
                use_mc = ck == "G" and mc is not None
                row = features(
                    V=out["V"], logp=logp, dlnv=dlnv, y1=record["y1"] if record["y1"] is not None else NAN,
                    lnV=record["lnV"], y2=labels.get("y2"),
                    mc_best=float(np.mean(mc["best"])) if use_mc else NAN,
                    mc_boot=float(np.mean(mc["boot"])) if use_mc else NAN,
                    child_V=out["V_child"] if child_y1 is not None else None, child_y1=child_y1)
                row.update(set=set_name, h11=record["h11"], polytope_index=record["polytope_index"],
                           state_id=record["state_id"])
                rows[(ck, set_name)].append(row)
    return rows


def load_probe(run_dir, variant):
    """Rows for the probe: S_sup train, S_sup validation and S_dev states."""
    path = run_dir / "probe" / f"{variant}_pred.jsonl"
    rows = {"sup_train": [], "sup_val": [], "dev": []}
    if not path.exists():
        return rows
    for line in path.open():
        record = json.loads(line)
        key = {"train": "sup_train", "val": "sup_val", "dev": "dev"}[record["role"]]
        row = features(V=record["V"], logp=np.asarray(record["logp"]), dlnv=record["dlnV"], y1=record["y1"])
        row.update(set=key, h11=record["h11"], polytope_index=record["polytope_index"], state_id=record["state_id"])
        rows[key].append(row)
    return rows


# ---------------------------------------------------------------- tables

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def h11_choices(groups):
    return [("all", None), ("15", 15), ("20", 20)]


def q1_table(rl):
    out_rows, results = [], {}
    for set_name, label in (("crit", "train polytopes (S_crit_train)"), ("dev", "dev (S_dev)")):
        rows = rl[("G", set_name)]
        for h_label, h in h11_choices(rows):
            groups = make_groups(rows, h)
            if not groups:
                continue
            pooled = lambda d: mask_pair(d, "V", "mc_boot")
            pear = bootstrap(groups, lambda d: pearson(*pooled(d)), seed=1)
            ev = bootstrap(groups, lambda d: explained_variance(*reversed(pooled(d))), seed=2)
            n_states = sum(len(g["V"]) for g in groups)
            n_valid = len(pooled(concat(groups))[0])
            results[(set_name, h_label)] = dict(pearson=pear, ev=ev, n=n_valid, polytopes=len(groups))
            out_rows.append([label, h_label, n_valid, len(groups), cell(pear), cell(ev)])
    gap_rows = []
    for h_label, h in h11_choices(rl[("G", "dev")]):
        train_groups, dev_groups = make_groups(rl[("G", "crit")], h), make_groups(rl[("G", "dev")], h)
        if not train_groups or not dev_groups:
            continue
        for metric, fn, seed in (("Pearson", lambda d: pearson(*mask_pair(d, "V", "mc_boot")), 3),
                                 ("EV", lambda d: explained_variance(*reversed(mask_pair(d, "V", "mc_boot"))), 4)):
            point_t, ci_t = bootstrap(train_groups, fn, seed)
            point_d, ci_d = bootstrap(dev_groups, fn, seed + 10)
            diff_point = point_t - point_d
            rng = np.random.default_rng(seed + 20)
            draws = []
            for _ in range(B):
                a = bootstrap_draw(train_groups, fn, rng)
                b = bootstrap_draw(dev_groups, fn, rng)
                draws.append(a - b)
            draws = np.asarray(draws)
            draws = draws[np.isfinite(draws)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            gap_rows.append([metric, h_label, f"{diff_point:.3f} [{lo:.3f}, {hi:.3f}]"])
    return out_rows, gap_rows, results


def bootstrap_draw(groups, stat, rng):
    index = rng.integers(0, len(groups), len(groups))
    return stat(concat([groups[i] for i in index]))


def q2_neighborhood(rl):
    rows_out = []
    for ck in RL_CKPTS:
        for set_name in ("crit", "dev"):
            for label in LABELS:
                if label == "mc_best" and ck != "G":
                    continue
                for h_label, h in h11_choices(rl[(ck, set_name)]):
                    groups = make_groups(rl[(ck, set_name)], h)
                    labelled = [g for g in groups]
                    n_states = int(sum(np.isfinite(g[label]).sum() for g in groups))
                    if n_states < 3:
                        continue

                    def stat_v(d, label=label):
                        return spearman(*mask_pair(d, "V", label))

                    def stat_b0(d, label=label):
                        m = np.isfinite(d["lnV"]) & np.isfinite(d[label])
                        return spearman(-d["lnV"][m], d[label][m])

                    def stat_partial(d, label=label):
                        m = np.isfinite(d["V"]) & np.isfinite(d[label]) & np.isfinite(d["lnV"])
                        return partial_spearman(d["V"][m], d[label][m], d["lnV"][m])

                    def stat_within(d, label=label):
                        return nanmean(d[f"w_{label}"])

                    cells = [bootstrap(labelled, f, seed=5 + i) for i, f in
                             enumerate((stat_v, stat_b0, stat_partial, stat_within))]
                    n_poly = sum(1 for g in groups if np.isfinite(g[f"w_{label}"]).all())
                    rows_out.append([RL_LABEL[ck], set_name, h_label, label, n_states, cell(cells[0]),
                                     cell(cells[1]), cell(cells[2]), f"{cell(cells[3])} (n_polytopes={n_poly})"])
    return rows_out


def q2_actor(rl, probe, set_names=("crit", "dev")):
    rows_out = []
    for ck in RL_CKPTS:
        for set_name in set_names:
            for h_label, h in h11_choices(rl[(ck, set_name)]):
                groups = make_groups(rl[(ck, set_name)], h)
                rows_out.append(actor_row(RL_LABEL[ck], set_name, h_label, groups))
    return rows_out


def actor_row(label, set_name, h_label, groups):
    n_states = int(sum(np.isfinite(g["a_r1"]).sum() for g in groups))
    cells = [bootstrap(groups, lambda d, k=k: nanmean(d[k]), seed=7 + i) for i, k in
             enumerate(("a_r1", "a_r4", "a_rand4", "a_rho"))]
    return [label, set_name, h_label, n_states, cell(cells[0]), cell(cells[1]), f"{cells[2][0]:.3f}",
            cell(cells[3])]


def q2_child(rl):
    rows_out = []
    for ck in RL_CKPTS:
        for h_label, h in h11_choices(rl[(ck, "dev")]):
            groups = make_groups(rl[(ck, "dev")], h)
            n_parents = int(sum(np.isfinite(g["c_rand"]).sum() for g in groups))
            if n_parents == 0:
                continue
            cells = []
            for i, key in enumerate(("c_hitA", "c_hitB", "c_hitC", "c_rhoA", "c_rhoB", "c_rhoC")):
                cells.append(cell(bootstrap(groups, lambda d, k=key: nanmean(d[k]), seed=40 + i)))
            random_one = bootstrap(groups, lambda d: nanmean(d["c_rand"]), seed=60)
            rows_out.append([RL_LABEL[ck], h_label, n_parents] + cells[:3] + [f"{random_one[0]:.3f}"] + cells[3:])
    return rows_out


def q3_table(rl, probe):
    rows_out = []
    models = []
    for ck in RL_CKPTS:
        models.append((RL_LABEL[ck], {"train": rl[(ck, "crit")], "dev": rl[(ck, "dev")]}))
    for variant in PROBE_VARIANTS:
        if probe[variant]["dev"]:
            models.append((f"probe {variant}", {"train": probe[variant]["sup_train"],
                                                 "val": probe[variant]["sup_val"],
                                                 "dev": probe[variant]["dev"]}))
    for name, splits in models:
        for split, rows in splits.items():
            if not rows:
                continue
            for h_label, h in h11_choices(rows):
                groups = make_groups(rows, h)
                n_states = int(sum(np.isfinite(g["a_r1"]).sum() for g in groups))
                actor = [bootstrap(groups, lambda d, k=k: nanmean(d[k]), seed=70 + i)
                         for i, k in enumerate(("a_r1", "a_r4", "a_rho"))]
                value_sp = bootstrap(groups, lambda d: spearman(*mask_pair(d, "V", "y1")), seed=80)
                value_ev = bootstrap(groups, lambda d: explained_variance(*reversed(mask_pair(d, "V", "y1"))),
                                     seed=81)
                rand4 = bootstrap(groups, lambda d: nanmean(d["a_rand4"]), seed=82)[0]
                rows_out.append([name, split, h_label, n_states, cell(actor[0]), cell(actor[1]),
                                 f"{rand4:.3f}", cell(actor[2]), cell(value_sp), cell(value_ev)])
    return rows_out


def counts_table(run_dir):
    rows = []
    for stage in ("A", "B"):
        path = run_dir / f"stage{stage}_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for set_name, info in summary["counts"].items():
            if set_name == "sanity":
                continue
            rows.append([f"stage {stage}", set_name, info.get("polytopes"), info.get("states"),
                         info.get("y2_states", "-")])
        rows.append([f"stage {stage} totals", "-", "-", "-", f"queries={summary['logical_queries']} "
                     f"cache_hits={summary['cache_hits']} wall={summary['wall_sec'] / 60:.1f} min"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default=str(MAIN_ROOT / "runs/probe_20261103"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    sections = []
    summary_json = {}

    sections.append("## Counts\n\n" + table(["stage", "set", "polytopes", "states", "y2 states / note"],
                                            counts_table(run_dir)))
    sanity_path = run_dir / "sanity_a_b.json"
    stage_a = json.loads((run_dir / "stageA_summary.json").read_text()) if (run_dir / "stageA_summary.json").exists() else {}

    rl = load_rl(run_dir)
    probe = {v: load_probe(run_dir, v) for v in PROBE_VARIANTS}

    q1_rows, gap_rows, q1_results = q1_table(rl)
    summary_json["q1"] = {f"{k[0]}_{k[1]}": {m: list(v) if isinstance(v, tuple) else v for m, v in d.items()}
                          for k, d in q1_results.items()}
    training_ev = None
    if RL_TRAINING_LOG.exists():
        logs = [json.loads(line) for line in RL_TRAINING_LOG.open()]
        training_ev = (logs[-1].get("ppo") or {}).get("explained_variance")
    sections.append("## Q1 critic generalization (G, V(s) vs mean MC boot)\n\n" + table(
        ["states", "h11", "n_states", "n_polytopes", "Pearson(V, boot)", "EV(V, boot)"], q1_rows)
        + "\n\nGap = train - dev:\n\n" + table(["metric", "h11", "train - dev [95% CI]"], gap_rows)
        + f"\n\nTraining-log EV of G at iteration 200 (ppo.explained_variance): {training_ev}")

    sanity_c = [e for e in [q1_results.get(("crit", "all"))] if e]
    if sanity_c and training_ev is not None:
        ev_point = sanity_c[0]["ev"][0]
        summary_json["sanity_c"] = dict(train_state_EV=ev_point, training_log_EV=training_ev,
                                        abs_diff=abs(ev_point - training_ev), passed=abs(ev_point - training_ev) <= 0.15)
        sections.append(f"Sanity (c): train-state EV {ev_point:.3f} vs training-log EV {training_ev:.3f}; "
                        f"|diff| = {abs(ev_point - training_ev):.3f} "
                        f"({'pass' if abs(ev_point - training_ev) <= 0.15 else 'FAIL, see README'}).")

    neighborhood = q2_neighborhood(rl)
    sections.append("## Q2 neighborhood: V(s) vs label, against B0 = -lnV(s)\n\n" + table(
        ["checkpoint", "set", "h11", "label", "n_states", "Spearman V", "Spearman B0", "partial Spearman V (ctrl lnV)",
         "within-polytope Spearman V"], neighborhood))
    sections.append("## Q2 actor (log pi vs dlnV, same states)\n\n" + table(
        ["checkpoint", "set", "h11", "n_states (>=2 children)", "recall@1", "recall@4", "random@4", "Spearman(log pi, dlnV)"],
        q2_actor(rl, probe)))
    child_rows = q2_child(rl)
    sections.append("## Q2 child level (dev, 2-hop subset; target = argmax dlnV + y1)\n\n" + table(
        ["checkpoint", "h11", "n_parents", "recall@1 A: dlnV+0.99V(c)", "recall@1 B: dlnV", "recall@1 C: V(c)",
         "random@1 (1/k)", "Spearman A", "Spearman B", "Spearman C"], child_rows))

    q3_rows = q3_table(rl, probe)
    sections.append("## Q3 supervised probe vs RL checkpoints (value: Spearman and EV vs y1)\n\n" + table(
        ["model", "split", "h11", "n_states", "recall@1", "recall@4", "random@4", "Spearman actor",
         "Spearman V vs y1", "EV V vs y1"], q3_rows))

    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
        summary_json["sanity_ab"] = dict(
            checked=len(sanity), max_lnV_fresh_diff=max(e["fresh_diff"] for e in sanity),
            max_V_single_vs_batch=max(e.get("V_diff", 0.0) for e in sanity),
            max_logp_single_vs_batch=max(e.get("logp_maxdiff", 0.0) for e in sanity))

    text = "\n\n".join(sections) + "\n"
    (run_dir / "analysis_tables.md").write_text(text)
    (run_dir / "analysis.json").write_text(json.dumps(summary_json, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
