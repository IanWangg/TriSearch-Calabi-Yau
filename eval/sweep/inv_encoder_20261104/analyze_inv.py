"""A3 analysis: Q3 rows (val and dev, all/15/20) for the invariant probes, next to the P1 probe rows.

Reuses the P1 functions in analyze.py (same directory): load_probe, make_groups, bootstrap, cell.
The P1 rows are copied from the P1 README table (Q3 section) so they are not retyped.
Writes <run_dir>/analysis_inv.md and <run_dir>/analysis_inv.json.

Example:
  python eval/sweep/inv_encoder_20261104/analyze_inv.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import analyze as A  # noqa: E402

MAIN_ROOT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
RUN_DIR = MAIN_ROOT / "runs/inv_encoder_20261104"
P1_README = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/"
                 "bridge-cse_01W7whPUEhsvUecX3jesrgYC/eval/sweep/probe_20261103/README.md")
INV_VARIANTS = ("inv_s0", "inv_s1", "inv_ctx_s0", "inv_ctx_s1")
P1_VARIANTS = ("plain_s0", "plain_s1", "ctx_s0", "ctx_s1")
HEADER = ["model", "split", "h11", "n_states", "recall@1", "recall@4", "random@4", "Spearman actor",
          "Spearman V vs y1", "EV V vs y1"]


def p1_rows():
    """Q3 rows of the P1 README for probe variants, verbatim, split val and dev only."""
    rows = []
    for line in P1_README.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0] in {f"probe {v}" for v in P1_VARIANTS} and cells[1] in ("val", "dev"):
            rows.append(cells)
    return rows


def inv_rows(run_dir):
    rows = []
    for variant in INV_VARIANTS:
        probe = A.load_probe(run_dir, variant)
        name = "probe " + variant
        for split, key in (("val", "sup_val"), ("dev", "dev")):
            for h_label, h in A.h11_choices(probe[key]):
                groups = A.make_groups(probe[key], h)
                n_states = int(sum(np.isfinite(g["a_r1"]).sum() for g in groups))
                actor = [A.bootstrap(groups, lambda d, k=k: A.nanmean(d[k]), seed=70 + i)
                         for i, k in enumerate(("a_r1", "a_r4", "a_rho"))]
                value_sp = A.bootstrap(groups, lambda d: A.spearman(*A.mask_pair(d, "V", "y1")), seed=80)
                value_ev = A.bootstrap(groups, lambda d: A.explained_variance(*reversed(A.mask_pair(d, "V", "y1"))),
                                       seed=81)
                rand4 = A.bootstrap(groups, lambda d: A.nanmean(d["a_rand4"]), seed=82)[0]
                rows.append([name, split, h_label, str(n_states), A.cell(actor[0]), A.cell(actor[1]),
                             f"{rand4:.3f}", A.cell(actor[2]), A.cell(value_sp), A.cell(value_ev)])
    return rows


def training_rows(run_dir):
    rows = []
    for variant in INV_VARIANTS:
        log = json.loads((run_dir / "probe" / f"{variant}_train_log.json").read_text())
        history = log["history"]
        rows.append([variant, str(len(history)), str(log["best_epoch"]),
                     f"{history[0]['train_loss']:.3f} -> {history[-1]['train_loss']:.3f}",
                     f"{log['best_val_loss']:.3f}", f"{log['wall_sec']:.0f}", f"{log.get('prep_sec', float('nan')):.1f}",
                     log["device"]])
    return rows


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    inv = inv_rows(RUN_DIR)
    p1 = p1_rows()
    order = {"val": 0, "dev": 1}
    h_order = {"all": 0, "15": 1, "20": 2}
    combined = []
    for model in [f"probe {v}" for v in P1_VARIANTS] + [f"probe {v}" for v in INV_VARIANTS]:
        block = [r for r in (p1 if model.split()[1] in P1_VARIANTS else inv) if r[0] == model]
        combined += sorted(block, key=lambda r: (order[r[1]], h_order[r[2]]))
    md = ["## Q3 probe rows (val = S_sup validation, dev = S_dev); 95% percentile CI over polytopes (B = 1000)",
          "", table(HEADER, combined), "",
          "## Training summary (from *_train_log.json)", "",
          table(["run", "epochs run", "best epoch", "train loss first -> last", "best val loss", "wall s",
                 "prep s", "device"], training_rows(RUN_DIR))]
    text = "\n".join(md) + "\n"
    (RUN_DIR / "analysis_inv.md").write_text(text)
    (RUN_DIR / "analysis_inv.json").write_text(json.dumps(dict(combined=combined), indent=1))
    print(text)


if __name__ == "__main__":
    main()
