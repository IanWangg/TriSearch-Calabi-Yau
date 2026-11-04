"""Pooling ablation table for the plain supervised probe (SPEC follow-up, 2026-11-03).

Reads probe prediction files and train logs written by probe_train.py (sum = plain_s<seed>,
others = plain_s<seed>_pool<mode>). Metrics on S_dev and S_sup_val, 95% percentile CI over
polytopes (B = 1000, from analyze.py). Writes runs/probe_20261103/pooling_ablation.md.

Example:
  python eval/sweep/probe_20261103/pooling_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze as an  # noqa: E402

RUN = an.MAIN_ROOT / "runs/probe_20261103"
MODES = [("sum", ""), ("mean", "_poolmean"), ("max", "_poolmax"), ("summax", "_poolsummax")]


def metrics(rows):
    groups = an.make_groups(rows, None)
    out = {}
    for key, fn in (
        ("recall1", lambda d: an.nanmean(d["a_r1"])),
        ("recall4", lambda d: an.nanmean(d["a_r4"])),
        ("spearman_actor", lambda d: an.nanmean(d["a_rho"])),
        ("spearman_value", lambda d: an.spearman(*an.mask_pair(d, "V", "y1"))),
    ):
        out[key] = an.bootstrap(groups, fn, seed=100 + len(out))
    n_states = sum(len(g["V"]) for g in groups)
    return out, n_states, len(groups)


def main():
    lines = ["| pooling (face / graph) | seed | split | n_states | n_polytopes | recall@1 | recall@4 | Spearman(actor, dlnV) | Spearman(V, y1) | best epoch | best val loss |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, suffix in MODES:
        for seed in (0, 1):
            variant = f"plain_s{seed}{suffix}"
            probe = an.load_probe(RUN, variant)
            log_path = RUN / "probe" / f"{variant}_train_log.json"
            log = json.loads(log_path.read_text()) if log_path.exists() else {}
            for split, label in (("dev", "dev"), ("sup_val", "S_sup_val")):
                rows = probe[split]
                if not rows:
                    continue
                m, n_states, n_poly = metrics(rows)
                cell = lambda k: an.cell(m[k])
                lines.append(f"| {mode} | {seed} | {label} | {n_states} | {n_poly} | {cell('recall1')} | "
                             f"{cell('recall4')} | {cell('spearman_actor')} | {cell('spearman_value')} | "
                             f"{log.get('best_epoch', 'n/a')} | {log.get('best_val_loss', float('nan')):.4f} |")
    text = "\n".join(lines) + "\n"
    (RUN / "pooling_ablation.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
