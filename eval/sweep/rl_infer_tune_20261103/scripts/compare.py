"""Paired mean log(best Kcup) ratio of each tune run vs plain BeFS (same budget)."""
import glob, json, math, os, sys
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(T), "fv1_dev_eval")
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
rng = np.random.default_rng(0)


def best(path):
    return {(r["polytope_index"], r["start_index"]): r["best_objective"]
            for r in map(json.loads, open(os.path.join(path, "rollouts.jsonl")))}


def stat(x, y):
    keys = sorted(set(x) & set(y))
    d = np.array([math.log(x[k]) - math.log(y[k]) for k in keys])
    boots = rng.choice(d, (2000, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, int((d > 1e-9).sum()), int((d < -1e-9).sum()), len(d)


rows = {}
for d in sorted(glob.glob(os.path.join(T, prefix + "*_h11_*_b*"))):
    if not os.path.exists(os.path.join(d, "summary.json")):
        continue
    tag = os.path.basename(d)
    name, rest = tag.split("_h11_")
    h, b = rest.split("_b")
    rows.setdefault((h, b), []).append((name, d))
for (h, b), items in sorted(rows.items()):
    base = best(os.path.join(D, f"befs_h11_{h}_b{b}"))
    print(f"\n== h11={h} budget={b}  (vs plain BeFS b{b})")
    print(f"{'variant':28s} {'mean':>7s} {'95% CI':>17s} win loss")
    for name, d in sorted(items, key=lambda it: -stat(best(it[1]), base)[0]):
        m, lo, hi, w, l, n = stat(best(d), base)
        print(f"{name:28s} {m:+.3f}  [{lo:+.3f},{hi:+.3f}] {w:3d} {l:3d}")
