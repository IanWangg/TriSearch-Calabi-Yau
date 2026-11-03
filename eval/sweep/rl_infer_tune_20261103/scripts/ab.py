"""A/B: plain BeFS vs BeFS + random 4 neighbors (no network), several proposal seeds.

Per start: d = mean over seeds of log(best_B) - log(best_A). Bootstrap over starts.
Also reports per-seed means (seed spread) and the actor top-4 run for reference.
"""
import glob, json, math, os
import numpy as np

T = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.path.dirname(T), "fv1_dev_eval")
rng = np.random.default_rng(0)
best = lambda p: {r["polytope_index"]: r["best_objective"] for r in map(json.loads, open(p + "/rollouts.jsonl"))}
for b, pat in ((256, "R1it200_rand4_c0{s}_h11_{h}_b256"), (1000, "AB_rand4_ps*_h11_{h}_b1000")):
    for h in (15, 20):
        A = best(f"{D}/befs_h11_{h}_b{b}")
        if b == 256:
            dirs = [f"{T}/R1it200_rand4_c0_h11_{h}_b256"] + sorted(glob.glob(f"{T}/R1it200_rand4_c0_ps*_h11_{h}_b256"))
        else:
            dirs = sorted(glob.glob(f"{T}/AB_rand4_ps*_h11_{h}_b1000"))
        dirs = [d for d in dirs if os.path.exists(d + "/summary.json")]
        if not dirs:
            continue
        keys = sorted(A)
        M = np.array([[math.log(best(d)[k]) - math.log(A[k]) for k in keys] for d in dirs])  # seeds x starts
        per_start = M.mean(0)
        boots = rng.choice(per_start, (5000, len(per_start))).mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        win = (per_start > 1e-9).sum(); loss = (per_start < -1e-9).sum()
        print(f"b={b} h11={h} seeds={len(dirs)}  mean={per_start.mean():+.3f} 95%CI[{lo:+.3f},{hi:+.3f}]  "
              f"win/loss(starts)={win}/{loss}  per-seed means={np.round(M.mean(1), 3).tolist()}")
