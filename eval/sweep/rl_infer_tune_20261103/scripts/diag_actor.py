"""Actor/critic ranking quality from a full-expansion search log (k=-1).

For each expansion where every unseen neighbor was queried, compare the
actor ranking with the true child log-volume gain.
"""
import json, sys, math
from collections import defaultdict
import numpy as np

path = sys.argv[1]
props, scores = [], {}
for line in open(path):
    r = json.loads(line)
    if r["event"] == "proposal":
        props.append(r)
    else:
        scores[(r["polytope_index"], r["start_index"], r["parent"], r["state"])] = r
parent_obj = {}
for (p, s, par, st), r in scores.items():
    parent_obj[(p, s, st)] = r["objective"]

K = [1, 2, 4, 8]
recall = defaultdict(list)       # best child in actor top-k
rand_recall = defaultdict(list)  # expected for random top-k
spearman, crit_corr, n_children = [], [], []
pos_gain_recall = defaultdict(list)
for r in props:
    key = (r["polytope_index"], r["start_index"])
    cands = [c for c in r["candidates"] if not c[1] and c[2]]
    rows = []
    for st, seen, sel, lp in cands:
        sc = scores.get((*key, r["parent"], st))
        if sc is not None:
            rows.append((lp, math.log(sc["objective"]), sc["critic"]))
    if len(rows) < 3:
        continue
    lp = np.array([x[0] for x in rows]); lv = np.array([x[1] for x in rows]); cr = np.array([x[2] for x in rows])
    n = len(rows); n_children.append(n)
    order = np.argsort(-lp, kind="stable")
    best = int(np.argmax(lv))
    rank = int(np.where(order == best)[0][0])
    for k in K:
        recall[k].append(rank < k)
        rand_recall[k].append(min(k, n) / n)
    ra = np.argsort(np.argsort(lp)); rb = np.argsort(np.argsort(lv))
    spearman.append(np.corrcoef(ra, rb)[0, 1])
    if cr.std() > 0:
        crit_corr.append(np.corrcoef(cr, lv)[0, 1])
print(f"expansions={len(n_children)}  mean children={np.mean(n_children):.1f}")
for k in K:
    print(f"recall@{k}: actor={np.mean(recall[k]):.3f}  random={np.mean(rand_recall[k]):.3f}")
print(f"Spearman(log pi, log V child): mean={np.nanmean(spearman):+.3f}")
print(f"corr(critic, log V child) within parent: mean={np.nanmean(crit_corr):+.3f}")
