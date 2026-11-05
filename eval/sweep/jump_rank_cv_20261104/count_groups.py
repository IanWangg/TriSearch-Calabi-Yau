"""Count (polytope, start, parent) groups in the B1 single-face candidate logs."""
import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
RESULTS = R / "runs/face_mutation_20261104/results"

groups = defaultdict(dict)  # key -> child_dna -> row
rows_total = Counter()
for path in sorted(RESULTS.glob("face_mutation_*_budget1000_seed0_poly*/face_mutation_candidates.jsonl")):
    algo = path.parent.name.split("_budget")[0]
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows_total[algo] += 1
            key = (row["polytope_index"], row["start_hash"], tuple(row["parent_dna"]))
            groups[key].setdefault(tuple(row["child_dna"]), row)
print("rows per algorithm", dict(rows_total))
print("raw groups", len(groups))
kept = Counter()
children_total = Counter()
feasible_total = Counter()
for key, children in groups.items():
    feas = sum(1 for r in children.values() if r["feasible"])
    if feas >= 4:
        kept[key[0]] += 1
        children_total[key[0]] += len(children)
        feasible_total[key[0]] += feas
print("kept groups (>=4 feasible) per polytope", dict(sorted(kept.items())))
print("children per polytope (deduped)", dict(sorted(children_total.items())))
print("feasible per polytope", dict(sorted(feasible_total.items())))
print("total kept", sum(kept.values()))
