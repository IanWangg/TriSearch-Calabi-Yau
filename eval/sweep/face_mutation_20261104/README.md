# B1: single-face jumps vs GA vs flip search, h11 = 50 (2026-11-04)

Question: GA (cyopt) searches the space "one FRT per 2-face" and beats flip search at h11 = 50.
Is the advantage from single-face jumps alone, or from population + crossover?

## Algorithms (all share CyoptEncoding: same per-face FRT codebook, same starts, same query accounting)

- `cyopt_ga`: existing GA (population 50, mutation 0.1, elitism 1, n-point crossover, mutation_k=1).
- `cyopt_ga_nocross`: same GA with an identity crossover (selection, mutation, elitism unchanged).
- `face_mutation_hc`: hill climbing. From the current best, draw M = 16 distinct single-face mutations
  (face uniform among faces with codebook size > 1, entry uniform among the other entries), evaluate the
  feasible ones, move to the best child if it improves.
- `face_mutation_befs`: best-first over DNA; each expansion draws M = 16 single-face mutations.
- `best_first`: flip best-first with all neighbors (plain; the random-k variant is not in this worktree).

Non-solid-cone (non-NTFE) children are free rejects for all codebook algorithms, as in GA.
Codebook: faces <= 12 points fully enumerated, larger faces 1000 sampled FRTs (same as the earlier h11 = 50 GA configs).

## Commands

```bash
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic PYTHONPATH=$W
python make_setups.py --seed 0 --num-polytopes 10 --num-starts 3 --shards
nohup python -u run_grid.py --budget 1000 --seeds 0 --workers 14 --tag grid_b1000 > $R/runs/face_mutation_20261104/logs/grid_b1000.out &
python analyze.py --budget 1000 --bootstrap 2000 --out $R/runs/face_mutation_20261104/analysis_b1000
```

Data: 10 h11 = 50 favorable polytopes (HF revision in common.py) x 3 FRST starts, seed 0, budget 1000 Kcup queries per run.
Wall time about 13 h on 14 CPU workers. One shard failed: `face_mutation_hc` polytope 9 (Kcup tip solver failed on one
state with all backends, after 182 queries). Paired comparisons below drop polytope 9 for all algorithms (27 units).

## Results (best lnV at budget 1000; unit = polytope x start)

| algorithm | units | mean best lnV [95% CI] | lnV at 100 / 250 / 500 queries |
|---|---|---|---|
| face_mutation_befs | 30 | 30.31 [29.51, 31.11] | 24.7 / 27.5 / 29.4 |
| face_mutation_hc | 27 | 30.21 [29.00, 31.51] | 24.7 / 26.9 / 28.8 |
| cyopt_ga | 30 | 28.01 [27.24, 28.76] | 23.9 / 25.8 / 26.9 |
| cyopt_ga_nocross | 30 | 26.71 [25.68, 27.60] | 22.8 / 23.5 / 25.4 |
| best_first (flip) | 30 | 24.16 [23.28, 25.10] | 17.0 / 18.9 / 21.4 |

Paired differences (27 common units, bootstrap over polytopes):

| comparison | mean diff lnV [95% CI] | win rate |
|---|---|---|
| face_mutation_befs - cyopt_ga | +2.38 [1.60, 3.16] | 0.85 |
| face_mutation_hc - cyopt_ga | +2.53 [1.43, 3.72] | 0.81 |
| cyopt_ga - cyopt_ga_nocross | +1.25 [0.38, 2.16] | 0.78 |
| face_mutation_befs - best_first | +6.37 [5.08, 7.78] | 1.00 |
| cyopt_ga - best_first | +3.99 [2.55, 5.46] | 0.96 |
| face_mutation_befs - face_mutation_hc | -0.14 [-1.14, 0.71] | 0.48 |

Single-face mutation children: feasible 55% (hc) / 63% (befs). Size of a jump (changed triangles in the mutated face / 2,
a lower bound on flips): median 8 (hc) / 6 (befs), p90 47 / 37. Child dlnV: median -4.0 (hc) / -2.8 (befs); most jumps are worse
than the parent, so candidate ranking is the obvious place for a learned model.

Caveats: one seed; 9-10 polytopes; flip baseline is plain best_first; large-face codebook is a 1000-FRT sample.
