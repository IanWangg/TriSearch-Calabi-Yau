# C1: model-guided single-face mutation BeFS, h11 = 50 (2026-11-04)

Question: can the RL-trained invariant actor (runs/inv_rl_full_20261104/checkpoints/final.pth), used zero-shot,
pick better single-face jumps than uniform sampling?

- `face_mutation_befs_guided`: each expansion draws 64 unseen single-face mutations, scores them with
  `eval/algorithm/face_move_scorer.py` (actor head on removed / added triangles of the face, "mean2" aggregation;
  no Kcup query, no feasibility check), evaluates the top 16 as in `face_mutation_befs`.
- `face_mutation_befs_random64`: same bookkeeping, keeps 16 of the 64 at random (control).

Same setups as B1 (`runs/face_mutation_20261104/setups/seed0_poly{0..9}_s3`), budget 1000 Kcup queries, seed 0,
10 workers. `run_grid.py --budget 1000 --seeds 0 --workers 10`, then
`analyze.py --budget 1000 --bootstrap 2000 --exclude-paired 7 9`.
One shard failed: guided polytope 7, start 1, geometry worker timeout at query 992. Paired comparisons exclude
polytopes 7 and 9 (24 units).

## Results (paired, 24 units, bootstrap over polytopes)

| comparison | mean diff lnV [95% CI] | win rate |
|---|---|---|
| guided - random64 | -1.51 [-3.55, +0.53] | 0.29 |
| guided - face_mutation_befs (B1) | -0.79 [-2.02, +0.65] | 0.38 |
| random64 - face_mutation_befs (B1) | +0.72 [-0.43, +1.89] | 0.62 |
| guided - cyopt_ga | +1.58 [+0.06, +3.73] | 0.58 |
| random64 - cyopt_ga | +3.09 [+1.84, +4.44] | 0.92 |

Evaluated-children feasibility: guided 46%, random64 62%. Spearman(score, dlnV) on evaluated feasible children:
guided +0.05. Scorer cost about 31 ms per expansion.

Reading: the zero-shot actor does not rank jumps; it prefers infeasible moves. This matches the C2 cross-validation
(eval/sweep/jump_rank_cv_20261104): zero-shot Spearman -0.06, feasibility AUC 0.33.
