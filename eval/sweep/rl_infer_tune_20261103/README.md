# RL BeFS inference tuning on favorable v1 dev (2026-11-03)

Checkpoint: `runs/fv1_R1_mix_s0_20261102_2207/checkpoints/200.pth` (two_face_deep_sets,
train h11=15+20, gamma 0.95). Dev: 30 starts per h11 (`runs/fv1_dev_eval/setup_dev_h11_{15,20}`).
Objective `max_kcup`; budget is logical Kcup queries. All numbers are the paired mean of
`log(best_variant) - log(best_plain_BeFS)` at the same budget, with a 95% bootstrap CI over starts.

## Tools

- `RLTunedBestFirst` (`eval/algorithm/rl.py`) keeps the `rl_value_best_first` name and adds:
  `proposal_mode` (`actor` top-k, `random` k unseen neighbors, `canonical`), `logp_coef`
  (score = ln V + value_discount * critic + logp_coef * log pi), `widen_penalty` (progressive
  widening: an expanded parent with unproposed neighbors returns to the frontier with
  score - penalty) and `proposal_seed`.
- `tools/tune_rl_search.py` runs one variant on a saved setup. A shared SQLite Kcup cache
  (keyed by full FRST state key) saves physical work only; every lookup still counts as one
  logical query. The default variant (k=4, c=0.9) reproduces the earlier R1 it200 dev results
  exactly (30/30 starts).
- `scripts/`: `run_variant.sh` (driver), `compare.py` (vs plain BeFS), `ab.py` (seed-averaged
  A/B), `diag_actor.py` (actor recall from a full-expansion log), `sweep1.txt` (variants).

## Actor diagnostic (plain BeFS order, all neighbors queried, budget 1000)

Fraction of expansions where the best child (by Kcup) is in the actor top k:

| h11 | recall@1 | recall@2 | recall@4 | recall@8 | random@4 |
|---|---|---|---|---|---|
| 15 (3985 expansions, 7.5 unseen children) | 0.25 | 0.43 | 0.72 | 0.98 | 0.57 |
| 20 (2334 expansions, 12.9 unseen children) | 0.18 | 0.29 | 0.49 | 0.76 | 0.32 |

Spearman(log pi, child ln V) is about +0.2 to +0.24. Within one parent, the critic is
negatively correlated with child ln V (about -0.12 to -0.15).

## Sweep 1 (budget 256)

Full table: `sweep_b256.txt`. Main rows:

| variant | h11=15 | h11=20 |
|---|---|---|
| actor top 4, c=0.9 (previous default) | +0.04 | +0.27 |
| actor top 8, c=0.9 | +0.03 | +0.39 [+0.09,+0.72] |
| actor top 2, c=0.9 | +0.12 | +0.02 |
| actor top 1, c=0.9 | -0.77 | -1.00 |
| all neighbors, c=0.9 | +0.02 | -0.01 |
| actor top 4, c in {0, 0.3, 2} | +0.01 to +0.06 | +0.24 to +0.32 |
| actor top 4, + logp bonus 0.5 / 1 | -0.13 / -0.28 | -0.16 / -0.14 |
| actor top 4, widening 0.5 / 2 | +0.02 / +0.04 | +0.30 / +0.27 |

The critic weight, a log pi bonus and progressive widening do not help.

## A/B: plain BeFS vs BeFS + random 4 neighbors (no network)

`random` mode with value_discount 0 ranks only by ln V and ignores network outputs.
Per start, the log ratio is averaged over proposal seeds first (`ab_test.txt`).

| budget | h11=15 | h11=20 |
|---|---|---|
| 256 (8 seeds) | +0.07 [-0.02,+0.18] | +0.27 [+0.08,+0.48] |
| 1000 (4 seeds) | +0.05 [-0.02,+0.13] | -0.03 [-0.21,+0.15] |

Seed-to-seed spread of the mean is about +-0.1, as large as most differences between variants.

Median depth at budget 256 (max / best found):

| | h11=15 | h11=20 |
|---|---|---|
| plain BeFS | 14 / 10 | 14 / 12 |
| all neighbors, c=0.9 | 13 / 9 | 13 / 12 |
| actor top 4 | 29 / 17 | 36 / 28 |
| random 4 | 26 / 18 | 36 / 26 |

## Conclusions

- At budget 256, the gain of this checkpoint over plain BeFS is reproduced by random k=4
  filtering. The actor ranking adds no measurable gain. The gain comes with about 2x deeper
  search; k=1 is deeper still but much worse.
- At budget 1000 the narrowing gain disappears.
- Report RL against a "BeFS + random k" baseline at budgets 256 and 1000, with several seeds.
- 30 dev starts per h11 are too few to tune many variants; use a larger held-out tune set.
