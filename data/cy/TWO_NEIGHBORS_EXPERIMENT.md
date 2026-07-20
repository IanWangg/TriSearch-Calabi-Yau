# Two-Neighbor CY Volume Experiment

## Setup

- Dataset: `data/cy/two_neighbors_h11_12.samples.jsonl`
- Split: 8 training and 4 held-out h11=12 polytopes
- Held-out polytope indices: 28, 22, 23, 26
- Seed: 0
- Training: PPO, 32 environments, rollout length 10
- Navigation: `two_neighbors`
- Objective: `max_cy_volume` from the stretched Kcup cone tip
- Evaluation: deterministic, 50 steps, identical initial FRSTs

The run was initially configured for 50 PPO iterations and was stopped at the
user-requested limit after iteration 25 completed. The 25 completed iterations
took 4,136.02 seconds (68.93 minutes), excluding setup. Iteration 25 had mean
initial, final, and best volumes of 408.98, 2,053.54, and 3,136.24. Its mean
best-volume improvement was 2,727.26 and all 32 trajectories improved.

The forced stop occurred before the next automatic checkpoint boundary, so the
latest recoverable trained checkpoint is iteration 20. Results below compare
that checkpoint against the seed-matched untrained checkpoint.

## Held-Out Results

| Checkpoint | Initial mean | Final mean | Best mean | Improved fraction | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained seed-0 baseline | 975.72 | 1,135.97 | 1,227.49 | 0.75 | 12.58 s |
| Trained, iteration 20 | 975.72 | 975.72 | 1,115.17 | 0.75 | 11.00 s |

The short trained checkpoint did not outperform the baseline: its held-out best
volume mean was 112.32 lower and its mean best-volume improvement was 139.45,
versus 251.77 for the baseline.

Evaluation JSON files and checkpoints are under
`ckpt/runs/two_neighbors_h11_12_seed0/`.
