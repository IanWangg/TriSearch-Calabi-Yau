# Value generalization and supervised upper-bound probe (2026-11-03)

Spec: `eval/sweep/probe_20261103/SPEC.md`. Data: favorable v1 train (S_sup, S_crit_train) and dev (S_dev).
Checkpoints: G = `runs/fv1_mix_g099_s0_20261103_0604/checkpoints/200.pth`, R = `runs/fv1_R1_mix_s0_20261102_2207/checkpoints/200.pth`,
G50 = `runs/fv1_mix_g099_s0_20261103_0604/checkpoints/50.pth`. The three RL checkpoints are compared on the same states.
All data and outputs are under `runs/probe_20261103/` (not committed). The worktree has no `data/` or `runs/`, so the scripts read
from the main checkout at `/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/` by default.

## Commands

```bash
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
PY=/home/yiranwang/anaconda3/envs/sage/bin/python
OUT=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/probe_20261103

# Smoke tests (2 dev polytopes x 2 states, 2 workers; sup/crit smoke with --sets sup,crit --limit_per_set 1)
$PY tools/probe_collect.py --stage A --sets dev --limit_per_set 1 --states 2 --workers 2 --sanity_n 4 --out_dir $OUT/smoke
$PY tools/probe_collect.py --stage B --sets dev --limit_per_set 1 --states 2 --workers 2 --y2_frac 0.5 --out_dir $OUT/smoke

# Stage A: S_sup (300 + 150 train polytopes, 12 states each), S_crit_train (60 + 40, 8 each), S_dev (30 + 30, 16 each)
$PY tools/probe_collect.py --stage A --workers 16 --out_dir $OUT
# Stage B: y2 (25% of crit and dev states) and MC (M=4, 20 steps, checkpoint G)
$PY tools/probe_collect.py --stage B --workers 16 --out_dir $OUT

# Supervised probe (fresh TwoFaceAgent; GPU picked by torch.cuda.mem_get_info, never GPU 0 or 3)
for v in "plain 0" "plain 1" "ctx 0" "ctx 1"; do set -- $v
  $PY eval/sweep/probe_20261103/probe_train.py --variant $1 --seed $2 > $OUT/probe/$1_s$2.log 2>&1; done

# Analysis (about 9 minutes; bootstrap B = 1000 over polytopes)
$PY eval/sweep/probe_20261103/analyze.py
```

Stage A and stage B were run with `nohup` and logs in `runs/probe_20261103/stageA.log` and `stageB.log`.

## Counts and wall time

| step | polytopes | states | logical Kcup queries | Kcup cache hits | expansions | wall |
|---|---|---|---|---|---|---|
| stage A (S_sup, S_crit_train, S_dev) | 450 + 100 + 60 | 5400 + 800 + 960 | 105,168 | 1,803 (1.7%) | 114,178 | 83.7 min |
| stage B (y2, MC) | 100 + 60 | 800 + 960 | 252,755 | 49,933 (19.8%) | 174,138 | 140.2 min |
| probe training (4 runs) | 450 (S_sup) | 4860 train, 540 val | | | | 54 to 86 s each on GPU |
| analysis | | | | | | about 9 min |

Stage A plus stage B took 3.7 hours, under the 6-hour budget, so the spec sizes were kept (no reduction to 200+100 or M=2).
Sizes: 16 geometry workers (the pool; plus the main process). S_dev has 960 states over all 60 dev polytopes; y2 is on 240 dev
states; MC is on all 960 dev and 800 train-polytope states (M=4 each, 0 stuck rollouts).
The cache counts are logical lookups: hits + misses = logical queries. Every lookup still counts as one query.

## Sanity checks

| check | result |
|---|---|
| (a) lnV(s) from the probe equals an uncached geometry recomputation and the cache, 20 random dev states | max abs diff 0.0 (fresh) and 0.0 (cache). Pass |
| (b) V(s) and log pi from the batched probe path equal single-state `eval/policy.py` calls, 20 dev states (the spec asked for 10; the cap is per group, so all 20 ran) | max abs diff V 7.2e-07, log pi 6.3e-07 (< 1e-4). Pass |
| (c) G, train-polytope states: EV(V(s), mean MC boot) vs training-log EV at iteration 200 | 0.179 [0.070, 0.270] vs 0.238 (`ppo.explained_variance`), |diff| = 0.059 <= 0.15. Pass |

## Probe training (Q3)

| run | device | epochs run | best epoch | train loss first -> last | best val loss | wall |
|---|---|---|---|---|---|---|
| plain s0 | cuda:1 | 17 | 8 | 2.682 -> 2.437 | 2.425 | 63 s |
| plain s1 | cuda:1 | 15 | 6 | 2.673 -> 2.441 | 2.421 | 54 s |
| ctx s0 | cuda:1 | 22 | 13 | 2.664 -> 2.404 | 2.417 | 86 s |
| ctx s1 | cuda:1 | 20 | 11 | 2.670 -> 2.404 | 2.419 | 71 s |

Early stop on S_sup_val (patience 8, max 60 epochs). Adam lr 3e-4, batch 64 states, tau 0.1, value coefficient 0.5.
All four runs were completed (2 seeds for each variant), so the time-saving fallback was not needed.

## Deviations and details

1. Scale: spec sizes as written. No reduction was needed; A + B = 3.7 h.
2. Starts: each `frst_list` holds one FRST start per polytope, so each state's walk starts from that start (index 0 is the only choice; the seeded draw is kept in the code).
3. Walk: L ~ Uniform{0..30} uniform-random flips; each walk has its own seeded RNG, so stage B replays the stage A states exactly. Replayed keys are checked against stage A (the run would have failed on a mismatch).
4. Q1 critic generalization uses MC boot, which exists only for G (MC was specified under G only). R and G50 have no Q1 rows.
5. y2 = max lnV over the children and grandchildren of s (excluding s itself) minus lnV(s). y1(child) = max lnV over that child's children (excluding s) minus lnV(child). Only the 25% seeded subset has these labels.
6. MC: M=4 rollouts of 20 steps under G, sampled from the policy softmax. boot = G20 + 0.99^20 V_G(s_20); a rollout that gets stuck would use boot = G20 (V = 0), and none did. Within-polytope Spearman needs >= 5 y2 states per polytope, so the y2 rows use few polytopes (n_polytopes is shown in each cell; S_crit y2 has 2 polytopes).
7. Probe scope: the probe predicts only S_sup (train and val) and S_dev. It has no children's V, so the child-level analysis (Q2 child table) is not computed for it.
8. Probe ctx variant: lnV(s) is standardized (mean and std on S_sup train), h11/20 is literal, and both are concatenated to the global features and projected back to width 64 (`Linear(66 -> 64)`) before the actor and value MLPs. The ctx variant therefore takes lnV(s) as an input, which the plain variant does not.
9. Probe GPU: cuda:1 was picked by `torch.cuda.mem_get_info` among the non-excluded devices. CPU was used for collection, and the geometry pool used 16 workers.
10. Commit trailer: the task asked for the `Co-Authored-By: Claude Opus 5.5` line; that is used here (it is the user's explicit instruction; the harness reminder named a different model).
11. Analysis sanity line for (c) was fixed mid-run (the training-log key is nested under `ppo`). Rerun output is the one below.

## Output files

- `runs/probe_20261103/stageA/records_{sup,crit,dev}.jsonl` and `shards/*.pt` (TwoFaceData with actions), `stageB/{crit,dev}.jsonl`
- `runs/probe_20261103/probe/{plain,ctx}_s{0,1}_{train_log.json,pred.jsonl,.pth}` and `.log`
- `runs/probe_20261103/sanity_a_b.json`, `stageA_summary.json`, `stageB_summary.json`, `analysis.json`, `analysis_tables.md`

Tables (verbatim copy of `runs/probe_20261103/analysis_tables.md`). Every cell is point estimate and 95% percentile CI over polytopes (B = 1000);
n_states and n_polytopes are in each row. "train" = S_crit_train for RL checkpoints and S_sup train for probes.

## Counts

| stage | set | polytopes | states | y2 states / note |
|---|---|---|---|---|
| stage A | sup | 450 | 5400 | - |
| stage A | crit | 100 | 800 | - |
| stage A | dev | 60 | 960 | - |
| stage A totals | - | - | - | queries=105168 cache_hits=1803 wall=83.7 min |
| stage B | crit | 100 | 800 | 200 |
| stage B | dev | 60 | 960 | 240 |
| stage B totals | - | - | - | queries=252755 cache_hits=49933 wall=140.2 min |

## Q1 critic generalization (G, V(s) vs mean MC boot)

| states | h11 | n_states | n_polytopes | Pearson(V, boot) | EV(V, boot) |
|---|---|---|---|---|---|
| train polytopes (S_crit_train) | all | 800 | 100 | 0.502 [0.415, 0.572] | 0.179 [0.070, 0.270] |
| train polytopes (S_crit_train) | 15 | 480 | 60 | 0.280 [0.149, 0.402] | 0.050 [-0.050, 0.151] |
| train polytopes (S_crit_train) | 20 | 320 | 40 | 0.481 [0.368, 0.567] | 0.121 [-0.062, 0.265] |
| dev (S_dev) | all | 960 | 60 | 0.487 [0.363, 0.567] | 0.199 [0.077, 0.303] |
| dev (S_dev) | 15 | 480 | 30 | 0.284 [0.159, 0.402] | 0.062 [-0.010, 0.155] |
| dev (S_dev) | 20 | 480 | 30 | 0.539 [0.405, 0.632] | 0.236 [0.080, 0.359] |

Gap = train - dev:

| metric | h11 | train - dev [95% CI] |
|---|---|---|
| Pearson | all | 0.015 [-0.105, 0.157] |
| EV | all | -0.020 [-0.174, 0.130] |
| Pearson | 15 | -0.004 [-0.183, 0.184] |
| EV | 15 | -0.012 [-0.149, 0.115] |
| Pearson | 20 | -0.058 [-0.201, 0.094] |
| EV | 20 | -0.116 [-0.346, 0.096] |

Training-log EV of G at iteration 200 (ppo.explained_variance): 0.23827427625656128

Sanity (c): train-state EV 0.179 vs training-log EV 0.238; |diff| = 0.059 (pass).

## Q2 neighborhood: V(s) vs label, against B0 = -lnV(s)

| checkpoint | set | h11 | label | n_states | Spearman V | Spearman B0 | partial Spearman V (ctrl lnV) | within-polytope Spearman V |
|---|---|---|---|---|---|---|---|---|
| G (g=0.99, it200) | crit | all | y1 | 800 | 0.202 [0.094, 0.293] | 0.174 [0.077, 0.268] | 0.213 [0.101, 0.309] | 0.082 [0.005, 0.157] (n_polytopes=100) |
| G (g=0.99, it200) | crit | 15 | y1 | 480 | 0.124 [0.005, 0.242] | 0.323 [0.204, 0.430] | 0.063 [-0.070, 0.187] | 0.055 [-0.044, 0.150] (n_polytopes=60) |
| G (g=0.99, it200) | crit | 20 | y1 | 320 | 0.103 [-0.053, 0.253] | 0.280 [0.129, 0.417] | 0.002 [-0.155, 0.142] | 0.122 [-0.001, 0.244] (n_polytopes=40) |
| G (g=0.99, it200) | crit | all | y2 | 200 | 0.227 [0.066, 0.372] | 0.220 [0.050, 0.375] | 0.246 [0.088, 0.399] | 0.450 [0.300, 0.600] (n_polytopes=2) |
| G (g=0.99, it200) | crit | 15 | y2 | 117 | 0.084 [-0.125, 0.289] | 0.519 [0.353, 0.648] | 0.025 [-0.176, 0.211] | 0.600 [0.600, 0.600] (n_polytopes=1) |
| G (g=0.99, it200) | crit | 20 | y2 | 83 | 0.194 [-0.072, 0.460] | 0.177 [-0.121, 0.416] | 0.131 [-0.191, 0.419] | 0.300 [0.300, 0.300] (n_polytopes=1) |
| G (g=0.99, it200) | crit | all | mc_best | 800 | 0.306 [0.225, 0.388] | 0.318 [0.225, 0.416] | 0.338 [0.238, 0.433] | 0.262 [0.184, 0.343] (n_polytopes=100) |
| G (g=0.99, it200) | crit | 15 | mc_best | 480 | 0.192 [0.079, 0.303] | 0.485 [0.376, 0.590] | 0.109 [-0.033, 0.248] | 0.197 [0.097, 0.302] (n_polytopes=60) |
| G (g=0.99, it200) | crit | 20 | mc_best | 320 | 0.233 [0.109, 0.360] | 0.570 [0.445, 0.683] | 0.033 [-0.139, 0.189] | 0.360 [0.237, 0.477] (n_polytopes=40) |
| G (g=0.99, it200) | dev | all | y1 | 960 | 0.090 [-0.032, 0.207] | 0.060 [-0.042, 0.173] | 0.083 [-0.044, 0.197] | 0.167 [0.090, 0.244] (n_polytopes=60) |
| G (g=0.99, it200) | dev | 15 | y1 | 480 | 0.057 [-0.068, 0.198] | 0.208 [0.078, 0.353] | 0.043 [-0.091, 0.180] | 0.143 [0.032, 0.248] (n_polytopes=30) |
| G (g=0.99, it200) | dev | 20 | y1 | 480 | 0.048 [-0.093, 0.192] | 0.102 [-0.073, 0.287] | -0.004 [-0.157, 0.144] | 0.192 [0.078, 0.306] (n_polytopes=30) |
| G (g=0.99, it200) | dev | all | y2 | 240 | 0.141 [-0.040, 0.300] | 0.019 [-0.150, 0.200] | 0.141 [-0.027, 0.305] | 0.177 [0.009, 0.336] (n_polytopes=23) |
| G (g=0.99, it200) | dev | 15 | y2 | 120 | 0.000 [-0.196, 0.209] | 0.176 [-0.031, 0.419] | -0.034 [-0.251, 0.184] | 0.161 [-0.148, 0.425] (n_polytopes=12) |
| G (g=0.99, it200) | dev | 20 | y2 | 120 | 0.161 [-0.051, 0.365] | 0.129 [-0.140, 0.383] | 0.108 [-0.115, 0.330] | 0.195 [-0.001, 0.394] (n_polytopes=11) |
| G (g=0.99, it200) | dev | all | mc_best | 960 | 0.224 [0.112, 0.348] | 0.247 [0.119, 0.379] | 0.202 [0.068, 0.333] | 0.322 [0.245, 0.391] (n_polytopes=60) |
| G (g=0.99, it200) | dev | 15 | mc_best | 480 | 0.132 [-0.015, 0.288] | 0.356 [0.170, 0.564] | 0.114 [-0.070, 0.281] | 0.221 [0.119, 0.321] (n_polytopes=30) |
| G (g=0.99, it200) | dev | 20 | mc_best | 480 | 0.242 [0.113, 0.374] | 0.397 [0.217, 0.573] | 0.053 [-0.164, 0.251] | 0.424 [0.334, 0.511] (n_polytopes=30) |
| R (g=0.95, it200) | crit | all | y1 | 800 | 0.162 [0.055, 0.254] | 0.174 [0.077, 0.268] | 0.199 [0.098, 0.287] | 0.092 [0.015, 0.170] (n_polytopes=100) |
| R (g=0.95, it200) | crit | 15 | y1 | 480 | 0.079 [-0.053, 0.215] | 0.323 [0.204, 0.430] | 0.068 [-0.068, 0.193] | 0.069 [-0.030, 0.166] (n_polytopes=60) |
| R (g=0.95, it200) | crit | 20 | y1 | 320 | 0.023 [-0.118, 0.163] | 0.280 [0.129, 0.417] | -0.048 [-0.193, 0.093] | 0.126 [0.002, 0.242] (n_polytopes=40) |
| R (g=0.95, it200) | crit | all | y2 | 200 | 0.189 [0.038, 0.328] | 0.220 [0.050, 0.375] | 0.253 [0.103, 0.390] | 0.600 [0.600, 0.600] (n_polytopes=2) |
| R (g=0.95, it200) | crit | 15 | y2 | 117 | 0.010 [-0.198, 0.217] | 0.519 [0.353, 0.648] | 0.063 [-0.167, 0.253] | 0.600 [0.600, 0.600] (n_polytopes=1) |
| R (g=0.95, it200) | crit | 20 | y2 | 83 | 0.164 [-0.107, 0.418] | 0.177 [-0.121, 0.416] | 0.123 [-0.171, 0.386] | 0.600 [0.600, 0.600] (n_polytopes=1) |
| R (g=0.95, it200) | dev | all | y1 | 960 | 0.110 [-0.008, 0.219] | 0.060 [-0.042, 0.173] | 0.117 [0.004, 0.227] | 0.165 [0.084, 0.236] (n_polytopes=60) |
| R (g=0.95, it200) | dev | 15 | y1 | 480 | 0.059 [-0.070, 0.193] | 0.208 [0.078, 0.353] | 0.087 [-0.050, 0.229] | 0.157 [0.050, 0.259] (n_polytopes=30) |
| R (g=0.95, it200) | dev | 20 | y1 | 480 | 0.054 [-0.077, 0.192] | 0.102 [-0.073, 0.287] | 0.019 [-0.116, 0.175] | 0.173 [0.048, 0.288] (n_polytopes=30) |
| R (g=0.95, it200) | dev | all | y2 | 240 | 0.180 [0.018, 0.321] | 0.019 [-0.150, 0.200] | 0.181 [0.026, 0.341] | 0.079 [-0.127, 0.271] (n_polytopes=23) |
| R (g=0.95, it200) | dev | 15 | y2 | 120 | 0.084 [-0.135, 0.293] | 0.176 [-0.031, 0.419] | 0.084 [-0.147, 0.314] | -0.061 [-0.410, 0.254] (n_polytopes=12) |
| R (g=0.95, it200) | dev | 20 | y2 | 120 | 0.073 [-0.147, 0.306] | 0.129 [-0.140, 0.383] | 0.033 [-0.188, 0.264] | 0.232 [0.040, 0.410] (n_polytopes=11) |
| G50 (g=0.99, it50) | crit | all | y1 | 800 | 0.164 [0.062, 0.258] | 0.174 [0.077, 0.268] | 0.140 [0.029, 0.238] | 0.065 [-0.007, 0.142] (n_polytopes=100) |
| G50 (g=0.99, it50) | crit | 15 | y1 | 480 | 0.162 [0.033, 0.286] | 0.323 [0.204, 0.430] | 0.062 [-0.069, 0.191] | 0.050 [-0.039, 0.144] (n_polytopes=60) |
| G50 (g=0.99, it50) | crit | 20 | y1 | 320 | 0.109 [-0.024, 0.236] | 0.280 [0.129, 0.417] | 0.044 [-0.100, 0.184] | 0.089 [-0.035, 0.208] (n_polytopes=40) |
| G50 (g=0.99, it50) | crit | all | y2 | 200 | 0.206 [0.048, 0.354] | 0.220 [0.050, 0.375] | 0.163 [-0.017, 0.325] | 0.450 [0.400, 0.500] (n_polytopes=2) |
| G50 (g=0.99, it50) | crit | 15 | y2 | 117 | 0.276 [0.059, 0.474] | 0.519 [0.353, 0.648] | 0.058 [-0.206, 0.290] | 0.500 [0.500, 0.500] (n_polytopes=1) |
| G50 (g=0.99, it50) | crit | 20 | y2 | 83 | 0.055 [-0.152, 0.272] | 0.177 [-0.121, 0.416] | 0.030 [-0.197, 0.239] | 0.400 [0.400, 0.400] (n_polytopes=1) |
| G50 (g=0.99, it50) | dev | all | y1 | 960 | 0.089 [-0.024, 0.218] | 0.060 [-0.042, 0.173] | 0.080 [-0.052, 0.206] | 0.179 [0.108, 0.246] (n_polytopes=60) |
| G50 (g=0.99, it50) | dev | 15 | y1 | 480 | 0.084 [-0.076, 0.257] | 0.208 [0.078, 0.353] | -0.009 [-0.199, 0.197] | 0.168 [0.091, 0.250] (n_polytopes=30) |
| G50 (g=0.99, it50) | dev | 20 | y1 | 480 | 0.030 [-0.137, 0.211] | 0.102 [-0.073, 0.287] | 0.010 [-0.199, 0.213] | 0.190 [0.064, 0.304] (n_polytopes=30) |
| G50 (g=0.99, it50) | dev | all | y2 | 240 | 0.116 [-0.085, 0.307] | 0.019 [-0.150, 0.200] | 0.115 [-0.075, 0.293] | 0.108 [-0.044, 0.279] (n_polytopes=23) |
| G50 (g=0.99, it50) | dev | 15 | y2 | 120 | 0.067 [-0.186, 0.314] | 0.176 [-0.031, 0.419] | -0.017 [-0.303, 0.258] | 0.025 [-0.239, 0.299] (n_polytopes=12) |
| G50 (g=0.99, it50) | dev | 20 | y2 | 120 | 0.111 [-0.156, 0.365] | 0.129 [-0.140, 0.383] | 0.073 [-0.238, 0.371] | 0.198 [0.034, 0.376] (n_polytopes=11) |

## Q2 actor (log pi vs dlnV, same states)

| checkpoint | set | h11 | n_states (>=2 children) | recall@1 | recall@4 | random@4 | Spearman(log pi, dlnV) |
|---|---|---|---|---|---|---|---|
| G (g=0.99, it200) | crit | all | 800 | 0.150 [0.125, 0.176] | 0.485 [0.443, 0.522] | 0.317 | 0.257 [0.235, 0.275] |
| G (g=0.99, it200) | crit | 15 | 480 | 0.152 [0.119, 0.192] | 0.523 [0.477, 0.565] | 0.378 | 0.232 [0.206, 0.256] |
| G (g=0.99, it200) | crit | 20 | 320 | 0.147 [0.106, 0.191] | 0.428 [0.372, 0.494] | 0.226 | 0.294 [0.263, 0.325] |
| G (g=0.99, it200) | dev | all | 960 | 0.155 [0.130, 0.183] | 0.483 [0.444, 0.523] | 0.306 | 0.257 [0.230, 0.284] |
| G (g=0.99, it200) | dev | 15 | 480 | 0.158 [0.119, 0.202] | 0.531 [0.471, 0.594] | 0.379 | 0.241 [0.194, 0.287] |
| G (g=0.99, it200) | dev | 20 | 480 | 0.152 [0.119, 0.183] | 0.435 [0.388, 0.487] | 0.232 | 0.273 [0.248, 0.300] |
| R (g=0.95, it200) | crit | all | 800 | 0.149 [0.122, 0.176] | 0.464 [0.419, 0.505] | 0.317 | 0.234 [0.213, 0.253] |
| R (g=0.95, it200) | crit | 15 | 480 | 0.165 [0.133, 0.200] | 0.529 [0.479, 0.577] | 0.378 | 0.218 [0.190, 0.244] |
| R (g=0.95, it200) | crit | 20 | 320 | 0.125 [0.078, 0.172] | 0.366 [0.306, 0.425] | 0.226 | 0.257 [0.231, 0.285] |
| R (g=0.95, it200) | dev | all | 960 | 0.156 [0.128, 0.186] | 0.471 [0.425, 0.516] | 0.306 | 0.253 [0.220, 0.280] |
| R (g=0.95, it200) | dev | 15 | 480 | 0.167 [0.121, 0.215] | 0.506 [0.429, 0.583] | 0.379 | 0.233 [0.182, 0.282] |
| R (g=0.95, it200) | dev | 20 | 480 | 0.146 [0.113, 0.179] | 0.435 [0.377, 0.490] | 0.232 | 0.273 [0.250, 0.298] |
| G50 (g=0.99, it50) | crit | all | 800 | 0.068 [0.050, 0.085] | 0.336 [0.291, 0.384] | 0.317 | 0.036 [0.015, 0.058] |
| G50 (g=0.99, it50) | crit | 15 | 480 | 0.075 [0.054, 0.100] | 0.412 [0.352, 0.471] | 0.378 | 0.050 [0.020, 0.082] |
| G50 (g=0.99, it50) | crit | 20 | 320 | 0.056 [0.028, 0.091] | 0.222 [0.163, 0.284] | 0.226 | 0.016 [-0.011, 0.044] |
| G50 (g=0.99, it50) | dev | all | 960 | 0.083 [0.064, 0.104] | 0.352 [0.307, 0.397] | 0.306 | 0.047 [0.023, 0.068] |
| G50 (g=0.99, it50) | dev | 15 | 480 | 0.094 [0.067, 0.123] | 0.415 [0.348, 0.481] | 0.379 | 0.045 [0.011, 0.079] |
| G50 (g=0.99, it50) | dev | 20 | 480 | 0.073 [0.050, 0.098] | 0.290 [0.242, 0.338] | 0.232 | 0.049 [0.020, 0.077] |

## Q2 child level (dev, 2-hop subset; target = argmax dlnV + y1)

| checkpoint | h11 | n_parents | recall@1 A: dlnV+0.99V(c) | recall@1 B: dlnV | recall@1 C: V(c) | random@1 (1/k) | Spearman A | Spearman B | Spearman C |
|---|---|---|---|---|---|---|---|---|---|
| G (g=0.99, it200) | all | 240 | 0.650 [0.582, 0.709] | 0.658 [0.592, 0.715] | 0.067 [0.038, 0.096] | 0.075 | 0.694 [0.669, 0.718] | 0.715 [0.691, 0.738] | -0.188 [-0.230, -0.149] |
| G (g=0.99, it200) | 15 | 120 | 0.658 [0.564, 0.748] | 0.642 [0.549, 0.726] | 0.058 [0.018, 0.107] | 0.093 | 0.690 [0.649, 0.728] | 0.705 [0.664, 0.740] | -0.198 [-0.265, -0.129] |
| G (g=0.99, it200) | 20 | 120 | 0.642 [0.552, 0.735] | 0.675 [0.593, 0.742] | 0.075 [0.037, 0.116] | 0.056 | 0.698 [0.668, 0.723] | 0.725 [0.698, 0.750] | -0.179 [-0.226, -0.136] |
| R (g=0.95, it200) | all | 240 | 0.633 [0.570, 0.695] | 0.658 [0.592, 0.715] | 0.096 [0.058, 0.138] | 0.075 | 0.707 [0.682, 0.731] | 0.715 [0.691, 0.738] | -0.143 [-0.185, -0.101] |
| R (g=0.95, it200) | 15 | 120 | 0.650 [0.559, 0.740] | 0.642 [0.549, 0.726] | 0.108 [0.057, 0.175] | 0.093 | 0.701 [0.659, 0.741] | 0.705 [0.664, 0.740] | -0.151 [-0.225, -0.075] |
| R (g=0.95, it200) | 20 | 120 | 0.617 [0.530, 0.703] | 0.675 [0.593, 0.742] | 0.083 [0.037, 0.136] | 0.056 | 0.712 [0.681, 0.739] | 0.725 [0.698, 0.750] | -0.136 [-0.182, -0.091] |
| G50 (g=0.99, it50) | all | 240 | 0.658 [0.598, 0.714] | 0.658 [0.592, 0.715] | 0.050 [0.017, 0.087] | 0.075 | 0.715 [0.689, 0.739] | 0.715 [0.691, 0.738] | -0.162 [-0.203, -0.120] |
| G50 (g=0.99, it50) | 15 | 120 | 0.642 [0.550, 0.732] | 0.642 [0.549, 0.726] | 0.075 [0.017, 0.147] | 0.093 | 0.704 [0.663, 0.743] | 0.705 [0.664, 0.740] | -0.163 [-0.234, -0.092] |
| G50 (g=0.99, it50) | 20 | 120 | 0.675 [0.598, 0.748] | 0.675 [0.593, 0.742] | 0.025 [0.000, 0.060] | 0.056 | 0.725 [0.696, 0.750] | 0.725 [0.698, 0.750] | -0.161 [-0.203, -0.119] |

## Q3 supervised probe vs RL checkpoints (value: Spearman and EV vs y1)

| model | split | h11 | n_states | recall@1 | recall@4 | random@4 | Spearman actor | Spearman V vs y1 | EV V vs y1 |
|---|---|---|---|---|---|---|---|---|---|
| G (g=0.99, it200) | train | all | 800 | 0.150 [0.124, 0.180] | 0.485 [0.446, 0.520] | 0.317 | 0.257 [0.237, 0.277] | 0.202 [0.102, 0.293] | -2.023 [-3.035, -1.201] |
| G (g=0.99, it200) | train | 15 | 480 | 0.152 [0.117, 0.188] | 0.523 [0.479, 0.569] | 0.378 | 0.232 [0.207, 0.258] | 0.124 [-0.003, 0.247] | -0.447 [-0.741, -0.180] |
| G (g=0.99, it200) | train | 20 | 320 | 0.147 [0.106, 0.191] | 0.428 [0.362, 0.491] | 0.226 | 0.294 [0.262, 0.326] | 0.103 [-0.046, 0.246] | -3.006 [-4.638, -1.808] |
| G (g=0.99, it200) | dev | all | 960 | 0.155 [0.130, 0.181] | 0.483 [0.446, 0.522] | 0.306 | 0.257 [0.229, 0.286] | 0.090 [-0.021, 0.205] | -1.533 [-2.321, -0.873] |
| G (g=0.99, it200) | dev | 15 | 480 | 0.158 [0.121, 0.196] | 0.531 [0.471, 0.590] | 0.379 | 0.241 [0.192, 0.287] | 0.057 [-0.075, 0.201] | -0.472 [-0.913, -0.133] |
| G (g=0.99, it200) | dev | 20 | 480 | 0.152 [0.123, 0.181] | 0.435 [0.390, 0.485] | 0.232 | 0.273 [0.247, 0.299] | 0.048 [-0.096, 0.182] | -2.471 [-3.912, -1.270] |
| R (g=0.95, it200) | train | all | 800 | 0.149 [0.120, 0.179] | 0.464 [0.424, 0.507] | 0.317 | 0.234 [0.213, 0.253] | 0.162 [0.061, 0.256] | -0.757 [-1.086, -0.448] |
| R (g=0.95, it200) | train | 15 | 480 | 0.165 [0.131, 0.200] | 0.529 [0.477, 0.583] | 0.378 | 0.218 [0.190, 0.245] | 0.079 [-0.066, 0.210] | -0.241 [-0.428, -0.051] |
| R (g=0.95, it200) | train | 20 | 320 | 0.125 [0.081, 0.172] | 0.366 [0.303, 0.422] | 0.226 | 0.257 [0.230, 0.284] | 0.023 [-0.121, 0.153] | -1.085 [-1.667, -0.675] |
| R (g=0.95, it200) | dev | all | 960 | 0.156 [0.129, 0.186] | 0.471 [0.422, 0.514] | 0.306 | 0.253 [0.224, 0.282] | 0.110 [-0.001, 0.215] | -0.770 [-1.154, -0.450] |
| R (g=0.95, it200) | dev | 15 | 480 | 0.167 [0.125, 0.210] | 0.506 [0.438, 0.581] | 0.379 | 0.233 [0.180, 0.284] | 0.059 [-0.079, 0.187] | -0.527 [-1.018, -0.094] |
| R (g=0.95, it200) | dev | 20 | 480 | 0.146 [0.115, 0.181] | 0.435 [0.383, 0.490] | 0.232 | 0.273 [0.248, 0.297] | 0.054 [-0.098, 0.191] | -0.977 [-1.615, -0.492] |
| G50 (g=0.99, it50) | train | all | 800 | 0.068 [0.050, 0.088] | 0.336 [0.289, 0.386] | 0.317 | 0.036 [0.015, 0.058] | 0.164 [0.066, 0.262] | 0.014 [-0.047, 0.074] |
| G50 (g=0.99, it50) | train | 15 | 480 | 0.075 [0.052, 0.100] | 0.412 [0.356, 0.471] | 0.378 | 0.050 [0.023, 0.080] | 0.162 [0.041, 0.277] | 0.016 [-0.058, 0.095] |
| G50 (g=0.99, it50) | train | 20 | 320 | 0.056 [0.028, 0.091] | 0.222 [0.159, 0.288] | 0.226 | 0.016 [-0.012, 0.045] | 0.109 [-0.019, 0.235] | -0.027 [-0.119, 0.042] |
| G50 (g=0.99, it50) | dev | all | 960 | 0.083 [0.064, 0.102] | 0.352 [0.306, 0.400] | 0.306 | 0.047 [0.024, 0.069] | 0.089 [-0.039, 0.210] | -0.001 [-0.050, 0.041] |
| G50 (g=0.99, it50) | dev | 15 | 480 | 0.094 [0.067, 0.127] | 0.415 [0.350, 0.475] | 0.379 | 0.045 [0.011, 0.079] | 0.084 [-0.076, 0.247] | -0.003 [-0.075, 0.065] |
| G50 (g=0.99, it50) | dev | 20 | 480 | 0.073 [0.050, 0.098] | 0.290 [0.237, 0.342] | 0.232 | 0.049 [0.021, 0.076] | 0.030 [-0.141, 0.202] | -0.021 [-0.084, 0.032] |
| probe plain_s0 | train | all | 4860 | 0.198 [0.186, 0.212] | 0.560 [0.541, 0.580] | 0.332 | 0.280 [0.270, 0.291] | 0.213 [0.169, 0.256] | 0.043 [0.030, 0.056] |
| probe plain_s0 | train | 15 | 3240 | 0.215 [0.198, 0.234] | 0.601 [0.579, 0.623] | 0.384 | 0.276 [0.262, 0.288] | 0.151 [0.096, 0.203] | 0.020 [0.009, 0.033] |
| probe plain_s0 | train | 20 | 1620 | 0.164 [0.144, 0.186] | 0.479 [0.449, 0.510] | 0.227 | 0.289 [0.274, 0.306] | 0.142 [0.083, 0.202] | 0.022 [0.006, 0.039] |
| probe plain_s0 | val | all | 540 | 0.202 [0.165, 0.239] | 0.589 [0.526, 0.650] | 0.348 | 0.294 [0.260, 0.333] | 0.230 [0.117, 0.340] | 0.037 [-0.011, 0.077] |
| probe plain_s0 | val | 15 | 360 | 0.214 [0.169, 0.261] | 0.658 [0.586, 0.731] | 0.404 | 0.300 [0.247, 0.356] | 0.181 [0.031, 0.330] | 0.029 [-0.003, 0.061] |
| probe plain_s0 | val | 20 | 180 | 0.178 [0.139, 0.222] | 0.450 [0.372, 0.533] | 0.235 | 0.283 [0.233, 0.332] | -0.065 [-0.192, 0.083] | -0.054 [-0.134, -0.007] |
| probe plain_s0 | dev | all | 960 | 0.141 [0.117, 0.166] | 0.499 [0.460, 0.535] | 0.306 | 0.269 [0.242, 0.297] | 0.191 [0.068, 0.294] | 0.030 [-0.005, 0.066] |
| probe plain_s0 | dev | 15 | 480 | 0.148 [0.112, 0.181] | 0.542 [0.477, 0.602] | 0.379 | 0.255 [0.213, 0.295] | 0.163 [0.028, 0.289] | 0.016 [-0.008, 0.040] |
| probe plain_s0 | dev | 20 | 480 | 0.133 [0.106, 0.163] | 0.456 [0.412, 0.500] | 0.232 | 0.282 [0.245, 0.320] | 0.107 [-0.022, 0.235] | 0.016 [-0.013, 0.051] |
| probe plain_s1 | train | all | 4860 | 0.203 [0.189, 0.218] | 0.559 [0.539, 0.578] | 0.332 | 0.293 [0.283, 0.304] | 0.182 [0.140, 0.222] | 0.027 [0.018, 0.037] |
| probe plain_s1 | train | 15 | 3240 | 0.221 [0.202, 0.239] | 0.607 [0.584, 0.628] | 0.384 | 0.289 [0.275, 0.303] | 0.126 [0.074, 0.177] | 0.013 [0.003, 0.025] |
| probe plain_s1 | train | 20 | 1620 | 0.169 [0.148, 0.190] | 0.462 [0.432, 0.492] | 0.227 | 0.302 [0.286, 0.317] | 0.060 [0.002, 0.120] | 0.003 [-0.006, 0.012] |
| probe plain_s1 | val | all | 540 | 0.204 [0.163, 0.243] | 0.594 [0.535, 0.650] | 0.348 | 0.315 [0.280, 0.353] | 0.165 [0.036, 0.270] | 0.031 [0.001, 0.062] |
| probe plain_s1 | val | 15 | 360 | 0.208 [0.161, 0.261] | 0.650 [0.583, 0.719] | 0.404 | 0.312 [0.261, 0.367] | 0.108 [-0.056, 0.267] | 0.019 [-0.019, 0.053] |
| probe plain_s1 | val | 20 | 180 | 0.194 [0.144, 0.244] | 0.483 [0.406, 0.561] | 0.235 | 0.321 [0.276, 0.364] | -0.064 [-0.201, 0.100] | -0.024 [-0.060, 0.001] |
| probe plain_s1 | dev | all | 960 | 0.153 [0.127, 0.180] | 0.482 [0.443, 0.519] | 0.306 | 0.285 [0.258, 0.314] | 0.148 [0.024, 0.261] | 0.018 [-0.015, 0.049] |
| probe plain_s1 | dev | 15 | 480 | 0.175 [0.135, 0.215] | 0.517 [0.456, 0.575] | 0.379 | 0.273 [0.230, 0.314] | 0.117 [-0.048, 0.272] | 0.009 [-0.022, 0.038] |
| probe plain_s1 | dev | 20 | 480 | 0.131 [0.098, 0.169] | 0.448 [0.404, 0.496] | 0.232 | 0.297 [0.263, 0.332] | 0.060 [-0.096, 0.209] | 0.001 [-0.034, 0.032] |
| probe ctx_s0 | train | all | 4860 | 0.212 [0.198, 0.227] | 0.578 [0.558, 0.599] | 0.332 | 0.298 [0.287, 0.309] | 0.379 [0.345, 0.413] | 0.123 [0.103, 0.143] |
| probe ctx_s0 | train | 15 | 3240 | 0.229 [0.212, 0.248] | 0.623 [0.601, 0.645] | 0.384 | 0.296 [0.283, 0.310] | 0.349 [0.310, 0.393] | 0.101 [0.079, 0.124] |
| probe ctx_s0 | train | 20 | 1620 | 0.178 [0.157, 0.201] | 0.488 [0.457, 0.519] | 0.227 | 0.303 [0.289, 0.318] | 0.358 [0.315, 0.406] | 0.115 [0.086, 0.146] |
| probe ctx_s0 | val | all | 540 | 0.200 [0.161, 0.235] | 0.591 [0.531, 0.646] | 0.348 | 0.314 [0.283, 0.348] | 0.358 [0.232, 0.483] | 0.093 [0.028, 0.165] |
| probe ctx_s0 | val | 15 | 360 | 0.214 [0.169, 0.258] | 0.644 [0.575, 0.714] | 0.404 | 0.319 [0.270, 0.366] | 0.365 [0.195, 0.520] | 0.113 [0.030, 0.193] |
| probe ctx_s0 | val | 20 | 180 | 0.172 [0.122, 0.228] | 0.483 [0.411, 0.561] | 0.235 | 0.304 [0.261, 0.343] | 0.173 [0.032, 0.312] | -0.003 [-0.079, 0.092] |
| probe ctx_s0 | dev | all | 960 | 0.151 [0.125, 0.177] | 0.481 [0.441, 0.520] | 0.306 | 0.280 [0.253, 0.307] | 0.295 [0.218, 0.370] | 0.070 [0.023, 0.115] |
| probe ctx_s0 | dev | 15 | 480 | 0.165 [0.127, 0.202] | 0.521 [0.458, 0.583] | 0.379 | 0.257 [0.216, 0.299] | 0.270 [0.152, 0.385] | 0.063 [0.011, 0.117] |
| probe ctx_s0 | dev | 20 | 480 | 0.138 [0.104, 0.173] | 0.442 [0.394, 0.494] | 0.232 | 0.302 [0.270, 0.335] | 0.246 [0.150, 0.358] | 0.050 [-0.008, 0.111] |
| probe ctx_s1 | train | all | 4860 | 0.221 [0.207, 0.236] | 0.575 [0.557, 0.594] | 0.332 | 0.297 [0.286, 0.308] | 0.373 [0.339, 0.410] | 0.122 [0.100, 0.144] |
| probe ctx_s1 | train | 15 | 3240 | 0.236 [0.217, 0.256] | 0.615 [0.593, 0.638] | 0.384 | 0.292 [0.279, 0.306] | 0.343 [0.301, 0.390] | 0.098 [0.074, 0.124] |
| probe ctx_s1 | train | 20 | 1620 | 0.191 [0.169, 0.214] | 0.495 [0.466, 0.523] | 0.227 | 0.307 [0.293, 0.323] | 0.350 [0.302, 0.400] | 0.111 [0.080, 0.145] |
| probe ctx_s1 | val | all | 540 | 0.200 [0.159, 0.239] | 0.600 [0.531, 0.663] | 0.348 | 0.311 [0.276, 0.348] | 0.363 [0.243, 0.480] | 0.095 [0.027, 0.171] |
| probe ctx_s1 | val | 15 | 360 | 0.214 [0.164, 0.267] | 0.667 [0.589, 0.745] | 0.404 | 0.319 [0.269, 0.370] | 0.349 [0.177, 0.501] | 0.105 [0.011, 0.194] |
| probe ctx_s1 | val | 20 | 180 | 0.172 [0.128, 0.222] | 0.467 [0.389, 0.556] | 0.235 | 0.294 [0.247, 0.343] | 0.191 [0.040, 0.347] | 0.002 [-0.080, 0.103] |
| probe ctx_s1 | dev | all | 960 | 0.167 [0.138, 0.197] | 0.484 [0.445, 0.522] | 0.306 | 0.280 [0.253, 0.310] | 0.302 [0.218, 0.383] | 0.073 [0.022, 0.121] |
| probe ctx_s1 | dev | 15 | 480 | 0.190 [0.148, 0.229] | 0.521 [0.454, 0.579] | 0.379 | 0.259 [0.217, 0.303] | 0.288 [0.181, 0.387] | 0.071 [0.023, 0.120] |
| probe ctx_s1 | dev | 20 | 480 | 0.144 [0.108, 0.179] | 0.448 [0.402, 0.500] | 0.232 | 0.301 [0.267, 0.336] | 0.237 [0.142, 0.346] | 0.046 [-0.017, 0.107] |

## Memorization check (plain variant, seed 0)

Question: can the same architecture fit a small train set, and where does the training loss sit relative to its floor?
Setup: `probe_train.py --variant plain --seed 0 --n_polytopes 50 --epochs 300 --no_early_stop` (50 seeded S_sup train
polytopes, 600 states; no validation, no early stop; the final weights are reported). The entropy floor is the mean entropy
of the targets softmax(dlnV / 0.1), which lower-bounds the listwise cross entropy for these states. Single seed, so no CI.
Metrics are on the same 600 train states. Recall and Spearman(actor) use states with >= 2 children. Logs and predictions are
in `runs/probe_20261103/memo/` (not committed).

| config | lr | hidden/out | final train CE | entropy floor | final train loss (CE + 0.5 MSE) | Spearman(actor, dlnV) | recall@1 | recall@4 | Spearman(V, y1) | EV(V, y1) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3e-4 | 64 | 1.076 | 0.563 | 1.123 | 0.391 | 0.813 | 0.988 | 0.767 | 0.638 |
| lr | 1e-3 | 64 | 0.693 | 0.563 | 0.720 | 0.505 | 0.923 | 1.000 | 0.875 | 0.816 |
| 2x width | 3e-4 | 128 | 0.669 | 0.563 | 0.681 | 0.487 | 0.923 | 1.000 | 0.936 | 0.900 |

Train CE by epoch (base): 1.80 (epoch 99), 1.38 (199), 1.22 (249), 1.08 (299), so the base run had not plateaued at 300 epochs.
The lr 1e-3 run is at 0.70 (249) and 0.69 (299), and the 2x-width run at 0.72 (249) and 0.67 (299). Both are within 0.13 of the floor.
