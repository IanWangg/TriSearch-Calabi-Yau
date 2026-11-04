# GL(4,Z)-invariant encoder for the two-face policy (2026-11-04)

Code: `core/two_face_invariant_features.py` (add_invariant_fields), `models/invariant_two_face_agent.py`
(InvariantTwoFaceAgent), `test/test_invariant_two_face.py`. Scripts in this directory. Data and checkpoints in
`runs/inv_encoder_20261104/` of the main checkout (not committed).

## Commands

```bash
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
PY=/home/yiranwang/anaconda3/envs/sage/bin/python
W=<worktree>   # /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf
cd $W
PYTHONPATH=$W CUDA_VISIBLE_DEVICES= $PY -m pytest -q test/test_invariant_two_face.py          # 14 passed, 8 s
PYTHONPATH=$W CUDA_VISIBLE_DEVICES= $PY -m pytest -q test/test_cy_two_face_data.py test/test_two_face_agent.py test/test_invariant_two_face.py   # 34 passed, 22 s
PYTHONPATH=$W OMP_NUM_THREADS=4 $PY eval/sweep/inv_encoder_20261104/gl4z_invariance_inv.py   # A2, 43 s
bash eval/sweep/inv_encoder_20261104/run_probe_inv.sh   # A3: four runs in parallel, all on cuda:4
PYTHONPATH=$W $PY eval/sweep/inv_encoder_20261104/analyze_inv.py   # A3 tables, about 1 min
```

Outputs: `runs/inv_encoder_20261104/gl4z_invariance_results.json`, `runs/inv_encoder_20261104/probe/{inv,inv_ctx}_s{0,1}*`,
`runs/inv_encoder_20261104/analysis_inv.{md,json}`.

## Pytest

14 passed for test/test_invariant_two_face.py (facet normals of the 4-cube, hand counts of point invariants,
GL(4,Z) and relabelling invariance of the features, point-graph and face counts, PyG batch offsets for point_index and
point_edge_index with batched-vs-single outputs, batch round trip, no mutation of the input, real shard facets on 20
states, agent invariance to GL(4,Z) and node/face/triangle relabelling, batch.x = NaN, rejection of data without
invariant fields, backward pass). Existing two-face tests (test_cy_two_face_data.py, test_two_face_agent.py) still pass:
34 passed in total with the new file.

## A2: invariance (fresh model, seed 0, eval mode)

200 states; each with 1 identity, 20 signed permutations, 20 single shears, 20 double shears, plus 2 random
node/face/triangle permutations and a batch.x = NaN run. Max over states and U of the absolute differences from the
unmodified state. Max abs logit over all states is 0.599. Acceptance (all < 1e-5) passes.

| condition | pairs | max abs logit diff | max abs V diff | top-1 changed |
|---|---|---|---|---|
| A identity | 200 | 0.00e+00 | 0.00e+00 | 0 |
| B signed perms | 4000 | 2.98e-07 | 1.19e-07 | 0 |
| C1 single shears | 4000 | 0.00e+00 | 0.00e+00 | 0 |
| C2 double shears | 4000 | 2.98e-07 | 8.94e-08 | 0 |
| perm (nodes/faces/triangles) | 400 | 4.17e-07 | 1.49e-07 | 0 |
| batch.x = NaN | 200 | 3.58e-07 | 1.49e-07 | n/a |

U sets: identical to the branch run (checked against `runs/gl4z_invariance_20261103/results.json` U_matrices). The
state sample is not the branch sample (see Deviations). Threshold: all max diffs < 1e-5, pass.

## A3: supervised probe (P1 protocol, width 64, same hyperparameters)

Same data as P1 (`runs/probe_20261103`): S_sup train and val, S_dev evaluation. Adam lr 3e-4, batch 64, tau 0.1,
value coefficient 0.5, max 60 epochs, early stopping on S_sup_val with patience 8. Seeds 0 and 1.
Metrics as in the P1 Q3 table, with 95% percentile bootstrap CIs over polytopes (B = 1000). The P1 rows are copied from
the P1 README by `analyze_inv.py`.

## Q3 probe rows (val = S_sup validation, dev = S_dev); 95% percentile CI over polytopes (B = 1000)

| model | split | h11 | n_states | recall@1 | recall@4 | random@4 | Spearman actor | Spearman V vs y1 | EV V vs y1 |
|---|---|---|---|---|---|---|---|---|---|
| probe plain_s0 | val | all | 540 | 0.202 [0.165, 0.239] | 0.589 [0.526, 0.650] | 0.348 | 0.294 [0.260, 0.333] | 0.230 [0.117, 0.340] | 0.037 [-0.011, 0.077] |
| probe plain_s0 | val | 15 | 360 | 0.214 [0.169, 0.261] | 0.658 [0.586, 0.731] | 0.404 | 0.300 [0.247, 0.356] | 0.181 [0.031, 0.330] | 0.029 [-0.003, 0.061] |
| probe plain_s0 | val | 20 | 180 | 0.178 [0.139, 0.222] | 0.450 [0.372, 0.533] | 0.235 | 0.283 [0.233, 0.332] | -0.065 [-0.192, 0.083] | -0.054 [-0.134, -0.007] |
| probe plain_s0 | dev | all | 960 | 0.141 [0.117, 0.166] | 0.499 [0.460, 0.535] | 0.306 | 0.269 [0.242, 0.297] | 0.191 [0.068, 0.294] | 0.030 [-0.005, 0.066] |
| probe plain_s0 | dev | 15 | 480 | 0.148 [0.112, 0.181] | 0.542 [0.477, 0.602] | 0.379 | 0.255 [0.213, 0.295] | 0.163 [0.028, 0.289] | 0.016 [-0.008, 0.040] |
| probe plain_s0 | dev | 20 | 480 | 0.133 [0.106, 0.163] | 0.456 [0.412, 0.500] | 0.232 | 0.282 [0.245, 0.320] | 0.107 [-0.022, 0.235] | 0.016 [-0.013, 0.051] |
| probe plain_s1 | val | all | 540 | 0.204 [0.163, 0.243] | 0.594 [0.535, 0.650] | 0.348 | 0.315 [0.280, 0.353] | 0.165 [0.036, 0.270] | 0.031 [0.001, 0.062] |
| probe plain_s1 | val | 15 | 360 | 0.208 [0.161, 0.261] | 0.650 [0.583, 0.719] | 0.404 | 0.312 [0.261, 0.367] | 0.108 [-0.056, 0.267] | 0.019 [-0.019, 0.053] |
| probe plain_s1 | val | 20 | 180 | 0.194 [0.144, 0.244] | 0.483 [0.406, 0.561] | 0.235 | 0.321 [0.276, 0.364] | -0.064 [-0.201, 0.100] | -0.024 [-0.060, 0.001] |
| probe plain_s1 | dev | all | 960 | 0.153 [0.127, 0.180] | 0.482 [0.443, 0.519] | 0.306 | 0.285 [0.258, 0.314] | 0.148 [0.024, 0.261] | 0.018 [-0.015, 0.049] |
| probe plain_s1 | dev | 15 | 480 | 0.175 [0.135, 0.215] | 0.517 [0.456, 0.575] | 0.379 | 0.273 [0.230, 0.314] | 0.117 [-0.048, 0.272] | 0.009 [-0.022, 0.038] |
| probe plain_s1 | dev | 20 | 480 | 0.131 [0.098, 0.169] | 0.448 [0.404, 0.496] | 0.232 | 0.297 [0.263, 0.332] | 0.060 [-0.096, 0.209] | 0.001 [-0.034, 0.032] |
| probe ctx_s0 | val | all | 540 | 0.200 [0.161, 0.235] | 0.591 [0.531, 0.646] | 0.348 | 0.314 [0.283, 0.348] | 0.358 [0.232, 0.483] | 0.093 [0.028, 0.165] |
| probe ctx_s0 | val | 15 | 360 | 0.214 [0.169, 0.258] | 0.644 [0.575, 0.714] | 0.404 | 0.319 [0.270, 0.366] | 0.365 [0.195, 0.520] | 0.113 [0.030, 0.193] |
| probe ctx_s0 | val | 20 | 180 | 0.172 [0.122, 0.228] | 0.483 [0.411, 0.561] | 0.235 | 0.304 [0.261, 0.343] | 0.173 [0.032, 0.312] | -0.003 [-0.079, 0.092] |
| probe ctx_s0 | dev | all | 960 | 0.151 [0.125, 0.177] | 0.481 [0.441, 0.520] | 0.306 | 0.280 [0.253, 0.307] | 0.295 [0.218, 0.370] | 0.070 [0.023, 0.115] |
| probe ctx_s0 | dev | 15 | 480 | 0.165 [0.127, 0.202] | 0.521 [0.458, 0.583] | 0.379 | 0.257 [0.216, 0.299] | 0.270 [0.152, 0.385] | 0.063 [0.011, 0.117] |
| probe ctx_s0 | dev | 20 | 480 | 0.138 [0.104, 0.173] | 0.442 [0.394, 0.494] | 0.232 | 0.302 [0.270, 0.335] | 0.246 [0.150, 0.358] | 0.050 [-0.008, 0.111] |
| probe ctx_s1 | val | all | 540 | 0.200 [0.159, 0.239] | 0.600 [0.531, 0.663] | 0.348 | 0.311 [0.276, 0.348] | 0.363 [0.243, 0.480] | 0.095 [0.027, 0.171] |
| probe ctx_s1 | val | 15 | 360 | 0.214 [0.164, 0.267] | 0.667 [0.589, 0.745] | 0.404 | 0.319 [0.269, 0.370] | 0.349 [0.177, 0.501] | 0.105 [0.011, 0.194] |
| probe ctx_s1 | val | 20 | 180 | 0.172 [0.128, 0.222] | 0.467 [0.389, 0.556] | 0.235 | 0.294 [0.247, 0.343] | 0.191 [0.040, 0.347] | 0.002 [-0.080, 0.103] |
| probe ctx_s1 | dev | all | 960 | 0.167 [0.138, 0.197] | 0.484 [0.445, 0.522] | 0.306 | 0.280 [0.253, 0.310] | 0.302 [0.218, 0.383] | 0.073 [0.022, 0.121] |
| probe ctx_s1 | dev | 15 | 480 | 0.190 [0.148, 0.229] | 0.521 [0.454, 0.579] | 0.379 | 0.259 [0.217, 0.303] | 0.288 [0.181, 0.387] | 0.071 [0.023, 0.120] |
| probe ctx_s1 | dev | 20 | 480 | 0.144 [0.108, 0.179] | 0.448 [0.402, 0.500] | 0.232 | 0.301 [0.267, 0.336] | 0.237 [0.142, 0.346] | 0.046 [-0.017, 0.107] |
| probe inv_s0 | val | all | 540 | 0.328 [0.276, 0.378] | 0.759 [0.707, 0.807] | 0.348 | 0.533 [0.510, 0.558] | 0.238 [0.130, 0.330] | 0.047 [0.019, 0.074] |
| probe inv_s0 | val | 15 | 360 | 0.350 [0.292, 0.411] | 0.786 [0.725, 0.839] | 0.404 | 0.539 [0.508, 0.570] | 0.096 [-0.039, 0.241] | 0.010 [-0.015, 0.036] |
| probe inv_s0 | val | 20 | 180 | 0.283 [0.200, 0.378] | 0.706 [0.639, 0.778] | 0.235 | 0.521 [0.481, 0.556] | -0.063 [-0.210, 0.089] | -0.007 [-0.028, 0.005] |
| probe inv_s0 | dev | all | 960 | 0.315 [0.277, 0.351] | 0.713 [0.673, 0.751] | 0.306 | 0.537 [0.506, 0.568] | 0.190 [0.074, 0.292] | 0.024 [-0.006, 0.054] |
| probe inv_s0 | dev | 15 | 480 | 0.342 [0.292, 0.385] | 0.754 [0.700, 0.810] | 0.379 | 0.532 [0.488, 0.577] | 0.111 [-0.061, 0.260] | 0.010 [-0.006, 0.028] |
| probe inv_s0 | dev | 20 | 480 | 0.287 [0.235, 0.340] | 0.671 [0.623, 0.723] | 0.232 | 0.542 [0.497, 0.583] | 0.133 [-0.010, 0.275] | 0.009 [-0.002, 0.020] |
| probe inv_s1 | val | all | 540 | 0.333 [0.283, 0.381] | 0.765 [0.722, 0.807] | 0.348 | 0.517 [0.488, 0.546] | 0.226 [0.111, 0.330] | 0.040 [0.003, 0.080] |
| probe inv_s1 | val | 15 | 360 | 0.353 [0.292, 0.411] | 0.803 [0.753, 0.847] | 0.404 | 0.518 [0.474, 0.558] | 0.154 [0.016, 0.284] | 0.020 [-0.034, 0.070] |
| probe inv_s1 | val | 20 | 180 | 0.294 [0.217, 0.378] | 0.689 [0.622, 0.767] | 0.235 | 0.515 [0.476, 0.552] | -0.035 [-0.158, 0.140] | -0.020 [-0.053, 0.009] |
| probe inv_s1 | dev | all | 960 | 0.307 [0.267, 0.345] | 0.698 [0.652, 0.741] | 0.306 | 0.531 [0.502, 0.560] | 0.201 [0.082, 0.307] | 0.027 [-0.013, 0.062] |
| probe inv_s1 | dev | 15 | 480 | 0.356 [0.304, 0.406] | 0.767 [0.710, 0.825] | 0.379 | 0.533 [0.489, 0.574] | 0.140 [-0.048, 0.297] | 0.012 [-0.029, 0.049] |
| probe inv_s1 | dev | 20 | 480 | 0.258 [0.212, 0.308] | 0.629 [0.573, 0.688] | 0.232 | 0.529 [0.484, 0.571] | 0.138 [-0.011, 0.271] | 0.013 [-0.007, 0.032] |
| probe inv_ctx_s0 | val | all | 540 | 0.343 [0.291, 0.393] | 0.757 [0.709, 0.802] | 0.348 | 0.529 [0.505, 0.555] | 0.332 [0.209, 0.458] | 0.080 [0.024, 0.144] |
| probe inv_ctx_s0 | val | 15 | 360 | 0.358 [0.297, 0.422] | 0.781 [0.722, 0.836] | 0.404 | 0.540 [0.509, 0.569] | 0.347 [0.191, 0.491] | 0.091 [0.012, 0.175] |
| probe inv_ctx_s0 | val | 20 | 180 | 0.311 [0.233, 0.394] | 0.711 [0.628, 0.789] | 0.235 | 0.509 [0.463, 0.548] | 0.203 [0.059, 0.356] | 0.027 [-0.037, 0.102] |
| probe inv_ctx_s0 | dev | all | 960 | 0.308 [0.278, 0.339] | 0.710 [0.669, 0.752] | 0.306 | 0.525 [0.493, 0.555] | 0.277 [0.187, 0.377] | 0.056 [0.012, 0.102] |
| probe inv_ctx_s0 | dev | 15 | 480 | 0.335 [0.292, 0.377] | 0.769 [0.717, 0.821] | 0.379 | 0.528 [0.481, 0.570] | 0.294 [0.185, 0.396] | 0.066 [0.021, 0.114] |
| probe inv_ctx_s0 | dev | 20 | 480 | 0.281 [0.244, 0.323] | 0.652 [0.598, 0.710] | 0.232 | 0.522 [0.477, 0.563] | 0.186 [0.059, 0.325] | 0.017 [-0.042, 0.084] |
| probe inv_ctx_s1 | val | all | 540 | 0.348 [0.300, 0.394] | 0.774 [0.737, 0.809] | 0.348 | 0.530 [0.502, 0.557] | 0.386 [0.280, 0.484] | 0.114 [0.052, 0.177] |
| probe inv_ctx_s1 | val | 15 | 360 | 0.367 [0.311, 0.422] | 0.808 [0.761, 0.850] | 0.404 | 0.535 [0.497, 0.569] | 0.343 [0.199, 0.485] | 0.093 [0.010, 0.177] |
| probe inv_ctx_s1 | val | 20 | 180 | 0.311 [0.233, 0.400] | 0.706 [0.667, 0.750] | 0.235 | 0.519 [0.483, 0.554] | 0.212 [0.049, 0.371] | 0.035 [-0.043, 0.120] |
| probe inv_ctx_s1 | dev | all | 960 | 0.315 [0.278, 0.351] | 0.713 [0.670, 0.755] | 0.306 | 0.536 [0.505, 0.567] | 0.311 [0.208, 0.411] | 0.074 [0.014, 0.131] |
| probe inv_ctx_s1 | dev | 15 | 480 | 0.346 [0.298, 0.390] | 0.762 [0.702, 0.821] | 0.379 | 0.533 [0.490, 0.576] | 0.320 [0.219, 0.424] | 0.082 [0.037, 0.127] |
| probe inv_ctx_s1 | dev | 20 | 480 | 0.283 [0.231, 0.338] | 0.662 [0.606, 0.717] | 0.232 | 0.539 [0.496, 0.579] | 0.236 [0.104, 0.375] | 0.040 [-0.031, 0.116] |

## Training summary (from *_train_log.json)

| run | epochs run | best epoch | train loss first -> last | best val loss | wall s | prep s | device |
|---|---|---|---|---|---|---|---|
| inv_s0 | 29 | 20 | 2.408 -> 1.935 | 2.078 | 124 | 13.1 | cuda:4 |
| inv_s1 | 33 | 24 | 2.426 -> 1.898 | 2.087 | 140 | 14.1 | cuda:4 |
| inv_ctx_s0 | 31 | 22 | 2.433 -> 1.894 | 2.078 | 141 | 14.0 | cuda:4 |
| inv_ctx_s1 | 30 | 21 | 2.405 -> 1.899 | 2.073 | 136 | 13.1 | cuda:4 |

Wall times: each run took 124 to 141 s of training (after 13 to 14 s of data preparation, the add_invariant_fields
calls on 6,360 states); the four runs ran in parallel with 4 CPU threads each.

## Deviations and details

1. Records file changed. The branch A2 run used a 5184-record `records_sup.jsonl` (5164 eligible, sha256 869afb26...).
   The current file has 5400 records (5380 eligible, sha256 different), so the 200 states are drawn by the same rule
   from the current file and are **not** the branch sample. The U sets are the branch ones exactly: the state draw
   consumes the same seeded stream, so the transforms were replayed with n = 5164 and checked against the branch
   U_matrices.
2. P1 script changed after it was read. `eval/sweep/probe_20261103/probe_train.py` gained a `--pooling` option
   (default `sum`, documented as reproducing TwoFaceAgent exactly). The copy used here is the current version;
   the plain path is unchanged. The P1 rows are from the P1 README, produced before this change.
3. GPU selection: `pick_device` was changed to choose among GPUs 4, 5, 6 by free memory (the P1 rule excluded only 0
   and 3). All four runs were placed on cuda:4 by that rule.
4. Threads: 4 per process (16 in total).
5. The feature `triangles_containing` counts distinct merged triangles (merged point triples) containing the point.
6. The inv variants are not dispatched through the model factory; InvariantTwoFaceAgent is constructed directly. Its
   model_config records encoder_variant = "gl4z_invariant_v1".
7. Git: `eval/sweep/*` is gitignored in this repo (`.gitignore:15`), so the new files in this directory need
   `git add -f`. Not changed here.
8. A stray file `runs/inv_encoder_20261104_start.txt` was written in the main checkout by an earlier command; it is
   empty of results and can be removed.

## A4: two_face_invariant in the PPO pipeline and eval/policy.py

Changes (no default behavior change; the two_face_deep_sets path is the same code):
- models/subcomplex_policy_config.py: "two_face_invariant" is a canonical actor type; TWO_FACE_ACTOR_TYPES groups it
  with two_face_deep_sets for encoder-option validation and observation kind (two_face).
- models/subcomplex_policy_factory.py: builds InvariantTwoFaceAgent for two_face_invariant (same arguments).
- models/invariant_two_face_agent.py: subcomplex_actor_type = "two_face_invariant" (saved in model_config, so the
  checkpoint round trip goes through the factory); observation_builder = "two_face_invariant".
- core/two_face_invariant_features.py: create_invariant_two_face_data_from_state (default create_two_face_data_from_state
  plus invariant fields, invariant fields cached per (configuration, restrictions), LRU 50,000 entries).
- core/cy_policy_inference.py: build_cy_data_list accepts observation_kind "two_face_invariant"; policy_build_kind(policy)
  is used at the build call sites (action selection, scoring, values, action evaluation); the step-result observation_kind
  stays the data kind "two_face". core/cy_ppo.py mini-batch rebuild uses policy_build_kind.
- core/cy_checkpointing.py, eval/policy.py, core/cy_training_runner.py, core/cy_training_config.py: accept the new type
  wherever two_face_deep_sets is accepted (checkpoint model_config encoder tag, EvaluationPolicy.from_spec, the training
  config path, best_so_far validation).

Tests: test/test_invariant_rl_path.py (3 passed): factory and config registration; checkpoint save/load and
EvaluationPolicy.from_spec round trip, with a mismatched architecture refused; the rollout observation path on a real
4D state (real_representatives) equals add_invariant_fields(create_two_face_data_from_state(...)) for all keys, and the
value and logits forward are equal to 1e-6; the second observation of the same topology is a cache hit.
Existing suites run with the change: test_cy_two_face_policy.py, test_two_face_agent.py, test_invariant_two_face.py,
test_cy_bounded_ppo.py, test_cy_two_face_data.py: 83 passed (plus the 3 new tests above).

Observation overhead (eval/sweep/inv_encoder_20261104/measure_obs_overhead.py, 12 real states, 1 thread, medians):
default create_two_face_data_from_state 5.2 ms; invariant, cache miss 9.3 ms (+4.1 ms); invariant, cache hit 6.2 ms
(+1.0 ms). Output: runs/inv_encoder_20261104/obs_overhead.json.

Smoke (run_inv_rl_smoke.sh): see the status section below.

### Smoke status (2026-11-04)

The first smoke launch (3 iterations, 16 states, 2 transition workers, GPU 4, eval_interval 1) did not complete an
iteration in 34 minutes, while the machine load average was about 60. The process was alive, mostly blocked on futex
waits, with no worker child processes visible. Stopping it and relaunching unbuffered was denied by the permission
layer, so the timing comparison with the baseline (iteration median 96.6 s at 128 states, 24 workers) is not available.
The command is in run_inv_rl_smoke.sh. That script has no -u flag; the stdout log is block-buffered, which is why the run looked silent.

### A4 full RL run (not launched)

Same as baseline G (fv1_mix_g099_s0_20261103_0604) except the actor type and transition workers (14, per the coordinator
brief). Paths follow the baseline layout; the run is not launched.

```bash
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
W=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf
R=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau
RUN=$R/runs/inv_rl_full_20261104
cd $W
PYTHONPATH=$W /home/yiranwang/anaconda3/envs/sage/bin/python scripts/train_cy.py \
  --dataset_path $R/data/cy/favorable_v1/train/h11_15.samples.jsonl $R/data/cy/favorable_v1/train/h11_20.samples.jsonl \
  --eval_dataset_path $R/data/cy/favorable_v1/dev/h11_15.samples.jsonl $R/data/cy/favorable_v1/dev/h11_20.samples.jsonl \
  --in_channels 4 --subcomplex_actor_type two_face_invariant --neighbor_mode two_neighbors --no-include_points_interior_to_facets \
  --reward_function max_kcup --hidden_channels 64 --out_channels 64 --num_layers 3 --count_bonus_coef 0 --no-vertex_aug_enable \
  --seed 0 --num_iterations 200 --num_epochs 1 --num_states 128 --rollout_length 20 --batch_size 128 --lr 0.0001 \
  --use_multiprocessing --transition_num_workers 14 --memory_budget_gb 72 --runtime_cache_gb 16 \
  --num_eval_states 128 --eval_steps 30 --eval_interval 50 --deterministic_eval \
  --checkpoint_path $RUN/checkpoints --iteration_metrics_path $RUN/iteration_metrics.jsonl \
  --latest_checkpoint_interval 10 --save_interval 50 --wandb_project calabi_yau_two_face_kcup \
  --name_suffix inv_rl_full_20261104 --gpu_index 4 --seed 0 --use_wandb --gamma 0.99 --gae_lambda 0.95
```
