# GL(4,Z) invariance test of TwoFaceAgent (CPU)

Measures how much the actor logits and the value of `TwoFaceAgent` change when the
raw lattice coordinates are transformed as `x -> x @ U.T` for unimodular integer `U`.
The geometry is invariant under these maps, so every change measured here is error.

Script: `eval/sweep/gl4z_invariance_20261103/gl4z_test.py`
Results: `runs/gl4z_invariance_20261103/results.json` (outside the repo, in the main checkout)

## Command

```
cd /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf
PYTHONPATH=$PWD /home/yiranwang/anaconda3/envs/sage/bin/python eval/sweep/gl4z_invariance_20261103/gl4z_test.py
```

CPU only, `torch.set_num_threads(8)`, seed 0. Runtime 72.7 s.

## Counts

- Records in `runs/probe_20261103/stageA/records_sup.jsonl`: 5184 (sha256 `869afb26...e861334`)
  - Note: the file was still being appended to while this ran (4992 records at first look,
    5184 at run time, 27 shards). The sample is drawn from the 5184-record snapshot identified by the hash.
- Eligible (>= 5 actions): 5164
- States sampled uniformly (seed 0): 200
- Skipped (child count != `num_available_subcomplexes`): 0
- States used: 200
- Transforms per condition: A = 1 (identity), B = 20 (signed permutations), C1 = 20 (single elementary shears), C2 = 20 (products of two elementary shears)
- All U: integer entries, det = ±1 (A: +1; B: ±1; C1: +1; C2: +1)
- Condition A check (U = I vs original, both checkpoints): max |logit diff| = 0.0, max |V diff| = 0.0

## Results

Pairs: 200 states for A (1 U each), 200 x 20 = 4000 (state, U) pairs for B, C1, C2.
CI = 95% bootstrap over states (1000 resamples). `rho_self` = Spearman(logits_orig, logits_U) per state.
`rho_true` = Spearman(logits, lnV(child) - lnV(s)), mean over states (orig) and over pairs (U).
`max|x|` = mean over pairs of max |x| after transform (original mean is 3.155).
`V std` = std over the 200 original states of V.

| ckpt | cond | n (states / pairs) | rho_self mean [95% CI] | rho_self p10 | top1 agree [95% CI] | top4 overlap | mean abs dV | V std | rho_true orig / U | max\|x\| after |
|---|---|---|---|---|---|---|---|---|---|---|
| G | A | 200 / 200 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.0000 | 0.589 | 0.245 / 0.245 | 3.155 |
| G | B | 200 / 4000 | 0.751 [0.734, 0.767] | 0.520 | 0.608 [0.563, 0.653] | 0.735 | 0.489 | 0.589 | 0.245 / 0.214 | 3.155 |
| G | C1 | 200 / 4000 | 0.882 [0.874, 0.889] | 0.761 | 0.750 [0.719, 0.780] | 0.809 | 0.242 | 0.589 | 0.245 / 0.225 | 3.656 |
| G | C2 | 200 / 4000 | 0.820 [0.809, 0.830] | 0.626 | 0.670 [0.635, 0.702] | 0.767 | 0.327 | 0.589 | 0.245 / 0.220 | 3.930 |
| G50 | A | 200 / 200 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.0000 | 0.095 | 0.041 / 0.041 | 3.155 |
| G50 | B | 200 / 4000 | 0.224 [0.208, 0.240] | -0.525 | 0.169 [0.152, 0.187] | 0.427 | 0.012 | 0.095 | 0.041 / 0.025 | 3.155 |
| G50 | C1 | 200 / 4000 | 0.871 [0.861, 0.882] | 0.682 | 0.658 [0.628, 0.686] | 0.820 | 0.005 | 0.095 | 0.041 / 0.049 | 3.656 |
| G50 | C2 | 200 / 4000 | 0.758 [0.743, 0.772] | 0.343 | 0.533 [0.509, 0.561] | 0.733 | 0.007 | 0.095 | 0.041 / 0.039 | 3.930 |

Per-pair maxima of the logit and V changes (from results.json, `max_abs_logit_diff`, `max_abs_V_diff`):
G B 7.03 / 4.05; G C1 4.03 / 2.13; G C2 5.96 / 2.84; G50 B 1.49 / 0.070; G50 C1 0.76 / 0.030; G50 C2 1.47 / 0.052.
