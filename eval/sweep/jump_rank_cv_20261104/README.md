# Ranking single-face jump moves with a learned scorer: leave-two-polytopes-out CV (2026-11-04)

Question: given a parent FRST and its single-face jump children (from the B1 logs), how well does a learned
scorer rank children by dlnV, and how well does it predict feasibility?

## Data

Source: `runs/face_mutation_20261104/results/face_mutation_{befs,hc}_budget1000_seed0_poly*/face_mutation_candidates.jsonl`.
Rows: befs 47,400; hc 49,037. hc has no polytope 9 (failed shard in B1), so polytope 9 groups come from befs only.

- Group = (polytope, start_hash, parent_dna), pooled over both algorithms. Children deduplicated by child_dna.
- Kept: groups with >= 4 feasible children (3277 of 3527 raw groups).
- Parent decode: `CyoptEncoding` rebuilt per polytope from the setup starts with the logged seed. The rebuilt codebook
  SHA-256 equals the logged `codebook_sha256` for all 10 polytopes. Parents decoded with `CyoptEncoding.decode`,
  cached under `runs/jump_rank_cv_20261104/parents/`.
- Per-parent check: every restriction of the decoded parent equals the logged codebook entry (0 mismatches).
- Face mapping: `row["face"]` is the codebook face; `row["dna_position"]` indexes the DNA (faces with codebook size > 1).
  New triangles = `codebook[face][child_dna[dna_position]]`.

Counts (groups kept / deduplicated children / feasible children):

| poly | groups | children | feasible |
|---|---|---|---|
| 0 | 521 | 17414 | 5414 |
| 1 | 493 | 15708 | 5914 |
| 2 | 352 | 9871 | 5993 |
| 3 | 242 | 8723 | 5942 |
| 4 | 267 | 6938 | 5987 |
| 5 | 283 | 6752 | 5992 |
| 6 | 305 | 7836 | 5986 |
| 7 | 323 | 8808 | 5997 |
| 8 | 269 | 6903 | 5988 |
| 9 | 222 | 3536 | 2997 |
| total | 3277 | 92489 | 56210 |

Mean per group: F = 17.15 feasible, N = 28.22 children.

## Scripts

- `count_groups.py`: raw group counts.
- `build_dataset.py`: builds `runs/jump_rank_cv_20261104/dataset/poly{i}.pt` (per-parent invariant-encoder batch and
  move index tensors in the layout of `FaceMoveScorer.score`, plus per-child metadata).
- `rank_cv.py`: `--verify`, `--zero-shot`, `--train`, `--rescore`, `--report`.

Commands (cwd = this directory; `PYTHONPATH=$W`, `MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic`, sage env python):

```bash
python count_groups.py
python -u build_dataset.py --check-scorer                      # run 1 stopped after poly 2 (background limit)
python -u build_dataset.py --polytopes 3 4 5 6 7 8 9 --check-scorer   # resumed
python -u rank_cv.py --verify
python -u rank_cv.py --zero-shot
python -u rank_cv.py --train --folds 0 1 2 3 4 --variants rank rank+feas   # stopped during fold 2 rank+feas
python -u rank_cv.py --rescore --folds 0 1 2 --variants rank rank+feas     # held-out scores from saved checkpoints
python -u rank_cv.py --train --folds 2 3 4 --variants rank+feas            # resumed
python -u rank_cv.py --train --folds 3 4 --variants rank                   # resumed
python -u rank_cv.py --report
```

Logs: `runs/jump_rank_cv_20261104/logs/`, `runs/jump_rank_cv_20261104/dataset/build_log.jsonl`,
`runs/jump_rank_cv_20261104/results/train_log.jsonl` (per fold/variant epochs and wall time).

Wall times:
- Build, polytopes 0..9 (`build_log.jsonl`): 33 s to 1393 s per polytope. Polytopes 3, 8, 9 took 800 to 1400 s but
  include idle time caused by the interruption and resume; decoding itself took about 6 to 15 s per polytope.
- Zero-shot and heuristics: 32 s. Verify: about 10 s.
- Training: 1126 to 1840 s per fold/variant (single GPU, 4 torch threads). Several runs were stopped at the 2 h
  background limit and resumed; the reported held-out scores come from saved checkpoints for those runs.

## Metrics

Per group, then averaged over groups. 95% CI: cluster bootstrap over the 10 polytopes (2000 resamples).

- Spearman: rank correlation of score and dlnV over the feasible children (scipy, average ranks).
- recall@1: 1 if the top-scored feasible child has the maximum dlnV among feasible children.
- top-16 gain: `k = min(F, max(1, round(16*F/N)))` (round half up), F = feasible children, N = all children.
  Top-k = the k feasible children with highest score (ties broken at random, averaged over 200 draws). Gain = max dlnV of
  the top-k minus the exact expected max dlnV over a uniformly random k-subset of the feasible children.
  The random method's gain is about 0 by construction.
- AUC (score): ROC AUC of score vs feasibility over all N children. Groups where all or none are feasible are skipped.
- AUC (feas head): same with the feasibility head logit (rank+feas only).

## Methods

- random: uniform scores (200 draws per group).
- heur_smaller_symdiff / heur_larger_symdiff: score = -/+ face_symdiff.
- zeroshot_mean2 / zeroshot_sum: pretrained `runs/inv_rl_full_20261104/checkpoints/final.pth`, actor logit with the
  `mean2` or `sum` aggregation (mean2 is the FaceMoveScorer default; `sum` is the raw sum).
- rank: encoder + actor head initialized from final.pth, listwise cross entropy between softmax(score) over feasible
  children and softmax(dlnV / 0.5).
- rank+feas: same, plus a feasibility head (Linear-ReLU-Linear, hidden 64 on the actor input) trained with BCE over all
  children. Ranking score is still the actor logit.

## Training protocol

- 5 folds; fold i tests polytopes {2i, 2i+1}. Every polytope is held out exactly once, so the pooled table covers
  all 3277 groups. Each group is scored by the model that did not train on its polytope.
- In each fold, 10% of the training groups (seed 1000+fold) are early-stopping groups and are not used for gradients.
  Fold 0 example: 2036 training, 227 early-stopping groups.
- Adam, lr 1e-4, 16 groups per optimizer step, max 30 epochs, early stop on the early-stopping loss with patience 5
  (patience not specified in the task; chosen here).
- Best epochs (rank / rank+feas): fold 0: 30 / 29; fold 1: 27 / 27; fold 2: 29 / 29; fold 3: 30 / 28; fold 4: 27 / 29.
  Most runs hit or nearly hit the 30-epoch cap, and validation loss was still decreasing at the cap, so these models are
  not converged. Early stopping never triggered.
- Checkpoints: `runs/jump_rank_cv_20261104/checkpoints/{rank,rank_feas}_fold{i}.pt`.

## Verification

`rank_cv.py --verify`: batched logits (index layout copied from FaceMoveScorer) vs `FaceMoveScorer.score`, 40 comparisons
(2 parents per polytope, both aggregations). Max absolute difference 7.6e-5 (float32 accumulation; threshold 1e-4).

## Results

Pooled held-out predictions over all 3277 groups (10 polytopes, 5 folds), mean with 95% cluster-bootstrap CI.
Source: `runs/jump_rank_cv_20261104/results/table.md` and `summary.json`.

| method | Spearman | recall@1 | top-16 gain | AUC (score) | AUC (feas head) |
|---|---|---|---|---|---|
| random | -0.003 [-0.014, 0.008] | 0.094 [0.068, 0.120] | 0.001 [-0.008, 0.016] | 0.502 [0.494, 0.509] | - |
| heur_smaller_symdiff | 0.473 [0.399, 0.565] | 0.239 [0.195, 0.275] | 0.031 [0.009, 0.052] | 0.627 [0.610, 0.642] | - |
| heur_larger_symdiff | -0.473 [-0.568, -0.400] | 0.044 [0.026, 0.062] | -0.131 [-0.180, -0.086] | 0.373 [0.358, 0.390] | - |
| zeroshot_mean2 | -0.061 [-0.186, 0.043] | 0.157 [0.111, 0.200] | -0.002 [-0.027, 0.020] | 0.329 [0.291, 0.376] | - |
| zeroshot_sum | -0.103 [-0.295, 0.052] | 0.092 [0.029, 0.153] | -0.088 [-0.134, -0.047] | 0.330 [0.295, 0.372] | - |
| rank (fine-tuned) | 0.394 [0.326, 0.475] | 0.264 [0.235, 0.289] | 0.033 [0.020, 0.048] | 0.461 [0.393, 0.530] | - |
| rank+feas (fine-tuned) | 0.371 [0.309, 0.443] | 0.248 [0.211, 0.276] | 0.040 [0.019, 0.061] | 0.470 [0.432, 0.511] | 0.755 [0.721, 0.796] |

Averages per group: mean k = 10.82, mean random expected max dlnV = -0.0865.

## Deviations from the task

- Patience 5 for early stopping (not specified). The 30-epoch cap was reached or nearly reached in most runs and
  validation loss was still falling, so the learned models are under-trained relative to convergence.
- Early stopping set = 10% of training groups, excluded from gradients (task did not say whether to exclude).
- Build, training and rescoring were interrupted and resumed (background time limit): polytope 3 build (first run
  stopped after polytope 2), fold 2 rank+feas (first training run), and then fold 2 to 4 rank+feas and folds 3 and 4 rank
  in separate jobs. Scores for folds 0 to 2 rank come from `--rescore` of saved checkpoints; the rest come from the
  resumed runs' caches. Same code and seeds, no change in protocol.
- Polytope 9 has only befs candidates (hc failed in B1).
- Per-group AUC is skipped when a group has only one class.
- Top-k ties are broken randomly; recall and top-k for the random method use 200 draws.
- Not run: any other seed or hyperparameter; no interpretation beyond the numbers.
