# favorable_large_v1 (2026-11-04)

Large-h11 train/dev set for face-level search and invariant-encoder training. No test split.
Location: `data/cy/favorable_large_v1/` (not tracked in git; regenerate with the scripts below).

| split | h11 | polytopes | starts per polytope | wall (6 workers) |
|---|---|---|---|---|
| train | 35, 40, 45, 50 | 60 each (240) | 4 | 30, 32, 34, 34 min |
| dev | 35, 40, 45, 50 | 15 each (60) | 1 | < 1 min each |

All 300 selected polytopes were kept (no generation failures, no Kcup failures).
Loads with `core.cy_training_runner.load_dataset_rows`; `h11` matches the file name in every row.

## Procedure

Same as favorable_v1 (same HF source and revision, favorable only, `select_polytopes` then
`generate_diverse_starts` with train 4 starts / min 3, dev 1 start / min 1, seed 0) except:

1. Selection seed 20261104, 60 train + 15 dev per h11, excluding every favorable_v1 polytope (all splits,
   matched by source id and convex-hull vertex set) and the 10 h11=50 polytopes of
   `runs/face_mutation_20261104/setups/seed0_p10_s3`. See `selection_manifest.json`.
2. `generate_diverse_starts --skip-ntfe-enumeration`: no `n_2face_triangs()` count and no NTFE enumeration.
   In v1 the count only gates the exact NTFE enumeration (skipped when the bound exceeds 600k), so at h11 >= 35
   we expect it to cost time without changing the starts (not verified per polytope: the first attempt was
   stopped before it wrote per-polytope NTFE info). A first attempt with the count (`build_attempt1_with_counting.log`)
   was stopped: two h11=35 workers stuck in the count/enumeration grew to about 70 GB RSS each, and the h11=50
   count did not finish in 40 min. Starts therefore come from the fast, fair and walk pools only.

## Commands

```bash
bash data/cy/make_favorable_large_v1.sh        # first attempt; only its select_polytopes step is still used (stop it after selection)
bash data/cy/make_favorable_large_v1_rest.sh   # generation for h11 35/40/45/50 with --skip-ntfe-enumeration + manifest
```
