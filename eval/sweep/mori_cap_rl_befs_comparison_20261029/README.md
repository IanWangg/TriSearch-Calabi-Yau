# CPU Mori cap RL + BeFS comparison

Compare incremental RL + BeFS, best_so_far RL + BeFS, vanilla BeFS and cyopt GA
on the same h11=30 setup: five polytopes, five distinct full FRST starts each,
seed 0, and 1000 logical objective queries per rollout. Smaller stages are a
1×1 smoke at 32 queries, followed by the full 5×5 setup at 128 queries. The full
setup is `eval/data/setups/h11_30_5_polytopes_5_starts_seed_0`, with HF commit
`60c0e119a03608418df538191f65da3f43b5b819`, no favorable filter, and the recorded
`skip_insufficient_starts` selection policy. This tests transfer from favorable
h11=15 training. These are source-order selections, not uniform samples.

Both models use `400.pth` from the `incremental_retry_2` / `best_so_far_retry_2`
directories of `runs/min_mori_cap_h11_15_mosek_restart_20261026_144514_2595371`.
They have 9 and 10 warm-start iterations respectively, followed by 400 restart
iterations. Frozen local copies and hashes are in `checkpoints/` and
`configs/checkpoint_manifest.json`; weights are excluded from Git. Checkpoints are fixed
before evaluation and are not selected using the test results.

The objective is the raw number of Mori cap extremal rays, minimized using the
shared MOSEK backend. All runs use full-FRST identity (`two_face_state=false`),
bounded caches, and exactly the same starts. GA retains its native 2-face DNA
representation and default population 50 / mutation probability 0.1 / elitism 1.
RL proposes up to four unseen targets per parent. Both critic coefficients are
1; incremental uses `-n(state)+V(state)`, best_so_far uses
`-session_best+V(state, session_best, min(64, remaining_queries))` and refreshes
the entire pending queue after each round. The count bonus in training remains
part of both learned critics; these are heuristic scores. Completing a final
parent can exceed the query budget; curves and endpoints exclude that overshoot.
Successful early stops carry their last minimum forward. Failures are not filled.

All inference and geometry run on CPU. Each method receives five distinct CPUs,
four geometry workers, a 16 GiB process-tree budget, and 4 GiB of runtime cache.
No GPU is assigned. Before starting, confirm that the chosen CPU ranges are
appropriate for the machine. The opt-in CPU startup is documented in
`eval/README.md`; use the sage interpreter and keep the variables inherited:

```bash
export TRISEARCH_CPU_ONLY=1
export PYTHONPATH="$PWD/tools/cpu_runtime:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python eval/sweep/mori_cap_rl_befs_comparison_20261029/scripts/run_study.py \
  --phase all --cpu_ids 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
```

`--phase smoke`, `pilot` or `full` runs one stage; output directories must be
new. A failed method stops the other owned jobs and leaves failure records and
logs. Smoke artifacts live under `eval/smoke_tests/`; pilot/full results live
under `results/<phase>/`, plots under `plots/<phase>/`, and stdout/stderr under
`logs/`. Configs are in `configs/`, launcher/analyzer source in `scripts/`.
Restore the frozen weights locally before reproducing; the manifest records their hashes.

Each method is validated with the shared offline reader. Explicit method labels
allow comparing the two checkpoints without weakening ordinary merge checks.
Plots retain all source directories and policy hashes. Outputs include per
polytope median/IQR curves, endpoint distributions, all per-start endpoints,
actual queries/overshoot/termination, and paired win/tie/loss plus raw count
differences versus both baselines. Pooled raw-count summaries are descriptive;
use per-polytope curves and paired comparisons for conclusions across polytopes.

The smoke, pilot and full benchmark all completed successfully. The full phase
finished on October 31, 2026 at 18:00 UTC, with all 100 trajectories complete.
CPU regression checks passed (171 passed, 5 CUDA cases skipped; 12 additional
contextual/provenance checks passed). `validation.json` preserves those outcomes.
`scripts/finish_study.py` supervised the pilot and launched the full benchmark
only after successful validation.

At 1000 queries, incremental RL + BeFS beats/ties/loses to vanilla BeFS on
15/7/3 paired starts and to GA on 5/12/8. Its mean endpoint ray count is 0.64
lower than vanilla BeFS and 0.28 higher than GA. Best_so_far RL + BeFS records
0/19/6 against vanilla BeFS and 0/7/18 against GA. These are paired descriptive
results for this five-polytope cohort; they are not significance tests.

| Method | Process wall time (hours) | Queries including overshoot |
| --- | ---: | ---: |
| Incremental RL + BeFS | 11.88 | 25050 |
| Best_so_far RL + BeFS | 13.36 | 25052 |
| Vanilla BeFS | 9.36 | 25183 |
| GA | 36.01 | 25000 |

Times include process startup and GA codebook preparation, with five CPUs
allocated to each method. Each curve and endpoint excludes queries above 1000.
See [the full paired statistics](plots/full/paired_vs_baselines.csv),
[per-start endpoints](plots/full/benchmark_summary.csv),
[runtime statistics](plots/full/runtime_summary.csv), and
[per-polytope comparison](plots/full/best_mori_cap_by_polytope.png).

The published archive includes pilot/full plots and numeric curves, per-start
endpoint CSVs, completed study manifests, and each method's config, summary and
rollout records. Checkpoints, shared setups, raw query/expansion logs, SQLite
runtime databases, console logs, patches and source snapshot ZIPs remain local.
Consequently, the compact archive supports reviewing the reported statistics;
native offline revalidation/replotting requires restoring the raw query logs
at their original paths. The full manifest retains code hashes and checkpoint
provenance. Archived result metadata identifies the original run's code and
working tree; it is not relabeled with the later publication commit.
