# Offline inventory of RL attempts

This script synthesizes the existing SNN, count_bonus, critic_weight,
two_face identity, triangle_dual and finite best_so_far experiments. It keeps
different cohorts, budgets and state identities separate. For each available
paired comparison it reports win/tie/loss counts, ratio of volume means,
median paired ratio and geometric mean paired ratio. These quantities answer
different questions; raw means can be dominated by one polytope.

No new training, inference or geometry is run. Source tables are previously
validated local evidence; native runs are revalidated where used. The original
h50 native_run selection is saved in `configs/h50_native_runs.json`, so the
script does not depend on the generated 313_run inventory. The evaluated
best_so_far checkpoint has 598 total iterations. It does not represent the
latest state of the continuing training job.

Required local inputs include the archived `eval/results/runs/` and earlier
`eval/sweep/` tables, plus outputs of the trajectory, current_search and
historical_comparison analyses linked below. Generated data is not included
in Git. Missing files fail with their path; do not substitute unmatched runs.

Run from the repository root in `sage`, after regenerating those analyses:

```bash
study=eval/sweep/rl_attempts_summary_20261023_201927
python "$study/scripts/summarize_attempts.py" \
  --historical_root /path/to/historical_archive \
  --current_root /path/to/current_archive \
  --history_dir /path/to/regenerated_history \
  --search_analysis_dir /path/to/regenerated_search_analysis \
  --trajectory_analysis_dir /path/to/regenerated_trajectory_analysis \
  --output_dir "$study/results/reproduction_01"
```

All output paths must be new. Generated tables and `summary.json` record input
hashes and statistical definitions. Reusing the same polytopes across settings
does not create independent observations; retrospectively chosen coefficients
are not independent validation. Historical step_reward + value results retain
their recorded scoring semantics even where algorithm names were later reused.

Prerequisites: [trajectory analysis](../two_face_trajectory_analysis_20261018/README.md),
[current search](../best_so_far_search_20261022_194142/README.md),
[historical comparison](../best_so_far_history_comparison_20261022_211226/README.md).
