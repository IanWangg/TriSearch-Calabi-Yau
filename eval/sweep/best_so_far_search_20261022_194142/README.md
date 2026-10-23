# Finite best_so_far search comparison

This frozen study evaluates the simplified `deep_sets` encoder trained with
`best_so_far_log_volume`. It compares ordinary BeFS/Beam with RL rollout,
policy Beam, value BeFS and value Beam, using identical full FRST starts and
two_face search identity. It measures logical objective queries, not equal
wall time. The budget is 128, beam width and proposal count are 4, critic
weight is 1, and search seed is 0.

Completed cohorts are h11=15 (5 held_out polytopes × 2 starts) and h11=30
(5 polytopes × 5 starts). `configs/evaluation.json` records the initially
requested 10 starts for h11=15; sampling could only provide 2 for one polytope,
so the completed run uses `configs/h11_15_two_starts.json`. That change is
explicit and does not fill missing starts. The h11=30 starts reuse the saved
source-order setup, with `skip_insufficient_starts=true`.

The frozen checkpoint file is named `iteration_592.pth`, with SHA256
`4e1355583bad3bec9c215ea96d28438f6e90add70e2a71f5920ce2b03ae34b0a`.
It has 598 total training iterations including its warm start. Training code
was `bca49ca`; evaluation code was `40cbb58`. Its critic predicts remaining
best_log_volume gain. Value search refreshes session context after each round;
it does not use the historical context_free critic formula.

Only source, configuration and documentation are published. Restore the local
archive under `eval/sweep/best_so_far_search_20261022_194142/` before offline
analysis. Raw results, shared setups, held_out input, weights and generated
plots/tables are separate local inputs or outputs. `configs/checkpoint_model_config.json`
records the frozen weight metadata; the matching sidecar must accompany weights.

From the repository root in `sage`, regenerate statistics in a new directory:

```bash
study=eval/sweep/best_so_far_search_20261022_194142
python "$study/scripts/analyze_results.py" h11_30 \
  --current_root /path/to/current_archive --output_dir "$study/analysis/reproduction_01"
```

Repeat with `h11_15` for the smaller cohort. Endpoints exclude budget overshoot;
the script validates native logs and pairing before computing means, paired
ratios and win/tie/loss counts. An optional summary averages equivalent initial
two_face classes before comparison, because FRST starts need not be distinct
two_face classes.

To run a new evaluation, supply local checkpoint and setup explicitly:

```bash
bash "$study/scripts/run_evaluation.sh" h11_30 /path/to/iteration_592.pth \
  /path/to/shared_setup "$study/results/reproduction_01" --force_cpu
```

The setup's saved scientific parameters must match the config, including the
h11=15 `polytope_file` path; use a CLI override matching the archived setup when
loading one. `configs/resources.json` is the original machine's parallel
allocation, including GPU indices. Adapt it before using `--parallel_resources`.
Save persistent command output under this study's `logs/` directory.
