# Checkpoint 600 evaluation on h11=50

Evaluate the entropy_coef_0_01 and num_states_512 checkpoints at iteration 600, each with rl_value_beam_search and rl_value_best_first. The user confirmed the num_states_512 path after the original request repeated the entropy checkpoint directory.

| Model | Iteration 600 checkpoint | SHA256 |
| --- | --- | --- |
| Entropy 0.01 | `runs/cy_snn_kcup_h11_15_entropy_coef_0_01_20260926_123537_2829755/checkpoints/600.pth` | `669df736229c6f4b58247db7196e2ad198a7f9f807e88612c78f330da9c86925` |
| Parallel env 512 | `runs/cy_snn_kcup_h11_15_num_states_512_20260926_123528_2829301/checkpoints/600.pth` | `0c90e10717ac95e98a97b5ee6b861c7a680050ddf98bb2ac133b2e316a889724` |

Reuse eval/data/setups/h11_50_5_polytopes_1_start_seed_0: five h11=50 polytopes, one identical start per polytope across all four jobs, seed 0, 2000 logical objective queries per trajectory. Fixed search settings: beam_width=4, policy_proposal_count=4, value_discount=0.9, max_kcup. Score: ln(volume_child) + 0.9 * critic(child). Complete the last admitted parent expansion; report endpoints using only query_index <= 2000.

## Search logic

The rules are defined in [rl.py](../../algorithm/rl.py) and executed by [batched_rollout.py](../../batched_rollout.py). Both methods enumerate FRST neighbors, discard already queried states and duplicate targets, then propose up to four unvisited candidates in actor probability order. They evaluate each proposal's actual volume and critic value. Ranking uses the absolute child score above; the parent score and actor probability are not accumulated into this score.

- **Beam:** expand the current layer's beam parents, pool their proposed children, and retain the four highest scoring children for the next layer. Pruned states stay in the queried-state set and are not reconsidered.
- **BeFS:** retain proposed children on a persistent global priority queue and expand its highest scoring node each round. It uses four proposals per parent; beam_width does not limit its frontier. Tied scores use query order.

The initial volume is query 0 and is uncharged. Enumeration and cached score reuse do not consume objective queries. A new proposed state's logical query is charged even when the geometry runtime serves it from cache. The final admitted parent expansion is completed, so raw runs may exceed the budget by at most three queries; the comparison curves and budget endpoints exclude that overshoot. Each trajectory has an independent budget and queried-state set. All queried candidates contribute to the tracked best volume, including candidates not retained in the frontier.

The initial CUDA probe with all GPUs visible blocked in the host driver. Exposing only one authorized GPU per process restored CUDA. A real GPU search smoke passed and matched all 45 initial/query records against the CPU prefix. The formal jobs use physical GPUs 3, 4, 5 and 6, respectively, each exposed as logical cuda:0. CPU attempts are preserved and excluded from formal results. Jobs have disjoint CPU affinity: eight physical cores and their two threads per job, 15 geometry workers, 32 GiB process tree budget and 8 GiB runtime cache per job. No installed packages or search implementations are modified; no GPU reset is performed.

The original probe PID 3631769 remained in uninterruptible driver sleep with SIGKILL pending at evaluation completion. All four formal evaluation processes and the analysis process exited; the driver-blocked probe did not prevent their successful CUDA execution. Its cleanup state is recorded in experiment_notes.json.

See experiment_notes.json for exact checkpoint and setup hashes, configurations, commands, CPU sets, and run/log paths. Each result directory contains the standard evaluation logs, summary and per-job plots. Cross-checkpoint comparisons are saved under plots/.

## Reproduction

The formal commands are saved in [run_20261005_184214.sh](scripts/run_20261005_184214.sh), with the four `*_gpu.json` configurations in [configs/](configs/). Their GPUs are, in order, entropy Beam = 3, entropy BeFS = 4, parallel env Beam = 5, parallel env BeFS = 6. For a new evaluation, copy these commands and choose new output and log directories so existing results are preserved. The CPU configurations and earlier launcher describe the interrupted CPU attempt and are excluded from the comparison.

Offline validation and comparison can be repeated from the repository root without running geometry:

```bash
CUDA_VISIBLE_DEVICES='' /home/yiranwang/anaconda3/envs/sage/bin/python -B \
  eval/sweep/checkpoint_600_h11_50/scripts/analyze_results.py
```

The analyzer verifies all 20 paired trajectories, checkpoint/setup/source hashes, CUDA inference, query deduplication, expansion counts and budget accounting. It writes absolute volume endpoints, checkpoint win/tie comparisons, PNG/PDF search curves and `validation.json`. `scripts/solver_bootstrap/sitecustomize.py` was used only for an optional solver thread smoke; it is not enabled in the formal runs.

## Ordinary BeFS baseline comparison

Reuse the completed `best_first` run at `eval/results/runs/h11_50_metric_value_policy_top_4_20260927_044127/algorithms/best_first`. Its setup ID, five initial state keys and volumes, scientific settings and 2000-query endpoints match this study. Ordinary Beam is omitted at the user's request. No search or geometry is rerun.

Ordinary BeFS enumerates all unvisited neighbors and ranks its persistent global frontier by absolute volume. The RL methods use actor top-four proposals and the metric-plus-critic score described above. Query deduplication and complete-parent budget accounting are validated for the reused baseline; its overshoot is excluded from endpoints.

| RL method | Wins against ordinary BeFS | Losses |
| --- | ---: | ---: |
| Entropy RL Beam | 0 | 5 |
| Entropy RL BeFS | 1 | 4 |
| Parallel env 512 RL Beam | 2 | 3 |
| Parallel env 512 RL BeFS | 1 | 4 |

Across these 20 paired comparisons, ordinary BeFS wins 16; there are no ties. The 512-env RL Beam wins on polytopes 0 and 1; entropy RL BeFS wins on polytope 4. Ordinary BeFS achieves the highest endpoint among all five methods on polytopes 2 and 3.

[Five-method search curves](plots/baseline_comparison/best_volume_by_polytope.png) · [PDF](plots/baseline_comparison/best_volume_by_polytope.pdf) · [All endpoints](plots/baseline_comparison/endpoints.csv) · [Paired comparisons](plots/baseline_comparison/paired_vs_befs.csv) · [Provenance and validation](plots/baseline_comparison/plot_config.json)

Reproduce this offline comparison with:

```bash
CUDA_VISIBLE_DEVICES='' /home/yiranwang/anaconda3/envs/sage/bin/python -B \
  eval/sweep/checkpoint_600_h11_50/scripts/compare_baseline.py
```

## Completed results

Budget endpoints exclude queries above 2000.

| Polytope | Entropy Beam | Entropy BeFS | Parallel env Beam | Parallel env BeFS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6.3780958e+08 | 5.7261104e+08 | 6.0179583e+10 | 1.0595122e+08 |
| 1 | 23676464 | 4.380887e+09 | 1.0593559e+10 | 9.4550052e+09 |
| 2 | 4.0826593e+10 | 8.0573986e+10 | 5.7425794e+10 | 1.440742e+10 |
| 3 | 7.8410703e+08 | 3.4979093e+10 | 1.2468873e+13 | 2.5463127e+11 |
| 4 | 8.1619313e+08 | 1.2991566e+11 | 1.2218274e+09 | 3.3058151e+09 |

[Paired endpoints](plots/paired_endpoints.csv) · [Checkpoint comparison](plots/checkpoint_comparison.csv) · [Search curves](plots/best_volume_by_polytope.png) · [Validation](plots/validation.json)
