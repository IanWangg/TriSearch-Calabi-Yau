# Two_face model comparison

The first comparison changes the network from `snn_simplex` to
`two_face_deep_sets`, with the same training split, PPO parameters, seeds, raw
coordinates and zero count bonus. Both evaluations use `two_face_state=true`,
`rl_value_best_first`, 4 proposals, value discount 0.9, and 1000 logical queries
per start. Defaults select 5 held-out h11=15 polytopes and 1 start each. Larger
h11 generalization and the local-actor ablation are separate follow-up studies.
These are runnable experiment templates, not claims of improved performance.

Run from the repository root with `conda activate sage`. The large dataset is
ignored by Git and is not copied into a new worktree. Set `dataset_path` to the
same existing input file for both models; the default training input is
`data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl`
in the original checkout.

```bash
dataset_path=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl
RUN_ID=snn_seed_0 bash train_cy_two_face_kcup.sh \
  --dataset_path "$dataset_path" --subcomplex_actor_type snn_simplex --seed 0
RUN_ID=two_face_seed_0 bash train_cy_two_face_kcup.sh \
  --dataset_path "$dataset_path" --seed 0
```

The shared launcher fixes hidden/out channels to 64, point layers to 3,
128 environments, rollout length 20, one PPO epoch per iteration, minibatch
size 128, learning rate 0.0001, and 100 held-out polytopes. Other PPO settings
retain the training CLI defaults. It writes weights, metrics and logs to the
named `runs/` directory. Wandb is opt-in through `--use_wandb`. Use new run names
for each training seed; seeds 0, 1 and 2 form the initial multi-seed protocol.
Record exact checkpoint iterations when comparing models. New two-face weights
require their adjacent `model_config.json`; optimizer/iteration state is not saved.

Export held-out inputs with the same splitter used by training, then prepare one
shared setup. The export records the source hash and all split indices. The
evaluation loader independently validates N-lattice geometry and h11. Existing
export directories are rejected to prevent accidental replacement.

```bash
python eval/sweep/two_face_model/scripts/prepare_inputs.py --dataset_path "$dataset_path"
python scripts/eval_cy.py --config eval/sweep/two_face_model/configs/two_face.json \
  --setup_only --output_dir eval/data/setups/two_face_h11_15_seed_0
```

Run each model on that setup with distinct result directories. The example uses
CPU; remove `--force_cpu` and select an available `--gpu_index` for GPU timing.

```bash
study_run_id=$(date +%Y%m%d_%H%M%S)
mkdir -p eval/sweep/two_face_model/logs
for model in snn two_face; do
  python scripts/eval_cy.py --config "eval/sweep/two_face_model/configs/${model}.json" \
    --setup_path eval/data/setups/two_face_h11_15_seed_0 --force_cpu \
    --output_dir "eval/sweep/two_face_model/results/${model}_${study_run_id}" \
    --plot_results > "eval/sweep/two_face_model/logs/${model}_${study_run_id}.log" 2>&1
done
```

Use `--policy_checkpoint` for a different iteration or training seed. Search
seed, setup, proposal count, discount and budget must remain paired. To isolate
the critic, repeat both runs with `--policy_proposal_count -1` in new directories.
The same two-face model can also run with `--no_two_face_state` to test the
independence of network observation and search identity.

Compare budget-limited best raw Kcup volume and complete wall time, parameter
counts, data construction/inference timings, CUDA peak allocation and retained
cache statistics. Each run's standard plots and tables are generated separately:
the existing cross-run algorithm merger intentionally rejects different policy
checkpoints. Do not relax that guard to combine models with the same algorithm.
The inference timings are host-side unless `--profile_cuda_timing` is enabled.

Disposable smoke outputs belong under `eval/smoke_tests/`. Run a one-iteration
CLI smoke using the tracked `data/cy/two_neighbors_h11_12.samples.jsonl` with
two input rows, one held-out polytope, two environments and two rollout steps;
set both checkpoint intervals to 1. `--dry_run` disables checkpoint saving and
therefore does not validate the save/load cycle. Automated coverage is in
`test/test_cy_two_face_policy.py`.

## Stress checks

`scripts/stress_test.py` runs deterministic CPU checks at the production model
size (64 channels, 3 point layers, 2 triangle layers). Run it from the repository
root in the `sage` environment. Each invocation requires a new output directory
and writes incremental `report.json` evidence, including failures, timing and
peak process RSS. Synthetic face grids exercise tensor scaling; they are not
claimed to be realizable reflexive polytopes.

```bash
stress_run_id=$(date +%Y%m%d_%H%M%S)
mkdir -p eval/smoke_tests/logs
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for mode in tensors ppo geometry; do
  python eval/sweep/two_face_model/scripts/stress_test.py "$mode" \
    --output_dir "eval/smoke_tests/results/two_face_stress_${mode}_${stress_run_id}" \
    > "eval/smoke_tests/logs/two_face_stress_${mode}_${stress_run_id}.log" 2>&1
done
```

The tensor checks compare single-state, full and chunked batches up to 512
states; permute all index spaces 100 times; backpropagate through an 8,192-node,
14,400-triangle, 7,200-action observation; and churn 2,000 observation identities
under a 1 MiB cache allowance. The PPO check compares losses, gradients and SGD
updates on 288 samples (60 without actions), then checks observation release.
Pass `--checkpoint path/to/8.pth` to `tensors` to repeat these checks with trained
weights; otherwise the test uses a seeded initialization.
The geometry check walks 12 steps on each of the 12 tracked h11=12 polytopes,
checks every candidate against the real destination restriction, rebuilds after
cache clearing and verifies worker cleanup. Pass `--extra_dataset "$dataset_path"`
to `geometry` to additionally check the 8 input rows with most vertices (one
initial FRST each); that selection measures scale, not held-out performance.

For real continuous PPO, run the standard launcher with bounded caches and
small physical batches. This uses Adam and the real KCUP reward:

```bash
RUN_DIR="$PWD/eval/smoke_tests/results/two_face_stress_train_${stress_run_id}" \
  bash train_cy_two_face_kcup.sh \
  --dataset_path data/cy/two_neighbors_h11_12.samples.jsonl \
  --num_eval_polytopes 2 --num_iterations 12 --num_states 64 --rollout_length 16 \
  --num_epochs 2 --batch_size 128 --num_eval_states 8 --eval_steps 8 --eval_interval 4 \
  --save_interval 4 --latest_checkpoint_interval 1 --transition_num_workers 2 \
  --transition_mp_min_batch 1 --memory_budget_gb 12 --runtime_cache_gb .02 \
  --policy_max_graph_size 20000 --max_hot_states 64 --cache_prune_interval 1 \
  --shared_cache_max_entries 64 --force_cpu
```

Use a fixed checkpoint file, rather than a concurrently updated `latest.pth`,
for paired search checks. The following checks all four RL algorithms and then
critic-only BeFS on three shared h11=12 starts, with 100 queries per start and
both warm and disabled caches. It validates every result through the existing
offline reader and compares all query, expansion, transition and rollout
records; objective floats allow the existing `1e-6` solver tolerance.

```bash
python eval/sweep/two_face_model/scripts/stress_test.py search \
  --checkpoint "eval/smoke_tests/results/two_face_stress_train_${stress_run_id}/checkpoints/12.pth" \
  --objective_budget 100 \
  --output_dir "eval/smoke_tests/results/two_face_stress_search_${stress_run_id}" \
  > "eval/smoke_tests/logs/two_face_stress_search_${stress_run_id}.log" 2>&1
```

The tests use CPU explicitly. CUDA throughput, peak GPU memory, long training
convergence and multi-seed model comparisons require separate measurements.

CPU validation on 2026_10_07 passed all four modes and a repeat of `tensors`
using the iteration-8 checkpoint. The largest mixed batch contained 512 states,
71,944 nodes and 87,330 triangles. The real geometry walk checked 1,536 flips
across 178 states on 20 polytopes. Continuous Adam training completed 12 rounds
of 64 × 16 samples on h11=12 and 4 rounds of 32 × 8 samples on a 12-polytope
h11=15 subset: 13,312 rollout samples total, finite metrics, no worker restarts,
and about 1.44 GiB sampled peak job RSS. Paired cache-on/off searches agreed over
3,040 logical queries; the final iteration-12 checkpoint then completed another
60 queries. All saved checkpoints strict-loaded, encoder/actor/critic weights
changed during training, and final/latest weights matched the final iteration.
This validates execution and resource behavior at these loads, not convergence
or a quality advantage over the SNN baseline. CUDA was unavailable for this run
because existing processes were blocked in the NVIDIA UVM driver.
