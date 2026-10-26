# TriSearch Calabi-Yau

This repository is the Calabi-Yau extraction.

## Environment

Create the project environment from the checked-in YAML. It mirrors the
Linux/CUDA 11.8 `sage` environment used for development and installs this
repository in editable mode.

```bash
conda env create -f environment.yml
conda activate trisearch_calabi_yau
python scripts/train_cy.py --help
```

To update an environment that was already created from the file:

```bash
conda env update -f environment.yml --prune
```

Do not install `requirements.txt` into a plain Python environment as a
substitute for the YAML. The training and geometry code expects Sage, the
matching CUDA/PyTorch/PyG stack, and the Python packages to be available in the
same environment.

CYTools regularity checks default to Mosek. Configure the license/backend with environment variables when needed:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
export CYTOOLS_REGULARITY_BACKEND=mosek
```

If `MOSEKLM_LICENSE_FILE` is not set, `core/cytools_config.py` falls back to `/home/yiranwang/mosek/mosek.lic` only when that file exists.

## Included Data And Checkpoint

Curated runnable artifacts are included:

- 3D sample dataset: `data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl`
- 4D sample dataset: `data/cy/output4d/cy4d_random_flip_100_3_random_flip.samples.jsonl`
- Compact two-neighbor h11=12 experiment dataset: `data/cy/two_neighbors_h11_12.samples.jsonl`
- 4D GNN policy checkpoint: `ckpt/cy_subcomplex_ppo_improved_512state_20rollout_actor_gnn_rollout_aug_count_bonus0p1_exp0p5_randomflipdata_d4/final.pth`
- SNN policy checkpoint: `ckpt/cy_train_h15_snn/latest.pth`
- K3 source data: `cy_data/k3.txt`

Bulk generated outputs from the source repo are intentionally not copied.

## Common Commands

CY policy APIs default to the simplex-topology-aware `snn_simplex` actor and
value features. Pass `--subcomplex_actor_type gnn` when training or loading a
legacy GNN checkpoint.

Training and rollout now isolate geometry in managed workers, bound retained
graphs and tensor caches, and keep exact visitation history on disk. `main.sh`
enables parallel geometry with a 64 GiB job budget and a 16 GiB retained-cache
allowance. Canonical action ordering is the new default; it preserves geometry
but can change trajectories relative to earlier native-order seeded runs.
See [managed rollout runtime](docs/cy_managed_runtime.md) for controls, process
cleanup guarantees, memory accounting, and reproducible benchmarks.

### Two_face model training

Use [train_cy_two_face_kcup.sh](train_cy_two_face_kcup.sh) to train
`two_face_deep_sets` on the local 4D h11=15 dataset:

```bash
conda activate sage
bash train_cy_two_face_kcup.sh --gpu_index 0 --seed 0
```

The launcher uses 64 channels and 3 point EGNN layers, followed by triangle
readout and Deep Sets pooling. Training uses
128 environments, 20 rollout steps, 1000 PPO iterations, and learning rate
`0.0001`. It uses `two_neighbors` and `max_kcup`, with raw coordinates,
augmentation disabled and count bonus zero. The default input is
`data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl`;
pass `--dataset_path` for another 4D dataset and adjust `--num_eval_polytopes`
for small inputs. Training defaults to a 128 GiB job budget and 32 GiB cache
allowance; override `--memory_budget_gb` and `--runtime_cache_gb` as needed.

Every launch creates a named directory under `runs/`, containing `checkpoints/`,
`iteration_metrics.jsonl`, `logs/commands.log`, and `logs/train_performance.log`.
The command log records the actual arguments; performance logs append when a
run directory is reused. Set `RUN_ID` or `RUN_DIR` to name the output explicitly.
An existing checkpoint directory reloads weights with a fresh optimizer and
iteration counter; `iteration_metrics.jsonl` is rewritten for each launch.
Append `--use_wandb` to enable Wandb or use `--help` for all
options. Paired SNN comparisons and stress commands are documented in the
[two_face study](eval/sweep/two_face_model/README.md).

The encoder and reward are independent switches:

| Option | Values | Default for fresh training |
| --- | --- | --- |
| `--two_face_encoder` | `deep_sets`, `triangle_dual` | `deep_sets` |
| `--two_face_triangle_layers` | Positive integer, only for `triangle_dual` | 2 |
| `--training_reward_mode` | `step_log_volume`, `best_so_far_log_volume` | `step_log_volume` |

For the original triangle_dual network, append `--two_face_encoder triangle_dual`.
Both encoders support either reward. The generic training default actor remains
`snn_simplex`; two_face encoder flags require `--subcomplex_actor_type two_face_deep_sets`.
`--two_face_state` independently controls evaluation search identity.
Training reloads and evaluation infer omitted encoder options from checkpoint
metadata: historical `triangle_num_layers` identifies the original dual network,
while its absence identifies the simplified encoder. Explicit incompatible
options fail. Loading remains strict and does not rewrite existing sidecars.

Published research sources: [trajectory analysis](eval/sweep/two_face_trajectory_analysis_20261018/README.md),
[best_so_far search](eval/sweep/best_so_far_search_20261022_194142/README.md),
[historical comparison](eval/sweep/best_so_far_history_comparison_20261022_211226/README.md),
and [RL attempts](eval/sweep/rl_attempts_summary_20261023_201927/README.md).
Generated data and model weights remain local; each study documents its inputs.

### Best_so_far reward

Use the finite-episode variant with the same dataset and resource controls:

```bash
conda activate sage
bash train_cy_two_face_best_so_far_kcup.sh --gpu_index 0 --seed 0
```

This sets `--training_reward_mode best_so_far_log_volume`, `--gamma 1`,
`--rollout_length 128`, and `--eval_steps 128`. Other launcher defaults are
unchanged; caller arguments override them. Each environment tracks its own
best volume, initialized from its starting state. The training reward is
`log(best_next) - log(best_current)`: returning to an earlier record earns zero.
The total episode reward is `log(best_final / initial_volume)`. This optimizes
expected best log volume; improvements in raw volume means or baseline win
rates must be measured separately.

An episode ends at its horizon or a dead end, with zero bootstrap. Finished
slots are not reset within that rollout, and padding is excluded from PPO
losses and sample statistics. Both actor and critic receive
`[log(current_volume), log(best_volume / current_volume), log(1 + remaining_steps)]`
through a shared context MLP after the geometric encoder. Context is stored
with each sampled observation, separately from cached geometry.

Finite-episode training and periodic evaluation submit initial volumes and each
step's selected destinations as ordered batches to the existing geometry worker
pool. Known current volumes are reused; unselected neighbors and inactive slots
are never queried. Batching preserves trajectory order, rewards and cache
identity. `objective_query_sec` in rollout logs and iteration metrics measures
the initial and destination objective queries, separately from PPO updates.
`tools/benchmark_cy_volume_queries.py --output_dir <new_directory>` compares
scalar and batched queries in fresh runtimes with identical seeded rollouts;
it checks actions, terminal states, volumes, rewards and policy outputs before
reporting speedup. Choose a free GPU with `--gpu_index` and store smoke artifacts
under `eval/smoke_tests/results/`.

The new observation schema is version 2. Checkpoints record the context fields,
training horizon, reward mode and critic target (expected remaining improvement
in best log volume); old checkpoints retain their existing architecture and
schema. Use a fresh run directory when changing modes or the training horizon.
Training metrics retain raw initial/final/best volumes and add the reward mode,
episode lengths and termination reasons. `return.training_mean` contains the
new training return; `return.mean` retains the original step-reward diagnostic.

External evaluation supports these checkpoints with the RL walk, policy beam,
value beam and value BeFS algorithms and `--subcomplex_actor_type two_face_deep_sets`;
use `--objective_budget 128`
to match the default training horizon and `--policy_checkpoint` to select the
new checkpoint directory. Context comes from each evaluation session's known
objectives and remaining logical queries, including a free initial objective.
The existing evaluation no-revisit rule remains in effect, whereas training
can revisit states. Evaluation does not query unselected candidates to build
context. Value BeFS/Beam score nodes by `log(best_volume) + value_discount * V`
using the latest session record and remaining query budget; queued BeFS nodes
are rescored after each round without further objective queries. Use
`--value_discount 1` for the unweighted expected final best log volume heuristic.
Policy proposals use context at the start of the scheduling round. This adapts
a critic trained on single walks to tree search and must be validated empirically.
See [evaluation instructions](eval/README.md)
for setup and output-directory options.

### Training smoke

One-iteration training smoke:

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --max_rows 4 \
  --num_eval_polytopes 1 \
  --num_iterations 1 \
  --num_epochs 1 \
  --num_states 2 \
  --rollout_length 1 \
  --num_eval_states 2 \
  --eval_steps 1 \
  --batch_size 2 \
  --force_cpu \
  --checkpoint_path /tmp/trisearch_cy_smoke_ckpt \
  --dry_run
```

To optimize the number of simplices over the reachable regular-triangulation
graph, use `--reward min_tri` to minimize or `--reward max_tri` to maximize.
Both objectives use the signed change in simplex count as their dense reward.
Unlike CY sampling mode, fine regular targets remain traversable and
one-simplex destinations receive their objective reward before ending as
no-action states.

```bash
python scripts/train_cy.py \
  --reward min_tri \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --max_rows 4 \
  --num_eval_polytopes 1 \
  --num_iterations 1 \
  --num_epochs 1 \
  --num_states 2 \
  --rollout_length 1 \
  --num_eval_states 2 \
  --eval_steps 1 \
  --batch_size 2 \
  --deterministic_rollout \
  --deterministic_eval \
  --force_cpu \
  --checkpoint_path /tmp/trisearch_cy_min_tri_smoke \
  --dry_run
```

Checkpoint evaluation smoke:

```bash
python scripts/eval_cy.py \
  --dataset_path data/cy/output4d/cy4d_random_flip_100_3_random_flip.samples.jsonl \
  --checkpoint_path ckpt/cy_subcomplex_ppo_improved_512state_20rollout_actor_gnn_rollout_aug_count_bonus0p1_exp0p5_randomflipdata_d4/final.pth \
  --subcomplex_actor_type gnn \
  --max_rows 4 \
  --num_eval_polytopes 1 \
  --eval_steps 1 \
  --force_cpu \
  --summary_path /tmp/trisearch_cy_eval_summary.json
```

Objective checkpoints use the same model format, so pass the objective again
when evaluating. The saved summary includes initial, final, and best simplex
counts for every trajectory.

```bash
python scripts/eval_cy.py \
  --reward min_tri \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --checkpoint_path /tmp/trisearch_cy_min_tri_smoke/final.pth \
  --max_rows 4 \
  --num_eval_polytopes 1 \
  --eval_steps 2 \
  --deterministic_eval \
  --force_cpu \
  --summary_path /tmp/trisearch_cy_min_tri_smoke/eval_summary.json
```

### FRST-only CY volume optimization

Use `--neighbor_mode two_neighbors` to navigate only between CYTools FRST
representatives whose 2-face restrictions differ by one diagonal flip. This
mode requires `--no-include_points_interior_to_facets`, matching the point
configuration on which CYTools constructs two-neighbor representatives. The
model action is the changed four-vertex 2-face circuit; the rollout state still
stores the complete FRST representative returned by CYTools.

`max_cy_volume` maximizes the CY threefold volume at the stretched-cone tip
computed from
`cy.mori_cone_cap(in_basis=True).dual().tip_of_stretched_cone(c=1)`. The dense
reward defaults to the raw potential difference `V(next_state) - V(state)`.
Pass `--cy_volume_reward_transform log` to train on
`log(V(next_state)) - log(V(state))`; raw Kcup volumes remain the metric shown
in console and JSON summaries.

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/two_neighbors_h11_12.samples.jsonl \
  --neighbor_mode two_neighbors \
  --no-include_points_interior_to_facets \
  --reward max_cy_volume \
  --cy_volume_reward_transform log \
  --num_eval_polytopes 4 \
  --num_states 32 \
  --rollout_length 5 \
  --seed 0 \
  --force_cpu \
  --checkpoint_path /tmp/trisearch_cy_volume
```

Kcup is used instead of `toric_kahler_cone()` because its value is invariant
across complete FRST representatives with the same 2-face restriction. The
toric-cone construction is cheaper, but can assign different values to those
representatives and therefore is not a well-defined objective on the
two-face-equivalence state space. CYTools currently labels the two-neighbor and
non-favorable CY paths as experimental; failures are surfaced directly, with
no alternate volume formula or toric-cone fallback.

To reproduce the `cyopt` objective instead, use `--reward max_toric_cy_volume`.
Its reported objective is
`log10(cy.compute_cy_volume(cy.toric_kahler_cone().tip_of_stretched_cone(c=1)))`,
and its dense transition reward is the next-state value minus the current-state
value. This reward deliberately uses the full toric Kähler cone and does not
accept `--cy_volume_reward_transform log`, because the base-10 logarithm is
already part of the objective:

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/two_neighbors_h11_12.samples.jsonl \
  --neighbor_mode two_neighbors \
  --no-include_points_interior_to_facets \
  --reward max_toric_cy_volume \
  --num_eval_polytopes 4 \
  --num_states 32 \
  --rollout_length 5 \
  --seed 0 \
  --force_cpu \
  --checkpoint_path /tmp/trisearch_max_toric_cy_volume
```

Unlike `max_cy_volume`, this exact `cyopt` objective can vary between complete
FRST representatives with the same 2-face restrictions.

Use `--reward min_mori_cap` to minimize
`float(len(cy.mori_cone_cap(in_basis=True).extremal_rays()))`.
The reported objective is the raw extremal-ray count, and the transition reward
is the current count minus the next count, so reducing the count gives positive
reward. This objective uses complete triangulation states.

Pass `--mori_cap_backend mosek` to training or evaluation to compute the same
objective with reusable, single-thread MOSEK primal-simplex LPs. The backend
checks reconstruction residuals and infeasibility certificates, retries an
inconclusive solve once, and preserves CYTools' lineality decomposition for
non-pointed cones. The general CLI default remains `cytools`; the dedicated
`train_cy_snn_min_mori_cap.sh` launcher selects `mosek`. The selected backend is
recorded in W&B/configuration and included in worker/engine objective cache keys.
Existing training processes retain their loaded backend until restarted.

For a cold-geometry comparison of both simplex variants against CYTools, run
`python tools/benchmark_mori_extremal_rays.py --cpu_id 16 --output_dir runs/mori_lp_benchmark`.
The benchmark verifies normalized ray-set equality and separately records cap
construction, extremal-ray extraction, and complete objective timings.

`train_cy_snn_min_mori_cap.sh incremental` trains with the per-step count decrease;
`train_cy_snn_min_mori_cap.sh best_so_far` rewards only new episode minima.
The latter conditions the SNN actor and critic on the current count, gap to the
best count, and remaining steps, uses a finite rollout with zero terminal
bootstrap and `gamma=1`, and adds the exploration bonus only to training rewards.
Both launchers use the h11=15 baseline dataset, 64-step rollouts, online W&B,
entropy/count bonus coefficients of 0.001, 16 geometry workers, and a 48 GiB
process-tree memory budget. Incremental training keeps the baseline `gamma=0.95`.
Caller arguments override defaults; use a detached session for unattended runs.

## Training Logs

Training reports one cumulative return summary after each complete rollout:

```text
Rollout: return=3.2734 return_std=1.2040 return_min=0.0000 return_max=6.0000 ...
```

For each parallel rollout slot, `return` sums the undiscounted extrinsic rewards
over the full configured rollout horizon, including steps after a terminal
reset. The displayed value is the mean across rollout slots; `return_std`,
`return_min`, and `return_max` describe the same distribution. When count-based
exploration is enabled, `training_return` additionally includes the intrinsic
bonus. The older `discounted_reward` remains available as a first-episode
diagnostic, but it is not the primary cumulative-return metric.

Use `--use_wandb` for online experiment tracking and `tee` for a persistent
local performance log. Do not combine this with `--dry_run`, which disables
W&B.

```bash
RUN_ID="max_tri_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/wandb" "${RUN_DIR}/checkpoints"
set -o pipefail

WANDB_MODE=online \
WANDB_DIR="${RUN_DIR}/wandb" \
PYTHONUNBUFFERED=1 \
python scripts/train_cy.py \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --reward max_tri \
  --num_iterations 1000 \
  --num_states 128 \
  --rollout_length 20 \
  --use_multiprocessing \
  --memory_budget_gb 64 \
  --runtime_cache_gb 16 \
  --batch_size 128 \
  --num_eval_polytopes 20 \
  --num_eval_states 128 \
  --eval_steps 30 \
  --eval_interval 100 \
  --checkpoint_path "${RUN_DIR}/checkpoints" \
  --latest_checkpoint_interval 10 \
  --save_interval 500 \
  --use_wandb \
  --wandb_project calabi_yau_max_tri \
  --name_suffix "${RUN_ID}" \
  2>&1 | tee "${RUN_DIR}/train_performance.log"
```

W&B records the primary metric as `rollout/return`, its distribution under
`rollout/return_std`, `rollout/return_min`, and `rollout/return_max`, and held-out
statistics under `eval/return_mean`, `eval/return_std`, `eval/return_min`, and
`eval/return_max`.

For crash-resilient local tracking, `--iteration_metrics_path PATH.jsonl`
writes and flushes one record after every PPO iteration. Each max-CY-volume
record contains every train and held-out slot's raw initial, final, and best
volume, best-volume improvement, aggregate volume statistics, return
statistics, PPO losses, and timing.

Random rollout sampling:

```bash
python tools/rollout_cy_random.py --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl --dry_run
```

## 4D Data Generation

Run every command in this section from the repository root after activating
the project environment. Four-dimensional reflexive polytopes produce
Calabi-Yau threefolds, which are the inputs required by `max_cy_volume`.

```bash
conda activate trisearch_calabi_yau
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
export CYTOOLS_REGULARITY_BACKEND=mosek
```

The Mosek Python package and the license are separate: every machine must have
its own valid license. CYTools fetching also requires network access. Use
`--help` to see every generator option:

```bash
python data/cy/generate_4d_dataset.py --help
python data/cy/generate_eval_dataset.py --help
```

### Generate an FRST dataset for volume optimization

`generate_4d_dataset.py` fetches 4D N-lattice reflexive polytopes with
`cytools.fetch_polytopes`, generates FRST seeds, and optionally collects nearby
non-fine regular triangulations. The following small seeded run requests
favorable polytopes with `h11=12` and uses the point configuration required by
CYTools two-neighbor navigation:

```bash
python data/cy/generate_4d_dataset.py \
  --num-polytopes 12 \
  --h11 12 \
  --favorable \
  --frsts-per-polytope 1 \
  --num-triangulations-per-frst 1 \
  --no-include-points-interior-to-facets \
  --bfs-max-depth 2 \
  --bfs-max-nodes 100 \
  --num-workers 4 \
  --seed 0 \
  --compact-output \
  --output-dir data/cy/output4d_volume \
  --output-name cy4d_h11_12_frst
```

The requested counts are upper bounds. CYTools may return fewer polytopes,
FRSTs, or nearby triangulations, and the final console summary reports the
actual counts. `--compact-output` keeps the main training artifact only:

```text
data/cy/output4d_volume/cy4d_h11_12_frst.samples.jsonl
```

The same directory also contains
`cy4d_h11_12_frst.checkpoint.json`. If generation is interrupted, rerun the
same command with `--resume`; completed polytope rows will not be regenerated.
Without `--compact-output`, the generator additionally writes a larger
`cy4d_h11_12_frst.json` file containing dataset metadata.

By default, nearby non-fine states are collected with bounded BFS. Pass
`--collection-depths 1 2` to retain only selected depths, or `--random-flip`
to replace BFS collection with regular random flips. Two-neighbor volume
training starts from the generated `frst_list`, so the
`--no-include-points-interior-to-facets` flag is the important compatibility
requirement for that workflow.

To smoke-test volume optimization on the generated file:

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/output4d_volume/cy4d_h11_12_frst.samples.jsonl \
  --neighbor_mode two_neighbors \
  --no-include_points_interior_to_facets \
  --reward max_cy_volume \
  --cy_volume_reward_transform log \
  --max_rows 2 \
  --num_eval_polytopes 1 \
  --force_cpu \
  --checkpoint_path /tmp/trisearch_cy_volume_smoke \
  --dry_run
```

After this succeeds, remove `--dry_run` and choose the production rollout,
evaluation, and checkpoint settings described in the earlier FRST-only volume
optimization section.

### Reuse a fixed 4D polytope set

For an exactly shared set of polytopes, pass a JSON or JSONL file instead of
fetching again. Each record must contain `vertices`, `n_points`, or
`n_vertices`; `polytope_index`, `h11`, `favorable`, and
`requested_num_vertices` are optional. A previously generated
`.samples.jsonl` file can also be reused directly because it contains
`vertices`:

```bash
python data/cy/generate_4d_dataset.py \
  --polytope-file data/cy/fixed_4d_polytopes.jsonl \
  --num-polytopes 12 \
  --frsts-per-polytope 1 \
  --num-triangulations-per-frst 1 \
  --no-include-points-interior-to-facets \
  --num-workers 4 \
  --seed 0 \
  --compact-output \
  --output-dir data/cy/output4d_fixed \
  --output-name cy4d_fixed_frst
```

When `--polytope-file` is supplied, `--num-polytopes` is only an optional cap;
the fetch filters such as `--h11`, `--num-vertices`, and `--favorable` are not
used.

### Generate ordinary 4D evaluation states

`generate_eval_dataset.py` fetches 4D polytopes, samples random heights, and
keeps unique non-fine regular triangulations. This is useful for evaluating
ordinary `neighbor_mode=regular` policies; it does not generate the FRST seed
dataset used by two-neighbor volume optimization.

```bash
python data/cy/generate_eval_dataset.py \
  --num-polytopes 4 \
  --h11 12 \
  --favorable \
  --num-triangulations 8 \
  --max-tries 100 \
  --num-workers 4 \
  --seed 1 \
  --compact-output \
  --output-dir data/cy/output4d_eval \
  --output-name cy4d_h11_12_eval
```

Generated `data/cy/output*` directories are ignored by Git because the files
can become large. To give a generated dataset to a collaborator, explicitly
publish or transfer the resulting `.samples.jsonl` file; pushing the code
alone will not include newly generated output.

## Tests

Run the focused CY tests from the repo root:

```bash
pytest test
```
