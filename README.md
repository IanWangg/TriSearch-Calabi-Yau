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
- 4D policy checkpoint: `ckpt/cy_subcomplex_ppo_improved_512state_20rollout_actor_gnn_rollout_aug_count_bonus0p1_exp0p5_randomflipdata_d4/final.pth`
- K3 source data: `cy_data/k3.txt`

Bulk generated outputs from the source repo are intentionally not copied.

## Common Commands

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
