# TriSearch Calabi-Yau

This repository is the Calabi-Yau extraction.

## Environment

Use the existing `sage` conda environment. Sage itself must be provided by the
environment; the Python package requirements, including CYTools, Mosek, and the
PyTorch stack, are listed in `requirements.txt` and `setup.py`.

```bash
conda activate sage
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train_cy.py --help
```

Do not install the requirements into a plain Python environment as a substitute
for Sage. The training and geometry code expects both Sage and the packages
above to be available in the same environment.

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

Dataset generation entrypoints:

```bash
python data/cy/generate_dataset.py --help
python data/cy/generate_4d_dataset.py --help
python data/cy/generate_eval_dataset.py --help
python data/cy/generate_k3_eval_dataset.py --help
```

## Tests

Run the focused CY tests from the repo root:

```bash
pytest test
```
