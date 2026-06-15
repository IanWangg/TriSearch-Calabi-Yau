# TriSearch Calabi-Yau

This repository is the Calabi-Yau extraction.

## Environment

Use the existing `sage` conda environment. The repo expects Sage/CYTools and the PyTorch stack to be available there.

```bash
conda activate sage
python scripts/train_cy.py --help
```

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
