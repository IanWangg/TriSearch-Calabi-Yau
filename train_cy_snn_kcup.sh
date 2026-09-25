#!/usr/bin/env bash
# Train on 4D h11=15 polytopes (CY threefolds).
# max_kcup uses log(V_next) - log(V_current); metrics report raw Kcup volume.
# Usage: bash train_cy_snn_kcup.sh [training arguments]
# Example: bash train_cy_snn_kcup.sh --num_iterations 100 --gpu_index 1
# Optional experiment tracking: append --use_wandb.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/yiranwang/anaconda3/envs/sage/bin/python}"
for arg in "$@"; do
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    exec "${PYTHON_BIN}" scripts/train_cy.py --help
  fi
done

RUN_ID="${RUN_ID:-cy_snn_kcup_h11_15_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/${RUN_ID}}"
mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd)"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/wandb"
export WANDB_DIR="${RUN_DIR}/wandb"
export PYTHONUNBUFFERED=1

# Arguments supplied by the caller override the defaults below.
"${PYTHON_BIN}" scripts/train_cy.py \
  --dataset_path data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl \
  --in_channels 4 \
  --subcomplex_actor_type snn_simplex \
  --neighbor_mode two_neighbors \
  --no-include_points_interior_to_facets \
  --reward_function max_kcup \
  --seed 0 \
  --num_iterations 1000 \
  --num_epochs 1 \
  --num_states 128 \
  --rollout_length 20 \
  --batch_size 128 \
  --lr 0.0001 \
  --use_multiprocessing \
  --memory_budget_gb 128 \
  --runtime_cache_gb 32 \
  --num_eval_polytopes 100 \
  --num_eval_states 128 \
  --eval_steps 30 \
  --eval_interval 100 \
  --deterministic_eval \
  --checkpoint_path "${RUN_DIR}/checkpoints" \
  --iteration_metrics_path "${RUN_DIR}/iteration_metrics.jsonl" \
  --latest_checkpoint_interval 10 \
  --save_interval 100 \
  --wandb_project calabi_yau_snn_kcup \
  --use_wandb \
  --name_suffix "${RUN_ID}" \
  "$@" \
  2>&1 | tee "${RUN_DIR}/train_performance.log"
