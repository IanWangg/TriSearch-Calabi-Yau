#!/usr/bin/env bash
# Usage: bash train_cy_snn_min_mori_cap.sh [incremental|best_so_far] [training arguments]
# Use nohup/setsid (or subprocess start_new_session) for unattended training.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
REWARD_MODE="${1:-incremental}"
if [[ $# -gt 0 ]]; then shift; fi
case "${REWARD_MODE}" in
  incremental) DEFAULT_GPU=0; DEFAULT_GAMMA=0.95 ;;
  best_so_far) DEFAULT_GPU=1; DEFAULT_GAMMA=1 ;;
  *) printf 'Expected incremental or best_so_far, got %s\n' "${REWARD_MODE}" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/home/yiranwang/anaconda3/envs/sage/bin/python}"
RUN_ID="${RUN_ID:-cy_snn_min_mori_cap_${REWARD_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/${RUN_ID}}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/wandb" "${RUN_DIR}/logs"
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd)"
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TRAIN_COMMAND=("${PYTHON_BIN}" scripts/train_cy.py
  --dataset_path data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl
  --in_channels 4 --subcomplex_actor_type snn_simplex
  --neighbor_mode two_neighbors --no-include_points_interior_to_facets
  --reward min_mori_cap --training_reward_mode "${REWARD_MODE}" --gamma "${DEFAULT_GAMMA}"
  --mori_cap_backend mosek
  --gpu_index "${DEFAULT_GPU}" --seed 0
  --num_iterations 1000 --num_epochs 1 --num_states 128 --rollout_length 64
  --batch_size 128 --lr 0.0001 --entropy_coef 0.001 --count_bonus_coef 0.001
  --no-vertex_aug_enable
  --use_multiprocessing --transition_num_workers 16
  --torch_num_threads 2 --torch_num_interop_threads 1
  --memory_budget_gb 48 --runtime_cache_gb 32
  --transition_task_timeout_sec 1800
  --num_eval_polytopes 100 --num_eval_states 128 --eval_steps 64 --eval_interval 100 --deterministic_eval
  --checkpoint_path "${RUN_DIR}/checkpoints"
  --iteration_metrics_path "${RUN_DIR}/iteration_metrics.jsonl"
  --latest_checkpoint_interval 1 --save_interval 100 --report_every 1
  --wandb_project calabi_yau_snn_min_mori_cap --use_wandb --name_suffix "${RUN_ID}"
  "$@")
{
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
} >> "${RUN_DIR}/logs/commands.log"
exec "${TRAIN_COMMAND[@]}"
