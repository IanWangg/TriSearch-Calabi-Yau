cd /home/yiranwang/combinartorics/TriSearch-Calabi-Yau

RUN_ID="min_tri_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/wandb" "${RUN_DIR}/checkpoints"
set -o pipefail

WANDB_MODE=online \
WANDB_DIR="${RUN_DIR}/wandb" \
PYTHONUNBUFFERED=1 \
/home/yiranwang/anaconda3/envs/sage/bin/python scripts/train_cy.py \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --reward_function max_tri \
  --num_iterations 1000 \
  --num_epochs 1 \
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
  --report_every 5 \
  --use_wandb \
  --wandb_project calabi_yau_min_tri \
  --name_suffix "${RUN_ID}" \
  2>&1 | tee "${RUN_DIR}/train_performance.log"
