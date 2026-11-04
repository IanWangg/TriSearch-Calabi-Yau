#!/bin/bash
# A4 smoke: two_face_invariant PPO with the G-run command, 3 iterations, 16 states, 2 transition workers.
set -e
W=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf
R=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau
PY=/home/yiranwang/anaconda3/envs/sage/bin/python
export PYTHONPATH=$W MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
RUN=$R/runs/inv_rl_smoke_20261104
mkdir -p $RUN
cd $W
$PY scripts/train_cy.py \
  --dataset_path $R/data/cy/favorable_v1/train/h11_15.samples.jsonl $R/data/cy/favorable_v1/train/h11_20.samples.jsonl \
  --eval_dataset_path $R/data/cy/favorable_v1/dev/h11_15.samples.jsonl $R/data/cy/favorable_v1/dev/h11_20.samples.jsonl \
  --in_channels 4 --subcomplex_actor_type two_face_invariant --neighbor_mode two_neighbors --no-include_points_interior_to_facets \
  --reward_function max_kcup --hidden_channels 64 --out_channels 64 --num_layers 3 --count_bonus_coef 0 --no-vertex_aug_enable \
  --seed 0 --num_iterations 3 --num_epochs 1 --num_states 16 --rollout_length 20 --batch_size 128 --lr 0.0001 \
  --use_multiprocessing --transition_num_workers 2 --memory_budget_gb 24 --runtime_cache_gb 4 \
  --num_eval_states 8 --eval_steps 30 --eval_interval 1 --deterministic_eval \
  --checkpoint_path $RUN/checkpoints --iteration_metrics_path $RUN/iteration_metrics.jsonl \
  --latest_checkpoint_interval 1 --save_interval 1 --gpu_index ${GPU:-4} --gamma 0.99 --gae_lambda 0.95 \
  --name_suffix inv_rl_smoke_20261104
