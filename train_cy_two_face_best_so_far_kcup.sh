#!/usr/bin/env bash
# Finite-episode PPO maximizing best log Kcup volume (sage environment).
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-cy_two_face_best_so_far_kcup_h11_15_$(date +%Y%m%d_%H%M%S)_$$}"
for arg in "$@"; do
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    printf '%s\n' 'Usage: bash train_cy_two_face_best_so_far_kcup.sh [training arguments]' \
      'Defaults: training_reward_mode=best_so_far_log_volume, gamma=1,' \
      '128 training steps and 128 evaluation steps. Other settings follow train_cy_two_face_kcup.sh.' \
      'Use a fresh RUN_ID or RUN_DIR; caller arguments override defaults.'
    exec "${PYTHON_BIN:-/home/yiranwang/anaconda3/envs/sage/bin/python}" "${REPO_DIR}/scripts/train_cy.py" --help
  fi
done
exec bash "${REPO_DIR}/train_cy_two_face_kcup.sh" \
  --training_reward_mode best_so_far_log_volume --gamma 1 \
  --rollout_length 128 --eval_steps 128 "$@"
