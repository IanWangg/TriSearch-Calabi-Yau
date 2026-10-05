#!/usr/bin/env bash
# Kcup SNN baseline with num_states=512 on GPU 0.
# Usage: bash train_cy_snn_kcup_num_states_512.sh [training arguments]
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-cy_snn_kcup_h11_15_num_states_512_$(date +%Y%m%d_%H%M%S)_$$}"

# Reuse the baseline settings; caller arguments can override these defaults.
exec bash "${REPO_DIR}/train_cy_snn_kcup.sh" \
  --gpu_index 0 \
  --num_states 512 \
  --memory_budget_gb 64 \
  --runtime_cache_gb 32 \
  "$@"
