#!/usr/bin/env bash
# Kcup SNN baseline with entropy_coef=0.01 on GPU 2.
# Usage: bash train_cy_snn_kcup_entropy_coef_0_01.sh [training arguments]
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-cy_snn_kcup_h11_15_entropy_coef_0_01_$(date +%Y%m%d_%H%M%S)_$$}"

# Reuse the baseline settings; caller arguments can override these defaults.
exec bash "${REPO_DIR}/train_cy_snn_kcup.sh" \
  --gpu_index 2 \
  --entropy_coef 0.01 \
  --memory_budget_gb 64 \
  --runtime_cache_gb 32 \
  "$@"
