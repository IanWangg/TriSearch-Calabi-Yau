#!/usr/bin/env bash
# Continue the interrupted run from its iteration 420 policy checkpoint.
# Usage: bash resume_cy_snn_kcup.sh [training arguments]
# Preview without creating files or starting training: DRY_RUN=1 bash resume_cy_snn_kcup.sh
# This is a weights-only restart: optimizer, RNG, and rollout state are reset.
# Local iteration 1 corresponds to overall iteration 421; W&B starts a new run.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
export PYTHON_BIN="${PYTHON_BIN:-/home/yiranwang/anaconda3/envs/sage/bin/python}"
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
for arg in "$@"; do
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    exec "${PYTHON_BIN}" scripts/train_cy.py --help
  fi
done

SOURCE_RUN="${REPO_DIR}/runs/cy_snn_kcup_h11_15_20260921_123000_1584461"
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/latest.pth"
export RUN_ID="cy_snn_kcup_h11_15_resume_420_$(date +%Y%m%d_%H%M%S)_$$"
export RUN_DIR="${REPO_DIR}/runs/${RUN_ID}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${SOURCE_CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Output directory already exists: ${RUN_DIR}" >&2; exit 1; }

# Fail before creating a run if the checkpoint or fallback solver is unusable.
"${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" <<'PY'
import sys
import torch
from qpsolvers import available_solvers

if "cvxopt" not in available_solvers:
    raise SystemExit("The cvxopt fallback solver is unavailable in this environment.")
weights = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
    raise SystemExit("Checkpoint contains invalid policy weights.")
print(f"Validated policy checkpoint: {sys.argv[1]}")
PY

echo "Continuing from iteration 420 for 580 more iterations (unless overridden)."
echo "Optimizer and rollout state will restart; iteration numbering starts at 1."
echo "Output directory: ${RUN_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command: bash train_cy_snn_kcup.sh --num_iterations 580'
  if (( $# > 0 )); then
    printf ' %q' "$@"
  fi
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_DIR}/checkpoints"
cp -- "${SOURCE_CHECKPOINT}" "${RUN_DIR}/checkpoints/latest.pth"
printf 'source_checkpoint=%s\ncompleted_iterations=420\nweights_only=true\n' \
  "${SOURCE_CHECKPOINT}" > "${RUN_DIR}/resume_info.txt"
# Prevent inherited W&B settings from appending reset steps to the old run.
unset WANDB_RUN_ID WANDB_RESUME
exec bash "${REPO_DIR}/train_cy_snn_kcup.sh" --num_iterations 580 "$@"
