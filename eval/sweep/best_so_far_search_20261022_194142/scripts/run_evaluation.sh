#!/usr/bin/env bash
# Run from an activated sage environment; all data/checkpoints remain local.
set -euo pipefail
if [[ "$#" -lt 4 ]]; then
  printf '%s\n' 'Usage: bash run_evaluation.sh {h11_15|h11_30} checkpoint setup_path new_output_dir [eval arguments]'
  exit 2
fi
dataset="$1"
checkpoint="$2"
setup_path="$3"
output_dir="$4"
shift 4
case "${dataset}" in
  h11_15) config_name=h11_15_two_starts ;;
  h11_30) config_name=h11_30 ;;
  *) printf 'Unknown dataset: %s\n' "${dataset}" >&2; exit 2 ;;
esac
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "${study_dir}/../../.." && pwd)"
cd "${repo_dir}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" scripts/eval_cy.py \
  --config "${study_dir}/configs/${config_name}.json" \
  --policy_checkpoint "${checkpoint}" --setup_path "${setup_path}" \
  --output_dir "${output_dir}" --plot_results "$@"
