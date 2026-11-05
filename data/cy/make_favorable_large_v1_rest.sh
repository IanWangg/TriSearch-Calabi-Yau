#!/usr/bin/env bash
# Generate favorable_large_v1 starts for all h11 with no 2-face triangulation counting / NTFE enumeration
# (--skip-ntfe-enumeration), as decided on 2026-11-04. The first attempt (with counting) was stopped: two h11=35
# workers stuck in counting/enumeration grew to ~70 GB RSS each.
set -euo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA=${DATA:-/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/data/cy}
OUT=$DATA/favorable_large_v1
PY=${PY:-/home/yiranwang/anaconda3/envs/sage/bin/python}
WORKERS=${WORKERS:-6}
WAIT_PID=${WAIT_PID:-}
export PYTHONPATH=$CODE
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
G="$PY -m data.cy.generate_diverse_starts --num-workers $WORKERS --seed 0 --skip-ntfe-enumeration"
for h in 35 40 45 50; do
  $G --polytopes "$OUT/train/polytopes_h11_$h.jsonl" --output "$OUT/train/h11_$h.samples.jsonl" \
     --mode train --starts 4 --min-starts 3
  $G --polytopes "$OUT/dev/polytopes_h11_$h.jsonl" --output "$OUT/dev/h11_$h.samples.jsonl" \
     --mode eval --starts 1 --min-starts 1
done
$PY -m data.cy.write_dataset_manifest --dataset-dir "$OUT" --name trisearch-cy-favorable-large-v1 \
  --index "$HOME/.cache/trisearch_h12_index/h12_index_60c0e119a036.json"
echo "favorable_large_v1 generation finished"
