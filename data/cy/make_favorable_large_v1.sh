#!/usr/bin/env bash
# Build favorable_large_v1: favorable 4D polytopes with h11 in {35,40,45,50}, 60 train + 15 dev each.
# Same procedure as favorable_v1 (see data/cy/FAVORABLE_V1_DATASET.md), with the changes listed in
# favorable_large_v1/README.md. Generation is per (split, h11) step; no checkpointing inside a step.
# Usage: bash data/cy/make_favorable_large_v1.sh   (GEN_H11S, WORKERS, CODE, DATA, PY override defaults)
set -euo pipefail

CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA=${DATA:-/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/data/cy}
RUNS=${RUNS:-/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs}
OUT=$DATA/favorable_large_v1
PY=${PY:-/home/yiranwang/anaconda3/envs/sage/bin/python}
WORKERS=${WORKERS:-6}
GEN_H11S=${GEN_H11S:-"35 40 45"}   # h11=50 generation is held pending a decision (see README)

export PYTHONPATH=$CODE
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"

EXCL=(
  "$DATA/favorable_v1/train/polytopes_h11_15.jsonl" "$DATA/favorable_v1/train/polytopes_h11_20.jsonl"
  "$DATA/favorable_v1/dev/polytopes_h11_15.jsonl" "$DATA/favorable_v1/dev/polytopes_h11_20.jsonl"
  "$DATA/favorable_v1/test/polytopes_h11_25.jsonl" "$DATA/favorable_v1/test/polytopes_h11_30.jsonl"
  "$RUNS/face_mutation_20261104/setups/seed0_p10_s3/setup.json"
)

# 1. Selection (seed 20261104) for all four h11 values, with the exclusion check written to selection_manifest.json.
$PY -m data.cy.select_polytopes \
  --splits train:35:60 dev:35:15 train:40:60 dev:40:15 train:45:60 dev:45:15 train:50:60 dev:50:15 \
  --seed 20261104 --exclude-from "${EXCL[@]}" --output-dir "$OUT"

# 2. Starts: same settings as favorable_v1 (train: 4 starts, min 3; dev: 1 start, min 1).
G="$PY -m data.cy.generate_diverse_starts --num-workers $WORKERS --seed 0"
for h in $GEN_H11S; do
  $G --polytopes "$OUT/train/polytopes_h11_$h.jsonl" --output "$OUT/train/h11_$h.samples.jsonl" \
     --mode train --starts 4 --min-starts 3
  $G --polytopes "$OUT/dev/polytopes_h11_$h.jsonl" --output "$OUT/dev/h11_$h.samples.jsonl" \
     --mode eval --starts 1 --min-starts 1
done

# 3. Manifest (hashes, generation summaries, versions, h12 population counts).
$PY -m data.cy.write_dataset_manifest --dataset-dir "$OUT" --name trisearch-cy-favorable-large-v1 \
  --index "$HOME/.cache/trisearch_h12_index/h12_index_60c0e119a036.json"
echo "favorable_large_v1 generation finished"
