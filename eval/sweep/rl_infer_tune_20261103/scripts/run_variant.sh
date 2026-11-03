#!/usr/bin/env bash
# Usage: run_variant.sh <name> <h11> <gpu> <ckpt> <budget> [tune_rl_search options]
set -u
name=$1; h=$2; gpu=$3; ck=$4; b=$5; shift 5
R=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau; T=$R/runs/fv1_infer_tune
export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
cd $R/.claude/worktrees/bridge-cse_01W7whPSzVLJ8fMkSfoGKzAY
out=$T/${name}_h11_${h}_b$b
[ -f $out/summary.json ] && exit 0
/home/yiranwang/anaconda3/envs/sage/bin/python tools/tune_rl_search.py --setup_path $R/runs/fv1_dev_eval/setup_dev_h11_$h \
  --polytope_file $R/data/cy/favorable_v1/dev/h11_$h.samples.jsonl --h11 $h --checkpoint $ck --budget $b \
  --gpu_index $gpu --workers ${WORKERS:-6} --output_dir $out --cache $T/kcup_cache.sqlite3 "$@" > $out.log 2>&1
