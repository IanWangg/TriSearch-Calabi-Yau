#!/bin/bash
# A3: supervised probe for the invariant encoder. Four runs in parallel (4 CPU threads each, GPUs 4-6 chosen by free memory).
set -e
W=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf
R=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau
PY=/home/yiranwang/anaconda3/envs/sage/bin/python
export PYTHONPATH=$W MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic OMP_NUM_THREADS=4
mkdir -p $R/runs/inv_encoder_20261104/probe
cd $W
for v in "inv 0" "inv 1" "inv_ctx 0" "inv_ctx 1"; do set -- $v
  $PY eval/sweep/inv_encoder_20261104/probe_train_inv.py --variant $1 --seed $2 \
      --run_dir $R/runs/probe_20261103 --out_dir $R/runs/inv_encoder_20261104/probe \
      > $R/runs/inv_encoder_20261104/probe/$1_s$2.log 2>&1 &
  sleep 5
done
wait
echo done
