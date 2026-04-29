#!/bin/bash
# Targeted v2: AEGIS-perc + critic_v4 with MAX_STEPS=280 + WARMUP=20
# on the same 2 problem cells (t1 lvII cabinet, t3 lvI drawer).
set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
PYTHON=/root/miniconda3/envs/Evo1/bin/python
CKPT=$ROOT/aegis_runs/critic_v4/critic_v4.pt
cd $ROOT/scripts

run() {
  local TASK=$1; local LV=$2
  local TAG=aegisperc_v2_goal_t${TASK}_lv${LV}
  echo "=== [$TAG] start $(date) ==="
  env DEBUG_TASK=$TASK \
      SAFELIBERO_SUITE=safelibero_goal \
      EVAL_RESULTS_DIR=$RUN_ROOT/$TAG \
      MAX_STEPS=280 WARMUP_STEPS=20 \
      CRITIC_MARGIN_GAIN=0.10 \
      H_SAFETY_MARGIN=0.005 \
      GRIPPER_INFLATION=1.0 \
      CRITIC_CKPT=$CKPT \
      $PYTHON -u eval_pi05_safelibero.py safemole_critic $LV 10 \
      > $RUN_ROOT/$TAG.log 2>&1
  rc=$?
  echo "=== [$TAG] done rc=$rc $(date) ==="
}

run 1 II  # cabinet
run 3 I   # drawer
echo "=== AEGISPERC V2 DONE $(date) ==="