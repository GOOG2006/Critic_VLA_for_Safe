#!/bin/bash
# critic_v4 with reduced CRITIC_MARGIN_GAIN=0.10 (vs default 0.30):
# v4 has positive rate 0.60 (vs v3's 0.09), so smaller gain offsets the
# 6× more frequent risk predictions to avoid over-conservatism.
set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
PYTHON=/root/miniconda3/envs/Evo1/bin/python
CKPT=$ROOT/aegis_runs/critic_v4/critic_v4.pt
cd $ROOT/scripts

run() {
  local SUITE=$1; local LV=$2
  local SHORT=${SUITE#safelibero_}
  local TAG=critic_v4_g010_${SHORT}_lv${LV}
  echo "=== [$TAG] start $(date) ==="
  env SAFELIBERO_SUITE=$SUITE \
      EVAL_RESULTS_DIR=$RUN_ROOT/$TAG \
      OBSTACLE_REPRESENTATION=multi \
      OBSTACLE_PADDING=0.005 \
      USE_PERCEPTION_PROXIMITY=1 \
      CRITIC_MARGIN_GAIN=0.10 \
      GRIPPER_INFLATION=1.0 \
      CRITIC_CKPT=$CKPT \
      $PYTHON -u eval_pi05_safelibero.py safemole_multi_critic $LV 10 \
      > $RUN_ROOT/$TAG.log 2>&1
  rc=$?
  echo "=== [$TAG] done rc=$rc $(date) ==="
}

run safelibero_goal I
run safelibero_goal II
run safelibero_object I
run safelibero_object II
echo "=== g010 DONE $(date) ==="
