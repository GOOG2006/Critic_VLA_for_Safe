#!/bin/bash
# Full object suite with alpha=50 + 280 steps + critic_v4 + multi.
set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
PYTHON=/root/miniconda3/envs/Evo1/bin/python
CKPT=$ROOT/aegis_runs/critic_v4/critic_v4.pt
cd $ROOT/scripts

run() {
  local LV=$1
  local TAG=alpha50_obj_full_lv${LV}
  echo "=== [$TAG] start $(date) ==="
  env SAFELIBERO_SUITE=safelibero_object \
      EVAL_RESULTS_DIR=$RUN_ROOT/$TAG \
      OBSTACLE_REPRESENTATION=multi \
      OBSTACLE_PADDING=0.005 \
      Q_INFLATION=1.0 \
      USE_PERCEPTION_PROXIMITY=1 \
      MAX_STEPS=280 WARMUP_STEPS=20 \
      ALPHA_H=50.0 \
      CRITIC_MARGIN_GAIN=0.10 \
      H_SAFETY_MARGIN=0.005 \
      GRIPPER_INFLATION=1.0 \
      CRITIC_CKPT=$CKPT \
      $PYTHON -u eval_pi05_safelibero.py safemole_multi_critic $LV 10 \
      > $RUN_ROOT/$TAG.log 2>&1
  rc=$?
  echo "=== [$TAG] done rc=$rc $(date) ==="
}

run I
run II
echo "=== ALPHA50 OBJECT FULL DONE $(date) ==="
