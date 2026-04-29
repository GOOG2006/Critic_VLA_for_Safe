#!/bin/bash
# Collect long-suite ONLY data via CLOSED-LOOP CBF (uses critic_v4 + multi-CBF).
# Gives balanced safe/unsafe samples: when CBF prevents collision -> negative samples,
# when CBF fails -> positive samples. Self-bootstrapped, the same way critic_v3 -> v4 was done.
# Init state offset 10 keeps eval (default [0..9]) untouched.
set -u
ROOT=/root/autodl-tmp/MysafeVLA
PYTHON=/root/miniconda3/envs/Evo1/bin/python
cd $ROOT/scripts

OUT=$ROOT/aegis_runs/critic_v5_long
mkdir -p $OUT

echo "=== collect+train critic_v5_long (closed-loop) start $(date) ==="
env BENCH_SUITES=safelibero_long     BENCH_LEVELS=I,II     COLLECT_EPS_PER_TASK=3     INIT_STATE_OFFSET=10     COLLECTOR_MODE=safemole_multi_critic     CRITIC_INIT_CKPT=$ROOT/aegis_runs/critic_v4/critic_v4.pt     OBSTACLE_REPRESENTATION=multi     MAX_PRIMITIVES_PER_OBSTACLE=11     Q_INFLATION=1.3     ALPHA_H=50.0     H_SAFETY_MARGIN=0.015     GRIPPER_INFLATION=1.0     CRITIC_OUT_DIR=$OUT     CRITIC_CKPT_NAME=critic_v5_long.pt     HF_ENDPOINT=https://hf-mirror.com     USE_TF=0 MUJOCO_GL=egl     $PYTHON -u collect_and_train_critic.py 2>&1 | tee $OUT/collect_train.log

rc=$?
echo "=== collect+train rc=$rc $(date) ==="
ls -lh $OUT/
