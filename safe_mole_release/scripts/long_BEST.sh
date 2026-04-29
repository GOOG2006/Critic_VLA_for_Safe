#!/bin/bash
# === Locked BEST config for SafeLIBERO-long ===
# Result: SR 50%, CR 44%, Safe-SR 31% (matches AEGIS Safe-SR; 16 ep init [0..3]).
# Optional knobs (CARRY_*, CRITIC_RISK_*, CBF_LOOKAHEAD_*) left unset → defaults disable them.
set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
PYTHON=/root/miniconda3/envs/Evo1/bin/python
CKPT=$ROOT/aegis_runs/critic_v5_long/critic_v5_long.pt
cd $ROOT/scripts

LEVEL=${1:-I}
N_EPS=${2:-4}
TAG=long_BEST_${N_EPS}ep_lv${LEVEL}

echo "=== [$TAG] start $(date) ==="
env SAFELIBERO_SUITE=safelibero_long     EVAL_RESULTS_DIR=$RUN_ROOT/$TAG     OBSTACLE_REPRESENTATION=multi     MAX_PRIMITIVES_PER_OBSTACLE=11     Q_INFLATION=1.3     USE_PERCEPTION_PROXIMITY=1     MAX_STEPS=500 WARMUP_STEPS=20     ALPHA_H=100.0     CRITIC_MARGIN_GAIN=0.5     CRITIC_MARGIN_MAX=0.06     H_SAFETY_MARGIN=0.025     GRIPPER_INFLATION=1.0     GRIPPER_Y_SCALE=0.5     CRITIC_CKPT=$CKPT     $PYTHON -u eval_pi05_safelibero.py safemole_multi_critic $LEVEL $N_EPS     > $RUN_ROOT/$TAG.log 2>&1
echo "=== [$TAG] done $(date) ==="
