#!/usr/bin/env bash
# v109 — SafeLIBERO Long Level II, task 2 ep 0 (dual-mug + obstacle)
# Achieves Safe-SR=100% across seeds {7, 11, 23, 42}.
#
# Usage: bash run_v109_t2ep0.sh [SEED]
# Default SEED=11.
set -e

SEED="${1:-11}"
cd "$(dirname "$0")/.."

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
/root/miniconda3/envs/Evo1/bin/python -u scripts/eval_mpc_cbf_single.py \
  --task-suite-name safelibero_long --safety-level II \
  --task-index 2 --episode-index 0 \
  --N 5 --alpha-h 10.0 --delta-min -0.05 \
  --tangent-threshold 0.10 --tangent-strength 3.0 --tangent-force-horizontal \
  --align-strength 2.0 --align-rotate-steps 15 --align-h-threshold 0.1 \
  --align-phase-grasp --no-rotate-bypass-qp --align-translate-scale 0.2 \
  --rotation-min-z 0.0 --rotation-min-z-grasp 0.0 \
  --grip-release-task-ids 2 \
  --grip-release-plate-d-max 0.05 --grip-release-eef-z-max 0.50 \
  --seed "${SEED}" --no-post-grasp-rise-enable \
  --stuck-window 30 --stuck-range-threshold 0.02 \
  --stuck-window-holding 60 --stuck-range-threshold-holding 0.02 \
  --stuck-rise-target-offset 0.10 --stuck-rise-target-max-z 0.65 \
  --rise-bias-z 0.5 --rise-bias-wrist-ry 0.1 \
  --stuck-max-eef-z 99.0 --stuck-cooldown-steps 120 \
  --clear-plan-after-maneuver \
  --release-q-z-shrink 0.10 --release-q-z-min-eef-z 0.50 --release-obstacle-q-scale 0.5 \
  --n-obstacle-clusters 4 --obstacle-cluster-min-pts 20 \
  --free-gripper-q-xy0 0.04 --free-gripper-q-xy1 0.06 --free-gripper-q-z 0.10 \
  --holding-gripper-q-xy0 0.06 --holding-gripper-q-xy1 0.12 --holding-gripper-q-z 0.20 \
  --gripper-q-override-task-ids 2 \
  --save-videos \
  --video-out-path "results/v109_s${SEED}_ep0"
