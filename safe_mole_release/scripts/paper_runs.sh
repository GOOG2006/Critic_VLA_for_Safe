#!/bin/bash
# Paper-supplementary runs: pi0.5 baseline + g-sweep on Lv II
# Runs sequentially in background. Logs to /root/autodl-tmp/MysafeVLA/aegis_runs/paper_runs/

set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
mkdir -p $RUN_ROOT
cd $ROOT/scripts
export USE_TF=0 MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
PYTHON=/root/miniconda3/envs/Evo1/bin/python

run() {
  local TAG=$1; local MODE=$2; local LEVEL=$3
  shift 3
  echo "=== [$TAG] start $(date) ==="
  EVAL_RESULTS_DIR=$RUN_ROOT/$TAG \
  GRIPPER_INFLATION=1.0 \
  CRITIC_CKPT=$ROOT/aegis_runs/critic_v3/critic_v3.pt \
  "$@" \
  $PYTHON -u eval_pi05_safelibero.py $MODE $LEVEL 10 > $RUN_ROOT/$TAG.log 2>&1
  local rc=$?
  echo "=== [$TAG] done rc=$rc $(date) ==="
}

# (1) pi0.5 baseline (no safety filter at all). Mode = baseline.
run baseline_lvI  baseline I
run baseline_lvII baseline II

# (2) g-sweep: vary CRITIC_MARGIN_GAIN (we already have g=0.3 = default).
# Lv II is the regime where the anticipatory critic has the most effect.
run g000_lvII safemole_multi_critic II CRITIC_MARGIN_GAIN=0.0  CRITIC_MARGIN_MAX=0.04
run g010_lvII safemole_multi_critic II CRITIC_MARGIN_GAIN=0.1  CRITIC_MARGIN_MAX=0.04
run g050_lvII safemole_multi_critic II CRITIC_MARGIN_GAIN=0.5  CRITIC_MARGIN_MAX=0.04
run g100_lvII safemole_multi_critic II CRITIC_MARGIN_GAIN=1.0  CRITIC_MARGIN_MAX=0.04

echo "=== ALL DONE $(date) ==="

# Summarise
$PYTHON - <<'PY' > $RUN_ROOT/paper_runs_summary.md
import json, os
ROOT='/root/autodl-tmp/MysafeVLA/aegis_runs/paper_runs'
def stats(p):
    if not os.path.exists(p): return None
    eps = json.load(open(p))['results']
    if not eps: return None
    n=len(eps); sr=sum(1 for e in eps if e['success'])/n
    cr=sum(1 for e in eps if e['collided'])/n
    safesr=sum(1 for e in eps if e['success'] and not e['collided'])/n
    ho=[e for e in eps if e.get('ep',-1) >= 3]
    if ho:
        ho_n=len(ho); ho_sr=sum(1 for e in ho if e['success'])/ho_n
        ho_cr=sum(1 for e in ho if e['collided'])/ho_n
        ho_safe=sum(1 for e in ho if e['success'] and not e['collided'])/ho_n
    else:
        ho_n=ho_sr=ho_cr=ho_safe=None
    return n,sr,cr,safesr,ho_n,ho_sr,ho_cr,ho_safe

cfgs = [
    ('pi0.5 baseline (no safety)', 'I',  'baseline_lvI/pi05_baseline_lvI/results.json'),
    ('pi0.5 baseline (no safety)', 'II', 'baseline_lvII/pi05_baseline_lvII/results.json'),
    ('Critic-VLA g=0.0',  'II', 'g000_lvII/pi05_safemole_multi_critic_lvII/results.json'),
    ('Critic-VLA g=0.1',  'II', 'g010_lvII/pi05_safemole_multi_critic_lvII/results.json'),
    ('Critic-VLA g=0.5',  'II', 'g050_lvII/pi05_safemole_multi_critic_lvII/results.json'),
    ('Critic-VLA g=1.0',  'II', 'g100_lvII/pi05_safemole_multi_critic_lvII/results.json'),
]
print('# Paper supplementary runs\n')
print('## All 40 ep')
print('| method | level | N | SR | CR | SafeSR |')
print('|---|---|---|---|---|---|')
for name, lv, path in cfgs:
    s = stats(f'{ROOT}/{path}')
    if s: print(f'| {name} | {lv} | {s[0]} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} |')
    else: print(f'| {name} | {lv} | - | missing | - | - |')

print('\n## Holdout (ep 3..9, 28 ep)')
print('| method | level | N | SR | CR | SafeSR |')
print('|---|---|---|---|---|---|')
for name, lv, path in cfgs:
    s = stats(f'{ROOT}/{path}')
    if s and s[4]:
        print(f'| {name} | {lv} | {s[4]} | {s[5]:.3f} | {s[6]:.3f} | {s[7]:.3f} |')
    else:
        print(f'| {name} | {lv} | - | missing | - | - |')
PY
cat $RUN_ROOT/paper_runs_summary.md
