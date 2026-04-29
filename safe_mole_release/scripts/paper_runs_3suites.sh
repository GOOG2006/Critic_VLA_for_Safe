#!/bin/bash
# 4-suite extension: run baseline + AEGIS + Critic-VLA on Goal/Object/Long.
# Sequential: finish all 3 methods × 2 levels for one suite before moving to next.
set -u
ROOT=/root/autodl-tmp/MysafeVLA
RUN_ROOT=$ROOT/aegis_runs/paper_runs
PYTHON=/root/miniconda3/envs/Evo1/bin/python
mkdir -p $RUN_ROOT
cd $ROOT/scripts

# (mode, label) pairs
MODES=(baseline:pi05 safemole:aegis safemole_multi_critic:critic)

run() {
  local SUITE=$1; local MODE_LABEL=$2; local LV=$3
  local MODE=${MODE_LABEL%:*}
  local LABEL=${MODE_LABEL#*:}
  local SUITE_SHORT=${SUITE#safelibero_}
  local TAG=${LABEL}_${SUITE_SHORT}_lv${LV}
  echo "=== [$TAG] start $(date) ==="
  env SAFELIBERO_SUITE=$SUITE \
      EVAL_RESULTS_DIR=$RUN_ROOT/$TAG \
      GRIPPER_INFLATION=1.0 \
      CRITIC_CKPT=$ROOT/aegis_runs/critic_v3/critic_v3.pt \
      $PYTHON -u eval_pi05_safelibero.py $MODE $LV 10 \
      > $RUN_ROOT/$TAG.log 2>&1
  rc=$?
  echo "=== [$TAG] done rc=$rc $(date) ==="
}

for SUITE in safelibero_goal safelibero_object safelibero_long; do
  echo
  echo "######## SUITE: $SUITE ########"
  for ML in "${MODES[@]}"; do
    for LV in I II; do
      run $SUITE "$ML" $LV
    done
  done
done

echo "=== ALL DONE $(date) ==="

# Summary
$PYTHON - <<'PY' > $RUN_ROOT/threesuite_summary.md
import json, os
ROOT='/root/autodl-tmp/MysafeVLA/aegis_runs/paper_runs'
def stats(p):
    if not os.path.exists(p): return None
    eps = json.load(open(p))['results']
    if not eps: return None
    n=len(eps)
    sr=sum(1 for e in eps if e['success'])/n
    cr=sum(1 for e in eps if e['collided'])/n
    safe=sum(1 for e in eps if e['success'] and not e['collided'])/n
    ets=sum(e['steps'] for e in eps)/n
    ho=[e for e in eps if e.get('ep',-1) >= 3]
    if not ho: return n, sr, cr, safe, ets, None, None, None, None, None
    m=len(ho)
    hsr=sum(1 for e in ho if e['success'])/m
    hcr=sum(1 for e in ho if e['collided'])/m
    hsafe=sum(1 for e in ho if e['success'] and not e['collided'])/m
    hets=sum(e['steps'] for e in ho)/m
    return n, sr, cr, safe, ets, m, hsr, hcr, hsafe, hets

print('# 3-suite extension (Goal / Object / Long), 28-ep holdout per (suite, method, level)\n')
print('| Suite | Method | Lv | N | SR | CR | SafeSR | ETS |')
print('|---|---|---|---|---|---|---|---|')
for suite in ['goal','object','long']:
    for mode_label, mode_lookup in [('pi05','baseline'),('aegis','safemole'),('critic','safemole_multi_critic')]:
        for lv in ['I','II']:
            tag = f'{mode_label}_{suite}_lv{lv}'
            sub = f'pi05_{mode_lookup}_lv{lv}'
            p = f'{ROOT}/{tag}/{sub}/results.json'
            s = stats(p)
            if s and s[5]:
                print(f'| {suite} | {mode_label} | {lv} | {s[5]} | {s[6]:.3f} | {s[7]:.3f} | {s[8]:.3f} | {s[9]:.1f} |')
            else:
                print(f'| {suite} | {mode_label} | {lv} | - | missing | - | - | - |')
PY
cat $RUN_ROOT/threesuite_summary.md
