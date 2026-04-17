#!/bin/bash
# Setup script for Critic-VLA-for-Safe
# Run on the GPU server after cloning the repo.
set -euo pipefail

echo "=== 1. Clone AEGIS (SafeLIBERO benchmark + utils) ==="
if [ ! -d "vlsa-aegis" ]; then
    git clone https://github.com/THU-RCSCT/vlsa-aegis.git
fi

echo "=== 2. Install LIBERO ==="
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO && pip install -e . && cd ..
fi

echo "=== 3. Overlay SafeLIBERO onto LIBERO ==="
SAFE=vlsa-aegis/safelibero/libero/libero
ORIG=LIBERO/libero/libero
cp -rn $SAFE/assets/obstacle_objects $ORIG/assets/ 2>/dev/null || true
cp -rn $SAFE/bddl_files/safelibero_* $ORIG/bddl_files/ 2>/dev/null || true
cp -rn $SAFE/init_files/safelibero_* $ORIG/init_files/ 2>/dev/null || true
cp $SAFE/benchmark/__init__.py $ORIG/benchmark/
cp $SAFE/benchmark/libero_suite_task_map.py $ORIG/benchmark/
cp $SAFE/envs/objects/obstacle_objects.py $ORIG/envs/objects/
grep -q 'obstacle_objects' $ORIG/envs/objects/__init__.py || echo 'from .obstacle_objects import *' >> $ORIG/envs/objects/__init__.py
cp -rn $SAFE/assets/scenes/*.xml $ORIG/assets/scenes/ 2>/dev/null || true
cp -rn $SAFE/assets/scenes/*.msh $ORIG/assets/scenes/ 2>/dev/null || true
echo "SafeLIBERO overlay done."

echo "=== 4. Install GroundingDINO ==="
if [ ! -d "vlsa-aegis/GroundingDINO/src" ]; then
    cd vlsa-aegis && mkdir -p GroundingDINO && cd GroundingDINO
    git clone https://github.com/IDEA-Research/GroundingDINO.git src
    cd src && pip install -e . --no-build-isolation && cd ..
    cp src/groundingdino/config/GroundingDINO_SwinT_OGC.py .
    # Download weights
    aria2c -x 8 -o groundingdino_swint_ogc.pth \
        'https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth'
    cd ../..
fi

echo "=== 5. Install Python deps ==="
pip install cvxpy open3d zhipuai zai-sdk sentencepiece tyro draccus

echo "=== 6. Install openpi (for pi0.5 serving) ==="
cd vlsa-aegis/openpi
sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' pyproject.toml 2>/dev/null || true
pip install --no-deps -e .
cd packages/openpi-client && pip install --no-deps -e . && cd ../../..
# Fix Python 3.10 compat
sed -i 's/tzinfo=datetime.UTC/tzinfo=datetime.timezone.utc/g' vlsa-aegis/openpi/src/openpi/shared/download.py 2>/dev/null || true

echo "=== 7. Set ZhipuAI API key ==="
echo "NOTE: Edit vlsa-aegis/main/utils.py and set api_key = 'YOUR_KEY'"
echo "      Get key from https://bigmodel.cn/"

echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Start pi0.5 server: cd vlsa-aegis && python openpi/scripts/serve_policy.py --env LIBERO"
echo "  2. Run baseline: python MysafeVLA/scripts/eval_pi05_safelibero.py baseline II 1"
echo "  3. Run with safety: python MysafeVLA/scripts/eval_pi05_safelibero.py safemole II 1"
echo "  4. Train critic: python MysafeVLA/scripts/collect_and_train_critic.py"
echo "  5. Run with critic: python MysafeVLA/scripts/eval_pi05_safelibero.py safemole_critic II 1"
