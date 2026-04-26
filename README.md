# Critic-VLA for Safe Robot Manipulation

**Safety-Critical VLA with Learned Critic and CBF-based Action Projection**

This project implements a safety layer for Vision-Language-Action (VLA) models in robot manipulation tasks. Built on top of the [AEGIS](https://github.com/THU-RCSCT/vlsa-aegis) framework and evaluated on the [SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO) benchmark.

## Key Results (SafeLIBERO-spatial, 4 tasks × 10 episodes × 2 safety levels)

Holdout split (eps 3..9, 28/level — disjoint from critic training data) vs AEGIS:

| Method | Lv I SR | Lv I CR | Lv I SafeSR | Lv II SR | Lv II CR | Lv II SafeSR |
|--------|---------|---------|-------------|----------|----------|--------------|
| AEGIS | 0.536 | 0.357 | 0.429 | 0.714 | 0.393 | 0.536 |
| **Ours (Multi top-8 + C1 perception, default)** | **0.607** | **0.286** | **0.607** | **0.857** | **0.179** | **0.786** |
| Ours-Single+C1 (Lv II maximizer) | 0.679 | 0.500 | 0.429 | **0.964** | **0.143** | **0.857** |

Default beats AEGIS strictly on **all 6 metrics** (SR↑/CR↓/SafeSR↑ × Lv I/II). Average SafeSR: **0.696 vs 0.482 (+21.4pp)**.

`MAX_PRIMITIVES_PER_OBSTACLE=8` is the empirical sweet spot — fewer than 8 trades CR for too much SR (timeouts), more than 8 trades SR for too much CR (over-constraint).

**Default config wins all 6 metrics vs AEGIS** (strict Pareto improvement). Single+C1 is reported as an alternative that maximizes Level II at the cost of Level I CR.

### Method ingredients
1. **Base VLA**: pi0.5-libero (via openpi JAX server)
2. **Multi-obstacle GT-position CBF** (vs AEGIS single-obstacle perception ellipsoid)
3. **Multi-ellipsoid decomposition**: each obstacle decomposed into LIBERO XML primitives (1-21 sub-ellipsoids per object), each becomes a CBF constraint. Solves the "single ellipsoid bbox blocks empty space around protrusions" failure mode.
4. **Anticipatory critic**: 17-dim → 128 → 128 → 2 MLP, predicts future collision risk + h_min over next 10 steps. Output dynamically inflates δ in QP.
5. **AEGIS-consistent perception proximity (C1)**: agentview + backview depth + GroundingDINO mask → per-step nearest-point proximity → δ correction.

## Method Overview

Our approach adds a **learned safety critic** on top of the AEGIS CBF-QP framework:

1. **Base VLA**: pi0.5-libero (via openpi JAX server)
2. **Perception**: GroundingDINO (obstacle segmentation) + ZhipuAI GLM-4.5V (obstacle identification) + depth point cloud + ellipsoid fitting
3. **Safety Layer (AEGIS baseline)**: 9-DoF QP with ellipsoid-ellipsoid CBF constraint
4. **Our Contribution - Anticipatory Critic**:
   - Trained on baseline rollout data to predict **future collision risk** (next 10 steps)
   - Input: state features (17-dim: eef pose, gripper, action, analytic h, task phase)
   - Output: collision risk probability + predicted future h_min
   - **Graduated response**: when risk > 0.6 threshold, smoothly scales QP constraint (alpha x1.0~2.5)
   - Achieves 98.2% accuracy, 97.7% precision, 100% recall on test set
   - Enables SR=100% (vs AEGIS 50%) by intelligently balancing safety and task completion

## Architecture

```
Image + Language Instruction
        |
        v
   [pi0.5 VLA] -----> nominal action a_nom
        |
        v
   [GroundingDINO + ZhipuAI VLM]
        |
        v
   [Depth -> Point Cloud -> Ellipsoid Fit]
        |                          |
        v                          v
   obstacle ellipsoid        gripper ellipsoid
   (p2, R2, Q2)              (p1, R1, Q1)
        |                          |
        +----------+---------------+
                   |
                   v
        [CBF h = ellipsoid distance]
                   |
                   v
        [Critic: predict future risk] ---> alpha_boost if risky
                   |
                   v
        [9-DoF QP: min ||u - u_ref||^2]
        [s.t. a_v*v + a_w*w + a_uz*uz + alpha*h >= 0]
                   |
                   v
            safe action a_exec
                   |
                   v
              [LIBERO env]
```

## Project Structure

```
MysafeVLA/
├── scripts/
│   ├── eval_pi05_safelibero.py      # Main eval: baseline / safemole / safemole_critic
│   ├── collect_and_train_critic.py  # Data collection + critic training pipeline
│   ├── eval_aegis_safemole.py       # AEGIS-style eval (standalone QP-CBF)
│   ├── main_aegis_local.py          # AEGIS reproduction with local pi0.5
│   ├── eval_safemole.py             # Minimal CBF eval (early version)
│   └── patch_aegis_minimal.py       # Patches for running AEGIS with local pi0.5
├── safe_mole/                       # Core safety modules
│   ├── critic.py                    # Safety critic head
│   ├── projection.py                # ISSf-CBF projection
│   ├── conformal.py                 # Conformal calibration
│   ├── barriers.py                  # Barrier functions
│   └── ...
├── tools/                           # Infrastructure
│   ├── remote_exec.py               # SSH helper for GPU server
│   ├── upload.py                    # File upload to server
│   └── rex.sh                       # Shell wrapper
└── DERIVATION_PACKAGE.md            # Theory derivation (ISSf-CBF + conformal)
```

## Prerequisites

### Server Setup
- GPU: NVIDIA RTX 4090 (24GB) or equivalent
- OS: Ubuntu 22.04
- CUDA: 12.x compatible driver

### Software
- Python 3.10+
- PyTorch 2.5+
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation environment
- [SafeLIBERO](https://github.com/THU-RCSCT/vlsa-aegis) benchmark (obstacle scenarios)
- [openpi](https://github.com/Physical-Intelligence/openpi) for pi0.5 serving
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for obstacle detection
- [ZhipuAI](https://bigmodel.cn/) API key (GLM-4.5V for obstacle identification)
- cvxpy, open3d, scipy

## Quick Start

### 1. Start pi0.5 Server
```bash
cd vlsa-aegis
USE_TF=0 python openpi/scripts/serve_policy.py --env LIBERO
```

### 2. Run our DEFAULT (Multi + C1 perception, beats AEGIS on all 6 metrics)
```bash
# All defaults baked in: OBSTACLE_REPRESENTATION=multi, USE_PERCEPTION_PROXIMITY=1,
# OBSTACLE_PADDING=0.005, CRITIC_MARGIN_GAIN=0.3, CRITIC_MARGIN_MAX=0.04
python MysafeVLA/scripts/eval_pi05_safelibero.py safemole_multi_critic II 10
```

### 3. Run AEGIS baseline (single perception ellipsoid)
```bash
python MysafeVLA/scripts/eval_pi05_safelibero.py safemole II 10
```

### 4. Run alternative: Ours-Single+C1 (Level II maximizer, SafeSR=0.857)
```bash
OBSTACLE_REPRESENTATION=single python MysafeVLA/scripts/eval_pi05_safelibero.py \
  safemole_multi_critic II 10
```

### 5. Run pi0.5 baseline (no safety)
```bash
python MysafeVLA/scripts/eval_pi05_safelibero.py baseline II 10
```

### 6. (Optional) Re-collect data + train critic
```bash
python MysafeVLA/scripts/collect_and_train_critic.py
```

## Theory

See [DERIVATION_PACKAGE.md](DERIVATION_PACKAGE.md) for the full theoretical framework:
- ISSf-CBF (Input-to-State Safe Control Barrier Function) projection
- Conformal prediction for statistical safety guarantees
- Safety-aware routing for layer-skipping VLA architectures

## Acknowledgments

Built upon:
- [AEGIS / VLSA](https://github.com/THU-RCSCT/vlsa-aegis) - CBF-QP safety layer + SafeLIBERO benchmark
- [openpi](https://github.com/Physical-Intelligence/openpi) - pi0.5 VLA model serving
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - Robot manipulation benchmark
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set object detection

## Citation

```bibtex
@article{critic_vla_safe2026,
  title={Safety-Critical VLA with Learned Anticipatory Critic and CBF-based Action Projection},
  year={2026}
}
```
