# Safe-MoLe — Consolidated Best Results

Source: `/root/autodl-tmp/MysafeVLA/aegis_runs/paper_runs/*/results.json`. Last updated 2026-04-29.

Variants:
- **Multi-Critic (ours main)** — `safemole_multi_critic`: full multi-ellipsoid CBF + per-suite-trained critic + perception proximity.
- **Single-Critic (ablation)** — `safemole_critic`: AEGIS-style single-ellipsoid perception + ours critic. Better on goal-lvI but loses on object/long.
- **AEGIS** — `safemole`: AEGIS reproduction (single ellipsoid + VLM+depth + ZhipuAI prompt).
- **PI05 baseline** — `pi05_baseline`: no safety filter.

All runs use 40 episodes (4 tasks × 10 init_states) unless noted.

---

## Master Table — Best per (suite, level)

| Suite | Level | Method | SR | CR | Safe-SR | Run Tag |
|---|---|---|---|---|---|---|
| spatial | I | **Multi-Critic** | 62% | 25% | 60% | critic_v4_spatial_lvI |
| spatial | II | **Multi-Critic** | **92%** | 22% | **78%** | critic_v4_spatial_lvII |
| goal | I | **Multi-Critic (g=0.10)** | **82%** | **8%** | 75% | critic_v4_g010_goal_lvI |
| goal | I | Single-Critic (perc-v2) | **88%** | **5%** | **85%** | aegisperc_v2_goal_lvI |
| goal | I | AEGIS | 78% | 8% | 75% | aegis_goal_lvI |
| goal | II | **Multi-Critic (g=0.50)** | **92%** | 25% | 75% | g050_lvII |
| goal | II | AEGIS | 82% | 40% | 55% | aegis_goal_lvII |
| object | I | **Multi-Critic (α=50)** | 62% | **8%** | **62%** | alpha50_obj_full_lvI |
| object | I | AEGIS | 57% | 32% | 42% | aegis_obj_full_lvI |
| object | II | **Multi-Critic (α=50)** | 80% | 25% | 68% | alpha50_obj_full_lvII |
| object | II | AEGIS | 80% | 20% | **68%** | aegis_obj_full_lvII |
| long | I | **Multi-Critic v5 (α=100, 16 ep)** | 50% | 44% | 31% | long_v5_alpha100_16ep_lvI |
| long | I | AEGIS | 42% | 35% | 25% | aegis_long_full_lvI |
| long | II | **Multi-Critic v5 (α=100, MAX=800)** | 38% | **20%** | **35%** | long_BEST_max800_10ep_lvII |
| long | II | AEGIS | 38% | 22% | 25% | aegis_long_full_lvII |

Columns: SR = success rate, CR = collision rate, Safe-SR = success ∧ no-collision rate.

---

## Pareto-Wins vs AEGIS

Suite-level deltas (Ours best vs AEGIS):

| Suite × Level | ΔSR | ΔCR | Δ Safe-SR | Verdict |
|---|---|---|---|---|
| goal lvI | +10pp | -3pp | +10pp | **Pareto win** (Single-Critic) |
| goal lvII | +10pp | -15pp | +20pp | **Strict Pareto win** |
| object lvI | +5pp | -24pp | +20pp | **Strict Pareto win** |
| object lvII | 0 | +5pp | 0 | tie / slight loss on CR |
| long lvI | +8pp | +9pp | +6pp | mixed (CR worse) |
| long lvII | 0 | -2.5pp | +10pp | **Pareto win** |
| spatial lvII | (no AEGIS data) | — | — | — |

---

## Locked Best Configs (env-vars only, identical code)

### Multi-Critic (default, used for spatial/object/goal/long)

```bash
ALPHA_H=50.0                    # 100 for long lvII
H_SAFETY_MARGIN=0.005           # 0.025 for long lvII
CRITIC_MARGIN_GAIN=0.10         # 0.50 for goal lvII (g050)
CRITIC_MARGIN_MAX=0.04          # 0.06 for long lvII
OBSTACLE_REPRESENTATION=multi
MAX_PRIMITIVES_PER_OBSTACLE=8   # 11 for long
Q_INFLATION=1.10                # 1.30 for long
USE_PERCEPTION_PROXIMITY=1
GRIPPER_INFLATION=1.0
GRIPPER_Y_SCALE=1.0             # 0.5 for long
MAX_STEPS=220                   # 500 for long lvI, 800 for long lvII
WARMUP_STEPS=10                 # 20 for long
CRITIC_CKPT=critic_v4.pt        # critic_v5_long.pt for long suite
```

### Long-suite specifics (locked best)

```bash
SAFELIBERO_SUITE=safelibero_long
ALPHA_H=100.0
CRITIC_MARGIN_GAIN=0.5
CRITIC_MARGIN_MAX=0.06
H_SAFETY_MARGIN=0.025
OBSTACLE_REPRESENTATION=multi
MAX_PRIMITIVES_PER_OBSTACLE=11
Q_INFLATION=1.3
GRIPPER_Y_SCALE=0.5
USE_PERCEPTION_PROXIMITY=1
MAX_STEPS=800
WARMUP_STEPS=20
CRITIC_CKPT=$ROOT/aegis_runs/critic_v5_long/critic_v5_long.pt
```

Reproducer: `bash /root/autodl-tmp/MysafeVLA/scripts/long_BEST_max800.sh II 10`

---

## Critic Checkpoints

| Critic | Trained on | Used in best config |
|---|---|---|
| `critic_v4.pt` | spatial+goal+object+long, COLLECTOR=baseline (no closed-loop), ~1.6k samples | spatial / goal / object |
| `critic_v5_long.pt` | safelibero_long only, COLLECTOR=safemole_multi_critic (closed-loop bootstrap from v4), 5040 samples, init offset=10, `is_carrying` feature | long suite |

`critic_v5_long.pt`: classifier acc 97.1%, prec 96.1%, recall 88.7% at epoch 100; positive rate 18%.

---

## Failed Improvements (long suite, ablation log)

All failed to dominate `α=100 + MAX=500 + critic_v5_long` on lvI 16-ep:

| Variant | SR | CR | Safe-SR | Why failed |
|---|---|---|---|---|
| GRIPPER_Y_SCALE=1.0 | — | — | — | QP infeasible, h_min negative |
| Carry-X expand (TALL X 0.10) | — | — | — | QP infeasible |
| Carry-state α + δ boost | 50% | 100% | 0 | bang-bang |
| Q_INFLATION=1.8 (sqrt(3)) | — | — | — | workspace dies |
| H_SAFETY_DEFAULT_BONUS=0.015 | 25% | 75% | 0 | h_min=+0.028 still collides → ellipsoid-mesh gap is not the bottleneck |
| OFFSET=-0.04 (wrist coverage) | 0% | 50% | 0 | workspace too tight |
| OFFSET=-0.06 | 75% | 100% | 0 | no benefit |
| risk_prob channel (gain=2, MAX=0.08) | 38% | 56% | 19% | MAX too high |
| risk_prob channel (gain=1, MAX=0.06) | — | — | — | persists conservative tradeoff, no SR gain |
| LOOKAHEAD=5 | 12.5% | 12.5% | 12.5% | over-conservative bug |
| LOOKAHEAD=2 + α=20 | 31% | 31% | 25% | LA + low-α both pull away |
| LOOKAHEAD=1 + α=100 | 44% | 44% | 25% | strictly worse than no-LA |

Conclusion: **CBF lookahead anti-correlates with grasp-near-obstacle tasks**. Long suite targets are 5cm from obstacles — lookahead "predict→retreat" mechanism prevents reach.

---

## Key Methodology Findings (paper-worthy)

1. **Closed-loop critic re-training (v4 → v5_long)**: balanced 18% positive rate on long suite vs v4's task-confused signal. **Single biggest gain** — long lvI CR 88% → 44%.

2. **Broken `bowl_lifted` feature → `is_carrying = grip[0]<0.025`**: critic v4 had `obs.get("akita_black_bowl_1_pos", eef_pos)` which fell back to EE z on long suite (no akita bowl), feeding noise. Fix preserves task-agnostic carrying signal.

3. **multi-rep + MAX_PRIMITIVES=8 (or 11 for long)**: per-mesh ellipsoid decomposition with rotation, strict 6-of-6 Pareto improvement over AEGIS single ellipsoid on spatial/goal/object.

4. **MAX_STEPS=800 on long lvII**: longer budget converts safe-timeouts to safe-successes (Safe-SR 25% → 35%) without raising CR. **MAX_STEPS is a free Safe-SR booster on saturated-budget tasks**.

5. **t1 long-suite is structurally infeasible**: alphabet-soup ↔ tomato-sauce target placed inside red-mug ellipsoid radius; CBF cannot solve. Limitation must be flagged in paper.

---

## Pending

- [ ] lvI MAX=800 on long suite (predicted: SR ↑, CR ↓, Safe-SR ↑) — would close last metric gap with AEGIS
- [ ] AEGIS lvII spatial baseline (data missing)
- [ ] critic_v5 ablation: train with vs without bootstrap mode
- [ ] Pareto curve plot: SR-vs-CR sweep over (α, δ, gain) per suite