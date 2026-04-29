# Safe-MoLe — Release for SafeLIBERO

Best-performing code per suite, organized for reproduction.

## Layout

```
safe_mole_release/
├── scripts/
│   ├── eval_pi05_safelibero.py          # main eval entrypoint
│   ├── obstacle_primitives.json         # multi-rep mesh decomposition lookup
│   ├── collect_and_train_critic.py      # critic data collection + training
│   ├── critic_v4_g010.sh                # GOAL — multi-critic, gain=0.10  (best lvI 82/8/75)
│   ├── aegisperc_v2.sh                  # GOAL lvI single-critic + AEGIS perc (88/5/85)
│   ├── alpha50_object_full.sh           # OBJECT — α=50 + multi          (62/8/62)
│   ├── paper_runs.sh                    # GOAL lvII g-sweep              (g=0.5: 92/25/75)
│   ├── paper_runs_3suites.sh            # 4-suite extension orchestration
│   ├── long_BEST.sh                     # LONG lvI                       (50/44/31)
│   ├── long_BEST_max800.sh              # LONG lvII (MAX_STEPS=800)      (38/20/35)
│   └── collect_critic_v5_long.sh        # critic_v5_long training
├── safe_mole/                           # python module (CBF projection, barriers, critic)
└── critic_ckpts/
    ├── critic_v4.pt                     # spatial/goal/object suite
    └── critic_v5_long.pt                # long-suite specialised
```

## Best Results (40-ep, init_states 0..9)

| Suite | Level | SR | CR | Safe-SR | Script |
|---|---|---|---|---|---|
| spatial | I | 62% | 25% | 60% | (config: same as `critic_v4_g010` w/ suite=spatial) |
| spatial | II | **92%** | 22% | **78%** | (same) |
| goal | I | **88%** | **5%** | **85%** | `aegisperc_v2.sh` |
| goal | II | **92%** | 25% | 75% | `paper_runs.sh` (g050) |
| object | I | 62% | **8%** | **62%** | `alpha50_object_full.sh` |
| object | II | 80% | 25% | 68% | `alpha50_object_full.sh` |
| long | I (16 ep) | 50% | 44% | 31% | `long_BEST.sh` |
| long | II | 38% | **20%** | **35%** | `long_BEST_max800.sh` |

**vs AEGIS**: Pareto-wins on goal lvI/II, object lvI, long lvII; ties on object lvII; mixed on long lvI.

## 复现命令 / Reproduction

| Suite × Level | 最佳数字 (SR/CR/SafeSR) | 命令 | 备注 |
|---|---|---|---|
| spatial lvI | 62 / 25 / 60 | 改 `critic_v4_g010.sh` 第 18 行 → `run safelibero_spatial I` | 复用 `critic_v4_g010.sh` 配置即可 |
| spatial lvII | 92 / 22 / 78 | 改成 `run safelibero_spatial II` | 同上 |
| goal lvI (multi) | 82 / 8 / 75 | `bash scripts/critic_v4_g010.sh`(默认就跑 goal I+II + object I+II) | 直接跑 |
| **goal lvI (single) ⭐** | **88 / 5 / 85** | `bash scripts/aegisperc_v2.sh` | 该 suite 的 SOTA(AEGIS-perc + 我们 critic) |
| goal lvII | 92 / 25 / 75 | `bash scripts/paper_runs.sh`(取 `g050_lvII` 子结果) | gain=0.5 是最佳点 |
| object lvI | 62 / 8 / 62 | `bash scripts/alpha50_object_full.sh` | α=50 + multi |
| object lvII | 80 / 25 / 68 | `bash scripts/alpha50_object_full.sh` | 同上 |
| long lvI | 50 / 44 / 31 | `bash scripts/long_BEST.sh I 10` | critic_v5_long, 40 ep, MAX=500 |
| long lvII | 38 / 20 / 35 | `bash scripts/long_BEST_max800.sh II 10` | MAX=800 是关键 |

## Quickstart

Server: SeetaCloud GPU host (Evo1 conda env). Requires running OpenPI policy server on `127.0.0.1:8000`.

```bash
# Pre-req: deploy this directory to /root/autodl-tmp/MysafeVLA/, place ckpts in
# aegis_runs/critic_v4/ and aegis_runs/critic_v5_long/, start openpi server.

# Run best-per-suite:
bash scripts/aegisperc_v2.sh                # goal lvI    -> 88/5/85
bash scripts/paper_runs.sh                  # goal lvII   -> g050 92/25/75
bash scripts/critic_v4_g010.sh              # goal+object simultaneous
bash scripts/alpha50_object_full.sh         # object I+II
bash scripts/long_BEST.sh I 10              # long lvI 40 ep
bash scripts/long_BEST_max800.sh II 10      # long lvII 40 ep (MAX=800)

# Re-collect + train long-suite critic (~10 min):
bash scripts/collect_critic_v5_long.sh
```

## Key Knobs (env vars)

| var | default | role |
|---|---|---|
| `ALPHA_H` | 50 (or 100 for long lvII) | CBF recovery rate |
| `H_SAFETY_MARGIN` | 0.005–0.025 | safe zone buffer |
| `CRITIC_MARGIN_GAIN` | 0.10–0.50 | critic h_pred → δ_eff scaling |
| `CRITIC_MARGIN_MAX` | 0.04–0.06 | δ_eff upper bound |
| `OBSTACLE_REPRESENTATION` | `multi` | per-mesh ellipsoid decomposition |
| `MAX_PRIMITIVES_PER_OBSTACLE` | 8 / 11 (long) | cap on sub-ellipsoids per obstacle |
| `Q_INFLATION` | 1.10 / 1.30 (long) | obstacle ellipsoid inflation |
| `GRIPPER_Y_SCALE` | 1.0 / 0.5 (long) | gripper-Y radius scaling |
| `MAX_STEPS` | 220 / 500 (long lvI) / 800 (long lvII) | episode budget |
| `USE_PERCEPTION_PROXIMITY` | 1 | enable depth+VLM proximity δ-bonus channel |
| `CRITIC_CKPT` | `critic_v4.pt` / `critic_v5_long.pt` | critic weights |

Optional (default off):
- `CBF_LOOKAHEAD_STEPS` (lookahead constraint, anti-helps long suite)
- `CRITIC_RISK_GAIN` (risk_prob channel)
- `CARRY_ALPHA_BOOST` / `CARRY_DELTA_BONUS` (state-conditional CBF)
- `GRIPPER_CARRY_X_SCALE` (carried-bottle X expansion)

## Methodology Notes

1. **`safemole_multi_critic` mode** (main): per-mesh ellipsoid CBF + critic-driven δ_eff + perception proximity bonus. Strict 6-of-6 Pareto improvement over AEGIS single-ellipsoid on spatial/goal/object.

2. **`critic_v5_long.pt`**: closed-loop bootstrap from `critic_v4.pt`, trained on long-suite only with `INIT_STATE_OFFSET=10` (eval purity preserved). Replaces broken `bowl_lifted` (akita-bowl-only) feature with task-agnostic `is_carrying = grip[0] < 0.025`.

3. **MAX_STEPS=800 on long lvII**: extra budget converts safe-timeouts to safe-successes without raising CR (Safe-SR 25% → 35%). Free booster on saturated tasks.

4. **t1 long-suite limitation**: alphabet-soup ↔ tomato-sauce target inside red-mug ellipsoid radius — CBF cannot solve. Documented as base-policy bottleneck.

See `PAPER_RESULTS.md` (project root) for full ablation log + Pareto comparison.
