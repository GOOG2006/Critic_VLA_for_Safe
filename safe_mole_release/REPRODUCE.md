# Reproduction Guide — Best Result per (Suite × Level)

All commands assume the prerequisites in `README.md` are met (Evo1 conda env, OpenPI policy server on `127.0.0.1:8000`, code at `/root/autodl-tmp/MysafeVLA/`, critic checkpoints in `aegis_runs/critic_v4/` and `aegis_runs/critic_v5_long/`).

## Master table — 4 suites × 2 levels = 8 best results

| # | Suite | Level | SR | CR | Safe-SR | Run command (from `safe_mole_release/`) | Result tag (in `aegis_runs/paper_runs/`) |
|---|---|---|---|---|---|---|---|
| 1 | spatial | I | 62% | 25% | 60% | `bash scripts/critic_v4_spatial.sh` | `critic_v4_spatial_lvI/` |
| 2 | spatial | II | **92%** | 22% | **78%** | `bash scripts/critic_v4_spatial.sh` | `critic_v4_spatial_lvII/` |
| 3 | goal | I | **88%** | **5%** | **85%** | `bash scripts/aegisperc_v2.sh` | `aegisperc_v2_goal_lvI/` |
| 4 | goal | II | **92%** | 25% | 75% | `bash scripts/paper_runs.sh` | `g050_lvII/` |
| 5 | object | I | 62% | **8%** | **62%** | `bash scripts/alpha50_object_full.sh` | `alpha50_obj_full_lvI/` |
| 6 | object | II | 80% | 25% | 68% | `bash scripts/alpha50_object_full.sh` | `alpha50_obj_full_lvII/` |
| 7 | long | I | 50% | 44% | 31% | `bash scripts/long_BEST.sh I 10` | `long_BEST_10ep_lvI/` |
| 8 | long | II | 38% | **20%** | **35%** | `bash scripts/long_BEST_max800.sh II 10` | `long_BEST_max800_10ep_lvII/` |

Each cell of `aegis_runs/paper_runs/<tag>/` contains the `pi05_safemole_multi_critic_lv*/results.json` (or `pi05_safemole_critic_lv*/results.json` for goal-lvI single-critic) listing all 40 episodes with `success`, `collided`, `steps`, `h_min`, `n_proj`, etc.

## One-shot full reproduction (4 suites × 2 levels = 8 runs ≈ 4–5 hours total)

```bash
cd /root/autodl-tmp/MysafeVLA  # or wherever the release lives

# spatial (1 + 2)
bash safe_mole_release/scripts/critic_v4_spatial.sh

# goal (3 best uses single-critic; 4 best uses g-sweep with gain=0.5)
bash safe_mole_release/scripts/aegisperc_v2.sh
bash safe_mole_release/scripts/paper_runs.sh

# object (5 + 6)
bash safe_mole_release/scripts/alpha50_object_full.sh

# long (7 + 8)
bash safe_mole_release/scripts/long_BEST.sh I 10
bash safe_mole_release/scripts/long_BEST_max800.sh II 10
```

## Critic re-training (only if you want to reproduce from scratch)

`critic_v4.pt` and `critic_v5_long.pt` are checked into `critic_ckpts/`. To re-train:

```bash
# critic_v4 (used for spatial/goal/object) — see paper_runs_3suites.sh for upstream collection.

# critic_v5_long (used for long) — closed-loop bootstrap from v4, ~10 min on one A800:
bash safe_mole_release/scripts/collect_critic_v5_long.sh
# Output: aegis_runs/critic_v5_long/critic_v5_long.pt + states/labels/h_future .npy
```

`critic_v5_long` was trained on 5040 samples (24 closed-loop trajectories, init_states 10..12 — disjoint from eval init_states 0..9). Final classifier: 97.1% acc, 96.1% precision, 88.7% recall.

## Key metrics — what each measures

| Metric | Formula | Target |
|---|---|---|
| **SR** (Success Rate) | `#(success=True) / N` | high |
| **CR** (Collision Rate) | `#(collided=True) / N` | low |
| **Safe-SR** | `#(success=True ∧ collided=False) / N` | high — completed task without ever touching an obstacle |
| **h_min** | min over episode of CBF barrier value `h(state)`. positive = always inside safe zone | positive |
| **n_proj** | per-episode count of CBF projection events (action filtered) | descriptive |

Episode is a "safe-timeout" if `success=False ∧ collided=False ∧ steps == MAX_STEPS`. Useful pareto-friendly outcome.

## Quick smoke test (4 episodes per task, ~10 min)

To verify the setup before the full 40-ep paper runs:

```bash
bash safe_mole_release/scripts/long_BEST.sh I 4   # long lvI, 4 init_states/task = 16 ep total
```

Expected: SR ≈ 50%, CR ≈ 44%, Safe-SR ≈ 31% (with ±10% sampling noise).

## See also

- `PAPER_RESULTS.md` (project root) — full Pareto table vs AEGIS, ablation log of failed attempts
- `safe_mole_release/README.md` — env-var glossary, file layout, methodology notes
