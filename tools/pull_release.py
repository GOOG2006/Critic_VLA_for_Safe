"""Pull all release files from remote MysafeVLA into safe_mole_release/."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from remote_exec import load_cfg, connect
from pull_via_ssh import pull_one

REMOTE_ROOT = "/root/autodl-tmp/MysafeVLA"
LOCAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "safe_mole_release"

PULLS = [
    # core eval + collector
    (f"{REMOTE_ROOT}/scripts/eval_pi05_safelibero.py", "scripts/eval_pi05_safelibero.py"),
    (f"{REMOTE_ROOT}/scripts/obstacle_primitives.json", "scripts/obstacle_primitives.json"),
    (f"{REMOTE_ROOT}/scripts/collect_and_train_critic.py", "scripts/collect_and_train_critic.py"),
    # winning shell scripts (one per suite)
    (f"{REMOTE_ROOT}/scripts/critic_v4_g010.sh", "scripts/critic_v4_g010.sh"),
    (f"{REMOTE_ROOT}/scripts/aegisperc_v2.sh", "scripts/aegisperc_v2.sh"),
    (f"{REMOTE_ROOT}/scripts/alpha50_object_full.sh", "scripts/alpha50_object_full.sh"),
    (f"{REMOTE_ROOT}/scripts/paper_runs.sh", "scripts/paper_runs.sh"),
    (f"{REMOTE_ROOT}/scripts/paper_runs_3suites.sh", "scripts/paper_runs_3suites.sh"),
    (f"{REMOTE_ROOT}/scripts/long_BEST.sh", "scripts/long_BEST.sh"),
    (f"{REMOTE_ROOT}/scripts/long_BEST_max800.sh", "scripts/long_BEST_max800.sh"),
    (f"{REMOTE_ROOT}/scripts/collect_critic_v5_long.sh", "scripts/collect_critic_v5_long.sh"),
    # safe_mole module
    (f"{REMOTE_ROOT}/safe_mole/__init__.py", "safe_mole/__init__.py"),
    (f"{REMOTE_ROOT}/safe_mole/barriers.py", "safe_mole/barriers.py"),
    (f"{REMOTE_ROOT}/safe_mole/calibration_collector.py", "safe_mole/calibration_collector.py"),
    (f"{REMOTE_ROOT}/safe_mole/calibrate_libero.py", "safe_mole/calibrate_libero.py"),
    (f"{REMOTE_ROOT}/safe_mole/cogact_adapter.py", "safe_mole/cogact_adapter.py"),
    (f"{REMOTE_ROOT}/safe_mole/conformal.py", "safe_mole/conformal.py"),
    (f"{REMOTE_ROOT}/safe_mole/critic.py", "safe_mole/critic.py"),
    (f"{REMOTE_ROOT}/safe_mole/integration.py", "safe_mole/integration.py"),
    (f"{REMOTE_ROOT}/safe_mole/openvla_adapter.py", "safe_mole/openvla_adapter.py"),
    (f"{REMOTE_ROOT}/safe_mole/projection.py", "safe_mole/projection.py"),
    (f"{REMOTE_ROOT}/safe_mole/routing.py", "safe_mole/routing.py"),
    (f"{REMOTE_ROOT}/safe_mole/sdf.py", "safe_mole/sdf.py"),
    (f"{REMOTE_ROOT}/safe_mole/train_critic_libero.py", "safe_mole/train_critic_libero.py"),
    # critic checkpoints
    (f"{REMOTE_ROOT}/aegis_runs/critic_v4/critic_v4.pt", "critic_ckpts/critic_v4.pt"),
    (f"{REMOTE_ROOT}/aegis_runs/critic_v5_long/critic_v5_long.pt", "critic_ckpts/critic_v5_long.pt"),
]


def main():
    cfg = load_cfg()
    c = connect(cfg)
    total = 0
    failed = []
    try:
        for remote, rel_local in PULLS:
            local = LOCAL_ROOT / rel_local
            try:
                n = pull_one(c, remote, str(local))
                total += n
                print(f"  ok {n:>8} B  {rel_local}")
            except Exception as e:
                failed.append(rel_local)
                print(f"  ERR  {rel_local}: {e}")
    finally:
        c.close()
    print(f"\nTotal: {total/1024:.1f} KiB across {len(PULLS) - len(failed)} files")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
