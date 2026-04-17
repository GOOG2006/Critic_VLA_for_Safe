"""Collect baseline rollout data on SafeLIBERO, train anticipatory critic, save weights."""
import os, json, math, sys, numpy as np, torch, torch.nn as nn
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_TF"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation as Rot
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os.path as op
from openpi_client import websocket_client_policy as _wcp
from openpi_client import image_tools

sys.path.insert(0, "/root/autodl-tmp/vlsa-aegis/main")
from utils import compute_h_ij, get_point_cloud, filtering_points, fit_ellipse, obstacle_detection

GRIPPER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
GRIPPER_Q_DIAG = np.array([0.06, 0.12, 0.11], dtype=np.float64)
REPLAN_STEPS = 5
LOOKAHEAD = 10  # predict collision within next K steps
OUT_DIR = Path("/root/autodl-tmp/MysafeVLA/critic_v2")


def quat2axisangle(quat):
    if quat[3] > 1: quat[3] = 1
    elif quat[3] < -1: quat[3] = -1
    den = math.sqrt(1 - quat[3]**2)
    if abs(den) < 1e-8: return np.zeros(3)
    return (quat[:3] * 2 * math.acos(quat[3])) / den


def build_element(obs, task_lang):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": str(task_lang)}


def compute_h(obs, active_keys, p2, Q2_diag, R2):
    eef = obs["robot0_eef_pos"]
    R1 = Rot.from_quat(obs["robot0_eef_quat"]).as_matrix()
    p1 = np.asarray(eef[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
    if p2 is not None:
        z = p2 - p1
        if np.linalg.norm(z) > 1e-6:
            return compute_h_ij(p1, GRIPPER_Q_DIAG, R1, p2, Q2_diag, R2, z)
    return 10.0


def collect_data():
    print("[collect] connecting to pi05 server...", flush=True)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", 8000)
    from groundingdino.util.inference import load_model
    gdino = load_model(
        "/root/autodl-tmp/vlsa-aegis/GroundingDINO/GroundingDINO_SwinT_OGC.py",
        "/root/autodl-tmp/vlsa-aegis/GroundingDINO/groundingdino_swint_ogc.pth",
    )
    print("[collect] models loaded", flush=True)

    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="II")
    all_samples = []

    for task_idx in range(suite.n_tasks):
        task = suite.get_task(task_idx)
        bddl = op.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                     camera_depths=True, camera_names=["agentview", "robot0_eye_in_hand", "backview"])
        except Exception as e:
            print(f"  t{task_idx} env fail: {e}", flush=True)
            continue
        env.seed(0)
        init_states = suite.get_task_init_states(task_idx)
        print(f"[collect] t{task_idx}: {task.language}", flush=True)

        for ep in range(3):
            try:
                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(10):
                    obs, _, _, _ = env.step([0]*6+[-1])
            except Exception as e:
                print(f"  ep{ep} reset fail: {e}", flush=True)
                continue

            # Perception (once per episode)
            obs_keys = [k for k in obs.keys() if "obstacle" in k and k.endswith("_pos") and "to_" not in k]
            active_keys = [k for k in obs_keys if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
            init_obs_pos = {k: obs[k].copy() for k in active_keys}

            import pathlib
            img_out = pathlib.Path(f"/tmp/critic_collect/t{task_idx}_ep{ep}")
            img_out.mkdir(parents=True, exist_ok=True)
            agv = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            agd = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
            obs_info = obstacle_detection(agv, task.language, "safelibero_spatial")
            apt = get_point_cloud(agv, agd, env, "agentview", obs_info, gdino, img_out)
            bv = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
            bd = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
            bpt = get_point_cloud(bv, bd, env, "backview", obs_info, gdino, img_out)
            if apt.shape[1] > 0 and bpt.shape[1] > 0:
                fp = np.vstack([apt, bpt])
            elif apt.shape[1] > 0:
                fp = apt
            elif bpt.shape[1] > 0:
                fp = bpt
            else:
                fp = np.array([[]])
            p2 = R2 = Q2_diag = None
            if fp.ndim == 2 and fp.shape[0] > 0 and fp.shape[1] == 3:
                filt = filtering_points(fp, "safelibero_spatial")
            else:
                filt = np.array([]).reshape(0, 3)
            if filt.shape[0] > 0:
                p2, R2, Q2_diag = fit_ellipse(filt, plot=False)

            # Rollout (baseline, no safety)
            action_plan = deque()
            trajectory = []  # (state_vec, h, collided_at_step)

            for step in range(220):
                if not action_plan:
                    element = build_element(obs, task.language)
                    ac = client.infer(element)["actions"]
                    action_plan.extend(ac[:REPLAN_STEPS])
                a = np.asarray(action_plan.popleft(), dtype=np.float64)

                # State features
                eef = obs["robot0_eef_pos"]
                eef_q = obs["robot0_eef_quat"]
                grip = obs["robot0_gripper_qpos"]
                h = compute_h(obs, active_keys, p2, Q2_diag, R2)
                bowl_pos = obs.get("akita_black_bowl_1_pos", eef)[:3]
                bowl_lifted = float(bowl_pos[2]) - 0.85 > 0.05

                # Check collision
                collided_now = False
                for k in active_keys:
                    if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                        collided_now = True

                state_vec = np.concatenate([
                    eef[:3],                      # 3: eef pos
                    quat2axisangle(np.array(eef_q, dtype=np.float64)),  # 3: eef ori
                    grip[:2],                     # 2: gripper
                    a[:7],                        # 7: current action
                    [h],                          # 1: analytic h
                    [float(bowl_lifted)],         # 1: task phase
                ]).astype(np.float32)             # total: 17

                trajectory.append({
                    "state": state_vec,
                    "step": step,
                    "h": h,
                    "collided": collided_now,
                })

                obs, _, done, _ = env.step(a.tolist())
                if done:
                    break

            # Label: for each step, will collision happen within next LOOKAHEAD steps?
            for i, t in enumerate(trajectory):
                future = trajectory[i:i+LOOKAHEAD]
                t["future_collision"] = any(f["collided"] for f in future)
                t["h_future_min"] = min(f["h"] for f in future) if future else t["h"]

            all_samples.extend(trajectory)
            n_col = sum(1 for t in trajectory if t["collided"])
            n_fc = sum(1 for t in trajectory if t["future_collision"])
            print(f"  ep{ep}: {len(trajectory)} steps, {n_col} collided, {n_fc} future_collision", flush=True)

        try:
            env.close()
        except:
            pass

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states = np.stack([s["state"] for s in all_samples])
    labels = np.array([s["future_collision"] for s in all_samples], dtype=np.float32)
    h_future = np.array([s["h_future_min"] for s in all_samples], dtype=np.float32)
    np.save(OUT_DIR / "states.npy", states)
    np.save(OUT_DIR / "labels.npy", labels)
    np.save(OUT_DIR / "h_future.npy", h_future)
    print(f"\n[collect] saved {len(states)} samples to {OUT_DIR}")
    print(f"  positive rate (future_collision): {labels.mean():.3f}")
    print(f"  h_future_min range: [{h_future.min():.3f}, {h_future.max():.3f}]")
    return states, labels, h_future


def train_critic(states, labels, h_future):
    """Train binary classifier: predict whether collision will happen in next K steps."""
    print("\n[train] training critic...", flush=True)

    class SafetyCritic(nn.Module):
        def __init__(self, d_in=17, d_hid=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.ReLU(),
                nn.Linear(d_hid, d_hid), nn.ReLU(),
                nn.Linear(d_hid, 2),  # [collision_risk, h_predicted]
            )
        def forward(self, x):
            out = self.net(x)
            return out[:, 0], out[:, 1]  # risk_logit, h_pred

    N = len(states)
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr, te = idx[:n_train], idx[n_train:]

    X_tr = torch.from_numpy(states[tr]).float()
    y_tr_cls = torch.from_numpy(labels[tr]).float()
    y_tr_h = torch.from_numpy(h_future[tr]).float()
    X_te = torch.from_numpy(states[te]).float()
    y_te_cls = torch.from_numpy(labels[te]).float()
    y_te_h = torch.from_numpy(h_future[te]).float()

    critic = SafetyCritic()
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        critic.train()
        perm = torch.randperm(X_tr.size(0))
        losses = []
        for i in range(0, X_tr.size(0), 256):
            b = perm[i:i+256]
            risk, h_pred = critic(X_tr[b])
            loss_cls = bce(risk, y_tr_cls[b])
            loss_h = ((h_pred - y_tr_h[b]) ** 2).mean()
            loss = loss_cls + 0.5 * loss_h
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            critic.eval()
            with torch.no_grad():
                risk_te, h_te = critic(X_te)
                pred_cls = (risk_te > 0).float()
                acc = (pred_cls == y_te_cls).float().mean().item()
                # Precision/recall for collision prediction
                tp = ((pred_cls == 1) & (y_te_cls == 1)).sum().item()
                fp = ((pred_cls == 1) & (y_te_cls == 0)).sum().item()
                fn = ((pred_cls == 0) & (y_te_cls == 1)).sum().item()
                prec = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
            print(f"  epoch {epoch+1}: loss={np.mean(losses):.4f} acc={acc:.3f} prec={prec:.3f} recall={recall:.3f}")

    torch.save(critic.state_dict(), OUT_DIR / "critic_v2.pt")
    print(f"[train] saved to {OUT_DIR / 'critic_v2.pt'}")
    return critic


if __name__ == "__main__":
    states, labels, h_future = collect_data()
    train_critic(states, labels, h_future)
