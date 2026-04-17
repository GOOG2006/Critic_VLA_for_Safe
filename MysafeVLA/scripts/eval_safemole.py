"""Eval Safe-MoLe (critic + conformal + CBF) on SafeLIBERO.

Clean minimal version: every step, use critic to predict h, apply conformal margin,
and run 3D CBF projection on the action. No detour/escape/stage-machine logic.
"""
import os, json, sys, numpy as np, torch, torch.nn as nn
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_TF"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os.path as op
from transformers import AutoModelForVision2Seq, AutoProcessor
from safe_mole.openvla_adapter import prepare_openvla_input, openvla_predict
from safe_mole.train_critic_libero import CognitionHook

CKPT = "/root/autodl-tmp/openvla_libero_spatial"
CRITIC_DIR = Path("/root/autodl-tmp/safe_mole_libero/critic_safelibero")
SAFETY_RADIUS = 0.12            # gripper effective radius (AEGIS Q1_diag max ≈ 0.12)
GAMMA = 0.5
GRIPPER_OFFSET = np.array([0.0, 0.0, -0.08])  # EE → gripper center (AEGIS offset_local)


def gripper_center(eef_pos, eef_quat=None):
    """Transform eef_pos into gripper-body center (EE + [0,0,-0.08] in EE frame).

    If eef_quat is provided, the offset is rotated into world frame (AEGIS-style).
    If not, we assume gripper pointing down so offset ≈ [0, 0, -0.08] in world.
    """
    if eef_quat is None:
        return np.asarray(eef_pos, dtype=np.float32) + GRIPPER_OFFSET
    from scipy.spatial.transform import Rotation as R
    Rmat = R.from_quat(eef_quat).as_matrix()
    return np.asarray(eef_pos, dtype=np.float32) + (Rmat @ GRIPPER_OFFSET).astype(np.float32)


class Critic(nn.Module):
    def __init__(self, d_in=4096, d_hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_hid), nn.GELU(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cbf_project(a_nom, gripper_pos, obstacles_3d, h, sigma_h_bar=0.0):
    """Minimal 3D CBF projection applied every step using GRIPPER center.

    gripper_pos:  3D world position of gripper body center (EE + [0,0,-0.08]).
    Barrier direction = (gripper - nearest_obstacle)/||.||  (3D unit vector).
    CBF condition: predicted_dh + gamma * (h - sigma_h_bar) >= 0.
    """
    a = a_nom.copy().astype(np.float32)
    if len(obstacles_3d) == 0:
        return a, {"triggered": False, "h": float(h), "displacement": 0.0}
    g = np.asarray(gripper_pos[:3], dtype=np.float32)
    dists = np.linalg.norm(g[None] - obstacles_3d, axis=1)
    idx = int(np.argmin(dists))
    d = float(dists[idx])
    direction = (g - obstacles_3d[idx]) / max(d, 1e-6)
    h_safe = h - sigma_h_bar
    predicted_dh = float(direction.dot(a[:3]))
    slack = predicted_dh + GAMMA * h_safe
    triggered = slack < 0
    disp = 0.0
    if triggered:
        lam = -slack
        a[:3] = a[:3] + lam * direction
        disp = float(abs(lam))
    return a, {"triggered": bool(triggered), "h": float(h), "h_safe": float(h_safe), "displacement": disp}


def run(mode, level, n_eps, out_path):
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    print(f"[{mode}] VLA loaded", flush=True)

    critic = None
    sigma_h_bar = 0.0
    if mode == "safemole":
        critic = Critic().to("cuda:0")
        critic.load_state_dict(torch.load(CRITIC_DIR / "critic.pt", map_location="cuda:0"))
        critic.eval()
        cal = json.load(open(CRITIC_DIR / "calibration.json"))
        sigma_h_bar = cal["sigma_h_bar"]
        print(f"[{mode}] critic loaded, sigma_h_bar={sigma_h_bar:.4f}", flush=True)

    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level=level)
    results = []

    for task_idx in range(suite.n_tasks):
        task = suite.get_task(task_idx)
        bddl = op.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        init_states = suite.get_task_init_states(task_idx)
        print(f"[{mode}] lv{level} t{task_idx}: {task.language}", flush=True)

        for ep in range(n_eps):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(10):
                obs, _, _, _ = env.step([0] * 6 + [-1])

            obs_keys = [k for k in obs.keys() if "obstacle" in k and k.endswith("_pos") and "to_" not in k]
            active_keys = [k for k in obs_keys if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
            init_obs_pos = {k: obs[k].copy() for k in obs_keys}

            success, collided = False, False
            n_proj = 0
            disp_acc = 0.0

            if mode == "safemole":
                with CognitionHook(model) as hook:
                    for step in range(220):
                        inputs = prepare_openvla_input(obs, task.language, processor, "cuda:0")
                        hook.buffer = None
                        with torch.no_grad():
                            _ = model.generate(
                                inputs["input_ids"].to("cuda:0"),
                                max_new_tokens=1, do_sample=False,
                                pixel_values=inputs["pixel_values"],
                            )
                        feat = hook.pooled_feature(inputs["attention_mask"].to("cuda:0"))
                        with torch.no_grad():
                            h_pred = critic(feat).item()
                        a_nom = openvla_predict(model, inputs, "libero_spatial")
                        eef_pos = obs["robot0_eef_pos"]
                        g_center = gripper_center(eef_pos, obs.get("robot0_eef_quat"))
                        obstacles_3d = np.stack([obs[k][:3] for k in active_keys]) if active_keys else np.zeros((0, 3))
                        a_exec, aux = cbf_project(a_nom, g_center, obstacles_3d, h_pred, sigma_h_bar)
                        if aux["triggered"]:
                            n_proj += 1
                            disp_acc += aux["displacement"]
                        obs, _, done, _ = env.step(a_exec.tolist())
                        for k in obs_keys:
                            if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                                collided = True
                        if done:
                            success = True
                            break
            elif mode == "analytic_cbf":
                for step in range(220):
                    inputs = prepare_openvla_input(obs, task.language, processor, "cuda:0")
                    a_nom = openvla_predict(model, inputs, "libero_spatial")
                    eef_pos = obs["robot0_eef_pos"]
                    g_center = gripper_center(eef_pos, obs.get("robot0_eef_quat"))
                    obstacles_3d = np.stack([obs[k][:3] for k in active_keys]) if active_keys else np.zeros((0, 3))
                    if len(obstacles_3d) > 0:
                        d_min = float(np.min(np.linalg.norm(g_center[None] - obstacles_3d, axis=1)))
                        h_true = d_min - SAFETY_RADIUS
                    else:
                        h_true = 10.0
                    a_exec, aux = cbf_project(a_nom, g_center, obstacles_3d, h_true, 0.0)
                    if aux["triggered"]:
                        n_proj += 1
                        disp_acc += aux["displacement"]
                    obs, _, done, _ = env.step(a_exec.tolist())
                    for k in obs_keys:
                        if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                            collided = True
                    if done:
                        success = True
                        break
            else:  # baseline
                for step in range(220):
                    inputs = prepare_openvla_input(obs, task.language, processor, "cuda:0")
                    a_nom = openvla_predict(model, inputs, "libero_spatial")
                    obs, _, done, _ = env.step(a_nom.tolist())
                    for k in obs_keys:
                        if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                            collided = True
                    if done:
                        success = True
                        break

            print(f"  ep{ep}: success={success} steps={step + 1} collided={collided} proj={n_proj} disp={disp_acc:.3f}", flush=True)
            results.append({
                "task_idx": task_idx, "ep": ep, "success": success, "collided": collided,
                "steps": step + 1, "n_proj": n_proj, "disp_acc": disp_acc,
            })
        env.close()

    n = len(results)
    sr = sum(r["success"] for r in results)
    cr = sum(r["collided"] for r in results)
    print(f"\n=== {mode} on SafeLIBERO-spatial Level {level} ===")
    print(f"SR = {sr}/{n} = {sr/n:.3f}")
    print(f"CR = {cr}/{n} = {cr/n:.3f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"results": results, "suite": "safelibero_spatial", "level": level, "mode": mode},
        open(out_path, "w"), indent=2,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "II"
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out = f"/root/autodl-tmp/safe_mole_libero/eval/safelibero_{mode}_lv{level}/results.json"
    run(mode, level, n_eps, out)
