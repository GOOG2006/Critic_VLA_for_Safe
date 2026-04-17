"""Full AEGIS-style 9-DoF QP-CBF with ellipsoid geometry, adapted for our Safe-MoLe framework.

Usage:
    python eval_aegis_safemole.py <mode> <level> <n_eps>
    mode ∈ {baseline, aegis, safemole}
      - baseline: OpenVLA only, no safety
      - aegis: AEGIS-style 9-DoF CBF using GT obstacle positions (no critic)
      - safemole: aegis CBF + critic-predicted early-warning augmentation
"""
import os, json, sys, numpy as np, torch, torch.nn as nn
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_TF"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import cvxpy as cp
from scipy.spatial.transform import Rotation as Rot
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os.path as op
from transformers import AutoModelForVision2Seq, AutoProcessor
from safe_mole.openvla_adapter import prepare_openvla_input, openvla_predict
from safe_mole.train_critic_libero import CognitionHook

# ------------------------------------------------------------------
# Vendored AEGIS CBF math (from vlsa-aegis/main/utils.py)
# ------------------------------------------------------------------

def vector_hat(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def project_matrix(z):
    z = np.asarray(z, dtype=np.float64).reshape(3, 1)
    return np.eye(3) - z @ z.T


def compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z_ij):
    Q_i = np.diag(Q_i_diag); Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    z = z_ij / max(np.linalg.norm(z_ij), 1e-10)
    term1 = np.linalg.norm(Qbar_j @ Qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ Qbar_i_inv @ z
    denom = np.linalg.norm(Qbar_i_inv @ z) + 1e-10
    return float((-term1 + term2 - 1.0) / denom)


def compute_h_coeffs_3d(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z, eps=1e-10):
    Q_i = np.diag(Q_i_diag); Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    Qbar_i_inv2 = Qbar_i_inv @ Qbar_i_inv
    Qbar_j2 = Qbar_j @ Qbar_j
    z = z / (np.linalg.norm(z) + eps)
    a_vec = Qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = (1.0 - (p_j - p_i).T @ a_vec + term1)
    eta_row = -(1.0 / denom) * (z.T @ Qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ Qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ Qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ Qbar_i_inv @ Qbar_j2 @ Qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3
    tmp1 = z.T @ Qbar_i_inv2 @ vector_hat(z)
    left_vec = (z.T @ Qbar_i_inv @ Qbar_j2)
    Ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (Ja_vec - Qbar_i_inv @ vector_hat(z))
    part_a = (p_j - p_i).T @ Qbar_i_inv @ vector_hat(z)
    part_b = z.T @ Qbar_i_inv @ vector_hat(p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = rho * (1.0 / (denom**3 + eps)) * tmp1 + (1.0 / sigma) * tmp2 + (1.0 / denom) * tmp3
    a_v = (eta_row @ R_i).ravel()
    a_omega = zeta_tilde @ R_i
    a_uz = (mu_row @ project_matrix(z)).ravel()
    h = compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z)
    return a_v, a_omega, a_uz, h, mu_row


# ------------------------------------------------------------------
# Safe-MoLe critic (trained on SafeLIBERO features)
# ------------------------------------------------------------------

CKPT = "/root/autodl-tmp/openvla_libero_spatial"
CRITIC_DIR = Path("/root/autodl-tmp/safe_mole_libero/critic_safelibero")

# Gripper ellipsoid (AEGIS defaults)
GRIPPER_OFFSET = np.array([0.0, 0.0, 0.10], dtype=np.float64)    # 10cm toward forearm (local +z = away from fingertip)
GRIPPER_Q_DIAG = np.array([0.10, 0.10, 0.25], dtype=np.float64)  # elongated 10×10×25cm to cover gripper+wrist+forearm
OBSTACLE_Q_DIAG = np.array([0.10, 0.10, 0.14], dtype=np.float64)
ALPHA_H = 10.0
H_SAFETY_MARGIN = 0.1       # light buffer so QP doesn't over-react when d is large


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


def solve_qp_cbf(v_ref_local, omega_ref, uz_ref, p1, Q1_diag, R1, obstacles_pos, obstacles_R, obstacles_Q, z_fixed, w_uz=0.1):
    """Solve AEGIS-style 9-DoF QP with CBF constraint. v_ref_local is in EE-local frame.

    Variables: u = [v(3), omega(3), uz(3)] (local frame) + slack (nonneg).
    Minimize: weighted(u - u_ref) + 1000 * slack^2.
    Subject to: sum over obstacles (0.2·a_v·v + 0.2·a_omega·omega + a_uz·uz + alpha*(h-margin) + slack) >= 0.
    """
    n_obs = len(obstacles_pos)
    if n_obs == 0:
        return v_ref_local, omega_ref, uz_ref, 10.0, 0.0

    u = cp.Variable(9)
    slack = cp.Variable(n_obs, nonneg=True)
    # Position weight raised to follow OpenVLA nominal (SR).
    # uz weight reduced (w_uz=0.1) so QP is free to steer z_fixed toward goal (detour around obstacle).
    W = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25, w_uz, w_uz, w_uz])
    u_ref = np.hstack([v_ref_local * 5.0, omega_ref * 5.0, uz_ref * 10.0])
    cost = cp.quad_form(u - u_ref, W) + 1000.0 * cp.sum_squares(slack)

    constraints = []
    h_values = []
    for i, (p_obs, R_obs, Q_obs) in enumerate(zip(obstacles_pos, obstacles_R, obstacles_Q)):
        a_v, a_omega, a_uz, h, _ = compute_h_coeffs_3d(p1, Q1_diag, R1, p_obs, Q_obs, R_obs, z_fixed)
        constraints.append(
            0.2 * a_v @ u[:3] + 0.2 * a_omega @ u[3:6] + a_uz @ u[6:] + ALPHA_H * (h - H_SAFETY_MARGIN) + slack[i] >= 0
        )
        h_values.append(h)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return v_ref_local, omega_ref, uz_ref, min(h_values), 0.0

    if u.value is None:
        return v_ref_local, omega_ref, uz_ref, min(h_values), 0.0
    total_slack = float(np.sum(slack.value)) if slack.value is not None else 0.0
    return u.value[:3], u.value[3:6], u.value[6:], min(h_values), total_slack


def run(mode, level, n_eps, out_path):
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    print(f"[{mode}] VLA loaded", flush=True)

    critic = None; sigma_h_bar = 0.0
    if mode == "safemole":
        critic = Critic().to("cuda:0")
        critic.load_state_dict(torch.load(CRITIC_DIR / "critic.pt", map_location="cuda:0"))
        critic.eval()
        cal = json.load(open(CRITIC_DIR / "calibration.json"))
        sigma_h_bar = cal["sigma_h_bar"]
        print(f"[{mode}] critic loaded, sigma_h_bar={sigma_h_bar:.4f}", flush=True)

    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level=level)
    results = []
    task_range = [int(os.environ.get("DEBUG_TASK", -1))] if int(os.environ.get("DEBUG_TASK", -1)) >= 0 else range(suite.n_tasks)

    for task_idx in task_range:
        task = suite.get_task(task_idx)
        bddl = op.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        except Exception as e:
            print(f"[{mode}] t{task_idx} env init failed: {e}", flush=True)
            continue
        env.seed(0)
        init_states = suite.get_task_init_states(task_idx)
        print(f"[{mode}] lv{level} t{task_idx}: {task.language}", flush=True)

        for ep in range(n_eps):
            try:
                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(10):
                    obs, _, _, _ = env.step([0] * 6 + [-1])
            except Exception as e:
                print(f"  ep{ep}: reset failed: {e}", flush=True)
                continue

            obs_keys = [k for k in obs.keys() if "obstacle" in k and k.endswith("_pos") and "to_" not in k]
            active_keys = [k for k in obs_keys if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
            # Only use ACTIVE obstacles for collision check (others are placed outside workspace)
            init_obs_pos = {k: obs[k].copy() for k in active_keys}

            success, collided = False, False
            n_proj = 0
            h_min_trace = 10.0

            # z_fixed: initial direction from gripper to nearest obstacle
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = Rot.from_quat(eef_quat).as_matrix()
            p1 = eef_pos + R1 @ GRIPPER_OFFSET
            if active_keys:
                obs_pos_list = [np.asarray(obs[k][:3], dtype=np.float64) for k in active_keys]
                idx0 = int(np.argmin([np.linalg.norm(p1 - p) for p in obs_pos_list]))
                z_fixed = obs_pos_list[idx0] - p1
                if np.linalg.norm(z_fixed) > 1e-6:
                    z_fixed = z_fixed / np.linalg.norm(z_fixed)
                else:
                    z_fixed = np.array([1.0, 0.0, 0.0])
            else:
                z_fixed = np.array([1.0, 0.0, 0.0])

            for step in range(220):
                inputs = prepare_openvla_input(obs, task.language, processor, "cuda:0")
                if mode == "safemole":
                    with CognitionHook(model) as hook:
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
                    h_critic_margin = max(0.0, sigma_h_bar - h_pred)  # extra safety when critic uncertain
                else:
                    h_critic_margin = 0.0

                a_nom = openvla_predict(model, inputs, "libero_spatial")

                if mode == "baseline":
                    a_exec = a_nom
                    hmin_step = 10.0
                else:
                    # Gripper ellipsoid pose
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    R1 = Rot.from_quat(eef_quat).as_matrix()
                    p1 = eef_pos + R1 @ GRIPPER_OFFSET
                    # Obstacles ellipsoids (identity rotation, uniform size)
                    if active_keys:
                        obs_pos_arr = [np.asarray(obs[k][:3], dtype=np.float64) for k in active_keys]
                        obs_R_arr = [np.eye(3)] * len(active_keys)
                        obs_Q_arr = [OBSTACLE_Q_DIAG] * len(active_keys)
                    else:
                        obs_pos_arr, obs_R_arr, obs_Q_arr = [], [], []

                    # a_nom[:3] is in WORLD frame. Convert to LOCAL for QP.
                    v_ref_local = (R1.T @ a_nom[:3].astype(np.float64))
                    omega_ref = a_nom[3:6].astype(np.float64)

                    # Goal-aware uz_ref: steer z_fixed toward obstacle→goal tangent so
                    # CBF escape direction gradually rotates AROUND the obstacle toward the goal.
                    bowl_pos_arr = np.asarray(obs.get("akita_black_bowl_1_pos", eef_pos)[:3], dtype=np.float64)
                    plate_pos_arr = np.asarray(obs.get("plate_1_pos", eef_pos)[:3], dtype=np.float64)
                    bowl_lifted = float(bowl_pos_arr[2]) - 0.85 > 0.05
                    goal_pos = plate_pos_arr if bowl_lifted else bowl_pos_arr
                    to_goal_world = goal_pos - p1
                    dg = float(np.linalg.norm(to_goal_world))
                    if dg > 1e-6:
                        g_hat_world = to_goal_world / dg
                        # Tangent = component perpendicular to z_fixed (in z_fixed's tangent plane)
                        tangent = g_hat_world - float(g_hat_world.dot(z_fixed)) * z_fixed
                        nt = float(np.linalg.norm(tangent))
                        uz_ref = (tangent / nt) if nt > 1e-6 else np.zeros(3)
                    else:
                        uz_ref = np.zeros(3)

                    u_v, u_omega, u_uz, h_min, total_slack = solve_qp_cbf(
                        v_ref_local, omega_ref, uz_ref, p1, GRIPPER_Q_DIAG, R1,
                        obs_pos_arr, obs_R_arr, obs_Q_arr, z_fixed,
                    )
                    hmin_step = h_min
                    h_min_trace = min(h_min_trace, h_min)

                    # Apply critic margin (safemole): tighten constraint by artificially reducing h
                    if mode == "safemole" and h_critic_margin > 0 and h_min < h_critic_margin:
                        # Safety activated by critic even if analytic h is ok — add lift bias
                        u_uz = u_uz + np.array([0.0, 0.0, 0.5])

                    # Update z_fixed toward goal tangent (AEGIS dynamic steering)
                    dz = (np.eye(3) - np.outer(z_fixed, z_fixed)) @ u_uz
                    z_fixed = z_fixed + dz * 0.2                      # faster rotation (0.05 → 0.2)
                    z_fixed = z_fixed / max(np.linalg.norm(z_fixed), 1e-6)

                    a_exec = a_nom.copy().astype(np.float32)
                    a_exec[:3] = (0.2 * R1 @ u_v).astype(np.float32)
                    a_exec[3:6] = (0.2 * u_omega).astype(np.float32)
                    if np.linalg.norm(a_exec[:3] - a_nom[:3]) > 1e-4:
                        n_proj += 1
                    # Debug log every 20 steps
                    if step % 20 == 0 and active_keys:
                        nearest = np.argmin([np.linalg.norm(p1 - np.asarray(obs[k][:3])) for k in active_keys])
                        obs_p = np.asarray(obs[active_keys[nearest]][:3])
                        d3 = float(np.linalg.norm(p1 - obs_p))
                        print(
                            f"  [s{step}] eef={eef_pos.round(3).tolist()} p1={p1.round(3).tolist()} "
                            f"obs={obs_p.round(3).tolist()} d={d3:.3f} h={h_min:.3f} slack={total_slack:.3f} "
                            f"a_nom_xyz={a_nom[:3].round(3).tolist()} a_exec_xyz={a_exec[:3].round(3).tolist()}",
                            flush=True,
                        )

                obs, _, done, _ = env.step(a_exec.tolist())
                for k in active_keys:
                    disp = np.sum(np.abs(obs[k] - init_obs_pos[k]))
                    if disp > 0.001:
                        if not collided:
                            print(f"  FIRST COLLISION at step {step}: obstacle {k} displaced {disp:.4f}", flush=True)
                        collided = True
                if done:
                    success = True
                    break

            print(f"  ep{ep}: success={success} steps={step + 1} collided={collided} proj={n_proj} h_min={h_min_trace:.3f}", flush=True)
            results.append({
                "task_idx": task_idx, "ep": ep, "success": success, "collided": collided,
                "steps": step + 1, "n_proj": n_proj, "h_min": h_min_trace,
            })
        try:
            env.close()
        except Exception:
            pass

    n = len(results)
    sr = sum(r["success"] for r in results)
    cr = sum(r["collided"] for r in results)
    print(f"\n=== {mode} on SafeLIBERO-spatial Level {level} ===")
    print(f"N={n}  SR = {sr}/{n} = {sr/max(1,n):.3f}  CR = {cr}/{n} = {cr/max(1,n):.3f}")
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
    out = f"/root/autodl-tmp/safe_mole_libero/eval/safelibero_aegis_{mode}_lv{level}/results.json"
    run(mode, level, n_eps, out)
