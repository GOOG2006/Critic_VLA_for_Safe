"""SafeLIBERO eval using PI0.5 (lerobot's pi05_libero_base) — baseline (no safety) + Safe-MoLe (AEGIS-style QP-CBF)."""
import os, json, sys, math, numpy as np, torch, torch.nn as nn
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_TF"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from collections import deque
import cvxpy as cp
from scipy.spatial.transform import Rotation as Rot
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os.path as op
from openpi_client import websocket_client_policy as _wcp
from openpi_client import image_tools

# Import config (paths, parameters)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import (
    PI05_HOST, PI05_PORT, GRIPPER_OFFSET, GRIPPER_Q_DIAG_DEFAULT,
    GRIPPER_Q_DIAG_TALL, ALPHA_H, CRITIC_ALPHA_BOOST, CRITIC_CKPT,
    VLSA_AEGIS_ROOT, GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT,
    REPLAN_STEPS,
)
H_SAFETY_MARGIN = 0.0

# AEGIS perception imports (vendored CBF math + AEGIS utils for perception)
from safe_mole.cbf_utils import compute_h_ij, compute_h_coeffs_3d
sys.path.insert(0, os.path.join(VLSA_AEGIS_ROOT, "main"))
from utils import get_point_cloud, filtering_points, fit_ellipse, obstacle_detection


def quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3]*quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def vector_hat(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float64)


def project_matrix(z):
    z = np.asarray(z, dtype=np.float64).reshape(3,1)
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


def solve_qp_cbf(v_ref_local, omega_ref, uz_ref, p1, Q1, R1, obs_pos, obs_R, obs_Q, z_fixed):
    n_obs = len(obs_pos)
    if n_obs == 0:
        return v_ref_local, omega_ref, uz_ref, 10.0, 0.0
    u = cp.Variable(9)
    slack = cp.Variable(n_obs, nonneg=True)
    W = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1])
    u_ref = np.hstack([v_ref_local * 5.0, omega_ref * 5.0, uz_ref * 10.0])
    cost = cp.quad_form(u - u_ref, W) + 1000.0 * cp.sum_squares(slack)
    constraints = []
    h_values = []
    for i, (p, R, Q) in enumerate(zip(obs_pos, obs_R, obs_Q)):
        a_v, a_om, a_uz, h, _ = compute_h_coeffs_3d(p1, Q1, R1, p, Q, R, z_fixed)
        constraints.append(0.2 * a_v @ u[:3] + 0.2 * a_om @ u[3:6] + a_uz @ u[6:] + ALPHA_H * (h - H_SAFETY_MARGIN) + slack[i] >= 0)
        h_values.append(h)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return v_ref_local, omega_ref, uz_ref, min(h_values), 0.0
    if u.value is None:
        return v_ref_local, omega_ref, uz_ref, min(h_values), 0.0
    return u.value[:3], u.value[3:6], u.value[6:], min(h_values), float(np.sum(slack.value)) if slack.value is not None else 0.0


def build_openpi_element(obs, task_lang, resize_size=224):
    """Convert LIBERO obs to openpi element dict (same as AEGIS main_aegis.py)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_lang),
    }


def run(mode, level, n_eps, out_path):
    print(f"[{mode}] connecting to openpi server at {PI05_HOST}:{PI05_PORT}...", flush=True)
    client = _wcp.WebsocketClientPolicy(PI05_HOST, PI05_PORT)
    print(f"[{mode}] connected to openpi pi05 server", flush=True)

    # Load GroundingDINO for perception (same as AEGIS)
    model_groundingdino = None
    critic_model = None
    if mode != "baseline":
        from groundingdino.util.inference import load_model
        model_groundingdino = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT)
        print(f"[{mode}] GroundingDINO loaded", flush=True)
    if mode == "safemole_critic":
        from safe_mole.critic import SafetyCritic
        critic_model = SafetyCritic()
        critic_model.load_state_dict(torch.load(CRITIC_CKPT, map_location="cpu"))
        critic_model.eval()
        print(f"[{mode}] Critic v2 loaded", flush=True)

    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level=level)
    results = []
    task_range = [int(os.environ.get("DEBUG_TASK", -1))] if int(os.environ.get("DEBUG_TASK", -1)) >= 0 else range(suite.n_tasks)

    for task_idx in task_range:
        task = suite.get_task(task_idx)
        bddl = op.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                     camera_depths=True, camera_names=["agentview", "robot0_eye_in_hand", "backview"])
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
                    obs, _, _, _ = env.step([0]*6+[-1])
            except Exception as e:
                print(f"  ep{ep}: reset failed: {e}", flush=True)
                continue

            obs_keys = [k for k in obs.keys() if "obstacle" in k and k.endswith("_pos") and "to_" not in k]
            active_keys = [k for k in obs_keys if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
            init_obs_pos = {k: obs[k].copy() for k in active_keys}

            success, collided = False, False
            n_proj = 0; h_min_trace = 10.0
            action_plan = deque()

            # AEGIS perception: detect obstacle + fit ellipsoid (once per episode)
            flag_safety_control = False
            p2 = R2 = Q2_diag = None
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = Rot.from_quat(eef_quat).as_matrix()
            p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET

            if mode != "baseline" and model_groundingdino is not None:
                import pathlib
                img_out_dir = pathlib.Path(out_path).parent / f"t{task_idx}_ep{ep}"
                img_out_dir.mkdir(parents=True, exist_ok=True)
                agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
                obstacle_info = obstacle_detection(agentview_img, task.language, "safelibero_spatial")
                print(f"  VLM detected: {obstacle_info}", flush=True)
                agent_pts = get_point_cloud(agentview_img, agentview_depth, env, "agentview", obstacle_info, model_groundingdino, img_out_dir)
                backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
                backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
                back_pts = get_point_cloud(backview_img, backview_depth, env, "backview", obstacle_info, model_groundingdino, img_out_dir)
                if agent_pts.shape[1] > 0 and back_pts.shape[1] > 0:
                    full_pts = np.vstack([agent_pts, back_pts])
                elif agent_pts.shape[1] > 0:
                    full_pts = agent_pts
                elif back_pts.shape[1] > 0:
                    full_pts = back_pts
                else:
                    full_pts = np.array([[]])
                filter_pts = filtering_points(full_pts, "safelibero_spatial")
                if filter_pts.shape[0] > 0:
                    p2, R2, Q2_diag = fit_ellipse(filter_pts, plot=True, save_path=img_out_dir)
                    z_fixed = (p2 - p1)
                    z_fixed = z_fixed / max(np.linalg.norm(z_fixed), 1e-6)
                    flag_safety_control = True
                    print(f"  Ellipsoid fitted: p2={p2.round(3)}, Q2={Q2_diag.round(3)}", flush=True)

            if not flag_safety_control:
                z_fixed = np.array([1.0, 0.0, 0.0])

            for step in range(220):
                # Action chunking (same as AEGIS): replan every REPLAN_STEPS
                if not action_plan:
                    element = build_openpi_element(obs, task.language)
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[:REPLAN_STEPS])
                a_nom = np.asarray(action_plan.popleft(), dtype=np.float64)

                if mode == "baseline":
                    a_exec = a_nom
                elif mode in ("safemole", "safemole_critic"):
                    # === AEGIS-faithful safety layer + optional critic ===
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    R1 = Rot.from_quat(eef_quat).as_matrix()
                    p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
                    Q1_diag = GRIPPER_Q_DIAG_TALL if any(w in task.language for w in ["orange juice", "milk", "alphabet soup"]) else GRIPPER_Q_DIAG_DEFAULT
                    if flag_safety_control:
                        v_ref = R1.T @ a_nom[:3]
                        u_v_ref = 5 * v_ref
                        omega_ref = a_nom[3:6]
                        u_omega_ref = 5 * omega_ref
                        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, Q1_diag, R1, p2, Q2_diag, R2, z_fixed)

                        # Critic: predict future collision risk → graduated constraint tightening
                        alpha_effective = ALPHA_H
                        h_margin = 0.0
                        if critic_model is not None:
                            grip = obs["robot0_gripper_qpos"]
                            bowl_pos = obs.get("akita_black_bowl_1_pos", eef_pos)[:3]
                            bowl_lifted = float(bowl_pos[2]) - 0.85 > 0.05
                            state_vec = np.concatenate([
                                eef_pos[:3],
                                quat2axisangle(np.array(eef_quat, dtype=np.float64)),
                                grip[:2], a_nom[:7], [h], [float(bowl_lifted)],
                            ]).astype(np.float32)
                            with torch.no_grad():
                                risk_logit, h_pred = critic_model(torch.from_numpy(state_vec).unsqueeze(0))
                                risk_prob = torch.sigmoid(risk_logit).item()
                                h_pred_val = h_pred.item()
                            # Graduated response: scale alpha smoothly with risk probability
                            RISK_THRESHOLD = 0.6
                            if risk_prob > RISK_THRESHOLD:
                                t = (risk_prob - RISK_THRESHOLD) / (1.0 - RISK_THRESHOLD)
                                boost = 1.0 + 1.5 * t  # max boost = 2.5x
                                alpha_effective = ALPHA_H * boost

                        a_u_v = 0.2 * a_v
                        a_u_omega = 0.2 * a_omega
                        u_z_nom = 10 * mu_row
                        u = cp.Variable(9)
                        W = np.diag([1./25]*6 + [1.]*3)
                        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
                        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                        constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + alpha_effective * (h - h_margin) >= 0]
                        prob = cp.Problem(objective, constraints)
                        try:
                            prob.solve(solver=cp.OSQP, verbose=False)
                            if u.value is not None:
                                u_v = u.value[:3]; u_omega = u.value[3:6]; u_z = u.value[6:]
                            else:
                                u_v = v_ref; u_omega = omega_ref; u_z = u_z_nom
                        except Exception:
                            u_v = v_ref; u_omega = omega_ref; u_z = u_z_nom
                        dz = (np.eye(3) - np.outer(z_fixed, z_fixed)) @ u_z
                        z_fixed = z_fixed + dz * 0.05
                        z_fixed = z_fixed / max(np.linalg.norm(z_fixed), 1e-6)
                        a_exec = np.zeros(7)
                        a_exec[:3] = 0.2 * R1 @ u_v
                        a_exec[3:6] = 0.2 * u_omega
                        a_exec[6] = a_nom[6]
                        h_min_trace = min(h_min_trace, h)
                        if np.linalg.norm(a_exec[:3] - a_nom[:3]) > 1e-4:
                            n_proj += 1
                    else:
                        a_exec = a_nom

                obs, _, done, _ = env.step(a_exec.tolist())
                for k in active_keys:
                    if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                        if not collided:
                            print(f"  FIRST COLLISION step {step}: {k}", flush=True)
                        collided = True
                if done:
                    success = True
                    break

            print(f"  ep{ep}: success={success} steps={step+1} collided={collided} proj={n_proj} h_min={h_min_trace:.3f}", flush=True)
            results.append({
                "task_idx": task_idx, "ep": ep, "success": success, "collided": collided,
                "steps": step+1, "n_proj": n_proj, "h_min": h_min_trace,
            })
        try: env.close()
        except: pass

    n = len(results)
    sr = sum(r["success"] for r in results)
    cr = sum(r["collided"] for r in results)
    print(f"\n=== {mode} on SafeLIBERO-spatial Level {level} ===")
    print(f"N={n}  SR = {sr}/{n} = {sr/max(1,n):.3f}  CR = {cr}/{n} = {cr/max(1,n):.3f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "level": level, "mode": mode}, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1]  # baseline | safemole
    level = sys.argv[2] if len(sys.argv) > 2 else "II"
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = f"/root/autodl-tmp/MysafeVLA/eval_results/pi05_{mode}_lv{level}/results.json"
    run(mode, level, n_eps, out)
