"""Collect rollout data on SafeLIBERO, train anticipatory critic, save weights.

COLLECTOR_MODE selects the policy used during rollout:
  - "baseline" (default): unfiltered pi05 actions — produces v2-style data.
  - "safemole_multi_critic": load CRITIC_INIT_CKPT, run Plan B QP per step, execute
    the filtered action. Self-bootstrap: the new critic learns under the closed-loop
    distribution it will actually be deployed in. Output goes to CRITIC_OUT_DIR
    (default critic_v3) so v2 is preserved as a baseline.
"""
import os, json, math, sys, numpy as np, torch, torch.nn as nn
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_TF"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
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

VLSA_AEGIS_ROOT = os.environ.get("VLSA_AEGIS_ROOT", "/root/autodl-tmp/vlsa-aegis")
GROUNDING_DINO_CONFIG = os.environ.get(
    "GROUNDING_DINO_CONFIG",
    os.path.join(VLSA_AEGIS_ROOT, "GroundingDINO/GroundingDINO_SwinT_OGC.py"),
)
GROUNDING_DINO_CKPT = os.environ.get(
    "GROUNDING_DINO_CKPT",
    os.path.join(VLSA_AEGIS_ROOT, "GroundingDINO/groundingdino_swint_ogc.pth"),
)
PI05_HOST = os.environ.get("PI05_HOST", "127.0.0.1")
PI05_PORT = int(os.environ.get("PI05_PORT", "8000"))

sys.path.insert(0, os.path.join(VLSA_AEGIS_ROOT, "main"))
from utils import get_point_cloud, filtering_points, fit_ellipse, obstacle_detection

COLLECTOR_MODE = os.environ.get("COLLECTOR_MODE", "baseline")
GRIPPER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
GRIPPER_Q_DIAG_DEFAULT = np.array([0.06, 0.12, 0.11], dtype=np.float64)
GRIPPER_Q_DIAG_TALL = np.array([0.06, 0.12, 0.20], dtype=np.float64)
GRIPPER_INFLATION = float(os.environ.get("GRIPPER_INFLATION", "1.0"))
REPLAN_STEPS = 5
LOOKAHEAD = 10  # predict collision within next K steps
ALPHA_H = float(os.environ.get("ALPHA_H", "10.0"))
H_SAFETY_MARGIN = float(os.environ.get("H_SAFETY_MARGIN", "0.015"))
Q_INFLATION = float(os.environ.get("Q_INFLATION", "1.10"))
CRITIC_INIT_CKPT = os.environ.get(
    "CRITIC_INIT_CKPT", "/root/autodl-tmp/MysafeVLA/critic_v2/critic_v2.pt")

DEFAULT_OBSTACLE_Q = np.array([0.06, 0.06, 0.12], dtype=np.float64)
OBSTACLE_Q_DIAG = {
    "moka_pot":         np.array([0.07,  0.07,  0.13],  dtype=np.float64),
    "milk":             np.array([0.045, 0.045, 0.11],  dtype=np.float64),
    "wine_bottle":      np.array([0.045, 0.045, 0.16],  dtype=np.float64),
    "orange_juice":     np.array([0.045, 0.045, 0.13],  dtype=np.float64),
    "alphabet_soup":    np.array([0.045, 0.045, 0.11],  dtype=np.float64),
    "tomato_sauce":     np.array([0.045, 0.045, 0.11],  dtype=np.float64),
    "ketchup":          np.array([0.04,  0.04,  0.12],  dtype=np.float64),
    "salad_dressing":   np.array([0.045, 0.045, 0.13],  dtype=np.float64),
    "cookies":          np.array([0.08,  0.06,  0.05],  dtype=np.float64),
    "butter":           np.array([0.05,  0.05,  0.04],  dtype=np.float64),
    "cream_cheese":     np.array([0.05,  0.05,  0.05],  dtype=np.float64),
    "chocolate_pudding":np.array([0.05,  0.05,  0.06],  dtype=np.float64),
}
_DEFAULT_OUT = (
    os.environ.get("CRITIC_OUT_DIR")
    or (os.path.join(os.environ["MYSAFEVLA_ROOT"],
                     "critic_v3" if COLLECTOR_MODE != "baseline" else "critic_v2")
        if os.environ.get("MYSAFEVLA_ROOT") else None)
    or ("/root/autodl-tmp/MysafeVLA/critic_v3" if COLLECTOR_MODE != "baseline"
        else "/root/autodl-tmp/MysafeVLA/critic_v2")
)
OUT_DIR = Path(_DEFAULT_OUT)


# === CBF helpers (duplicated from eval_pi05_safelibero.py to keep collector self-contained) ===
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


def _obstacle_type(key):
    import re
    s = key[:-len("_pos")] if key.endswith("_pos") else key
    return re.sub(r"_obstacle_\d+$", "", s)


def _build_gt_obstacles(obs, active_keys):
    obstacles = []
    for k in active_keys:
        p_obs = np.asarray(obs[k][:3], dtype=np.float64)
        name = _obstacle_type(k)
        Q_diag = OBSTACLE_Q_DIAG.get(name, DEFAULT_OBSTACLE_Q).copy() * Q_INFLATION
        obstacles.append((p_obs, Q_diag, np.eye(3, dtype=np.float64), name))
    return obstacles


# === Critic model class (defined at module scope so it can be loaded for bootstrap) ===
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
        return out[:, 0], out[:, 1]


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


def _gripper_Q(task_lang):
    base = (GRIPPER_Q_DIAG_TALL
            if any(w in task_lang for w in ["orange juice", "milk", "alphabet soup"])
            else GRIPPER_Q_DIAG_DEFAULT)
    return base * GRIPPER_INFLATION


def compute_h(obs, active_keys, p2, Q2_diag, R2, task_lang=""):
    eef = obs["robot0_eef_pos"]
    R1 = Rot.from_quat(obs["robot0_eef_quat"]).as_matrix()
    p1 = np.asarray(eef[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
    if p2 is not None:
        z = p2 - p1
        if np.linalg.norm(z) > 1e-6:
            return compute_h_ij(p1, _gripper_Q(task_lang), R1, p2, Q2_diag, R2, z)
    return 10.0


def compute_h_min_multi(obs, active_keys, task_lang):
    """Minimum analytic h across all GT obstacles (matches eval Plan B)."""
    eef = obs["robot0_eef_pos"]
    eef_q = obs["robot0_eef_quat"]
    R1 = Rot.from_quat(eef_q).as_matrix()
    p1 = np.asarray(eef[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
    Q1_diag = _gripper_Q(task_lang)
    obstacles = _build_gt_obstacles(obs, active_keys)
    if not obstacles:
        return 10.0
    h_min = float("inf")
    for (p_j, Q_j, R_j, _name) in obstacles:
        z = p_j - p1
        nz = np.linalg.norm(z)
        z = z / (nz if nz > 1e-6 else 1.0)
        h_min = min(h_min, compute_h_ij(p1, Q1_diag, R1, p_j, Q_j, R_j, z))
    return h_min


def cbf_filter_action(a_nom, obs, active_keys, task_lang, critic_model):
    """Plan B QP: nominal action -> filtered action. Mirrors eval safemole_multi_critic.

    Returns (a_exec, h_min, slack_sum, qp_ok). On QP failure, falls back to a_nom.
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    R1 = Rot.from_quat(eef_quat).as_matrix()
    p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
    Q1_diag = _gripper_Q(task_lang)
    obstacles = _build_gt_obstacles(obs, active_keys)
    if not obstacles:
        return a_nom.copy(), 10.0, 0.0, True

    v_ref_local = R1.T @ a_nom[:3]
    u_v_ref = 5 * v_ref_local
    omega_ref = a_nom[3:6]
    u_omega_ref = 5 * omega_ref

    coeffs = []
    h_values = []
    for (p_j, Q_j, R_j, _name) in obstacles:
        z_j = p_j - p1
        nz = np.linalg.norm(z_j)
        z_j = z_j / (nz if nz > 1e-6 else 1.0)
        a_v_j, a_om_j, a_uz_j, h_curr, mu_j = compute_h_coeffs_3d(
            p1, Q1_diag, R1, p_j, Q_j, R_j, z_j)
        coeffs.append((a_v_j, a_om_j, a_uz_j, h_curr, mu_j))
        h_values.append(h_curr)
    min_idx = int(np.argmin(h_values))
    mu_row_ref = coeffs[min_idx][4]
    u_z_ref = 10 * mu_row_ref

    # Critic-driven dynamic margin (Plan B)
    delta_eff = H_SAFETY_MARGIN
    if critic_model is not None:
        grip = obs["robot0_gripper_qpos"]
        # Task-agnostic carrying indicator (gripper closed below threshold).
        is_carrying = float(grip[0] < 0.025)
        h_min_curr = float(min(h_values))
        state_vec = np.concatenate([
            eef_pos[:3],
            quat2axisangle(np.array(eef_quat, dtype=np.float64)),
            grip[:2], a_nom[:7], [h_min_curr], [is_carrying],
        ]).astype(np.float32)
        with torch.no_grad():
            _risk_logit, h_pred = critic_model(torch.from_numpy(state_vec).unsqueeze(0))
            h_pred_v = float(h_pred.item())
        if np.isfinite(h_pred_v) and h_pred_v < H_SAFETY_MARGIN:
            gain = float(os.environ.get("CRITIC_MARGIN_GAIN", "0.3"))
            delta_eff = H_SAFETY_MARGIN + gain * (H_SAFETY_MARGIN - h_pred_v)
            delta_eff = min(delta_eff, float(os.environ.get("CRITIC_MARGIN_MAX", "0.04")))

    u = cp.Variable(9)
    slack = cp.Variable(len(obstacles), nonneg=True)
    W = np.diag([1./25]*6 + [1.]*3)
    u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_ref])
    slack_pen = float(os.environ.get("SLACK_PENALTY", "1e6"))
    cost = cp.quad_form(u - u_ref_vec, W) + 1e4 * cp.sum(slack) + slack_pen * cp.sum_squares(slack)
    cons = []
    for i, (a_v_j, a_om_j, a_uz_j, h_j, _) in enumerate(coeffs):
        cons.append(0.2 * a_v_j @ u[:3] + 0.2 * a_om_j @ u[3:6] + a_uz_j @ u[6:]
                    + ALPHA_H * (h_j - delta_eff) + slack[i] >= 0)
    prob = cp.Problem(cp.Minimize(cost), cons)
    qp_ok = False
    slack_sum = 0.0
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if u.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            u_v = u.value[:3]; u_omega = u.value[3:6]
            qp_ok = True
            if slack.value is not None:
                slack_sum = float(np.sum(slack.value))
        else:
            u_v = v_ref_local; u_omega = omega_ref
    except Exception:
        u_v = v_ref_local; u_omega = omega_ref

    if not qp_ok:
        return a_nom.copy(), float(min(h_values)), 0.0, False
    a_exec = np.zeros(7)
    a_exec[:3] = 0.2 * R1 @ u_v
    a_exec[3:6] = 0.2 * u_omega
    a_exec[6] = a_nom[6]
    return a_exec, float(min(h_values)), slack_sum, True


def collect_data():
    print(f"[collect] mode={COLLECTOR_MODE} out={OUT_DIR}", flush=True)
    print(f"[collect] connecting to pi05 server at {PI05_HOST}:{PI05_PORT}...", flush=True)
    client = _wcp.WebsocketClientPolicy(PI05_HOST, PI05_PORT)
    # GroundingDINO is only used by baseline mode (perception → fitted ellipsoid).
    # Plan B (safemole_multi_critic) uses GT obstacles, so skip the gdino load and
    # avoid the bert-base-uncased huggingface fetch entirely.
    gdino = None
    if COLLECTOR_MODE == "baseline":
        from groundingdino.util.inference import load_model
        gdino = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT)

    critic_model = None
    if COLLECTOR_MODE == "safemole_multi_critic":
        if not os.path.exists(CRITIC_INIT_CKPT):
            raise FileNotFoundError(f"CRITIC_INIT_CKPT not found: {CRITIC_INIT_CKPT}")
        critic_model = SafetyCritic()
        critic_model.load_state_dict(torch.load(CRITIC_INIT_CKPT, map_location="cpu"))
        critic_model.eval()
        print(f"[collect] loaded critic from {CRITIC_INIT_CKPT}", flush=True)
    print("[collect] models loaded", flush=True)

    # Multi-level support: BENCH_LEVELS comma-separated (default "II" preserves prior behavior).
    # Critic_v2 was trained on Level II only — that's the OOD bug for Level I deploy.
    bench_levels = [s.strip() for s in os.environ.get("BENCH_LEVELS", "II").split(",") if s.strip()]
    eps_per_task = int(os.environ.get("COLLECT_EPS_PER_TASK", "3"))
    print(f"[collect] levels={bench_levels} eps_per_task={eps_per_task}", flush=True)
    all_samples = []

    bench_suites = [s.strip() for s in os.environ.get("BENCH_SUITES", "safelibero_spatial").split(",") if s.strip()]
    print(f"[collect] suites={bench_suites}", flush=True)
    for SUITE_NAME in bench_suites:
     for level in bench_levels:
      suite = benchmark.get_benchmark_dict()[SUITE_NAME](safety_level=level)
      print(f"[collect] suite={SUITE_NAME} level={level}", flush=True)
      print(f"[collect] === level={level} === ({suite.n_tasks} tasks)", flush=True)
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
        print(f"[collect] lv{level} t{task_idx}: {task.language}", flush=True)

        for ep in range(eps_per_task):
            try:
                env.reset()
                _offset = int(os.environ.get("INIT_STATE_OFFSET", "0"))
                obs = env.set_init_state(init_states[(ep + _offset) % len(init_states)])
                for _ in range(10):
                    obs, _, _, _ = env.step([0]*6+[-1])
            except Exception as e:
                print(f"  ep{ep} reset fail: {e}", flush=True)
                continue

            # Active obstacles (GT keys; used by both modes for label/h_min).
            obs_keys = [k for k in obs.keys() if "obstacle" in k and k.endswith("_pos") and "to_" not in k]
            active_keys = [k for k in obs_keys if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
            init_obs_pos = {k: obs[k].copy() for k in active_keys}

            p2 = R2 = Q2_diag = None
            if COLLECTOR_MODE == "baseline":
                # Perception only used in baseline: fit a single AEGIS-style ellipsoid
                # from depth + GroundingDINO masks for the analytic h feature.
                import pathlib
                img_out = pathlib.Path(f"/tmp/critic_collect/t{task_idx}_ep{ep}")
                img_out.mkdir(parents=True, exist_ok=True)
                agv = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agd = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
                obs_info = obstacle_detection(agv, task.language, SUITE_NAME)
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
                if fp.ndim == 2 and fp.shape[0] > 0 and fp.shape[1] == 3:
                    filt = filtering_points(fp, SUITE_NAME)
                else:
                    filt = np.array([]).reshape(0, 3)
                if filt.shape[0] > 0:
                    p2, R2, Q2_diag = fit_ellipse(filt, plot=False)

            # Rollout — baseline (unfiltered) or closed-loop Plan B per COLLECTOR_MODE
            action_plan = deque()
            trajectory = []
            n_qp_fail_ep = 0

            for step in range(220):
                if not action_plan:
                    element = build_element(obs, task.language)
                    ac = client.infer(element)["actions"]
                    action_plan.extend(ac[:REPLAN_STEPS])
                a_nom = np.asarray(action_plan.popleft(), dtype=np.float64)

                # State features (computed BEFORE filtering — features describe what
                # the policy sees, label captures what actually happens). Use multi-
                # obstacle h_min so labels are consistent across baseline/closed-loop.
                eef = obs["robot0_eef_pos"]
                eef_q = obs["robot0_eef_quat"]
                grip = obs["robot0_gripper_qpos"]
                if COLLECTOR_MODE == "baseline":
                    h = compute_h(obs, active_keys, p2, Q2_diag, R2, task.language)
                else:
                    h = compute_h_min_multi(obs, active_keys, task.language)
                # Task-agnostic carrying indicator (gripper closed below threshold).
                is_carrying = float(grip[0] < 0.025)

                # Choose executed action
                if COLLECTOR_MODE == "safemole_multi_critic":
                    a_exec, _h_min_qp, _slack, qp_ok = cbf_filter_action(
                        a_nom, obs, active_keys, task.language, critic_model)
                    if not qp_ok:
                        n_qp_fail_ep += 1
                else:
                    a_exec = a_nom

                # Check collision (post-step is the truth signal — done after env.step)
                collided_now = False
                for k in active_keys:
                    if np.sum(np.abs(obs[k] - init_obs_pos[k])) > 0.001:
                        collided_now = True

                # Feature vector uses the NOMINAL action a_nom — this is what the
                # critic will see at inference time inside the QP (action chunk
                # before filtering). Keeping this consistent across modes is what
                # makes the bootstrap correct: only the executed dynamics differ.
                state_vec = np.concatenate([
                    eef[:3], quat2axisangle(np.array(eef_q, dtype=np.float64)),
                    grip[:2], a_nom[:7], [h], [is_carrying],
                ]).astype(np.float32)

                trajectory.append({
                    "state": state_vec, "step": step, "h": h, "collided": collided_now,
                })

                obs, _, done, _ = env.step(a_exec.tolist())
                if done:
                    break
            if COLLECTOR_MODE != "baseline" and n_qp_fail_ep > 0:
                print(f"  ep{ep}: {n_qp_fail_ep} QP failures (fell back to nominal)", flush=True)

            # Label: for each step, will collision happen within next LOOKAHEAD steps?
            # FIX (label leakage): only keep samples whose full K-step future window exists.
            # Previously the last LOOKAHEAD steps had incomplete futures and h_future_min
            # silently fell back to the current h, contaminating labels with tail-truncated
            # pseudo-negatives. Now trajectory[:-LOOKAHEAD] keeps only samples with a
            # complete K-step future (required by exchangeability assumption).
            if len(trajectory) > LOOKAHEAD:
                usable = trajectory[:-LOOKAHEAD]
                for i, t in enumerate(usable):
                    future = trajectory[i:i+LOOKAHEAD]
                    assert len(future) == LOOKAHEAD, "truncation bug"
                    t["future_collision"] = any(f["collided"] for f in future)
                    t["h_future_min"] = min(f["h"] for f in future)
                n_dropped = len(trajectory) - len(usable)
            else:
                usable = []
                n_dropped = len(trajectory)

            all_samples.extend(usable)
            n_col = sum(1 for t in usable if t["collided"])
            n_fc = sum(1 for t in usable if t["future_collision"])
            print(f"  ep{ep}: {len(trajectory)} steps (kept {len(usable)}, "
                  f"dropped {n_dropped} tail w/o full {LOOKAHEAD}-step future), "
                  f"{n_col} collided, {n_fc} future_collision", flush=True)

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

    ckpt_name = os.environ.get("CRITIC_CKPT_NAME") or (
        "critic_v3.pt" if COLLECTOR_MODE != "baseline" else "critic_v2.pt")
    torch.save(critic.state_dict(), OUT_DIR / ckpt_name)
    print(f"[train] saved to {OUT_DIR / ckpt_name}")
    return critic


if __name__ == "__main__":
    states, labels, h_future = collect_data()
    train_critic(states, labels, h_future)
