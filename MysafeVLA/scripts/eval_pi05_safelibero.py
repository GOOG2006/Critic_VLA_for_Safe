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

PI05_HOST = "127.0.0.1"
PI05_PORT = 8000

# Gripper ellipsoid (AEGIS original)
GRIPPER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
GRIPPER_Q_DIAG_DEFAULT = np.array([0.06, 0.12, 0.11], dtype=np.float64)
GRIPPER_Q_DIAG_TALL = np.array([0.06, 0.12, 0.20], dtype=np.float64)  # for milk/juice/soup tasks
# Obstacle ellipsoid: fitted from depth point cloud (AEGIS original), not fixed
ALPHA_H = float(os.environ.get("ALPHA_H", "10.0"))
H_SAFETY_MARGIN = float(os.environ.get("H_SAFETY_MARGIN", "0.015"))  # positive buffer on h >= δ
Q_INFLATION = float(os.environ.get("Q_INFLATION", "1.10"))            # inflate obstacle ellipsoid
# Multiplicative scale on the gripper ellipsoid semi-axes. Motivation: k=3 lookahead
# evidence (h_min=0.025>δ STILL collided) shows the analytic gripper ellipsoid
# under-approximates the real gripper-mesh + finger sweep. 1.0 = AEGIS original.
GRIPPER_INFLATION = float(os.environ.get("GRIPPER_INFLATION", "1.0"))
CRITIC_ALPHA_BOOST = 3.0  # multiply alpha when critic predicts danger
CRITIC_CKPT = os.environ.get("CRITIC_CKPT", "/root/autodl-tmp/MysafeVLA/critic_v2/critic_v2.pt")

# Default per-object ellipsoid sizes (semi-axes, meters). Used by `safemole_multi`
# mode which reads all active obstacle positions from the env and stacks one
# CBF constraint per obstacle. Values are rough over-approximations of LIBERO
# object meshes — Q_INFLATION adds an extra safety buffer on top.
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


def _obstacle_type(key):
    """'moka_pot_obstacle_1_pos' -> 'moka_pot'."""
    import re
    s = key[:-len("_pos")] if key.endswith("_pos") else key
    return re.sub(r"_obstacle_\d+$", "", s)


# Multi-ellipsoid: load LIBERO XML primitive decomposition (1-21 box geoms per
# obstacle, each becomes a sub-ellipsoid in CBF). Activated by OBSTACLE_REPRESENTATION=multi.
# JSON file lives next to this script. Override path with OBSTACLE_PRIMITIVES_PATH env var.
_PRIMITIVES_PATH = os.environ.get(
    "OBSTACLE_PRIMITIVES_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacle_primitives.json"),
)
try:
    with open(_PRIMITIVES_PATH) as _f:
        OBSTACLE_PRIMITIVES = json.load(_f)
except FileNotFoundError:
    OBSTACLE_PRIMITIVES = {}


def _build_gt_obstacles(obs, active_keys):
    """Read all active obstacles from the environment as (p, Q_diag, R, name).

    OBSTACLE_REPRESENTATION="single" (default, SOTA): one ellipsoid per obstacle from
    OBSTACLE_Q_DIAG, axis-aligned to world.
    OBSTACLE_REPRESENTATION="multi": decompose each obstacle into N primitive
    ellipsoids using LIBERO's XML box-geom union. Each sub-ellipsoid gets its own
    world-frame position+rotation via the obstacle's quat composed with the
    primitive's local quat. This avoids the "ellipsoid bbox covers spout-empty
    region" failure mode of orientation-aware single-ellipsoid (each protrusion
    is its OWN small ellipsoid, leaving the surrounding space free).
    """
    # Default = "multi": LIBERO XML primitive decomposition (1-21 sub-ellipsoids
    # per obstacle). Strict Pareto improvement over AEGIS on all 6 metrics
    # (SR/CR/SafeSR × Lv I/Lv II) when paired with C1 perception proximity.
    rep = os.environ.get("OBSTACLE_REPRESENTATION", "multi")
    pad = float(os.environ.get("OBSTACLE_PADDING", "0.005"))
    obstacles = []
    for k in active_keys:
        p_obs = np.asarray(obs[k][:3], dtype=np.float64)
        name = _obstacle_type(k)
        if rep == "multi" and name in OBSTACLE_PRIMITIVES:
            quat_key = k.replace("_pos", "_quat")
            R_world_obj = (Rot.from_quat(np.asarray(obs[quat_key], dtype=np.float64)).as_matrix()
                           if quat_key in obs else np.eye(3, dtype=np.float64))
            for i, prim in enumerate(OBSTACLE_PRIMITIVES[name]):
                offset_local = np.asarray(prim['pos'], dtype=np.float64)
                size_local = np.asarray(prim['size'], dtype=np.float64) + pad
                R_local = Rot.from_quat(np.asarray(prim['quat'], dtype=np.float64)).as_matrix()
                p_prim = p_obs + R_world_obj @ offset_local
                R_prim = R_world_obj @ R_local
                Q_prim = size_local * Q_INFLATION
                obstacles.append((p_prim, Q_prim, R_prim, f"{name}#{i}"))
        else:
            Q_diag = OBSTACLE_Q_DIAG.get(name, DEFAULT_OBSTACLE_Q).copy() * Q_INFLATION
            obstacles.append((p_obs, Q_diag, np.eye(3, dtype=np.float64), name))
    return obstacles

# AEGIS perception imports
sys.path.insert(0, "/root/autodl-tmp/vlsa-aegis/main")
from utils import (
    compute_h_ij, compute_h_coeffs_3d, get_point_cloud,
    filtering_points, fit_ellipse, obstacle_detection,
)


def get_full_pointcloud(image, depth, env, view):
    """Convert ALL pixels in the depth image to world-frame 3D points.

    Plan B: drops the GroundingDINO mask step from get_point_cloud — for proximity
    we want the densest possible scene coverage (including obstacles the VLM may
    have missed), not a single segmented obstacle.
    """
    from robosuite.utils.camera_utils import (
        get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix,
    )
    depth = get_real_depth_map(env.sim, depth).squeeze()
    h_full, w_full = image.shape[0], image.shape[1]
    K_inv = np.linalg.inv(get_camera_intrinsic_matrix(env.sim, view, h_full, w_full))
    T_cam_to_world = get_camera_extrinsic_matrix(env.sim, view)
    v_full, u_full = np.indices((h_full, w_full))
    v_full = (h_full - 1) - v_full  # flip vertical to match get_point_cloud convention
    u_flat = u_full.flatten()
    v_flat = v_full.flatten()
    depth_flat = depth.flatten()
    valid = (depth_flat > 1e-3) & np.isfinite(depth_flat)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pixels = np.stack([u_flat[valid], v_flat[valid], np.ones(valid.sum())], axis=0)
    points_cam = K_inv @ pixels * depth_flat[valid]
    points_cam_h = np.vstack([points_cam, np.ones(valid.sum())])
    points_world = (T_cam_to_world @ points_cam_h)[:3, :].T
    return points_world


def filter_proximity_pointcloud(pts, eef_init_pos, suite_name,
                                target_positions=(), target_radius=0.08,
                                gripper_radius=0.12):
    """Spatial filter for proximity pointcloud. Keep workspace, drop:
       - table surface (z below threshold)
       - far points (outside table region)
       - points near initial gripper pose (the gripper itself in the snapshot)
       - points near each TARGET object position (bowls/items the robot must grasp;
         keeping these in the pointcloud causes proximity to push gripper AWAY
         from the very object the policy is trying to reach)
    """
    if pts.shape[0] == 0:
        return pts
    if "spatial" in suite_name or "goal" in suite_name:
        keep = ((pts[:, 2] > 0.92) & (pts[:, 2] < 1.5)
                & (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3)
                & (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3))
    elif "object" in suite_name:
        keep = ((pts[:, 2] > 0.05) & (pts[:, 2] < 0.5)
                & (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3)
                & (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3))
    elif "long" in suite_name:
        keep = ((pts[:, 2] > 0.43) & (pts[:, 2] < 0.8)
                & (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3)
                & (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3))
    else:
        keep = np.ones(pts.shape[0], dtype=bool)
    pts = pts[keep]
    if pts.shape[0] == 0:
        return pts
    # Drop points near initial gripper pose (gripper/finger surfaces in snapshot)
    d_eef = np.linalg.norm(pts - eef_init_pos[:3], axis=1)
    pts = pts[d_eef > gripper_radius]
    # Drop points near each target object (bowl/item to be grasped)
    for tp in target_positions:
        if pts.shape[0] == 0:
            break
        d_t = np.linalg.norm(pts - np.asarray(tp[:3]), axis=1)
        pts = pts[d_t > target_radius]
    return pts


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


REPLAN_STEPS = 5  # same as AEGIS


def run(mode, level, n_eps, out_path):
    print(f"[{mode}] connecting to openpi server at {PI05_HOST}:{PI05_PORT}...", flush=True)
    client = _wcp.WebsocketClientPolicy(PI05_HOST, PI05_PORT)
    print(f"[{mode}] connected to openpi pi05 server", flush=True)

    # Load GroundingDINO for VLM-driven perception. Also loaded for safemole_multi_critic
    # when USE_PERCEPTION_PROXIMITY=1 AND PROXIMITY_FULL_POINTCLOUD=0 (uses VLM mask).
    # Plan B (PROXIMITY_FULL_POINTCLOUD=1) skips gdino entirely — full depth pointcloud
    # captures objects VLM misses (Level I unknown items: storage_box, book, etc).
    model_groundingdino = None
    critic_model = None
    # Default ON: AEGIS-consistent depth-pointcloud → per-step nearest-point
    # proximity signal. Used as δ correction in QP. Same input source as AEGIS.
    USE_PERCEPTION_PROXIMITY = os.environ.get("USE_PERCEPTION_PROXIMITY", "1") == "1"
    PROXIMITY_FULL_POINTCLOUD = os.environ.get("PROXIMITY_FULL_POINTCLOUD", "0") == "1"
    needs_gdino = mode in ("safemole", "safemole_critic") or (
        mode == "safemole_multi_critic" and USE_PERCEPTION_PROXIMITY and not PROXIMITY_FULL_POINTCLOUD
    )
    if needs_gdino:
        from groundingdino.util.inference import load_model
        model_groundingdino = load_model(
            "/root/autodl-tmp/vlsa-aegis/GroundingDINO/GroundingDINO_SwinT_OGC.py",
            "/root/autodl-tmp/vlsa-aegis/GroundingDINO/groundingdino_swint_ogc.pth",
        )
        print(f"[{mode}] GroundingDINO loaded", flush=True)
    elif mode in ("safemole_multi", "safemole_multi_critic"):
        print(f"[{mode}] using GT obstacles; margin δ={H_SAFETY_MARGIN:.3f} "
              f"Q_infl={Q_INFLATION:.2f} α={ALPHA_H}", flush=True)
    if mode in ("safemole_critic", "safemole_multi_critic"):
        class SafetyCritic(nn.Module):
            def __init__(self, d_in=17, d_hid=128):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_hid), nn.ReLU(),
                    nn.Linear(d_hid, d_hid), nn.ReLU(),
                    nn.Linear(d_hid, 2),
                )
            def forward(self, x):
                out = self.net(x)
                return out[:, 0], out[:, 1]
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
            n_qp_infeasible = 0  # count steps where CBF-QP fell back to reference action
            n_slack = 0          # count steps with any slack > 1e-4 (soft-constraint violated)
            action_plan = deque()

            # AEGIS perception: detect obstacle + fit ellipsoid (once per episode)
            flag_safety_control = False
            p2 = R2 = Q2_diag = None
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = Rot.from_quat(eef_quat).as_matrix()
            p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET

            # Episode-init perception. Three modes that use it:
            #   - safemole/safemole_critic: fit single ellipsoid → use as CBF obstacle
            #   - safemole_multi_critic + USE_PERCEPTION_PROXIMITY + PROXIMITY_FULL_POINTCLOUD=0:
            #     AEGIS-consistent gdino-masked pointcloud for per-step proximity
            #   - safemole_multi_critic + USE_PERCEPTION_PROXIMITY + PROXIMITY_FULL_POINTCLOUD=1
            #     (Plan B): full-scene depth pointcloud (no gdino mask)
            filter_pts_ep = None  # for proximity in safemole_multi_critic
            if (mode == "safemole_multi_critic" and USE_PERCEPTION_PROXIMITY
                and PROXIMITY_FULL_POINTCLOUD):
                # Plan B: dense full-scene pointcloud, gripper-init region removed.
                agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
                backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
                backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
                full_a = get_full_pointcloud(agentview_img, agentview_depth, env, "agentview")
                full_b = get_full_pointcloud(backview_img, backview_depth, env, "backview")
                if full_a.shape[0] > 0 and full_b.shape[0] > 0:
                    full_pts = np.vstack([full_a, full_b])
                elif full_a.shape[0] > 0:
                    full_pts = full_a
                elif full_b.shape[0] > 0:
                    full_pts = full_b
                else:
                    full_pts = np.zeros((0, 3))
                grip_radius = float(os.environ.get("PROX_GRIPPER_RADIUS", "0.12"))
                target_radius = float(os.environ.get("PROX_TARGET_RADIUS", "0.08"))
                # Targets: any non-obstacle item the policy might grasp. In LIBERO
                # spatial these are bowls (and similar) — we exclude their region
                # from the proximity pointcloud so the safety layer doesn't fight
                # the policy's reach-and-grasp toward the actual target.
                target_keys = [k for k in obs.keys()
                               if k.endswith("_pos") and "obstacle" not in k
                               and "to_" not in k  # exclude relative-coord keys
                               and any(t in k for t in ("bowl", "plate", "ramekin",
                                                         "stove", "cabinet", "akita"))]
                target_positions = [obs[k][:3] for k in target_keys
                                    if abs(obs[k][0]) < 0.5 and abs(obs[k][1]) < 0.5]
                filter_pts_ep = filter_proximity_pointcloud(
                    full_pts, eef_pos, "safelibero_spatial",
                    target_positions=target_positions,
                    target_radius=target_radius,
                    gripper_radius=grip_radius,
                ).astype(np.float64)
                print(f"  [proxB] full-pointcloud: {len(full_pts)} raw → "
                      f"{len(filter_pts_ep)} after filter (excluded "
                      f"{len(target_positions)} targets: {[k.replace('_pos','') for k in target_keys if abs(obs[k][0])<0.5 and abs(obs[k][1])<0.5]})",
                      flush=True)
            if needs_gdino and model_groundingdino is not None:
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
                    if mode in ("safemole", "safemole_critic"):
                        p2, R2, Q2_diag = fit_ellipse(filter_pts, plot=True, save_path=img_out_dir)
                        z_fixed = (p2 - p1)
                        z_fixed = z_fixed / max(np.linalg.norm(z_fixed), 1e-6)
                        flag_safety_control = True
                        print(f"  Ellipsoid fitted: p2={p2.round(3)}, Q2={Q2_diag.round(3)}", flush=True)
                    if mode == "safemole_multi_critic" and USE_PERCEPTION_PROXIMITY:
                        filter_pts_ep = filter_pts.astype(np.float64)
                        print(f"  proximity pointcloud: {len(filter_pts_ep)} points", flush=True)

            if not flag_safety_control:
                z_fixed = np.array([1.0, 0.0, 0.0])

            if mode == "safemole_multi":
                _obs_init = _build_gt_obstacles(obs, active_keys)
                print(f"  [multi] {len(_obs_init)} obstacles: "
                      + ", ".join(f"{n}@({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                                  for (p, _, _, n) in _obs_init), flush=True)

            for step in range(220):
                # Action chunking (same as AEGIS): replan every REPLAN_STEPS
                if not action_plan:
                    element = build_openpi_element(obs, task.language)
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[:REPLAN_STEPS])
                a_nom = np.asarray(action_plan.popleft(), dtype=np.float64)

                if mode == "baseline":
                    a_exec = a_nom
                elif mode in ("safemole_multi", "safemole_multi_critic"):
                    # === Multi-obstacle CBF-QP with GT positions + δ margin + Q inflation ===
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    R1 = Rot.from_quat(eef_quat).as_matrix()
                    p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
                    Q1_diag = (GRIPPER_Q_DIAG_TALL
                               if any(w in task.language for w in ["orange juice", "milk", "alphabet soup"])
                               else GRIPPER_Q_DIAG_DEFAULT) * GRIPPER_INFLATION
                    obstacles = _build_gt_obstacles(obs, active_keys)
                    if not obstacles:
                        a_exec = a_nom
                    else:
                        v_ref_local = R1.T @ a_nom[:3]
                        u_v_ref = 5 * v_ref_local
                        omega_ref = a_nom[3:6]
                        u_omega_ref = 5 * omega_ref

                        # Predictive CBF (approach A, rolled back to OFF by default after
                        # k=1..3 sweep showed net regression — pi05 is open-loop and fights
                        # preemptive braking, causing timeouts without reducing collisions).
                        # Set CBF_LOOKAHEAD_STEPS>0 to re-enable; 0 reproduces v1.
                        _lookahead_steps = int(os.environ.get("CBF_LOOKAHEAD_STEPS", "0"))
                        _dt_sim = float(os.environ.get("CBF_DT_SIM", "0.05"))
                        _danger_mult = float(os.environ.get("CBF_DANGER_MULT", "3.0"))
                        v_nom_world = np.asarray(a_nom[:3], dtype=np.float64)
                        p1_future = p1 + v_nom_world * _dt_sim * _lookahead_steps

                        # Per-obstacle CBF coeffs + barrier value (z_fixed dynamic per obstacle)
                        # is_default flag tracks obstacles for which we have no precise size
                        # entry in OBSTACLE_Q_DIAG — the algorithm uses a larger δ for these
                        # constraints to compensate for our representation uncertainty.
                        coeffs = []
                        h_values = []         # current h (for monitoring/trace)
                        h_future_values = []  # predicted h under nominal rollout
                        for (p_j, Q_j, R_j, name) in obstacles:
                            z_j = p_j - p1
                            nz = np.linalg.norm(z_j)
                            z_j = z_j / (nz if nz > 1e-6 else 1.0)
                            a_v_j, a_om_j, a_uz_j, h_curr, mu_j = compute_h_coeffs_3d(
                                p1, Q1_diag, R1, p_j, Q_j, R_j, z_j)
                            if _lookahead_steps > 0 and h_curr < _danger_mult * H_SAFETY_MARGIN:
                                z_fut = p_j - p1_future
                                nzf = np.linalg.norm(z_fut)
                                z_fut = z_fut / (nzf if nzf > 1e-6 else 1.0)
                                h_fut = compute_h_ij(p1_future, Q1_diag, R1, p_j, Q_j, R_j, z_fut)
                                h_j = min(h_curr, h_fut)
                            else:
                                h_fut = h_curr
                                h_j = h_curr
                            is_default = name not in OBSTACLE_Q_DIAG
                            coeffs.append((a_v_j, a_om_j, a_uz_j, h_j, mu_j, is_default))
                            h_values.append(h_curr)
                            h_future_values.append(h_fut)
                        # Steer u_z using mu_row of the most-active obstacle (min h)
                        min_idx = int(np.argmin(h_values))
                        mu_row_ref = coeffs[min_idx][4]
                        u_z_ref = 10 * mu_row_ref

                        # === Plan B: critic-driven dynamic margin ===
                        # The critic's h_pred is its estimate of the minimum h that will be
                        # observed over the next LOOKAHEAD env steps under the CURRENT policy
                        # distribution. It is a closed-loop learned estimator and therefore
                        # captures (a) action-chunking decoherence, (b) Taylor-error
                        # in the one-step CBF gradient, and (c) env integration overshoot.
                        # We use it to inflate δ: δ_eff = δ + max(0, δ - h_pred), so whenever
                        # the critic anticipates h dropping below δ, the QP tightens
                        # proportionally. If h_pred ≥ δ, δ_eff = δ (no extra conservatism).
                        delta_eff = H_SAFETY_MARGIN
                        risk_prob_v = None; h_pred_v = None
                        if mode == "safemole_multi_critic" and critic_model is not None:
                            grip = obs["robot0_gripper_qpos"]
                            bowl_pos = obs.get("akita_black_bowl_1_pos", eef_pos)[:3]
                            bowl_lifted = float(bowl_pos[2]) - 0.85 > 0.05
                            h_min_curr = float(min(h_values))
                            state_vec = np.concatenate([
                                eef_pos[:3],
                                quat2axisangle(np.array(eef_quat, dtype=np.float64)),
                                grip[:2], a_nom[:7], [h_min_curr], [float(bowl_lifted)],
                            ]).astype(np.float32)
                            with torch.no_grad():
                                risk_logit, h_pred = critic_model(torch.from_numpy(state_vec).unsqueeze(0))
                                risk_prob_v = float(torch.sigmoid(risk_logit).item())
                                h_pred_v = float(h_pred.item())
                            if np.isfinite(h_pred_v) and h_pred_v < H_SAFETY_MARGIN:
                                _critic_gain = float(os.environ.get("CRITIC_MARGIN_GAIN", "0.3"))
                                delta_eff = H_SAFETY_MARGIN + _critic_gain * (H_SAFETY_MARGIN - h_pred_v)
                                # Clamp to avoid runaway conservatism
                                _delta_max = float(os.environ.get("CRITIC_MARGIN_MAX", "0.04"))
                                delta_eff = min(delta_eff, _delta_max)

                        # === Plan A: AEGIS-consistent perception proximity ===
                        # Per-step nearest-point distance from gripper to filtered depth
                        # pointcloud (built once at episode init from agentview+backview
                        # depth + GroundingDINO mask — same input AEGIS uses, but queried
                        # per-step instead of fitted to one ellipsoid). When the real
                        # geometry says we're closer than PROX_THRESHOLD, inflate δ_eff
                        # uniformly to compensate for the ellipsoid undermodel.
                        if filter_pts_ep is not None and len(filter_pts_ep) > 0:
                            dists = np.linalg.norm(filter_pts_ep - p1, axis=1)
                            h_prox = float(dists.min())
                            _prox_thr = float(os.environ.get("PROX_THRESHOLD", "0.10"))
                            _prox_gain = float(os.environ.get("PROX_GAIN", "1.0"))
                            prox_bonus = max(0.0, _prox_thr - h_prox) * _prox_gain
                            if prox_bonus > 0.0:
                                _delta_max_prox = float(os.environ.get("PROX_DELTA_MAX", "0.10"))
                                delta_eff = min(delta_eff + prox_bonus, _delta_max_prox)

                        alpha_effective = ALPHA_H
                        u = cp.Variable(9)
                        slack = cp.Variable(len(obstacles), nonneg=True)
                        W = np.diag([1./25]*6 + [1.]*3)
                        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_ref])
                        # Penalty: 1e4 * L1(slack) + 1e6 * L2(slack). L1 term forces slack=0
                        # whenever feasible; L2 term regularizes magnitudes when infeasible.
                        slack_penalty = float(os.environ.get("SLACK_PENALTY", "1e6"))
                        cost = cp.quad_form(u - u_ref_vec, W) \
                            + 1e4 * cp.sum(slack) + slack_penalty * cp.sum_squares(slack)
                        # Uncertainty-aware per-constraint δ: when the obstacle's name is not
                        # in OBSTACLE_Q_DIAG (DEFAULT fallback), our ellipsoid is a generic
                        # under-approximation of the true mesh. H_SAFETY_MARGIN_DEFAULT_BONUS
                        # adds an extra buffer on those constraints only — pure algorithmic
                        # response to representation uncertainty, no per-object data tuning.
                        _default_bonus = float(os.environ.get("H_SAFETY_MARGIN_DEFAULT_BONUS", "0.0"))
                        constraints = []
                        for i, (a_v_j, a_om_j, a_uz_j, h_j, _, is_default) in enumerate(coeffs):
                            delta_local = delta_eff + (_default_bonus if is_default else 0.0)
                            constraints.append(
                                0.2 * a_v_j @ u[:3] + 0.2 * a_om_j @ u[3:6] + a_uz_j @ u[6:]
                                + alpha_effective * (h_j - delta_local) + slack[i] >= 0
                            )
                        prob = cp.Problem(cp.Minimize(cost), constraints)
                        qp_ok = False; slack_val = 0.0
                        try:
                            prob.solve(solver=cp.OSQP, verbose=False)
                            if u.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                                u_v = u.value[:3]; u_omega = u.value[3:6]; u_z = u.value[6:]
                                qp_ok = True
                                slack_val = float(np.sum(slack.value)) if slack.value is not None else 0.0
                            else:
                                u_v = v_ref_local; u_omega = omega_ref; u_z = u_z_ref
                                if n_qp_infeasible == 0:
                                    print(f"  QP INFEASIBLE step {step}: status={prob.status} "
                                          f"h_min={min(h_values):.4f} n_obs={len(obstacles)}", flush=True)
                        except Exception as e:
                            u_v = v_ref_local; u_omega = omega_ref; u_z = u_z_ref
                            if n_qp_infeasible == 0:
                                print(f"  QP SOLVER ERROR step {step}: {type(e).__name__}: {e}", flush=True)
                        if not qp_ok:
                            n_qp_infeasible += 1
                        if slack_val > 1e-4:
                            if n_slack == 0:
                                print(f"  SLACK ACTIVE step {step}: sum={slack_val:.4f} "
                                      f"h_min={min(h_values):.4f}", flush=True)
                            n_slack += 1
                        a_exec = np.zeros(7)
                        a_exec[:3] = 0.2 * R1 @ u_v
                        a_exec[3:6] = 0.2 * u_omega
                        a_exec[6] = a_nom[6]
                        h_min_trace = min(h_min_trace, min(h_values))
                        if np.linalg.norm(a_exec[:3] - a_nom[:3]) > 1e-4:
                            n_proj += 1
                elif mode in ("safemole", "safemole_critic"):
                    # === AEGIS-faithful safety layer + optional critic ===
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    R1 = Rot.from_quat(eef_quat).as_matrix()
                    p1 = np.asarray(eef_pos[:3], dtype=np.float64) + R1 @ GRIPPER_OFFSET
                    Q1_diag = (GRIPPER_Q_DIAG_TALL if any(w in task.language for w in ["orange juice", "milk", "alphabet soup"]) else GRIPPER_Q_DIAG_DEFAULT) * GRIPPER_INFLATION
                    if flag_safety_control:
                        v_ref = R1.T @ a_nom[:3]
                        u_v_ref = 5 * v_ref
                        omega_ref = a_nom[3:6]
                        u_omega_ref = 5 * omega_ref
                        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, Q1_diag, R1, p2, Q2_diag, R2, z_fixed)

                        # Critic: predict future collision risk → boost constraint
                        alpha_effective = ALPHA_H
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
                            if risk_prob > 0.5:
                                alpha_effective = ALPHA_H * CRITIC_ALPHA_BOOST  # tighter constraint

                        a_u_v = 0.2 * a_v
                        a_u_omega = 0.2 * a_omega
                        u_z_nom = 10 * mu_row
                        u = cp.Variable(9)
                        W = np.diag([1./25]*6 + [1.]*3)
                        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
                        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                        constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + alpha_effective * h >= 0]
                        prob = cp.Problem(objective, constraints)
                        qp_ok = False
                        try:
                            prob.solve(solver=cp.OSQP, verbose=False)
                            if u.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                                u_v = u.value[:3]; u_omega = u.value[3:6]; u_z = u.value[6:]
                                qp_ok = True
                            else:
                                u_v = v_ref; u_omega = omega_ref; u_z = u_z_nom
                                if n_qp_infeasible == 0:
                                    print(f"  QP INFEASIBLE at step {step}: status={prob.status} h={h:.4f} "
                                          f"alpha={alpha_effective:.2f} - barrier NOT enforced", flush=True)
                        except Exception as e:
                            u_v = v_ref; u_omega = omega_ref; u_z = u_z_nom
                            if n_qp_infeasible == 0:
                                print(f"  QP SOLVER ERROR at step {step}: {type(e).__name__}: {e}", flush=True)
                        if not qp_ok:
                            n_qp_infeasible += 1
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

            print(f"  ep{ep}: success={success} steps={step+1} collided={collided} "
                  f"proj={n_proj} qp_infeas={n_qp_infeasible} slack={n_slack} "
                  f"h_min={h_min_trace:.3f}", flush=True)
            results.append({
                "task_idx": task_idx, "ep": ep, "success": success, "collided": collided,
                "steps": step+1, "n_proj": n_proj, "n_qp_infeasible": n_qp_infeasible,
                "n_slack": n_slack, "h_min": h_min_trace,
            })
        try: env.close()
        except: pass

    n = len(results)
    sr = sum(r["success"] for r in results)
    cr = sum(r["collided"] for r in results)
    qp_inf_total = sum(r.get("n_qp_infeasible", 0) for r in results)
    qp_inf_eps = sum(1 for r in results if r.get("n_qp_infeasible", 0) > 0)
    slack_total = sum(r.get("n_slack", 0) for r in results)
    slack_eps = sum(1 for r in results if r.get("n_slack", 0) > 0)
    safe_success = sum(1 for r in results if r["success"] and not r["collided"])
    print(f"\n=== {mode} on SafeLIBERO-spatial Level {level} ===")
    print(f"N={n}  SR = {sr}/{n} = {sr/max(1,n):.3f}  CR = {cr}/{n} = {cr/max(1,n):.3f}  "
          f"SafeSR = {safe_success}/{n} = {safe_success/max(1,n):.3f}")
    print(f"QP infeasible: {qp_inf_total} steps across {qp_inf_eps}/{n} episodes "
          f"(barrier not enforced for those steps)")
    print(f"Slack active:  {slack_total} steps across {slack_eps}/{n} episodes "
          f"(soft-constraint violated; only meaningful for safemole_multi)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "level": level, "mode": mode}, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1]  # baseline | safemole | safemole_critic
    level = sys.argv[2] if len(sys.argv) > 2 else "II"
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    _default_root = (
        os.environ.get("EVAL_RESULTS_DIR")
        or (os.path.join(os.environ["MYSAFEVLA_ROOT"], "eval_results")
            if os.environ.get("MYSAFEVLA_ROOT") else None)
        or "/root/autodl-tmp/MysafeVLA/eval_results"
    )
    out = os.path.join(_default_root, f"pi05_{mode}_lv{level}", "results.json")
    run(mode, level, n_eps, out)
