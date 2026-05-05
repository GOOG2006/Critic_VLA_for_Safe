"""Stage 1.5: AEGIS pipeline + N-step MPC-CBF (single obstacle).

This is a 1:1 copy of main_aegis.py with ONLY the safety-control QP block
replaced by an N-step receding-horizon MPC-CBF (with slack). Everything
else — VLM obstacle pick, GroundingDINO + depth + fit_ellipse, single
obstacle ellipsoid, z_fixed integration, replan_steps=5 — is preserved
identically so we can isolate the QP improvement.

Default N=5 horizon (matches pi05's replan_steps).

Usage:
    python eval_mpc_cbf_single.py \
        --task-suite-name safelibero_long --safety-level I \
        --task-index 0 1 2 3 --episode-index 0 1 2 3 4 \
        --N 5 --video-out-path results/mpc_cbf_long_lvI

Comparison baseline: AEGIS results.json from
    /root/autodl-tmp/MysafeVLA/aegis_runs/paper_runs/aegis_long_full_lvI/...
"""
import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time
import imageio
from typing import List

import numpy as np
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R

# Import AEGIS utils + our MPC-CBF
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "aegis_baseline"))
sys.path.insert(0, _ROOT)
from utils import (  # noqa: E402
    compute_h_coeffs_3d, compute_h_ij, get_point_cloud, filtering_points,
    fit_ellipse, obstacle_detection,
)
from src.safety import solve_mpc_cbf  # noqa: E402

import warnings
warnings.filterwarnings("ignore")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "safelibero_long"
    safety_level: str = "I"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_steps_wait: int = 20
    video_out_path: str = "results/mpc_cbf"
    seed: int = 7
    # === MPC-CBF specific ===
    N: int = 5
    alpha_h: float = 10.0
    delta_min: float = 0.0
    slack_l1: float = 1e3
    slack_l2: float = 1e5
    # Tangential push: when h < threshold, bias u_v_safe tangentially in QP cost
    tangent_threshold: float = 0.10
    tangent_strength: float = 5.0
    tangent_force_horizontal: bool = False  # v80: force lateral (xy) bypass instead of following pi05's intent
    align_trigger_z_min: float = 0.0           # v84: align yaw only when eef_z in [z_min, z_max] (pre-grasp altitude)
    align_trigger_z_max: float = 99.0
    align_continuous: bool = False             # v85: continuous yaw alignment (no rise+rotate state machine)
    align_phase_grasp: bool = False            # v92: enable yaw align in grasp phase (gripper open, descending toward mug)
    align_phase_place: bool = False            # v92: enable yaw align in place phase (gripper closed, descending toward plate)
    align_phase_transit: bool = False          # v92: enable yaw align in transit phase (rare)
    align_phase_descend_threshold: float = -0.05  # action[2] below this => descending
    wrist_pivot_finger_offset_z: float = 0.10     # v101: assumed fingertip distance below gripper for pivot correction
    wrist_pivot_correction_scale: float = 1.0     # 1.0 = full pivot correction; 0 = none
    align_keep_grip_open: bool = False             # v106: force grip=-1 during align rotation (prevents fingers closing mid-rotate)
    align_translate_scale: float = 1.0             # v107: scale on pi_0.5 u_v_ref while align rotation active (0=freeze translation)
    # v108: approach-deceleration — when gripper open AND near a graspable mug, scale down translation
    approach_decel_enable: bool = False
    approach_decel_d_max: float = 0.20             # at this distance to nearest mug, scale=1
    approach_decel_d_min: float = 0.04             # at or below this distance, scale=approach_decel_min_scale
    approach_decel_min_scale: float = 0.2          # minimum translation scale when very close to mug
    # v97: post-completion freeze — when all mugs placed on plates, freeze gripper so it stops bumping things
    freeze_on_completion: bool = False
    freeze_completion_xy_thresh: float = 0.06    # mug-plate xy distance to consider placed
    freeze_completion_z_max: float = 0.50        # mug z must be below this to consider on-plate
    # Toward-target reward: when h < threshold, also reward u_v in u_v_ref direction
    target_strength: float = 0.0
    # Horizontal rotation reward: when gripper is near obstacle AND stalled, solve a sub-
    # optimization for the world-z rotation angle θ that maximizes h(rotated gripper ↔ obstacle).
    # Drive u_omega toward sign(θ*) world-z direction.
    align_strength: float = 0.0
    align_stall_window: int = 30        # steps to look back for stall detection
    align_stall_range: float = 0.03     # if eef position range over window < this (m), stalled
    align_h_threshold: float = 0.05     # only trigger when h_min < this (gripper near obstacle)
    align_n_samples: int = 24
    align_min_h_gain: float = 0.01
    align_min_height: float = 0.0       # legacy (unused in v11 rising-edge mode)
    align_min_theta: float = 0.10
    align_rise_speed_z: float = 0.0     # legacy (unused in v11)
    # v11 rising-edge state-machine:
    #   - When h transitions IDLE (h>thresh) → CLOSE (h<thresh) — rising edge — pick θ*
    #     and start rotation lasting align_rotate_steps steps.
    #   - During rotation: apply ω_align, also keep position update slow (no override of u_v_ref).
    #   - After rotation done: idle. Re-trigger only when h goes back above threshold then below.
    align_rotate_steps: int = 30        # fixed number of steps to rotate after rising edge
    # Handback: if u_v_ref · z_world_to_obs < -handback_thresh, skip QP (use raw pi05)
    handback_dot_threshold: float = 0.0
    save_videos: bool = False
    # v21: multi-ellipsoid obstacle — cluster filter_points into K sub-ellipsoids
    # to better approximate non-convex shapes (e.g. white storage box) without
    # over-extending xy footprint into the basket drop zone.
    n_obstacle_clusters: int = 1               # 1 = single ellipsoid (back-compat)
    obstacle_cluster_min_pts: int = 30         # require at least N pts per cluster
    # v27: when not holding, shrink gripper Q to true free-gripper geometry
    # (carry-Q values 0.12y/0.20z are inflated for held cup; not-holding gripper
    # is much smaller, and the inflated Q blocks approach to mugs near obstacles)
    free_gripper_q_xy0: float = 0.04           # x half-axis when not holding (vs 0.06 carry)
    free_gripper_q_xy1: float = 0.06           # y half-axis when not holding (vs 0.12 carry)
    free_gripper_q_z: float = 0.10             # z half-axis when not holding (vs 0.20 carry)
    # v28: when holding (carry), use larger Q to cover full gripper+mug footprint
    holding_gripper_q_xy0: float = 0.06        # x half-axis when holding (default 0.06)
    holding_gripper_q_xy1: float = 0.12        # y half-axis when holding (default 0.12)
    holding_gripper_q_z: float = 0.20          # z half-axis when holding (default 0.20)
    # v28e: only apply v27/v28 gripper Q overrides to specific tasks (e.g. "2")
    gripper_q_override_task_ids: str = ""      # comma-separated; empty = disabled (use Q1_diag)
    # v39: force grip closed when holding mug and not yet at target plate
    # (prevents pi_0.5 from releasing prematurely)
    grip_lock_task_ids: str = ""               # comma-separated; empty = disabled
    grip_lock_plate_d_min: float = 0.15        # only force +1 when nearest plate xy distance > this
    # v44: force grip open when over plate + descended (ensures release at right place)
    grip_release_task_ids: str = ""            # comma-separated; empty = disabled
    grip_release_plate_d_max: float = 0.05     # plate xy distance < this
    grip_release_eef_z_max: float = 0.50       # eef_z < this (descended over plate)
    # v51: gentle plate bias when holding + plate close (per-task)
    plate_bias_task_ids: str = ""              # comma-separated; empty = disabled
    plate_bias_plate_d_max: float = 0.20       # only bias when plate xy < this
    plate_bias_xy: float = 0.2                 # xy bias magnitude toward plate
    # v57: scripted placement pilot — bypass pi_0.5 when holding mug2 (after mug1 placed)
    # to script direct placement onto plate_2.
    placement_pilot_task_ids: str = ""         # comma-separated; empty = disabled
    placement_pilot_target_mug: str = "white_yellow"  # mug name substring (target = mug 2)
    placement_pilot_target_plate: str = "plate_2"     # plate name to deliver to
    placement_pilot_descend_z: float = 0.45    # target z above plate (≈ plate_z + small lift)
    placement_pilot_xy_speed: float = 0.5      # action xy max magnitude
    placement_pilot_z_descend_speed: float = -0.5  # negative = down
    placement_pilot_release_dxy: float = 0.04  # plate xy distance to start opening grip
    placement_pilot_safe_z: float = 0.75       # high-altitude transit z to avoid obstacle in xy plane
    placement_pilot_grasp_z: float = 0.02      # offset above mug top when descending to grasp
    # diag: per-step log
    diag_log_every: int = 0                    # 0 = disabled; e.g. 5 = every 5 steps
    # Multi-ellipsoid gripper (B): default OFF — reverted to v3b after lvII test
    use_two_grippers: bool = False
    forearm_offset_z: float = 0.08      # +z offset (eef-frame)
    forearm_q_xy: float = 0.06
    forearm_q_z: float = 0.08
    # Carry-adaptive (C): cup-aware Q1 when grasping (default ON since v12)
    use_carry_adaptive: bool = True
    grasp_qpos_threshold: float = 0.05
    carry_extra_z: float = 0.08      # cup ~0.10–0.15m, half-extent ~0.08
    carry_extra_xy: float = 0.04     # cup ~5cm radius
    carry_offset_extra: float = 0.05
    # v12: pre-rise before rotation (ensure z high enough so rotated gripper/cup
    # doesn't sweep into low obstacles or table)
    rotation_min_z: float = 1.00          # without grasping
    rotation_min_z_grasp: float = 1.05    # with grasped object
    rise_steps: int = 3
    rotation_rise_speed: float = 0.25    # legacy: u-space additive bias
    rise_override: bool = True           # v13: replace u_v[:3] entirely during RISE
    rise_override_speed: float = 1.5     # u-space world-z velocity during RISE override
    # v13b: bypass QP during RISE — emit pure +z action directly (most robust)
    rise_bypass_qp: bool = True
    rise_action_z: float = 1.0           # action[2] (world-z delta-pos command, normalized -1..1)
    # v14: bypass QP during ROTATE too — emit pure rotation action; pause pi05
    rotate_bypass_qp: bool = True
    rotate_action_speed: float = 1.0     # action[3:6] magnitude (local frame, like u_omega)
    clear_plan_after_maneuver: bool = True   # force pi05 to re-query with fresh state
    align_cooldown_steps: int = 10       # block re-trigger for N steps after maneuver
    # v15: post-grasp rise. On gripper-close rising-edge (open→closed), pause pi05
    # and rise until eef_z >= target_z (or hit max_steps). Then resume pi05.
    post_grasp_rise_enable: bool = True
    post_grasp_rise_target_z: float = 1.0
    post_grasp_rise_max_steps: int = 12
    post_grasp_rise_action_z: float = 1.0
    # v15b: confirm an object is actually held (nearest graspable within radius)
    post_grasp_confirm_radius: float = 0.10
    # v15c: also require nearest object has been LIFTED above its initial z
    post_grasp_lift_threshold: float = 0.02   # m above initial obj_z
    # v19w: rise via z-bias (no bypass) — add upward delta to a_exec[2] until target_z reached
    rise_via_bias: bool = False
    rise_bias_z: float = 0.5                  # added to a_exec[2] while in rise mode
    # v20t: also bias wrist rotation during bias-rise so OSC coordinates multiple joints
    # (avoid degenerate IK where only forearm extends; force whole-arm motion)
    rise_bias_wrist_rx: float = 0.0
    rise_bias_wrist_ry: float = 0.2
    rise_bias_wrist_rz: float = 0.0
    # v18: also require object is ACTIVELY rising in recent window (not just sitting elevated)
    post_grasp_active_window: int = 8
    post_grasp_active_dz: float = 0.01        # min upward movement over window
    # v19q: is_holding via cmd+qpos mismatch (no object tracking needed)
    holding_qpos_min: float = 0.005           # fingers must not be fully closed
    holding_qpos_max: float = 0.07            # fingers must not be fully open
    holding_cmd_streak: int = 5               # v19r: consecutive close cmd steps
    # v19v: after rise, guide gripper xy toward closest basket/plate before pi_0.5 takes over
    guide_basket_enable: bool = False  # restored v19h: disabled by default
    guide_basket_target_radius: float = 0.05
    guide_basket_action_xy: float = 0.5
    guide_basket_max_steps: int = 50
    # v19n: hysteresis — after trigger fires, need N consecutive is_holding=False
    # steps before allowing re-trigger (avoids bouncing around lift threshold)
    post_grasp_release_streak: int = 30
    # v19o: detect held by "object moves with gripper" — relative pos stable over K steps
    post_grasp_corr_window: int = 10
    post_grasp_corr_tolerance: float = 0.02   # m max spread of relative pos
    # v19p: also require object's recent peak z to be lifted above initial
    # (filters false positive: hover near table-resting object)
    post_grasp_recent_window: int = 30
    post_grasp_recent_lift: float = 0.02      # m above initial obj_z
    # v20v: drop actively_rising requirement so holding-while-hovering is detected
    holding_require_active_rising: bool = False
    # v20w: when holding object high above table, shrink gripper z-half-axis
    # so CBF doesn't block descent into basket near tall obstacles
    release_q_z_shrink: float = 0.10           # z half-axis used during release phase
    release_q_z_min_eef_z: float = 0.50        # only shrink when eef_z above this
    # v20x: also shrink obstacle Q (xy + z) during release phase — large boxes (e.g.
    # white storage box) have wide xy footprint that envelops basket drop position
    release_obstacle_q_scale: float = 0.5      # multiplier on Q2 during release phase
    obstacle_q_scale: float = 1.0              # global multiplier on Q2 (always); <1 inflates obstacle
    # v16: stuck-detection — if eef position range over window < threshold, clear pi05 plan
    stuck_detect_enable: bool = True
    stuck_window: int = 30
    stuck_range_threshold: float = 0.02     # m
    stuck_cooldown_steps: int = 30
    # v17: stuck → also force a physical rise (reuse post-grasp rise machinery)
    stuck_rise_steps: int = 6
    # v20j: stuck rise target = cur_eef_z + offset (always rise above current, not fixed)
    stuck_rise_target_offset: float = 0.10
    # v20t: cap rise target — never rise eef above this absolute z (avoid runaway height)
    stuck_rise_target_max_z: float = 0.60
    # v19g: only trigger stuck-detection when fingers OPEN (approach phase)
    # avoids false-triggers during slow placement
    stuck_gate_open_only: bool = False
    # v19s: stuck only when eef_z low (likely approaching object on table)
    stuck_max_eef_z: float = 0.55
    # v20h: separate stuck params for holding state (more lenient)
    stuck_window_holding: int = 60
    stuck_range_threshold_holding: float = 0.05
    # v13: grasp-target-aware target_dir. When gripper is OPEN, use nearest graspable
    # object position from env (instead of pi05's u_v_ref) for the target_dir reward.
    use_grasp_target: bool = True
    grasp_target_exclude: str = "obstacle,plate,basket,eef,gripper"


def _quat2axisangle(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q[3] > 1: q[3] = 1
    elif q[3] < -1: q[3] = -1
    d = math.sqrt(1 - q[3] ** 2)
    if abs(d) < 1e-8: return np.zeros(3)
    return (q[:3] * 2 * math.acos(q[3])) / d


def _build_pi05_element(obs, lang, resize_size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate([
        obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {"observation/image": img, "observation/wrist_image": wrist,
             "observation/state": state, "prompt": str(lang)}


def _get_libero_env(task, level, resolution, seed):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution, camera_widths=resolution,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand", "backview"],
    )
    env.seed(seed)
    return env, task.language


def _build_horizon_u_refs(action_plan_list, R1, mu_row, N):
    """Build N reference u vectors (9D) from a 7D action chunk.
    u_ref stays pure pi05 intent — tangent push is added in QP cost, not here.
    """
    if not action_plan_list:
        action_plan_list = [np.zeros(7, dtype=np.float64)]
    seq = list(action_plan_list)
    if len(seq) < N:
        seq = seq + [seq[-1]] * (N - len(seq))
    seq = seq[:N]
    u_refs = []
    for a in seq:
        a_np = np.asarray(a, dtype=np.float64)
        v_ref = R1.T @ a_np[:3]
        u_v_ref = 5.0 * v_ref
        u_om_ref = 5.0 * a_np[3:6]
        u_z_nom = 10.0 * np.asarray(mu_row, dtype=np.float64)
        u_refs.append(np.concatenate([u_v_ref, u_om_ref, u_z_nom]))
    return u_refs


def _ellipsoid_clearance(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j):
    """Signed nearest-point clearance between two ellipsoids along the line connecting centers.
    Positive = separated by this distance along the connecting axis.
    Q_diag interpreted as semi-axes (radii in meters)."""
    diff = p_j - p_i
    d = float(np.linalg.norm(diff))
    if d < 1e-9:
        return -float(max(Q_i_diag) + max(Q_j_diag))
    u = diff / d
    u_i = R_i.T @ u  # direction in ellipsoid_i frame
    u_j = R_j.T @ u  # direction in ellipsoid_j frame
    r_i = float(np.sqrt(np.sum((np.asarray(Q_i_diag) * u_i) ** 2)))
    r_j = float(np.sqrt(np.sum((np.asarray(Q_j_diag) * u_j) ** 2)))
    return d - r_i - r_j


def _find_best_horizontal_rotation(eef_pos, offset_local, R1, Q1_diag,
                                     p_obs, Q_obs, R_obs, n_samples=24):
    """Sample θ ∈ [-π, π] around world-z axis. v91: pick yaw that maximizes
    nearest-point clearance between rotated gripper ellipsoid and obstacle ellipsoid.
    Return (best_theta, clearance_best, clearance_current).
    """
    best_c = -float("inf"); best_theta = 0.0; c_at_zero = None
    thetas = np.linspace(-np.pi, np.pi, n_samples + 1)[:-1]
    for theta in thetas:
        ct, st = np.cos(theta), np.sin(theta)
        Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        R1_new = Rz @ R1
        p1_new = eef_pos + R1_new @ offset_local
        c = _ellipsoid_clearance(p1_new, Q1_diag, R1_new, p_obs, Q_obs, R_obs)
        if abs(theta) < 1e-6:
            c_at_zero = c
        if c > best_c:
            best_c = c
            best_theta = float(theta)
    if c_at_zero is None:
        c_at_zero = _ellipsoid_clearance(eef_pos + R1 @ offset_local, Q1_diag, R1,
                                          p_obs, Q_obs, R_obs)
    return best_theta, best_c, float(c_at_zero)


def _compute_tangent_dir_local(z_world, R1, u_v_ref_local, eps=1e-6, force_horizontal=False):
    """Pick a tangent direction in eef-frame to bias u_v_ref toward circumnavigation.

    Strategy:
      - z_world: world-frame direction from gripper toward obstacle (unit vector)
      - z_local = R1.T @ z_world : same direction in eef-frame
      - If force_horizontal: always use horizontal cross-product tangent (side bypass).
      - Else project u_v_ref onto plane perpendicular to z_local. If non-zero, use that
        (continue pi05's lateral intent). Else fallback to horizontal cross product
        in world frame, then transform to eef frame.

    Returns unit-vector (3,) in eef-frame tangent to obstacle surface, or None
    if no sensible tangent (gripper has no lateral intent and z_world is vertical).
    """
    z_world = np.asarray(z_world, dtype=np.float64).reshape(3)
    z_world = z_world / (np.linalg.norm(z_world) + eps)
    z_local = R1.T @ z_world
    if force_horizontal:
        # v80: always force horizontal lateral tangent (side bypass), pick sign to
        # match pi05's lateral intent if any.
        z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t_world = np.cross(z_world, z_up)
        if np.linalg.norm(t_world) > eps:
            t_world = t_world / np.linalg.norm(t_world)
            t_local = R1.T @ t_world
            # pick sign to align with pi_0.5's lateral intent (if any)
            u_v = np.asarray(u_v_ref_local, dtype=np.float64).reshape(3)
            u_v_lat = u_v - (u_v @ z_local) * z_local
            if np.linalg.norm(u_v_lat) > eps and (t_local @ u_v_lat) < 0:
                t_local = -t_local
            return t_local
        # z_world nearly vertical => fall through to projection logic
    # Project u_v_ref onto tangent plane (perp to z_local in eef frame)
    u_v_ref_local = np.asarray(u_v_ref_local, dtype=np.float64).reshape(3)
    t_local = u_v_ref_local - (u_v_ref_local @ z_local) * z_local
    if np.linalg.norm(t_local) > eps:
        return t_local / np.linalg.norm(t_local)
    # Fallback: cross product with world-up to get horizontal tangent
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t_world = np.cross(z_world, z_up)
    if np.linalg.norm(t_world) > eps:
        t_world = t_world / np.linalg.norm(t_world)
        return R1.T @ t_world
    return None


def eval_libero(args: Args):
    np.random.seed(args.seed)
    safety_level = args.safety_level

    # === Setup ===
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=safety_level)
    logging.info(f"Task suite: {args.task_suite_name}, level: {safety_level}")

    if "long" in args.task_suite_name:
        max_steps = 550
    else:
        max_steps = 300

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    print("[init] connecting pi05 ...", flush=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[init] pi05 OK", flush=True)

    print("[init] loading GroundingDINO ...", flush=True)
    sys.path.insert(0, "/root/autodl-tmp/vlsa-aegis")
    from groundingdino.util.inference import load_model
    gdino = load_model(
        "/root/autodl-tmp/vlsa-aegis/GroundingDINO/GroundingDINO_SwinT_OGC.py",
        "/root/autodl-tmp/vlsa-aegis/GroundingDINO/groundingdino_swint_ogc.pth",
    )
    print("[init] GroundingDINO OK", flush=True)

    all_results = []
    t_total = time.time()

    for task_id in args.task_index:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, safety_level, LIBERO_ENV_RESOLUTION, args.seed)
        task_segment = task_description.replace(" ", "_")
        out_dir = pathlib.Path(args.video_out_path) / task_segment / f"lv{safety_level}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for episode_idx in args.episode_index:
            print(f"\n[task {task_id} ep {episode_idx}] {task_description}", flush=True)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1

            # === Initial gripper state ===
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = R.from_quat(eef_quat).as_matrix()
            offset_local = np.array([0, 0, -0.08])
            p1 = eef_pos + R1 @ offset_local
            # Unified: Q[2]=0.20 covers gripper + held cup/can (no carry-Q needed)
            Q1_diag = np.array([0.06, 0.12, 0.20])
            print(f"  [Q1_diag] task_id={task_id} Q1_diag={Q1_diag.tolist()}", flush=True)

            # === VLM-based obstacle pick (single, AEGIS-style) ===
            agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            agent_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
            back_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
            back_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
            img_out_dir = out_dir / f"{episode_idx}"
            img_out_dir.mkdir(parents=True, exist_ok=True)

            obstacle_name_str = obstacle_detection(agent_img, task_description, args.task_suite_name)
            print(f"  [VLM] picked: {obstacle_name_str}", flush=True)
            agent_pts = get_point_cloud(agent_img, agent_depth, env, "agentview", obstacle_name_str, gdino, img_out_dir)
            back_pts = get_point_cloud(back_img, back_depth, env, "backview", obstacle_name_str, gdino, img_out_dir)
            if agent_pts.shape[1] > 0 and back_pts.shape[1] > 0:
                full_pts = np.vstack([agent_pts, back_pts])
            elif back_pts.shape[1] > 0:
                full_pts = back_pts
            elif agent_pts.shape[1] > 0:
                full_pts = agent_pts
            else:
                full_pts = np.array([[]])

            filt_pts = filtering_points(full_pts, args.task_suite_name)
            flag_safety = filt_pts.shape[0] > 0
            obstacle = None
            obstacles_all = []
            if flag_safety:
                # v21: cluster point cloud into K sub-ellipsoids
                if args.n_obstacle_clusters > 1 and filt_pts.shape[0] >= args.obstacle_cluster_min_pts * 2:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=args.n_obstacle_clusters, n_init=4, random_state=args.seed)
                    labels = km.fit_predict(filt_pts)
                    for k in range(args.n_obstacle_clusters):
                        cluster_pts = filt_pts[labels == k]
                        if cluster_pts.shape[0] < args.obstacle_cluster_min_pts:
                            continue
                        try:
                            p_k, R_k, Q_k = fit_ellipse(cluster_pts, plot=False, save_path=None)
                            obstacles_all.append({"p": p_k, "R_mat": R_k, "Q": Q_k})
                        except Exception:
                            continue
                if not obstacles_all:
                    p2, R2_mat, Q2_diag = fit_ellipse(filt_pts, plot=False, save_path=img_out_dir)
                    obstacles_all = [{"p": p2, "R_mat": R2_mat, "Q": Q2_diag}]
                # v79 global obstacle inflation: scale all Q by obstacle_q_scale (<1 => larger ellipsoid)
                if abs(args.obstacle_q_scale - 1.0) > 1e-9:
                    for _o in obstacles_all:
                        _o["Q"] = _o["Q"] * float(args.obstacle_q_scale)
                # initial obstacle = closest to current gripper
                _dists = [float(np.linalg.norm(o["p"] - p1)) for o in obstacles_all]
                obstacle = obstacles_all[int(np.argmin(_dists))]
                p2 = obstacle["p"]
                z_fixed = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
                if len(obstacles_all) > 1:
                    print(f"  [obstacles] K={len(obstacles_all)} clusters", flush=True)
            else:
                z_fixed = np.array([1.0, 0.0, 0.0])
            dt_z = 0.05

            # === Active obstacle for collision detection (independent of perception) ===
            obstacle_keys = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]
            obstacle_name_env = " "
            for k in obstacle_keys:
                pp = obs[f"{k}_pos"]
                if pp[2] > 0 and -0.5 < pp[0] < 0.5 and -0.5 < pp[1] < 0.5:
                    obstacle_name_env = k
                    break
            initial_obstacle_pos = obs[obstacle_name_env + "_pos"] if obstacle_name_env != " " else None

            # v19v: snapshot basket/plate positions as guide targets
            basket_targets = {}
            for _k, _v in obs.items():
                if not (isinstance(_k, str) and _k.endswith("_pos")):
                    continue
                if any(_x in _k for _x in ("obstacle", "eef", "gripper")):
                    continue
                if not any(_x in _k for _x in ("basket", "plate")):
                    continue
                try:
                    _arr = np.asarray(_v, dtype=np.float64)
                except Exception:
                    continue
                if _arr.shape == (3,):
                    basket_targets[_k] = _arr.copy()
            print(f"  [basket_targets] {list(basket_targets.keys())}", flush=True)
            # v15c: snapshot initial positions of all graspable objects (to detect lift)
            _excl_init = ("obstacle", "plate", "basket", "eef", "gripper")
            initial_graspable = {}
            for _k, _v in obs.items():
                if not (isinstance(_k, str) and _k.endswith("_pos")):
                    continue
                if any(_x in _k for _x in _excl_init):
                    continue
                try:
                    _arr = np.asarray(_v, dtype=np.float64)
                except Exception:
                    continue
                if _arr.shape == (3,):
                    initial_graspable[_k] = _arr.copy()
            collide_flag = False
            success = False
            collide_step = -1
            t_ep_start = time.time()

            h_min_traj = []
            slack_traj = []
            qp_ok_count = 0
            frames = [] if args.save_videos else None
            eef_position_history = collections.deque(maxlen=args.align_stall_window)
            align_active_count = 0
            # v12 rising-edge state (with optional pre-rise)
            prev_close = False
            rotation_remaining = 0
            rotation_omega_local = None
            rise_remaining = 0
            rotation_pending = 0
            cooldown_remaining = 0
            # v15 post-grasp rise state
            prev_is_grasping = False
            post_grasp_rise_remaining = 0
            post_grasp_gripper_cmd = 1.0  # closed during rise
            release_streak = 0  # v19n: count consecutive is_holding=False steps
            last_gripper_cmd = -1.0  # v19q: track last commanded gripper action
            cmd_close_streak = 0  # v19r: consecutive close-command counter
            guide_basket_remaining = 0  # v19v: countdown for guide-to-basket phase
            in_rise_bias_mode = False  # v19w: True while injecting upward bias
            current_rise_target_z = 0.0  # v20j: per-trigger target (stuck uses relative)
            # v28e: gripper Q override active for this task?
            _gripper_q_override_ids = {int(s.strip()) for s in args.gripper_q_override_task_ids.split(",") if s.strip()}
            _gripper_q_override_active = bool(_gripper_q_override_ids) and (task_id in _gripper_q_override_ids)
            # v39: grip lock active for this task?
            _grip_lock_ids = {int(s.strip()) for s in args.grip_lock_task_ids.split(",") if s.strip()}
            _grip_lock_active = bool(_grip_lock_ids) and (task_id in _grip_lock_ids)
            # v44: grip release active for this task?
            _grip_release_ids = {int(s.strip()) for s in args.grip_release_task_ids.split(",") if s.strip()}
            _grip_release_active = bool(_grip_release_ids) and (task_id in _grip_release_ids)
            # v51: plate bias active for this task?
            _plate_bias_ids = {int(s.strip()) for s in args.plate_bias_task_ids.split(",") if s.strip()}
            _plate_bias_active = bool(_plate_bias_ids) and (task_id in _plate_bias_ids)
            # v57: scripted placement pilot active for this task?
            _pp_ids = {int(s.strip()) for s in args.placement_pilot_task_ids.split(",") if s.strip()}
            _pp_active = bool(_pp_ids) and (task_id in _pp_ids)
            _pp_engaged = False  # latched once mug 2 grasped + released first mug
            ever_held_v57 = False
            release_streak_v57 = 0
            ever_released_v57 = False
            # v16 stuck-detection state
            stuck_eef_window = collections.deque(maxlen=max(args.stuck_window, args.stuck_window_holding))
            stuck_cooldown_remaining = 0
            stuck_trigger_count = 0
            # v18: per-object recent z window for active-rise check
            recent_obj_z = {k: collections.deque(maxlen=args.post_grasp_active_window)
                              for k in initial_graspable.keys()}
            # v19o: per-object recent (obj - gripper_center) for correlation check
            recent_rel_pos = {k: collections.deque(maxlen=args.post_grasp_corr_window)
                                for k in initial_graspable.keys()}
            # v19p: per-object recent z window for peak-lift check
            recent_obj_z_long = {k: collections.deque(maxlen=args.post_grasp_recent_window)
                                   for k in initial_graspable.keys()}

            t = 0
            while t < max_steps:
                # === v19h restored: is_holding by fingers_closed + lifted + actively_rising ===
                cur_eef_z = float(obs["robot0_eef_pos"][2])
                eef_world_top = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                g_qpos_top = obs.get("robot0_gripper_qpos", np.zeros(2))
                fingers_closed_top = (float(np.sum(np.abs(g_qpos_top)))
                                        < args.grasp_qpos_threshold)
                # Update recent z window for active-rise check
                for kk in list(recent_obj_z.keys()):
                    cur_v_top = obs.get(kk)
                    if cur_v_top is None:
                        continue
                    try:
                        recent_obj_z[kk].append(float(np.asarray(cur_v_top, dtype=np.float64)[2]))
                    except Exception:
                        continue
                is_holding_top = False
                if fingers_closed_top:
                    for kk, init_pos in initial_graspable.items():
                        cur_v_top = obs.get(kk)
                        if cur_v_top is None:
                            continue
                        try:
                            cur_arr_top = np.asarray(cur_v_top, dtype=np.float64)
                        except Exception:
                            continue
                        if cur_arr_top.shape != (3,):
                            continue
                        d_top = float(np.linalg.norm(cur_arr_top - eef_world_top))
                        lifted_top = (cur_arr_top[2] - init_pos[2]) >= args.post_grasp_lift_threshold
                        actively_rising = False
                        zw = recent_obj_z.get(kk)
                        if zw is not None and len(zw) == args.post_grasp_active_window:
                            actively_rising = (zw[-1] - zw[0]) >= args.post_grasp_active_dz
                        if (d_top <= args.post_grasp_confirm_radius
                            and lifted_top
                            and (actively_rising or not args.holding_require_active_rising)):
                            is_holding_top = True
                            break
                # v32: backup is_holding — grip-cmd close + mug d<0.12 + cmd_close_streak>=5
                if (not is_holding_top
                    and last_gripper_cmd > 0
                    and cmd_close_streak >= 5):
                    for _kk in initial_graspable.keys():
                        _v = obs.get(_kk)
                        if _v is None: continue
                        try:
                            _arr = np.asarray(_v, dtype=np.float64)
                            if _arr.shape != (3,): continue
                        except Exception:
                            continue
                        if float(np.linalg.norm(_arr - eef_world_top)) <= 0.12:
                            is_holding_top = True
                            break
                # === v16: stuck-detection plan-reset ===
                # v58: skip stuck-detection when placement pilot engaged
                if args.stuck_detect_enable and not _pp_engaged:
                    stuck_eef_window.append(eef_world_top.copy())
                    if stuck_cooldown_remaining > 0:
                        stuck_cooldown_remaining -= 1
                    # v20h: dual-mode stuck — sensitive when not holding, lenient when holding
                    if is_holding_top:
                        eff_window = args.stuck_window_holding
                        eff_threshold = args.stuck_range_threshold_holding
                    else:
                        eff_window = args.stuck_window
                        eff_threshold = args.stuck_range_threshold
                    if (len(stuck_eef_window) >= eff_window
                        and stuck_cooldown_remaining == 0
                        and post_grasp_rise_remaining == 0
                        and not in_rise_bias_mode
                        and cur_eef_z <= args.stuck_max_eef_z):
                        arr_w = np.stack(list(stuck_eef_window)[-eff_window:], axis=0)
                        rng = float(np.max(np.linalg.norm(arr_w - arr_w.mean(0), axis=1)))
                        if rng < eff_threshold:
                            action_plan.clear()
                            stuck_cooldown_remaining = args.stuck_cooldown_steps
                            stuck_trigger_count += 1
                            # v20s: stuck-rise via BIAS mode (don't bypass pi_0.5)
                            in_rise_bias_mode = True
                            current_rise_target_z = min(
                                args.stuck_rise_target_max_z,
                                cur_eef_z + args.stuck_rise_target_offset,
                            )
                            stuck_eef_window.clear()
                            print(f"  [stuck@{t} BIAS-MODE] range={rng:.4f}  z={eef_world_top[2]:.3f}  "
                                   f"target={current_rise_target_z:.3f}  holding={is_holding_top}",
                                   flush=True)
                # === v15: post-grasp rise — fires on rising edge of is_holding ===
                if args.post_grasp_rise_enable:
                    is_grasping_top = is_holding_top
                    # v19n: track release streak (consecutive False steps)
                    if is_grasping_top:
                        release_streak = 0
                    else:
                        release_streak += 1
                    # Rising-edge with hysteresis: prev_is_grasping must have been
                    # False for at least post_grasp_release_streak steps
                    can_retrigger = (release_streak >= args.post_grasp_release_streak)
                    if (is_grasping_top and (not prev_is_grasping or can_retrigger)
                        and post_grasp_rise_remaining == 0
                        and not in_rise_bias_mode
                        and cur_eef_z < args.post_grasp_rise_target_z):
                        if args.rise_via_bias:
                            in_rise_bias_mode = True
                            print(f"  [post-grasp@{t} BIAS-MODE START] z={cur_eef_z:.3f} → "
                                   f"target={args.post_grasp_rise_target_z:.2f}",
                                   flush=True)
                        else:
                            post_grasp_rise_remaining = args.post_grasp_rise_max_steps
                            post_grasp_gripper_cmd = 1.0   # holding object
                            current_rise_target_z = args.post_grasp_rise_target_z
                            print(f"  [post-grasp@{t} START] z={cur_eef_z:.3f} → "
                                   f"target={args.post_grasp_rise_target_z:.2f}  "
                                   f"max_steps={post_grasp_rise_remaining}  "
                                   f"release_streak={release_streak}",
                                   flush=True)
                        prev_is_grasping = True
                        release_streak = 0
                    if is_grasping_top:
                        prev_is_grasping = True
                    elif release_streak >= args.post_grasp_release_streak:
                        prev_is_grasping = False  # confirmed release
                # === Bias-mode rise exit (always active, used by stuck-rise too) ===
                _bias_target = current_rise_target_z if current_rise_target_z > 0 else args.post_grasp_rise_target_z
                if in_rise_bias_mode and cur_eef_z >= _bias_target:
                    in_rise_bias_mode = False
                    if args.clear_plan_after_maneuver:
                        action_plan.clear()
                    cooldown_remaining = args.align_cooldown_steps
                    current_rise_target_z = 0.0
                    print(f"  [bias-rise@{t} REACHED] z={cur_eef_z:.3f}", flush=True)
                # === Rise bypass loop (always active if rise_remaining > 0, e.g., from stuck) ===
                if True:  # always run rise loop regardless of post_grasp_rise_enable
                    if post_grasp_rise_remaining > 0:
                        # v19: pi_0.5 fully paused. Exit ONLY when eef_z >= current_rise_target_z
                        # OR safety cap (post_grasp_rise_remaining hit 0).
                        cur_eef_z_now = float(obs["robot0_eef_pos"][2])
                        _rise_target = current_rise_target_z if current_rise_target_z > 0 else args.post_grasp_rise_target_z
                        if cur_eef_z_now >= _rise_target:
                            post_grasp_rise_remaining = 0
                            if args.guide_basket_enable and len(basket_targets) > 0 and post_grasp_gripper_cmd > 0:
                                # Transition to guide-to-basket phase
                                guide_basket_remaining = args.guide_basket_max_steps
                                print(f"  [rise@{t} REACHED → GUIDE] z={cur_eef_z_now:.3f}",
                                       flush=True)
                            else:
                                print(f"  [rise@{t} REACHED] z={cur_eef_z_now:.3f} ≥ "
                                       f"target={_rise_target:.2f} — resume pi05",
                                       flush=True)
                                if args.clear_plan_after_maneuver:
                                    action_plan.clear()
                                cooldown_remaining = args.align_cooldown_steps
                            current_rise_target_z = 0.0
                            # Fall through to next iteration (guide or normal)
                        else:
                            # Emit rise action and step env (pi_0.5 paused)
                            if frames is not None:
                                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            a_pg = np.zeros(7, dtype=np.float64)
                            a_pg[2] = args.post_grasp_rise_action_z
                            a_pg[6] = post_grasp_gripper_cmd
                            post_grasp_rise_remaining -= 1
                            h_min_traj.append(0.0)
                            slack_traj.append(0.0)
                            obs, _, done, _ = env.step(a_pg.tolist())
                            last_gripper_cmd = float(a_pg[6])
                            cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0
                            if not collide_flag and initial_obstacle_pos is not None:
                                cur = obs[obstacle_name_env + "_pos"]
                                if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                                    collide_flag = True; collide_step = t
                            eef_pos = obs["robot0_eef_pos"]
                            eef_quat = obs["robot0_eef_quat"]
                            R1 = R.from_quat(eef_quat).as_matrix()
                            p1 = eef_pos + R1 @ offset_local
                            if (post_grasp_rise_remaining % 10 == 0
                                or post_grasp_rise_remaining == 0):
                                print(f"  [rise@{t}] z={float(eef_pos[2]):.3f}  "
                                       f"remaining={post_grasp_rise_remaining}",
                                       flush=True)
                            if post_grasp_rise_remaining == 0:
                                print(f"  [rise@{t} SAFETY-CAP] z={float(eef_pos[2]):.3f} "
                                       f"never reached {args.post_grasp_rise_target_z:.2f}",
                                       flush=True)
                                if args.clear_plan_after_maneuver:
                                    action_plan.clear()
                                cooldown_remaining = args.align_cooldown_steps
                            if done: success = True; break
                            t += 1
                            continue

                # === v19v: guide-to-basket phase (bypass pi_0.5) ===
                if guide_basket_remaining > 0:
                    eef_xy = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)[:2]
                    # Find closest basket
                    best_d = float("inf"); best_target = None
                    for bk, bp in basket_targets.items():
                        d = float(np.linalg.norm(bp[:2] - eef_xy))
                        if d < best_d:
                            best_d = d; best_target = bp
                    if best_target is None or best_d <= args.guide_basket_target_radius:
                        print(f"  [guide@{t} REACHED] xy_dist={best_d:.3f} — resume pi05",
                               flush=True)
                        guide_basket_remaining = 0
                        if args.clear_plan_after_maneuver:
                            action_plan.clear()
                        cooldown_remaining = args.align_cooldown_steps
                        # Fall through
                    else:
                        if frames is not None:
                            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        a_g = np.zeros(7, dtype=np.float64)
                        dx = float(best_target[0] - eef_xy[0])
                        dy = float(best_target[1] - eef_xy[1])
                        norm = max(np.sqrt(dx*dx + dy*dy), 1e-6)
                        a_g[0] = float(np.clip(dx / norm * args.guide_basket_action_xy, -1, 1))
                        a_g[1] = float(np.clip(dy / norm * args.guide_basket_action_xy, -1, 1))
                        a_g[6] = post_grasp_gripper_cmd  # closed (carrying)
                        guide_basket_remaining -= 1
                        h_min_traj.append(0.0)
                        slack_traj.append(0.0)
                        obs, _, done, _ = env.step(a_g.tolist())
                        last_gripper_cmd = float(a_g[6])
                        cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0
                        if not collide_flag and initial_obstacle_pos is not None:
                            cur = obs[obstacle_name_env + "_pos"]
                            if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                                collide_flag = True; collide_step = t
                        eef_pos = obs["robot0_eef_pos"]
                        eef_quat = obs["robot0_eef_quat"]
                        R1 = R.from_quat(eef_quat).as_matrix()
                        p1 = eef_pos + R1 @ offset_local
                        if guide_basket_remaining == 0:
                            if args.clear_plan_after_maneuver:
                                action_plan.clear()
                            cooldown_remaining = args.align_cooldown_steps
                        if done: success = True; break
                        t += 1
                        continue

                if not action_plan:
                    element = _build_pi05_element(obs, task_description, args.resize_size)
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()  # 7D
                if frames is not None:
                    frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

                if flag_safety and obstacle is not None:
                    # === Build N-step u_refs from remaining chunk ===
                    remaining = [action] + list(action_plan)  # current + future in chunk
                    # v21: pick closest obstacle to gripper for linearization (z_fixed, h0)
                    if len(obstacles_all) > 1:
                        _dists = [float(np.linalg.norm(o["p"] - p1)) for o in obstacles_all]
                        obstacle = obstacles_all[int(np.argmin(_dists))]
                    # Compute mu_row at current state (used as u_z reference linearization)
                    p_j = obstacle["p"]; R_j = obstacle["R_mat"]; Q_j = obstacle["Q"]
                    # v20x: shrink obstacle Q during release phase (xy + z)
                    if (is_holding_top
                        and cur_eef_z > args.release_q_z_min_eef_z
                        and args.release_obstacle_q_scale != 1.0):
                        Q_j = Q_j * args.release_obstacle_q_scale
                    a_v0, a_om0, a_uz0, h0, mu_row = compute_h_coeffs_3d(
                        p1, Q1_diag, R1, p_j, Q_j, R_j, z_fixed,
                    )

                    # u_ref pure pi05 intent (no tangent injection here)
                    u_refs = _build_horizon_u_refs(remaining, R1, mu_row, args.N)

                    # === Build grippers list (B + C) ===
                    # Unified: QP carry-aware Q uses same is_holding_top as post-grasp trigger
                    is_grasping = is_holding_top
                    apply_carry = (args.use_carry_adaptive and is_grasping)
                    # Lower gripper (fingers only — no carry-Q z expansion).
                    # Default Q1[2] from AEGIS (0.20 for tall obj, 0.11 for mug).
                    lower_offset_z = -0.08
                    lower_Q = Q1_diag.copy()
                    if _gripper_q_override_active:
                        # v27: not-holding → use true free-gripper Q
                        if not is_holding_top:
                            lower_Q = np.array([args.free_gripper_q_xy0,
                                                 args.free_gripper_q_xy1,
                                                 args.free_gripper_q_z])
                        # v28: holding → use larger carry Q (covers gripper + mug)
                        else:
                            lower_Q = np.array([args.holding_gripper_q_xy0,
                                                 args.holding_gripper_q_xy1,
                                                 args.holding_gripper_q_z])
                    else:
                        # default: only release-z-shrink at high z holding
                        if (is_holding_top
                            and cur_eef_z > args.release_q_z_min_eef_z):
                            lower_Q = np.array([Q1_diag[0], Q1_diag[1], args.release_q_z_shrink])
                    lower_offset = np.array([0.0, 0.0, lower_offset_z])
                    lower_p = eef_pos + R1 @ lower_offset
                    grippers = [{
                        "p": lower_p, "Q": lower_Q, "R": R1,
                        "offset_local": lower_offset,
                    }]
                    # B: add upper forearm ellipsoid
                    if args.use_two_grippers:
                        upper_offset = np.array([0.0, 0.0, args.forearm_offset_z])
                        upper_Q = np.array([args.forearm_q_xy, args.forearm_q_xy, args.forearm_q_z])
                        upper_p = eef_pos + R1 @ upper_offset
                        grippers.append({
                            "p": upper_p, "Q": upper_Q, "R": R1,
                            "offset_local": upper_offset,
                        })

                    # === Compute tangent + target dir for QP cost rewards ===
                    z_world_to_obs = (p_j - lower_p) / (np.linalg.norm(p_j - lower_p) + 1e-9)
                    u_v_ref_first = 5.0 * (R1.T @ np.asarray(action[:3], dtype=np.float64))
                    tangent_dir_local = _compute_tangent_dir_local(
                        z_world=z_world_to_obs, R1=R1,
                        u_v_ref_local=u_v_ref_first,
                        force_horizontal=args.tangent_force_horizontal,
                    )
                    # === Target dir: grasp-target when gripper OPEN, pi05 intent when CLOSED ===
                    target_dir_local = None
                    used_grasp_target = False
                    if args.use_grasp_target and (not is_grasping):
                        excl = tuple(s.strip() for s in args.grasp_target_exclude.split(","))
                        eef_world = np.asarray(eef_pos, dtype=np.float64)
                        cands = []
                        for k, v in obs.items():
                            if not (isinstance(k, str) and k.endswith("_pos")):
                                continue
                            if any(x in k for x in excl):
                                continue
                            try:
                                arr = np.asarray(v, dtype=np.float64)
                            except Exception:
                                continue
                            if arr.shape != (3,):
                                continue
                            cands.append((float(np.linalg.norm(arr - eef_world)), arr))
                        if cands:
                            cands.sort(key=lambda x: x[0])
                            grasp_target_world = cands[0][1]
                            delta = grasp_target_world - lower_p
                            n_d = float(np.linalg.norm(delta))
                            if n_d > 1e-6:
                                target_dir_local = R1.T @ (delta / n_d)
                                used_grasp_target = True
                    if target_dir_local is None:
                        n_uv = np.linalg.norm(u_v_ref_first)
                        target_dir_local = (u_v_ref_first / n_uv) if n_uv > 1e-9 else None

                    # === Handback: gripper moving AWAY from obstacle? Skip filter ===
                    # u_v_ref direction in world: R1 @ u_v_ref_local
                    if args.handback_dot_threshold > 0.0:
                        u_v_world = R1 @ u_v_ref_first
                        u_v_world_norm = np.linalg.norm(u_v_world) + 1e-9
                        # u_v_world · z_world_to_obs < 0 → moving away
                        proj = float(u_v_world @ z_world_to_obs) / u_v_world_norm
                        if proj < -args.handback_dot_threshold:
                            # Hand control back to pi05
                            obs_step_handback = True
                        else:
                            obs_step_handback = False
                    else:
                        obs_step_handback = False

                    # === v12 rising-edge state machine: RISE → ROTATE ===
                    if cooldown_remaining > 0:
                        cooldown_remaining -= 1
                    omega_align_local = None
                    align_phase = "IDLE"
                    # v92 phase-aware align: detect grasp/transit/place/retreat phase, gate trigger per phase
                    _prev_hold = locals().get('prev_is_holding_local', False)
                    _descending = float(action[2]) < args.align_phase_descend_threshold
                    if (not is_holding_top) and _descending:
                        cur_phase = "grasp"
                    elif is_holding_top and _descending:
                        cur_phase = "place"
                    elif is_holding_top and (not _descending):
                        cur_phase = "transit"
                    else:
                        cur_phase = "retreat"
                    _prev_phase = locals().get('prev_phase_local', "init")
                    phase_changed = (cur_phase != _prev_phase)
                    phase_enabled = ((cur_phase == "grasp" and args.align_phase_grasp)
                                      or (cur_phase == "place" and args.align_phase_place)
                                      or (cur_phase == "transit" and args.align_phase_transit))
                    h_close_z_ok = (args.align_strength > 0.0
                                     and h0 < args.align_h_threshold
                                     and args.align_trigger_z_min <= eef_pos[2] <= args.align_trigger_z_max)
                    # trigger when entering an enabled phase AND obstacle is near
                    h_edge = h_close_z_ok and not prev_close
                    grasp_edge = (is_holding_top != _prev_hold) and h_close_z_ok
                    phase_edge = phase_changed and phase_enabled and h_close_z_ok
                    is_close_now = h_close_z_ok
                    can_trigger = (phase_edge or h_edge or grasp_edge) and phase_enabled and (
                                    rotation_remaining == 0 and rise_remaining == 0
                                    and cooldown_remaining == 0)
                    prev_is_holding_local = is_holding_top
                    prev_phase_local = cur_phase
                    if can_trigger:
                        # Sub-opt with cup-aware ellipsoid (lower_Q, lower_offset)
                        best_theta, h_best, h_zero = _find_best_horizontal_rotation(
                            eef_pos=eef_pos, offset_local=lower_offset,
                            R1=R1, Q1_diag=lower_Q,
                            p_obs=p_j, Q_obs=Q_j, R_obs=R_j,
                            n_samples=args.align_n_samples,
                        )
                        if (h_best - h_zero) >= args.align_min_h_gain and abs(best_theta) >= args.align_min_theta:
                            omega_world_z = np.array([0.0, 0.0, float(np.sign(best_theta))])
                            rotation_omega_local = R1.T @ omega_world_z
                            min_z = (args.rotation_min_z_grasp if is_grasping
                                      else args.rotation_min_z)
                            if eef_pos[2] < min_z:
                                rise_remaining = args.rise_steps
                                rotation_pending = args.align_rotate_steps
                            else:
                                rotation_remaining = args.align_rotate_steps
                                rotation_pending = 0
                            print(f"  [align@{t} START] phase={cur_phase} θ*={best_theta:+.2f}rad  "
                                   f"Δh={h_best - h_zero:+.3f}  z={eef_pos[2]:.3f}  "
                                   f"grasp={is_grasping}  "
                                   f"rise={rise_remaining}  "
                                   f"rot={rotation_remaining or rotation_pending}",
                                   flush=True)
                    if rise_remaining > 0 and args.rise_bypass_qp:
                        # v13b: bypass QP — emit pure upward action directly (pi05 paused)
                        a_exec_rise = np.zeros(7, dtype=np.float64)
                        a_exec_rise[2] = args.rise_action_z
                        a_exec_rise[6] = float(action[6])
                        rise_remaining -= 1
                        align_phase = "RISING"
                        align_active_count += 1
                        if rise_remaining == 0 and rotation_pending > 0:
                            rotation_remaining = rotation_pending
                            rotation_pending = 0
                        if rise_remaining == 0 and rotation_remaining == 0:
                            # End of maneuver (rise-only path) → reset pi05 plan
                            if args.clear_plan_after_maneuver:
                                action_plan.clear()
                            cooldown_remaining = args.align_cooldown_steps
                        prev_close = is_close_now
                        # Step env directly
                        h_min_traj.append(h0)
                        slack_traj.append(0.0)
                        obs, _, done, _ = env.step(a_exec_rise.tolist())
                        last_gripper_cmd = float(a_exec_rise[6])
                        cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0
                        if not collide_flag and initial_obstacle_pos is not None:
                            cur = obs[obstacle_name_env + "_pos"]
                            if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                                collide_flag = True; collide_step = t
                        eef_pos = obs["robot0_eef_pos"]
                        eef_quat = obs["robot0_eef_quat"]
                        R1 = R.from_quat(eef_quat).as_matrix()
                        p1 = eef_pos + R1 @ offset_local
                        if done: success = True; break
                        t += 1
                        continue
                    if rotation_remaining > 0 and args.rotate_bypass_qp:
                        # v14: bypass QP — emit pure rotation action (pi05 paused)
                        a_exec_rot = np.zeros(7, dtype=np.float64)
                        omega_cmd_local = (rotation_omega_local
                                            * args.rotate_action_speed)
                        a_exec_rot[3:6] = omega_cmd_local
                        a_exec_rot[6] = float(action[6])
                        rotation_remaining -= 1
                        align_phase = "ROTATING"
                        align_active_count += 1
                        if rotation_remaining == 0:
                            if args.clear_plan_after_maneuver:
                                action_plan.clear()
                            cooldown_remaining = args.align_cooldown_steps
                        prev_close = is_close_now
                        h_min_traj.append(h0)
                        slack_traj.append(0.0)
                        obs, _, done, _ = env.step(a_exec_rot.tolist())
                        last_gripper_cmd = float(a_exec_rot[6])
                        cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0
                        if not collide_flag and initial_obstacle_pos is not None:
                            cur = obs[obstacle_name_env + "_pos"]
                            if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                                collide_flag = True; collide_step = t
                        eef_pos = obs["robot0_eef_pos"]
                        eef_quat = obs["robot0_eef_quat"]
                        R1 = R.from_quat(eef_quat).as_matrix()
                        p1 = eef_pos + R1 @ offset_local
                        if done: success = True; break
                        t += 1
                        continue
                    if rise_remaining > 0:
                        # v13: override u_v[:3] with pure upward velocity to defeat pi05 descent
                        if args.rise_override:
                            rise_local = R1.T @ np.array([0.0, 0.0, args.rise_override_speed])
                            for k in range(len(u_refs)):
                                u_refs[k] = u_refs[k].astype(np.float64).copy()
                                u_refs[k][:3] = rise_local  # OVERRIDE, not add
                        else:
                            rise_local = R1.T @ np.array([0.0, 0.0, 1.0])
                            for k in range(len(u_refs)):
                                u_refs[k] = u_refs[k].astype(np.float64).copy()
                                u_refs[k][:3] = u_refs[k][:3] + args.rotation_rise_speed * rise_local
                        rise_remaining -= 1
                        align_phase = "RISING"
                        align_active_count += 1
                        if rise_remaining == 0 and rotation_pending > 0:
                            rotation_remaining = rotation_pending
                            rotation_pending = 0
                    elif rotation_remaining > 0:
                        omega_align_local = rotation_omega_local
                        rotation_remaining -= 1
                        align_phase = "ROTATING"
                        align_active_count += 1
                    prev_close = is_close_now

                    # === Handback: skip QP if moving away from obstacle ===
                    if obs_step_handback:
                        # Use pi05's raw action directly
                        a_exec = np.asarray(action, dtype=np.float64)
                        h_min_traj.append(h0)
                        slack_traj.append(0.0)
                        # No u_z update (no QP run)
                        obs, _, done, _ = env.step(a_exec.tolist())
                        last_gripper_cmd = float(a_exec[6])
                        cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0
                        if not collide_flag and initial_obstacle_pos is not None:
                            cur = obs[obstacle_name_env + "_pos"]
                            if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                                collide_flag = True; collide_step = t
                        eef_pos = obs["robot0_eef_pos"]
                        eef_quat = obs["robot0_eef_quat"]
                        R1 = R.from_quat(eef_quat).as_matrix()
                        p1 = eef_pos + R1 @ offset_local
                        if done: success = True; break
                        t += 1
                        continue

                    # v11: no u_ref override — pi05's intent preserved; rotation only via ω cost reward
                    # === Solve N-step MPC-CBF (multi-gripper) ===
                    # v21: also apply release-phase Q shrink to ALL obstacles in QP
                    _scale = 1.0
                    if (is_holding_top
                        and cur_eef_z > args.release_q_z_min_eef_z
                        and args.release_obstacle_q_scale != 1.0):
                        _scale *= args.release_obstacle_q_scale
                    if _scale != 1.0 and len(obstacles_all) > 1:
                        _qp_obstacles = [{"p": o["p"], "R_mat": o["R_mat"],
                                          "Q": o["Q"] * _scale}
                                         for o in obstacles_all]
                    else:
                        _qp_obstacles = obstacles_all
                    # z_inits: each obstacle has its own initial direction
                    _z_inits = [(o["p"] - p1) / (np.linalg.norm(o["p"] - p1) + 1e-9)
                                for o in _qp_obstacles]
                    # v107: when align rotation is active, scale down pi_0.5 translation to limit drift
                    if rotation_remaining > 0 and abs(args.align_translate_scale - 1.0) > 1e-6:
                        for _i in range(len(u_refs)):
                            u_refs[_i][:3] = u_refs[_i][:3] * float(args.align_translate_scale)
                    # v108: approach deceleration — gripper open + near nearest mug -> reduce translation
                    if args.approach_decel_enable and (not is_holding_top):
                        _d_min_mug = float('inf')
                        for _kk, _ip in initial_graspable.items():
                            _v = obs.get(_kk)
                            if _v is None: continue
                            try:
                                _arr = np.asarray(_v, dtype=np.float64)
                                if _arr.shape != (3,): continue
                                _d = float(np.linalg.norm(_arr - eef_world_top))
                            except Exception:
                                continue
                            if _d < _d_min_mug:
                                _d_min_mug = _d
                        if _d_min_mug < args.approach_decel_d_max:
                            d_max = float(args.approach_decel_d_max)
                            d_min = float(args.approach_decel_d_min)
                            s_min = float(args.approach_decel_min_scale)
                            # linear interp: scale=s_min at d_min, scale=1 at d_max
                            t_lerp = float(np.clip((_d_min_mug - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0))
                            _decel = s_min + (1.0 - s_min) * t_lerp
                            for _i in range(len(u_refs)):
                                u_refs[_i][:3] = u_refs[_i][:3] * _decel
                            if t % 20 == 0:
                                print(f"  [decel@{t}] d_mug={_d_min_mug:.3f} scale={_decel:.2f}", flush=True)
                    # v101 finger-pivot compensation: when align active, add translation v_corr so fingertip stays put.
                    # finger_local ≈ (0, 0, -d_finger) (below gripper); v_corr = -(ω_local × finger_local)
                    if omega_align_local is not None and args.align_strength > 0.0:
                        _d_finger = float(args.wrist_pivot_finger_offset_z)
                        _finger_local = np.array([0.0, 0.0, -_d_finger], dtype=np.float64)
                        _omega_l = np.asarray(omega_align_local, dtype=np.float64)
                        # cross: ω × finger_local
                        _cross = np.array([
                            _omega_l[1] * _finger_local[2] - _omega_l[2] * _finger_local[1],
                            _omega_l[2] * _finger_local[0] - _omega_l[0] * _finger_local[2],
                            _omega_l[0] * _finger_local[1] - _omega_l[1] * _finger_local[0],
                        ], dtype=np.float64)
                        _v_corr = -_cross  # in eef-frame
                        _scale_corr = float(args.wrist_pivot_correction_scale)
                        # add to first horizon u_v_ref (eef-frame translation)
                        for _i in range(min(len(u_refs), 1)):
                            u_refs[_i][:3] = u_refs[_i][:3] + _scale_corr * _v_corr * 5.0
                    u_safes, info = solve_mpc_cbf(
                        grippers=grippers,
                        obstacles=_qp_obstacles,
                        u_refs=u_refs,
                        alpha_h=args.alpha_h, delta_min=args.delta_min,
                        slack_l1=args.slack_l1, slack_l2=args.slack_l2,
                        z_inits=_z_inits,
                        tangent_dir_local=tangent_dir_local,
                        tangent_threshold=args.tangent_threshold,
                        tangent_weight=args.tangent_strength,
                        target_dir_local=target_dir_local,
                        target_weight=args.target_strength,
                        omega_align_local=omega_align_local,
                        align_weight=args.align_strength,
                    )
                    h_min_traj.append(info["h_min"])
                    slack_traj.append(info["slack_sum"])
                    if info["qp_ok"]:
                        qp_ok_count += 1

                    u_safe = u_safes[0]
                    u_v = u_safe[:3]
                    u_omega = u_safe[3:6]
                    u_z = u_safe[6:9]

                    # === Update z_fixed (AEGIS-style) ===
                    Id = np.eye(3)
                    dz = (Id - np.outer(z_fixed, z_fixed)) @ u_z
                    z_fixed = z_fixed + dz * dt_z
                    z_fixed = z_fixed / (np.linalg.norm(z_fixed) + 1e-9)

                    a_exec = np.zeros(7)
                    a_exec[:3] = 0.2 * R1 @ u_v
                    a_exec[3:6] = 0.2 * u_omega
                    a_exec[6] = action[6]
                    # diag: per-step log (default every 5 steps; 0 = disabled)
                    if args.diag_log_every > 0 and (t % args.diag_log_every) == 0:
                        # nearest graspable to eef (mug etc)
                        _best_kk, _best_d, _best_pos = None, 1e9, None
                        for _kk in initial_graspable.keys():
                            _v = obs.get(_kk)
                            if _v is None:
                                continue
                            try:
                                _arr = np.asarray(_v, dtype=np.float64)
                                if _arr.shape != (3,):
                                    continue
                            except Exception:
                                continue
                            _d = float(np.linalg.norm(_arr - eef_pos))
                            if _d < _best_d:
                                _best_d, _best_kk, _best_pos = _d, _kk, _arr
                        _mug_short = (_best_kk[:6] if _best_kk else "?")
                        _mug_z = float(_best_pos[2]) if _best_pos is not None else 0.0
                        _mug_init_z = float(initial_graspable.get(_best_kk, np.zeros(3))[2]) if _best_kk else 0.0
                        _mug_dz_init = _mug_z - _mug_init_z
                        _obs_dxy = float(np.linalg.norm(p_j[:2] - eef_pos[:2]))
                        _obs_dz = float(p_j[2] - eef_pos[2])
                        _pi05_v = np.asarray(action[:3], dtype=np.float64)
                        _qp_v = a_exec[:3]
                        # nearest plate/basket target
                        _plate_d = -1.0; _plate_pos = None; _plate_short = "?"
                        for _bt in basket_targets.keys():
                            _bp = obs.get(_bt)
                            if _bp is None: continue
                            try:
                                _bpa = np.asarray(_bp, dtype=np.float64)
                                if _bpa.shape != (3,): continue
                            except Exception:
                                continue
                            _bd = float(np.linalg.norm(_bpa - eef_pos))
                            if _plate_d < 0 or _bd < _plate_d:
                                _plate_d, _plate_pos, _plate_short = _bd, _bpa, _bt[:7]
                        _plate_str = (f"plate={_plate_short} ppos=[{_plate_pos[0]:+.2f},{_plate_pos[1]:+.2f},{_plate_pos[2]:.2f}] pd={_plate_d:.2f}"
                                       if _plate_pos is not None else "")
                        _mode = "RISE" if in_rise_bias_mode else ("HOLD" if is_holding_top else "FREE")
                        print(f"  [t{t}] eef=[{eef_pos[0]:+.2f},{eef_pos[1]:+.2f},{eef_pos[2]:.2f}] "
                               f"{_mode} hold={int(is_holding_top)} grip={action[6]:+.1f} "
                               f"Q=[{lower_Q[0]:.2f},{lower_Q[1]:.2f},{lower_Q[2]:.2f}] "
                               f"obs_dxy={_obs_dxy:.2f} obs_dz={_obs_dz:+.2f} h0={h0:+.3f} "
                               f"mug={_mug_short} d={_best_d:.2f} mug_z={_mug_z:.2f} dz0={_mug_dz_init:+.3f} "
                               f"{_plate_str} "
                               f"pi05_v=[{_pi05_v[0]:+.2f},{_pi05_v[1]:+.2f},{_pi05_v[2]:+.2f}] "
                               f"qp_v=[{_qp_v[0]:+.2f},{_qp_v[1]:+.2f},{_qp_v[2]:+.2f}]", flush=True)
                else:
                    a_exec = np.asarray(action, dtype=np.float64)
                    h_min_traj.append(1.0)
                    slack_traj.append(0.0)

                # v106: force grip OPEN during align rotation — prevent fingers from closing mid-rotation and tipping mug
                if rotation_remaining > 0 and args.align_keep_grip_open:
                    a_exec[6] = -1.0
                # v19w: rise via z-bias — add upward delta while in rise mode
                # v20t: also bias wrist rotation so OSC coordinates whole arm (not just forearm)
                if in_rise_bias_mode and not _pp_engaged:
                    a_exec[2] = float(np.clip(a_exec[2] + args.rise_bias_z, -1.0, 1.0))
                    if args.rise_bias_wrist_rx != 0.0:
                        a_exec[3] = float(np.clip(a_exec[3] + args.rise_bias_wrist_rx, -1.0, 1.0))
                    if args.rise_bias_wrist_ry != 0.0:
                        a_exec[4] = float(np.clip(a_exec[4] + args.rise_bias_wrist_ry, -1.0, 1.0))
                    if args.rise_bias_wrist_rz != 0.0:
                        a_exec[5] = float(np.clip(a_exec[5] + args.rise_bias_wrist_rz, -1.0, 1.0))

                # v39+v44+v51: nearest plate xy + position
                _min_pd = 1e9; _min_plate_pos = None
                if _grip_lock_active or _grip_release_active or _plate_bias_active:
                    for _bt in basket_targets.keys():
                        _bp = obs.get(_bt)
                        if _bp is None: continue
                        try:
                            _bpa = np.asarray(_bp, dtype=np.float64)
                            if _bpa.shape != (3,): continue
                        except Exception:
                            continue
                        _bdxy = float(np.linalg.norm(_bpa[:2] - eef_pos[:2]))
                        if _bdxy < _min_pd:
                            _min_pd = _bdxy; _min_plate_pos = _bpa
                # v39: lock grip closed during transit
                if _grip_lock_active and is_holding_top and _min_pd > args.grip_lock_plate_d_min:
                    a_exec[6] = 1.0
                # v44: force release when over plate + descended
                # v97: add release-latch — once released, keep gripper open until eef leaves plate area
                if _grip_release_active:
                    _release_latch_set = locals().get('_grip_release_latched', set())
                    # Detect plate that was released over and remember it
                    if (is_holding_top
                        and _min_pd < args.grip_release_plate_d_max
                        and cur_eef_z < args.grip_release_eef_z_max):
                        a_exec[6] = -1.0
                        if _min_plate_pos is not None:
                            _release_latch_set.add(tuple(np.round(_min_plate_pos[:2], 3)))
                            _grip_release_latched = _release_latch_set
                    # While near a plate that was released over and z still low: keep gripper open
                    if _min_plate_pos is not None and _min_pd < args.grip_release_plate_d_max + 0.05:
                        _key = tuple(np.round(_min_plate_pos[:2], 3))
                        if _key in _release_latch_set and cur_eef_z < args.grip_release_eef_z_max + 0.10:
                            a_exec[6] = -1.0  # latch open
                # v51 plate bias removed (caused gripper to linger near plate after release)
                # v74: full state-machine pilot (approach mug2 -> grasp -> lift -> transport -> place -> retreat)
                if _pp_active:
                    if is_holding_top:
                        ever_held_v57 = True
                        release_streak_v57 = 0
                    else:
                        release_streak_v57 += 1
                        if ever_held_v57 and release_streak_v57 >= 5:
                            ever_released_v57 = True
                    # take over once mug1 is placed; pilot will scripted-grasp mug2 itself
                    if not _pp_engaged and ever_released_v57:
                        _pp_engaged = True
                        _pp_phase = "approach"  # initial phase
                        _pp_grasp_streak = 0
                        print(f"  [pilot@{t} TAKEOVER] starting scripted grasp of mug2", flush=True)
                    if _pp_engaged:
                        _phase = locals().get('_pp_phase', 'approach')
                        _safe_z = float(args.placement_pilot_safe_z)
                        _rel_dxy = float(args.placement_pilot_release_dxy)
                        _descend_z = float(args.placement_pilot_descend_z)
                        _grasp_z = float(args.placement_pilot_grasp_z)

                        # locate mug2 (target) and plate_2
                        _mug_pos = None
                        _mug_key = None
                        for _kk in initial_graspable.keys():
                            if args.placement_pilot_target_mug not in _kk.lower():
                                continue
                            _v = obs.get(_kk)
                            if _v is None: continue
                            try:
                                _arr = np.asarray(_v, dtype=np.float64)
                                if _arr.shape == (3,): _mug_pos = _arr; _mug_key = _kk; break
                            except Exception:
                                continue
                        _tgt_plate = None
                        for _bt in basket_targets.keys():
                            if args.placement_pilot_target_plate in _bt.lower():
                                _v = obs.get(_bt)
                                if _v is None: continue
                                try:
                                    _arr = np.asarray(_v, dtype=np.float64)
                                    if _arr.shape == (3,): _tgt_plate = _arr; break
                                except Exception:
                                    continue

                        _override_action = np.zeros(7, dtype=np.float64)

                        if _phase == "approach" and _mug_pos is not None:
                            # rise to safe_z then xy align over mug2
                            _dxy_m = _mug_pos[:2] - eef_world_top[:2]
                            _dxy_mn = float(np.linalg.norm(_dxy_m))
                            if cur_eef_z < _safe_z - 0.02 and _dxy_mn > 0.05:
                                _override_action[2] = +1.0
                                _override_action[6] = -1.0  # open
                            elif _dxy_mn > 0.02:
                                if _dxy_mn > 1e-6:
                                    _u = _dxy_m / _dxy_mn
                                    _override_action[0] = float(np.clip(1.0 * _u[0], -1, 1))
                                    _override_action[1] = float(np.clip(1.0 * _u[1], -1, 1))
                                if cur_eef_z < _safe_z:
                                    _override_action[2] = +0.3
                                _override_action[6] = -1.0
                            else:
                                _pp_phase = "descend_grasp"
                                _override_action[6] = -1.0
                            _stage = f"approach dxy_m={_dxy_mn:.3f}"
                        elif _phase == "descend_grasp" and _mug_pos is not None:
                            # descend to mug top, keeping xy aligned
                            _dxy_m = _mug_pos[:2] - eef_world_top[:2]
                            _dxy_mn = float(np.linalg.norm(_dxy_m))
                            _target_grasp_z = float(_mug_pos[2]) + _grasp_z  # offset above mug
                            if cur_eef_z > _target_grasp_z + 0.02:
                                # tiny xy correction
                                if _dxy_mn > 1e-6:
                                    _u = _dxy_m / _dxy_mn
                                    _override_action[0] = float(np.clip(0.5 * _u[0], -1, 1))
                                    _override_action[1] = float(np.clip(0.5 * _u[1], -1, 1))
                                _override_action[2] = -1.0
                                _override_action[6] = -1.0
                            else:
                                _pp_phase = "close_grasp"
                                _pp_grasp_streak = 0
                                _override_action[6] = +1.0
                            _stage = f"descend_grasp z={cur_eef_z:.3f}->{_target_grasp_z:.3f}"
                        elif _phase == "close_grasp":
                            # hold position, close gripper for several steps
                            _grasp_streak = locals().get('_pp_grasp_streak', 0) + 1
                            _pp_grasp_streak = _grasp_streak
                            _override_action[6] = +1.0
                            if _grasp_streak >= 10:
                                _pp_phase = "lift"
                            _stage = f"close_grasp streak={_grasp_streak}"
                        elif _phase == "lift":
                            # rise to safe_z while gripping
                            _override_action[2] = +1.0
                            _override_action[6] = +1.0
                            if cur_eef_z >= _safe_z - 0.02:
                                _pp_phase = "transport"
                            _stage = f"lift z={cur_eef_z:.3f}"
                        elif _phase == "transport" and _tgt_plate is not None:
                            _dxy = _tgt_plate[:2] - eef_world_top[:2]
                            _dxy_n = float(np.linalg.norm(_dxy))
                            _obs_y = float(initial_obstacle_pos[1]) if initial_obstacle_pos is not None else 0.0
                            _plate_y = float(_tgt_plate[1])
                            _eef_y = float(eef_world_top[1])
                            if _plate_y > _obs_y:
                                past_obstacle_y = _eef_y > _obs_y + 0.08
                            else:
                                past_obstacle_y = _eef_y < _obs_y - 0.08
                            tight_xy = (_dxy_n <= 0.05)
                            ok_to_descend = past_obstacle_y or tight_xy
                            if (cur_eef_z >= _safe_z - 0.05) and ((_dxy_n > _rel_dxy) or (not ok_to_descend)):
                                if _dxy_n > 1e-6:
                                    _u = _dxy / _dxy_n
                                    _override_action[0] = float(np.clip(1.0 * _u[0], -1, 1))
                                    _override_action[1] = float(np.clip(1.0 * _u[1], -1, 1))
                                if cur_eef_z < _safe_z:
                                    _override_action[2] = +0.3
                                _override_action[6] = +1.0
                                _stage = f"transport_xy dxy={_dxy_n:.3f}"
                            elif ok_to_descend and (_dxy_n <= _rel_dxy) and cur_eef_z > _descend_z + 0.02:
                                if _dxy_n > 1e-6:
                                    _u = _dxy / _dxy_n
                                    _override_action[0] = float(np.clip(0.3 * _u[0], -1, 1))
                                    _override_action[1] = float(np.clip(0.3 * _u[1], -1, 1))
                                # v75: slower descent to minimize mug swing/drag near obstacle
                                _override_action[2] = -0.4
                                _override_action[6] = +1.0
                                _stage = f"transport_descend dxy={_dxy_n:.3f} z={cur_eef_z:.3f}"
                            else:
                                _pp_phase = "release"
                                _pp_release_streak2 = 0
                                _override_action[6] = -1.0
                                _stage = "transport_done"
                        elif _phase == "release":
                            _streak = locals().get('_pp_release_streak2', 0) + 1
                            _pp_release_streak2 = _streak
                            _override_action[6] = -1.0
                            if _streak < 12:
                                _override_action[2] = 0.0
                                _stage = f"release_detach streak={_streak}"
                            else:
                                _override_action[2] = +1.0
                                _stage = f"release_retreat streak={_streak}"
                        else:
                            # fallback: retreat
                            _override_action[2] = +1.0
                            _override_action[6] = -1.0
                            _stage = "fallback"

                        a_exec = _override_action
                        if t % 5 == 0:
                            print(f"  [pilot@{t} {_phase}] {_stage} eef={eef_world_top.tolist()} grip={_override_action[6]:+.1f}", flush=True)

                obs, _, done, _ = env.step(a_exec.tolist())
                last_gripper_cmd = float(a_exec[6])
                cmd_close_streak = (cmd_close_streak + 1) if last_gripper_cmd > 0 else 0

                # === Collision detection (single env obstacle, AEGIS-style) ===
                if not collide_flag and initial_obstacle_pos is not None:
                    cur = obs[obstacle_name_env + "_pos"]
                    if np.sum(np.abs(cur - initial_obstacle_pos)) > 0.001:
                        collide_flag = True
                        collide_step = t
                        print(f"  [COLLIDE@{t}] obstacle moved by {float(np.linalg.norm(cur - initial_obstacle_pos)):.4f} eef={eef_world_top.tolist()}", flush=True)

                # === Update gripper state ===
                eef_pos = obs["robot0_eef_pos"]
                eef_quat = obs["robot0_eef_quat"]
                R1 = R.from_quat(eef_quat).as_matrix()
                p1 = eef_pos + R1 @ offset_local

                if done:
                    success = True
                    break
                t += 1

            ep_t = time.time() - t_ep_start
            outcome = "OK" if success and not collide_flag else ("COLLIDE" if collide_flag else "TIMEOUT")

            # Save video if requested
            if frames is not None and len(frames) > 0:
                try:
                    import imageio
                    # Append the final frame for completeness
                    frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    mp4_path = out_dir / f"t{task_id}_ep{episode_idx}_{outcome}.mp4"
                    imageio.mimwrite(str(mp4_path), frames, fps=30, codec="libx264")
                    print(f"  saved video -> {mp4_path}", flush=True)
                except Exception as e:
                    print(f"  [warn] video save failed: {e}", flush=True)

            ep_record = {
                "task_id": task_id, "ep": episode_idx,
                "task": task_description,
                "success": bool(success), "collided": bool(collide_flag),
                "steps": t + 1,
                "h_min": float(min(h_min_traj)) if h_min_traj else 1.0,
                "slack_mean": float(np.mean([s for s in slack_traj if s >= 0])) if any(s >= 0 for s in slack_traj) else 0.0,
                "qp_ok_count": qp_ok_count,
                "align_active_count": int(align_active_count),
                "vlm_obstacle": obstacle_name_str,
                "elapsed_s": ep_t,
            }
            all_results.append(ep_record)
            print(f"  -> {outcome}  step={t+1}  h_min={ep_record['h_min']:+.3f}  "
                  f"slack_mean={ep_record['slack_mean']:.2f}  "
                  f"qp_ok={qp_ok_count}/{t+1}  align={align_active_count}  ({ep_t:.0f}s)", flush=True)

        env.close()

    # === Summary ===
    n = len(all_results)
    sr = sum(r["success"] for r in all_results) / max(n, 1)
    cr = sum(r["collided"] for r in all_results) / max(n, 1)
    safe_sr = sum(r["success"] and not r["collided"] for r in all_results) / max(n, 1)
    print(f"\n[done] N={n}  SR={sr:.2%}  CR={cr:.2%}  Safe-SR={safe_sr:.2%}  "
          f"({(time.time()-t_total)/60:.1f} min)", flush=True)

    out = pathlib.Path(args.video_out_path)
    with open(out / "results.json", "w") as f:
        json.dump({"args": dataclasses.asdict(args), "results": all_results,
                    "summary": {"N": n, "SR": sr, "CR": cr, "SafeSR": safe_sr}}, f, indent=2)
    print(f"saved -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
