"""OpenVLA + Safe-MoLe + LIBERO rollout adapter.

Loads OpenVLA-7B (finetuned on a LIBERO suite), wraps it with Safe-MoLe CBF
projection, and runs rollout evaluation on a LIBERO benchmark suite.

Usage:
    python -m safe_mole.openvla_adapter eval \\
        --ckpt /root/autodl-tmp/openvla_libero_spatial \\
        --suite libero_spatial --tasks 0 1 2 --n_episodes 10 \\
        --out /root/autodl-tmp/eval_runs/openvla_spatial_safemole \\
        --apply_safe_mole

Reference: DERIVATION_PACKAGE.md v6; AEGIS/GRAPE/PACS comparison context.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# OpenVLA loader
# ---------------------------------------------------------------------------

def load_openvla(ckpt_path: str, device: str = "cuda:0"):
    """Load OpenVLA (HF trust_remote_code) + processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"[openvla] loading {ckpt_path}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    torch.cuda.synchronize()
    print(f"[openvla] loaded in {time.time()-t0:.1f}s; "
          f"GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# LIBERO → OpenVLA input conversion
# ---------------------------------------------------------------------------

def _jpeg_roundtrip(img_arr: np.ndarray) -> np.ndarray:
    """JPEG encode/decode to match OpenVLA training-time image quantization noise.

    This matches libero_utils.resize_image() which does:
        tf.image.encode_jpeg(img)  # Encode as JPEG (RLDS dataset builder)
        tf.io.decode_image(img)    # Decode back (quantization noise added)
        tf.image.resize(img, 224, method='lanczos3')
    """
    import cv2
    # JPEG encode/decode (default quality 95 in tf.image.encode_jpeg)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    # Resize to 224x224 with Lanczos (matches tf.image.resize method='lanczos3')
    img_arr = cv2.resize(img_arr, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return img_arr


def prepare_openvla_input(obs: dict, instruction: str, processor, device):
    """LIBERO obs dict + language → OpenVLA input tensors (on GPU, bf16).

    OpenVLA-LIBERO eval convention (from openvla/openvla repo):
      - Rotate image 180 degrees (`img[::-1, ::-1]`) to match training preprocessing.
      - JPEG encode/decode to match training-time quantization noise.
      - Resize to 224x224 with Lanczos (matches tf.image.resize method='lanczos3').
    """
    img = obs["agentview_image"][::-1, ::-1]   # 180-deg rotation
    img = _jpeg_roundtrip(np.ascontiguousarray(img))
    import cv2
    # GRAPE/OpenVLA protocol: center-crop scale 0.9 then resize back to 224.
    H, W = img.shape[:2]
    nH, nW = int(round(np.sqrt(0.9) * H)), int(round(np.sqrt(0.9) * W))
    y0, x0 = (H - nH) // 2, (W - nW) // 2
    img = img[y0:y0 + nH, x0:x0 + nW]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
    return inputs


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """OpenVLA gripper [0, 1] → LIBERO convention [-1, +1].

    LIBERO: gripper in [-1, 1], -1 = open, +1 = close.
    OpenVLA: gripper in [0, 1], 1 = open, 0 = close.

    Standard transformation (from openvla/experiments/robot/libero/run_libero_eval.py):
        action[-1] = 2 * action[-1] - 1       # [0,1] → [-1,1] but sign is FLIPPED
        action[-1] = -action[-1]               # invert so -1=open, +1=close
    Equivalent shorthand:  action[-1] = 1 - 2*action[-1]
    """
    action = action.copy()
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    # LIBERO convention: invert
    action[..., -1] = -action[..., -1]
    return action


@torch.no_grad()
def openvla_predict(model, inputs, unnorm_key: str, apply_gripper_norm: bool = True):
    """Run OpenVLA predict_action; returns 7-DoF numpy action.

    apply_gripper_norm: if True, converts [0,1] → [-1,+1] and inverts (official LIBERO eval).
    """
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    action = np.asarray(action, dtype=np.float32)
    raw_gripper = float(action[..., -1])
    if apply_gripper_norm:
        action = normalize_gripper_action(action, binarize=True)
    print(f"    [predict] raw gripper={raw_gripper:.3f}  final gripper={action[-1]:.3f}", flush=True)
    return action


# ---------------------------------------------------------------------------
# Safe-MoLe wrap over OpenVLA
# ---------------------------------------------------------------------------

@dataclass
class OpenVLASafeMoLeConfig:
    apply_safe_mole: bool = True
    # CBF parameters tuned for LIBERO EE increments (~1-5 cm per step)
    gamma: float = 0.5
    lip_h: float = 1.0
    d_max: float = 0.005
    a_max: float = 0.1               # max 7-DoF action norm in LIBERO
    eta: float = 1.0                 # 2nd-order Taylor bound (workspace small)
    # Barrier selects "table contact avoidance"
    z_min: float = 0.85               # safety height above table
    # Conformal buffers (pre-computed by calibration; default 0 = no conformal)
    sigma_h_bar: float = 0.0
    sigma_grad_bar: float = 0.0
    jac_norm: float = 1.0
    # Action-space Gaussian noise injected after VLA predict to stress safety (B3)
    action_noise_sigma: float = 0.0
    # GRAPE-style stage-1 pick barrier: z_floor = bowl_z_now when stage1 active
    use_pick_barrier: bool = True
    target_obj_key: str = "akita_black_bowl_1_pos"


def build_safe_mole_wrapper(device: str, cfg: OpenVLASafeMoLeConfig):
    """Build a Safe-MoLe extension (critic + projection) compatible with
    OpenVLA's output.

    Since OpenVLA has no CogKD cognition token, we attach the critic to the
    LLM's final hidden state mean-pooled over non-pad tokens. For the pilot
    we use analytic TableHeightBarrier, no learned critic yet — the projection
    is analytic (eef pos → h = z - z_min), not neural.
    """
    from safe_mole import TableHeightBarrier, CBFParams, issf_cbf_project
    import torch
    barrier = TableHeightBarrier(z_index=2, z_min=cfg.z_min)
    params = CBFParams(
        gamma=cfg.gamma, lip_h=cfg.lip_h, d_max=cfg.d_max,
        a_max=cfg.a_max, eta=cfg.eta, jac_norm=cfg.jac_norm,
    )
    return barrier, params


def safe_mole_project_eef(
    a_nominal: np.ndarray,
    eef_pos: np.ndarray,            # (3,) robot EE world position
    barrier,
    params,
    cfg: OpenVLASafeMoLeConfig,
    z_floor: float = None,          # dynamic safety floor (e.g., bowl_z during stage 1)
) -> tuple[np.ndarray, dict]:
    from safe_mole import issf_cbf_project
    """Apply ISSf-CBF projection on a predicted 7-DoF action using analytic
    barrier on end-effector position.

    Builds c(x) analytically: since action a[0:3] IS the delta-EE-pos,
        h(x_{t+1}) ≈ h(x_t) + [0 0 1] · a[0:3]   (z component only)
    so c = [0, 0, 1, 0, 0, 0, 0] in action space.
    """
    t = torch.from_numpy
    a = t(a_nominal).float()
    # True barrier at current state
    z_min_eff = cfg.z_min if z_floor is None else max(cfg.z_min, float(z_floor))
    h = float(eef_pos[2] - z_min_eff)                      # stage-aware floor
    c = torch.zeros(7)
    c[2] = 1.0                                              # action Δz is the safety axis
    xi = (cfg.gamma * cfg.sigma_h_bar
          + cfg.a_max * cfg.sigma_grad_bar * cfg.jac_norm
          + cfg.eta * cfg.a_max ** 2)
    a_safe = issf_cbf_project(
        a.unsqueeze(0),              # (1, 7)
        c.unsqueeze(0),              # (1, 7)
        torch.tensor([h]),           # (1,)
        xi=xi,
        params=params,
    ).squeeze(0).numpy()
    displacement = float(np.linalg.norm(a_safe - a_nominal))
    triggered = displacement > 1e-8
    return a_safe, {
        "h_true": h,
        "displacement": displacement,
        "triggered": triggered,
        "xi": xi,
    }


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------

NUM_STEPS_WAIT = 10  # LIBERO convention: dummy action first ~10 steps to let objects settle
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]   # 7-DoF no-op


def run_one_episode(
    env,
    model,
    processor,
    instruction: str,
    unnorm_key: str,
    cfg: OpenVLASafeMoLeConfig,
    barrier,
    params,
    init_state=None,
    max_steps: int = 220,
    device: str = "cuda:0",
) -> dict:
    """Single LIBERO episode. Returns stats dict."""
    env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    else:
        obs = env.reset()
    total_reward = 0.0
    success = False
    n_triggered = 0
    disp_acc = 0.0
    h_trace = []
    policy_step = -1

    # 1. Wait phase: let objects settle
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    # GRAPE-style stage-1 col_cost tracking (pick-and-place tasks).
    bowl_key = getattr(cfg, "target_obj_key", "akita_black_bowl_1_pos")
    bowl_z_init = float(obs[bowl_key][2]) if bowl_key in obs else None
    stage1_col_steps = 0
    stage1_total_steps = 0

    # 2. Policy rollout
    for step in range(max_steps):
        policy_step = step
        eef_pos = obs["robot0_eef_pos"]
        h = float(eef_pos[2] - cfg.z_min)
        h_trace.append(h)

        # Stage-1 check: EE must remain above target bowl until grasp lifts it.
        if bowl_z_init is not None:
            bowl_z_now = float(obs[bowl_key][2])
            lifted = (bowl_z_now - bowl_z_init) > 0.03  # 3cm lift threshold
            if not lifted:
                stage1_total_steps += 1
                if eef_pos[2] < bowl_z_now:
                    stage1_col_steps += 1

        inputs = prepare_openvla_input(obs, instruction, processor, device)
        a_nom = openvla_predict(model, inputs, unnorm_key)

        # (B3) Stress: inject Gaussian noise on delta-xyz.
        if cfg.action_noise_sigma > 0:
            noise = np.random.normal(0.0, cfg.action_noise_sigma, size=3).astype(a_nom.dtype)
            a_nom = a_nom.copy()
            a_nom[:3] = a_nom[:3] + noise

        if step < 5:
            print(f"    DEBUG step={step} eef={eef_pos.round(3).tolist()} "
                  f"a_nom={a_nom.round(3).tolist()}", flush=True)

        # Dynamic stage-1 safety floor: during approach, z_floor = bowl_z_now.
        z_floor = None
        if cfg.use_pick_barrier and bowl_z_init is not None:
            bowl_z_now_ = float(obs[bowl_key][2])
            if (bowl_z_now_ - bowl_z_init) <= 0.03:
                z_floor = bowl_z_now_

        if cfg.apply_safe_mole:
            a_exec, aux = safe_mole_project_eef(
                a_nom, eef_pos, barrier, params, cfg, z_floor=z_floor,
            )
            n_triggered += int(aux["triggered"])
            disp_acc += aux["displacement"]
        else:
            a_exec = a_nom

        obs, r, done, info = env.step(a_exec.tolist())
        total_reward += float(r)
        if done:
            success = bool(info.get("success", r > 0.5))
            break

    col_cost_rate = (stage1_col_steps / max(1, stage1_total_steps)) if stage1_total_steps > 0 else 0.0
    return {
        "success": success,
        "total_reward": total_reward,
        "steps": policy_step + 1,
        "projection_rate": n_triggered / max(1, policy_step + 1),
        "mean_displacement": disp_acc / max(1, n_triggered or 1),
        "h_min": float(min(h_trace)) if h_trace else 0.0,
        "h_viol_steps": sum(1 for h in h_trace if h < 0),
        # GRAPE stage-1 collision metrics (EE punches below bowl pre-grasp)
        "col_cost_any": int(stage1_col_steps > 0),
        "col_cost_rate": col_cost_rate,
        "stage1_steps": stage1_total_steps,
    }


def main_eval(args: argparse.Namespace) -> int:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    print(f"[eval] suite={args.suite}, n_tasks={suite.n_tasks}", flush=True)

    model, processor = load_openvla(args.ckpt)
    cfg = OpenVLASafeMoLeConfig(
        apply_safe_mole=args.apply_safe_mole,
        gamma=args.gamma, a_max=args.a_max, eta=args.eta,
        z_min=args.z_min, lip_h=args.lip_h, d_max=args.d_max,
        action_noise_sigma=getattr(args, "action_noise_sigma", 0.0),
    )
    barrier, params = build_safe_mole_wrapper("cuda:0", cfg)

    tasks_to_run = args.tasks or list(range(suite.n_tasks))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    unnorm_key = args.unnorm_key or args.suite   # default e.g. "libero_spatial"

    for task_idx in tasks_to_run:
        task = suite.get_task(task_idx)
        print(f"\n[eval] task {task_idx}: {task.language}", flush=True)
        bddl = os.path.join(get_libero_path("bddl_files"),
                           task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)  # reproducibility (official)
        init_states = suite.get_task_init_states(task_idx)
        print(f"  loaded {len(init_states)} predefined init states", flush=True)

        for ep in range(args.n_episodes):
            init_state = init_states[ep % len(init_states)]
            cfg._ep_seed = 1000 * int(task_idx) + int(ep)
            stats = run_one_episode(
                env, model, processor,
                instruction=task.language, unnorm_key=unnorm_key,
                cfg=cfg, barrier=barrier, params=params,
                init_state=init_state,
                max_steps=args.max_steps,
            )
            stats["task_idx"] = task_idx
            stats["ep"] = ep
            stats["apply_safe_mole"] = cfg.apply_safe_mole
            results.append(stats)
            print(f"  ep{ep}: success={stats['success']} steps={stats['steps']} "
                  f"proj_rate={stats['projection_rate']:.2f} "
                  f"h_min={stats['h_min']:.3f} h_viol={stats['h_viol_steps']}",
                  flush=True)
        env.close()

    # Aggregate
    succ = sum(r["success"] for r in results) / len(results)
    mean_proj = np.mean([r["projection_rate"] for r in results])
    print(f"\n[eval SUMMARY] success_rate = {succ:.3f} ({int(succ*len(results))}/{len(results)}) "
          f"projection_rate = {mean_proj:.3f}", flush=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "suite": args.suite,
            "tasks": tasks_to_run,
            "apply_safe_mole": cfg.apply_safe_mole,
            "config": cfg.__dict__,
            "results": results,
            "summary": {"success_rate": succ, "mean_projection_rate": mean_proj},
        }, f, indent=2)
    print(f"[eval] wrote {out_dir/'results.json'}", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("eval")
    pe.add_argument("--ckpt", required=True,
                    help="Path to OpenVLA checkpoint (local directory)")
    pe.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object",
                             "libero_goal", "libero_10", "libero_90"])
    pe.add_argument("--tasks", nargs="*", type=int, default=None,
                    help="task indices, default: all")
    pe.add_argument("--n_episodes", type=int, default=10)
    pe.add_argument("--max_steps", type=int, default=300)
    pe.add_argument("--out", required=True)
    pe.add_argument("--apply_safe_mole", action="store_true",
                    help="apply Safe-MoLe CBF projection (default off = baseline)")
    pe.add_argument("--unnorm_key", default=None,
                    help="dataset_statistics key; default <suite>_no_noops")
    # CBF params
    pe.add_argument("--gamma", type=float, default=0.5)
    pe.add_argument("--a_max", type=float, default=0.1)
    pe.add_argument("--eta", type=float, default=1.0)
    pe.add_argument("--z_min", type=float, default=0.85)
    pe.add_argument("--lip_h", type=float, default=1.0)
    pe.add_argument("--d_max", type=float, default=0.005)
    pe.add_argument("--action_noise_sigma", type=float, default=0.0,
                    help="std of Gaussian noise on delta-xyz action (B3)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "eval":
        raise SystemExit(main_eval(args))
