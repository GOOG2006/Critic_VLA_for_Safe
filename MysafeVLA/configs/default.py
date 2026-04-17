"""Default configuration for Critic-VLA safety evaluation."""
import os
import numpy as np

# ============ Paths (override via environment variables) ============
VLSA_AEGIS_ROOT = os.environ.get("VLSA_AEGIS_ROOT", "/root/autodl-tmp/vlsa-aegis")
GROUNDING_DINO_CONFIG = os.environ.get(
    "GROUNDING_DINO_CONFIG",
    os.path.join(VLSA_AEGIS_ROOT, "GroundingDINO/GroundingDINO_SwinT_OGC.py"),
)
GROUNDING_DINO_CKPT = os.environ.get(
    "GROUNDING_DINO_CKPT",
    os.path.join(VLSA_AEGIS_ROOT, "GroundingDINO/groundingdino_swint_ogc.pth"),
)
CRITIC_CKPT = os.environ.get(
    "CRITIC_CKPT",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "critic_v2.pt"),
)

# ============ pi0.5 server ============
PI05_HOST = os.environ.get("PI05_HOST", "127.0.0.1")
PI05_PORT = int(os.environ.get("PI05_PORT", "8000"))

# ============ Gripper ellipsoid (AEGIS defaults) ============
GRIPPER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
GRIPPER_Q_DIAG_DEFAULT = np.array([0.06, 0.12, 0.11], dtype=np.float64)
GRIPPER_Q_DIAG_TALL = np.array([0.06, 0.12, 0.20], dtype=np.float64)

# ============ CBF parameters ============
ALPHA_H = 10.0           # CBF gain
CRITIC_ALPHA_BOOST = 3.0  # multiplier when critic predicts danger
REPLAN_STEPS = 5          # action chunk replan interval
MAX_STEPS = 220           # max episode steps (spatial)
NUM_STEPS_WAIT = 10       # dummy steps for object settling
LOOKAHEAD = 10            # critic prediction horizon (steps)
