"""Minimal patches: keep AEGIS perception (GroundingDINO + ZhipuAI), only replace
openpi WebsocketClient with lerobot PI05 (jax not installed)
and stub eef_marker (not in SafeLIBERO env).
"""
import re

P = "/root/autodl-tmp/vlsa-aegis/main/main_aegis.py"
# Restore from backup first
import shutil
shutil.copy("/root/autodl-tmp/vlsa-aegis/main/main_aegis_orig.py.bak", P)
print("Restored from backup.")
s = open(P).read()

# patch 1: replace WebsocketClient with lerobot PI05 wrapper, but KEEP GroundingDINO loading
old1 = '''    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2
    CONFIG_PATH = "GroundingDINO/GroundingDINO_SwinT_OGC.py"    # Source code config file
    CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"   # Downloaded weights file
    DEVICE = "cuda"   # Select cpu/cuda
    BOX_TRESHOLD = 0.35     # Source code bounding box threshold
    TEXT_TRESHOLD = 0.25    # Source code text threshold for key attributes

    model_groundingdino = load_model(CONFIG_PATH, CHECKPOINT_PATH)'''
new1 = '''    # PATCHED: lerobot PI05 instead of openpi WebsocketClient (jax not installed)
    import torch
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    _local_policy = PI05Policy.from_pretrained("/root/autodl-tmp/pi05_libero_base").to("cuda:0").eval()
    _local_pre, _local_post = make_pre_post_processors(
        _local_policy.config, pretrained_path="/root/autodl-tmp/pi05_libero_base",
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": "/root/autodl-tmp/paligemma_tokenizer"}},
    )
    class _LocalClient:
        def infer(self, element):
            import numpy as _np, torch as _t
            img = element["observation/image"]
            wrist = element["observation/wrist_image"]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2,0,1)
                wrist = wrist.transpose(2,0,1)
            img_t = _t.from_numpy(_np.ascontiguousarray(img)).float().unsqueeze(0)/255.0
            wrist_t = _t.from_numpy(_np.ascontiguousarray(wrist)).float().unsqueeze(0)/255.0
            state_t = _t.from_numpy(element["observation/state"].astype(_np.float32)).unsqueeze(0)
            empty = _t.zeros((1,3,224,224), dtype=_t.float32)
            batch = {
                "observation.images.image": img_t.to("cuda:0"),
                "observation.images.image2": wrist_t.to("cuda:0"),
                "observation.images.empty_camera_0": empty.to("cuda:0"),
                "observation.state": state_t.to("cuda:0"),
                "task": [element["prompt"]],
            }
            batch = _local_pre(batch)
            with _t.no_grad():
                ac = _local_policy.predict_action_chunk(batch)
            ac = _local_post(ac).squeeze(0).cpu().numpy()
            return {"actions": ac}
    client = _LocalClient()
    # Original GroundingDINO loading (kept as-is)
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2
    CONFIG_PATH = "/root/autodl-tmp/vlsa-aegis/GroundingDINO/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT_PATH = "/root/autodl-tmp/vlsa-aegis/GroundingDINO/groundingdino_swint_ogc.pth"
    DEVICE = "cuda"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    model_groundingdino = load_model(CONFIG_PATH, CHECKPOINT_PATH)'''
assert old1 in s, "patch1 anchor not found"
s = s.replace(old1, new1)

# patch 2: stub eef_marker (not in SafeLIBERO)
s = s.replace(
    'eef_body_id = model.body_name2id("eef_marker")',
    'eef_body_id = -1  # PATCHED: eef_marker not in SafeLIBERO',
)
# Comment out body_pos / body_quat updates (would crash with eef_body_id = -1)
import re as _re
s = _re.sub(
    r'^(\s+)env\.sim\.model\.body_pos\[eef_body_id\]',
    r'\1pass # env.sim.model.body_pos[eef_body_id]',
    s, flags=_re.MULTILINE,
)
s = _re.sub(
    r'^(\s+)env\.sim\.model\.body_quat\[eef_body_id\]',
    r'\1pass # env.sim.model.body_quat[eef_body_id]',
    s, flags=_re.MULTILINE,
)

open(P, "w").write(s)
print("Patches applied (PI05 backend + eef_marker stub only). Perception KEPT original.")
