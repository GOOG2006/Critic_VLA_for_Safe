"""Train Safe-MoLe Safety Critic on LIBERO demonstration data.

Uses frozen OpenVLA + LIBERO spatial demos. For each step:
  1. Extract OpenVLA cognition feature (LLM final hidden state, mean-pooled
     over non-pad tokens).
  2. Ground-truth h(x) from TableHeightBarrier on eef_pos.
  3. Regress critic head on MSE + hinge CBF loss.

Held-out split used for conformal calibration (σ̄_h, σ̄_∇).

Usage:
  python -m safe_mole.train_critic_libero \\
      --openvla_ckpt /root/autodl-tmp/openvla_libero_spatial \\
      --demo_dir /root/autodl-tmp/libero_datasets/rlds/libero_spatial_no_noops \\
      --out /root/autodl-tmp/safe_mole_libero/critic_v1 \\
      --steps 1000 --batch 4 --z_min 0.85
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# OpenVLA cognition extraction
# ---------------------------------------------------------------------------

class CognitionHook:
    """Context manager that hooks OpenVLA's LLM to capture mean-pooled hidden states."""

    def __init__(self, model):
        self.model = model
        self.buffer = None
        self.handles = []

    def __enter__(self):
        # Try to locate the LLM backbone; differs between Prismatic / HF form
        llm = self._find_llm_module()
        if llm is None:
            raise RuntimeError("Couldn't find LLM submodule inside OpenVLA model")

        def capture(module, inputs, output):
            # Output is a tuple; first element is last_hidden_state (B, L, D)
            if isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, 'last_hidden_state'):
                h = output.last_hidden_state
            else:
                h = output
            self.buffer = h.detach()

        self.handles.append(llm.register_forward_hook(capture))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _find_llm_module(self):
        # Heuristic search
        for name, mod in self.model.named_modules():
            if name.endswith(("llm_backbone.llm", "language_model", "model.model")):
                if hasattr(mod, "embed_tokens"):
                    return mod
        # Fallback: any submodule with "LlamaModel" in class name
        for mod in self.model.modules():
            if "LlamaModel" in mod.__class__.__name__:
                return mod
        return None

    def pooled_feature(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden state over non-pad tokens. Returns (B, D)."""
        if self.buffer is None:
            raise RuntimeError("No hidden state captured yet")
        h = self.buffer                          # (B, L, D)
        # OpenVLA inserts 256 image patches into LLM; generate() may capture either
        # prefill (L = text_len + 256) or a decode step (L = 1). Use attention-aware
        # mean-pool when shapes align; otherwise plain mean.
        B, L, D = h.shape
        if L == attention_mask.shape[1]:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
        elif L > attention_mask.shape[1]:
            extra = L - attention_mask.shape[1]
            pad_ones = torch.ones((B, extra), dtype=attention_mask.dtype, device=attention_mask.device)
            mask = torch.cat([attention_mask, pad_ones], dim=1).unsqueeze(-1).to(h.dtype)
        else:
            # L < attention_mask: decode-step capture — fall back to plain mean over L
            return h.mean(dim=1).float()
        summed = (h * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1)
        return (summed / count).float()


# ---------------------------------------------------------------------------
# Demo data loading
# ---------------------------------------------------------------------------

def iter_rlds_shards(demo_dir: str):
    """Yield (image, instruction, eef_pos, action) tuples from RLDS TFRecords."""
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    ds = tfds.load("libero_spatial_no_noops",
                   data_dir=os.path.dirname(demo_dir), split="train")
    for ep in ds:
        for step in ep["steps"]:
            obs = step["observation"]
            yield {
                "image": obs["image"].numpy(),
                "instruction": step["language_instruction"].numpy().decode(),
                "eef_pos": obs["state"].numpy()[:3],  # first 3 dims = xyz
                "action": step["action"].numpy(),
            }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load OpenVLA (frozen)
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"[critic] loading OpenVLA from {args.openvla_ckpt}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.openvla_ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.openvla_ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    for p in model.parameters():
        p.requires_grad = False
    torch.cuda.synchronize()
    print(f"[critic] OpenVLA loaded in {time.time()-t0:.1f}s, "
          f"GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    # 2. Build Safety Critic + barrier
    from safe_mole import SafeMoLeConfig, SafetyCriticHead, TableHeightBarrier
    from safe_mole.projection import issf_cbf_project, CBFParams
    cfg = SafeMoLeConfig(
        feature_dim=4096, critic_hidden=args.critic_hidden,
        gamma=0.5, lip_h=1.0, d_max=0.005, a_max=0.1, eta=1.0,
        lambda_critic=1.0, lambda_cbf=0.5,
    )
    critic = SafetyCriticHead(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.critic_hidden,
        use_spectral_norm=True,
    ).to("cuda:0").float().train()
    barrier = TableHeightBarrier(z_index=2, z_min=args.z_min)

    optim = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"[critic] {sum(p.numel() for p in critic.parameters())/1e6:.2f} M params", flush=True)

    # 3. Data iterator
    iterator = iter_rlds_shards(args.demo_dir)
    buffer = []
    for step in range(args.steps):
        # Collect a batch
        while len(buffer) < args.batch:
            try:
                buffer.append(next(iterator))
            except StopIteration:
                iterator = iter_rlds_shards(args.demo_dir)
        batch = buffer[:args.batch]
        buffer = buffer[args.batch:]

        # Prepare OpenVLA inputs
        images = [Image.fromarray(ex["image"]) for ex in batch]
        prompts = [f"In: What action should the robot take to {ex['instruction'].lower()}?\nOut:" for ex in batch]
        inputs_list = [processor(p, im) for p, im in zip(prompts, images)]
        # Naive batching: pad input_ids to longest
        max_len = max(x["input_ids"].shape[-1] for x in inputs_list)
        pad_id = processor.tokenizer.pad_token_id or 0
        input_ids = torch.stack([
            torch.nn.functional.pad(x["input_ids"][0], (max_len - x["input_ids"].shape[-1], 0),
                                     value=pad_id) for x in inputs_list
        ]).to("cuda:0")
        attn_mask = (input_ids != pad_id).long()
        pixel_values = torch.stack([x["pixel_values"][0] for x in inputs_list]).to(
            "cuda:0", dtype=torch.bfloat16)

        # Ground-truth h
        eef = torch.tensor(np.stack([ex["eef_pos"] for ex in batch]), dtype=torch.float32,
                           device="cuda:0")
        h_true = eef[:, 2] - args.z_min  # (B,)

        # Extract cognition via forward hook
        with CognitionHook(model) as hook:
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    pixel_values=pixel_values,
                )
            cog = hook.pooled_feature(attn_mask)          # (B, 4096)

        # Critic forward + backward
        h_pred = critic(cog)
        L_critic = torch.nn.functional.mse_loss(h_pred, h_true)

        # No CBF loss at this stage (needs action + jac; keep pure regression)
        optim.zero_grad()
        L_critic.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optim.step()

        if step % 10 == 0 or step == args.steps - 1:
            with torch.no_grad():
                err = (h_pred - h_true).abs().mean().item()
            print(f"  step {step:5d} L_critic={L_critic.item():.5f} "
                  f"|h_pred - h_true|={err:.4f}", flush=True)

    # 4. Save
    ckpt = out_dir / "critic_head.pt"
    torch.save({
        "critic_state_dict": critic.state_dict(),
        "cfg": cfg.__dict__,
        "z_min": args.z_min,
        "n_steps": args.steps,
    }, ckpt)
    print(f"[critic] saved to {ckpt}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--openvla_ckpt", required=True)
    p.add_argument("--demo_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--z_min", type=float, default=0.85)
    p.add_argument("--critic_hidden", type=int, default=256)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
