"""Conformal calibration of Safe-MoLe critic on LIBERO held-out demonstration data.

Reads trained critic_head.pt, runs OpenVLA + critic on held-out demos, collects
nonconformity scores s_i = ĥ_φ(f_i) - h(x_i), computes quantile σ̂_h at
miscoverage δ, writes to calibration_result.json.

Usage:
    python -m safe_mole.calibrate_libero \\
        --openvla_ckpt /root/autodl-tmp/openvla_libero_spatial \\
        --critic_ckpt /root/autodl-tmp/safe_mole_libero/critic_v1/critic_head.pt \\
        --demo_dir /root/autodl-tmp/libero_datasets/rlds/libero_spatial_no_noops \\
        --out /root/autodl-tmp/safe_mole_libero/critic_v1/calibration \\
        --n_samples 2000 --delta 0.01
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image

from safe_mole import (
    SafetyCriticHead, SafeMoLeConfig,
    split_conformal_quantile, empirical_coverage,
    TableHeightBarrier,
)
from safe_mole.train_critic_libero import CognitionHook, iter_rlds_shards


@torch.no_grad()
def main(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.openvla_ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.openvla_ckpt, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()

    # Load critic
    state = torch.load(args.critic_ckpt, map_location="cuda:0")
    cfg = SafeMoLeConfig(**state["cfg"])
    critic = SafetyCriticHead(
        feature_dim=cfg.feature_dim, hidden_dim=cfg.critic_hidden,
        use_spectral_norm=True,
    ).to("cuda:0").float().eval()
    critic.load_state_dict(state["critic_state_dict"])
    z_min = state.get("z_min", args.z_min)
    barrier = TableHeightBarrier(z_index=2, z_min=z_min)

    print(f"[cal] critic loaded from {args.critic_ckpt}", flush=True)
    print(f"[cal] collecting {args.n_samples} samples ...", flush=True)

    scores, h_true_all, h_pred_all = [], [], []
    iterator = iter_rlds_shards(args.demo_dir)
    cal_buffer = []
    pad_id = processor.tokenizer.pad_token_id or 0

    while len(scores) < args.n_samples:
        while len(cal_buffer) < args.batch:
            try:
                cal_buffer.append(next(iterator))
            except StopIteration:
                break
        if not cal_buffer:
            break
        batch = cal_buffer[:args.batch]
        cal_buffer = cal_buffer[args.batch:]

        images = [Image.fromarray(ex["image"]) for ex in batch]
        prompts = [f"In: What action should the robot take to {ex['instruction'].lower()}?\nOut:" for ex in batch]
        inputs_list = [processor(p, im) for p, im in zip(prompts, images)]
        max_len = max(x["input_ids"].shape[-1] for x in inputs_list)
        input_ids = torch.stack([
            torch.nn.functional.pad(x["input_ids"][0],
                                     (max_len - x["input_ids"].shape[-1], 0),
                                     value=pad_id) for x in inputs_list
        ]).to("cuda:0")
        attn = (input_ids != pad_id).long()
        pixel_values = torch.stack([x["pixel_values"][0] for x in inputs_list]).to(
            "cuda:0", dtype=torch.bfloat16)
        eef = torch.tensor(np.stack([ex["eef_pos"] for ex in batch]), dtype=torch.float32,
                           device="cuda:0")
        h_true = (eef[:, 2] - z_min).cpu().numpy()

        with CognitionHook(model) as hook:
            _ = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values)
            cog = hook.pooled_feature(attn)
        h_pred = critic(cog).cpu().numpy()

        # Nonconformity score s_i = ĥ_φ - h (pessimistic direction — v3 fix)
        s = h_pred - h_true
        scores.extend(s.tolist())
        h_true_all.extend(h_true.tolist())
        h_pred_all.extend(h_pred.tolist())
        if len(scores) % 200 == 0:
            print(f"  collected {len(scores)}, running mean |h_pred-h_true| = "
                  f"{np.mean(np.abs(np.array(h_pred_all) - np.array(h_true_all))):.4f}",
                  flush=True)

    scores = np.array(scores[:args.n_samples])
    h_true_all = np.array(h_true_all[:args.n_samples])
    h_pred_all = np.array(h_pred_all[:args.n_samples])

    sigma_h_bar = split_conformal_quantile(scores, args.delta)
    # Empirical coverage on same split for sanity
    cov = empirical_coverage(h_true_all, h_pred_all, sigma_h_bar)

    result = {
        "n_samples": len(scores),
        "delta": args.delta,
        "sigma_h_bar": float(sigma_h_bar),
        "empirical_coverage": float(cov),
        "target_coverage": 1 - args.delta,
        "score_stats": {
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "q01": float(np.quantile(scores, 0.01)),
            "q05": float(np.quantile(scores, 0.05)),
            "q50": float(np.quantile(scores, 0.50)),
            "q95": float(np.quantile(scores, 0.95)),
            "q99": float(np.quantile(scores, 0.99)),
        },
        "h_true_stats": {
            "mean": float(np.mean(h_true_all)), "std": float(np.std(h_true_all)),
            "min": float(np.min(h_true_all)), "max": float(np.max(h_true_all)),
        },
    }
    with open(out_dir / "calibration_result.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(out_dir / "raw_scores.npz",
             scores=scores, h_true=h_true_all, h_pred=h_pred_all)
    print(f"\n[cal] σ̂_h = {sigma_h_bar:.4f}", flush=True)
    print(f"[cal] empirical coverage = {cov:.3f} (target ≥ {1-args.delta:.3f})",
          flush=True)
    print(f"[cal] written to {out_dir}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--openvla_ckpt", required=True)
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--demo_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--z_min", type=float, default=0.85)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
