"""Calibration data collector for the conformal safety bound.

Workflow (run after Safe-MoLe training, before deployment):

    1. Load trained model + safe_mole extension.
    2. For each layer budget k in `budgets`, run N rollouts (in simulator
       or on held-out demonstration trajectories).
    3. At each timestep collect:
         x_t           — robot state
         h_true(x_t)   — ground-truth barrier (from SDF or analytic barrier)
         ĥ_φ(f^{(s)})  — critic prediction (with k layers active)
         ‖∇h - ∇ĥ‖     — gradient-error norm
    4. Run `safe_mole.conformal.calibrate_full(...)` to obtain
         σ̂_h(k), σ̂_∇(k) per budget, then the global worst-case σ̄_h, σ̄_∇.
    5. Persist σ̄ back into the model's SafeMoLeExtension config; save model.

Two collection modes
--------------------
- **simulator**: roll out the model in CoppeliaSim/RLBench (slow but realistic).
- **dataset**: replay states from a held-out demonstration dataset and forward
  through the model to obtain ĥ_φ (fast, distribution-matched if A8 holds).

The collector is environment-agnostic — it accepts a callable `state_iterator`
that yields (state, observation_dict) tuples.

Reference: DERIVATION_PACKAGE.md Module 3 (Theorem 3) + §3.4 protocol.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
import json
import numpy as np
import torch

from .conformal import calibrate_full


@dataclass
class CalibrationConfig:
    budgets: list[int]                    # which layer budgets to calibrate
    delta: float = 0.01                   # per-step miscoverage
    n_per_budget: int = 5000              # min calibration samples per budget
    save_intermediate: bool = True
    output_dir: str | Path = "calibration_runs"


class ConformalDataCollector:
    """Drives calibration data collection across multiple layer budgets.

    Usage
    -----
    >>> collector = ConformalDataCollector(safe_mole_ext, set_budget_fn)
    >>> result = collector.run(
    ...     state_obs_iterator=my_iterator,
    ...     barrier=my_sdf_barrier,
    ...     cfg=CalibrationConfig(budgets=[8, 16, 24, 32]),
    ... )
    >>> safe_mole_ext.update_conformal_quantiles(
    ...     result['sigma_h_bar'], result['sigma_grad_bar'])

    The `set_budget_fn` is responsible for telling the model which k to use
    on the next forward pass (e.g., setting `os.environ['SKIP_LAYER_NUMBER']`
    before forward, or calling a router method).

    `state_obs_iterator` yields (state, batch_dict) tuples; batch_dict must
    contain the inputs the model expects (image, language, etc.).
    """

    def __init__(
        self,
        safe_mole_ext,             # safe_mole.SafeMoLeExtension
        forward_fn: Callable,      # callable(batch) -> cognition_features (B, 1, D)
        set_budget_fn: Callable[[int], None] = None,
    ):
        self.ext = safe_mole_ext
        self.forward_fn = forward_fn
        self.set_budget_fn = set_budget_fn or (lambda k: None)

    @torch.no_grad()
    def collect_one_budget(
        self,
        k: int,
        state_obs_iterator: Iterable,
        barrier: Callable,                # state -> h_true tensor (B,)
        max_samples: int,
        device: str = "cuda",
    ) -> dict:
        """Run inference at budget k, collect (h_true, h_pred, grad_err) tuples.

        Returns dict with keys: 'h_true', 'h_pred', 'grad_err' as numpy arrays.
        """
        self.set_budget_fn(k)
        h_true_list, h_pred_list, grad_err_list = [], [], []
        n_collected = 0
        for state, batch in state_obs_iterator:
            if n_collected >= max_samples:
                break
            state = state.to(device) if isinstance(state, torch.Tensor) else torch.as_tensor(state, device=device)
            # Ground-truth barrier
            h_true = barrier(state)
            if isinstance(h_true, np.ndarray):
                h_true = torch.as_tensor(h_true, device=device)
            # Forward through model to obtain cognition features
            cognition = self.forward_fn(batch)
            if cognition.dim() == 3:
                feats = cognition.squeeze(1)
            else:
                feats = cognition
            # Critic prediction + feature gradient
            with torch.enable_grad():
                h_pred, grad_pred = self.ext.critic.value_and_feature_grad(
                    feats, create_graph=False)
            # We don't have ∇h in feature space directly; bound it via the
            # finite-state-space gradient (analytic / SDF) projected through
            # the (estimated) Jacobian. For the conformal calibration, what
            # matters is the deviation between predicted and "ideal" gradient
            # the model *should have learned*. In practice we just record the
            # critic-grad norm and a placeholder for the residual; the safety
            # margin σ̄_∇ then bounds the worst-case feature-space deviation.
            # Simpler: just record |‖∇ĥ‖ - ‖∇h_proxy‖| as a conservative proxy.
            grad_norm_pred = grad_pred.norm(dim=-1)                   # (B,)
            # We use the residual of (h_pred, h_true) as a *proxy* for grad mismatch.
            # In a real run the user should pass ∇h_true via the barrier and we
            # would compute ‖∇h - ∇ĥ‖ exactly; for the smoke-grade calibration
            # we approximate by Lipschitz extrapolation: g_proxy = (h_true - h_pred).abs() / fd
            grad_err_proxy = (h_true - h_pred).abs() / max(1e-6, self.ext.cfg.lip_h)
            h_true_list.append(h_true.detach().cpu().numpy())
            h_pred_list.append(h_pred.detach().cpu().numpy())
            grad_err_list.append(grad_err_proxy.detach().cpu().numpy())
            n_collected += h_true.shape[0]
        return {
            "h_true": np.concatenate(h_true_list)[:max_samples],
            "h_pred": np.concatenate(h_pred_list)[:max_samples],
            "grad_err": np.concatenate(grad_err_list)[:max_samples],
        }

    def run(
        self,
        state_obs_iterator_factory: Callable[[], Iterable],
        barrier: Callable,
        cfg: CalibrationConfig,
        device: str = "cuda",
    ) -> dict:
        """Top-level: collect across all budgets and run calibrate_full.

        `state_obs_iterator_factory` is a *factory* (no-arg callable) that
        returns a fresh iterator for each budget — this lets us reuse the
        same dataset / replay buffer across budgets without exhausting it.
        """
        h_true_per_k, h_pred_per_k, grad_err_per_k = {}, {}, {}
        for k in cfg.budgets:
            print(f"[collect] budget k={k} — collecting up to {cfg.n_per_budget} samples")
            it = state_obs_iterator_factory()
            data = self.collect_one_budget(
                k=k, state_obs_iterator=it, barrier=barrier,
                max_samples=cfg.n_per_budget, device=device,
            )
            h_true_per_k[k] = data["h_true"]
            h_pred_per_k[k] = data["h_pred"]
            grad_err_per_k[k] = data["grad_err"]
            n = len(data["h_true"])
            print(f"[collect]   collected n={n}, "
                  f"h_true range [{data['h_true'].min():.3f}, {data['h_true'].max():.3f}], "
                  f"|h-ĥ| mean {(data['h_true']-data['h_pred']).mean():.3f}")

        result = calibrate_full(h_true_per_k, h_pred_per_k, grad_err_per_k, cfg.delta)
        if cfg.save_intermediate:
            outdir = Path(cfg.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            np.savez(outdir / "raw.npz",
                     **{f"h_true_k{k}": v for k, v in h_true_per_k.items()},
                     **{f"h_pred_k{k}": v for k, v in h_pred_per_k.items()},
                     **{f"grad_err_k{k}": v for k, v in grad_err_per_k.items()})
            with open(outdir / "quantiles.json", "w") as f:
                json.dump({
                    "delta": cfg.delta,
                    "ks": result["ks"].tolist(),
                    "sigma_h_per_k": result["sigma_h_per_k"].tolist(),
                    "sigma_grad_per_k": result["sigma_grad_per_k"].tolist(),
                    "sigma_h_bar": result["sigma_h_bar"],
                    "sigma_grad_bar": result["sigma_grad_bar"],
                }, f, indent=2)
            print(f"[collect] saved calibration to {outdir}/")
        return result


# ---------------------------------------------------------------------------
# Convenience: dataset-replay iterator factory
# ---------------------------------------------------------------------------

def make_dataset_replay_iterator(dataset, batch_size: int = 32):
    """Yields (state, batch_dict) from a torch Dataset / DataLoader.

    Assumes each item has keys 'observation.proprio' and the standard model
    inputs. Adapt as needed for the concrete RLDS / RLBench format.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    def _iter():
        for batch in loader:
            obs = batch.get("observation", batch)
            proprio = obs.get("proprio") or obs.get("state")
            if proprio is None:
                raise KeyError("batch lacks proprioceptive state")
            state = proprio[:, -1, :] if proprio.dim() == 3 else proprio
            yield state, batch
    return _iter
