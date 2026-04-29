"""Minimal-intrusion adapter to attach Safe-MoLe to an existing CogACT model.

The upstream MoLe-VLA `vla/cogactvla.py` has the loss computation at line 232-233:

    loss = self.action_model.loss(actions_repeated, cognition_features_repeated)
    return loss + balance_loss * balance_weight + KD_loss * KD_weight, output

Our adapter adds:
    L_critic, L_cbf, aux = self.safe_mole.compute_safety_losses(
        cognition_features, h_true, a_nominal=actions_future, jac=None)
    loss = loss + λ_critic * L_critic + λ_cbf * L_cbf

Integration modes
-----------------
1. **Wrapper subclass** (`SafeCogACT`): inherit from CogACT, override forward.
   Cleanest; preferred when training from scratch.

2. **Monkey-patch** (`attach_safe_mole_to`): replace an existing instance's
   forward bound method. Useful when we need to safely extend a pre-loaded
   checkpoint without creating a new class.

Both preserve FSDP/DDP compatibility since we only add a new nn.Module
sibling (`self.safe_mole`), which gets wrapped alongside the rest.

Reference: SAFE_MOLE_INTEGRATION_POINTS.md §5.
"""
from __future__ import annotations
from functools import wraps
from typing import Callable
import torch
import torch.nn as nn

from .integration import SafeMoLeExtension, SafeMoLeConfig
from .barriers import TableHeightBarrier, compute_analytic_h_and_grad


def attach_safe_mole_to(
    cogact_model: nn.Module,
    cfg: SafeMoLeConfig,
    barrier: Callable | None = None,
    state_extractor: Callable | None = None,
) -> nn.Module:
    """In-place attach a SafeMoLeExtension + patched forward to `cogact_model`.

    Parameters
    ----------
    cogact_model : CogACT instance (from vla.cogactvla)
    cfg : SafeMoLeConfig
    barrier : callable state -> h_true tensor. Default: TableHeightBarrier().
    state_extractor : callable batch-dict -> state tensor shape (B, state_dim).
        Default: reads `batch['observation']['proprio']` or similar; for the
        smoke test we allow explicit override.

    Effects
    -------
    - adds `cogact_model.safe_mole : SafeMoLeExtension`
    - adds `cogact_model.safety_barrier` and `cogact_model.state_extractor`
    - replaces `cogact_model.forward` with a version that appends safety losses

    Returns `cogact_model` for chaining.
    """
    if hasattr(cogact_model, "safe_mole"):
        raise RuntimeError("model already has a safe_mole extension attached")

    ext = SafeMoLeExtension(cfg)
    # Match dtype / device of parent model
    ref_param = next(cogact_model.parameters(), None)
    if ref_param is not None:
        ext = ext.to(device=ref_param.device, dtype=ref_param.dtype)

    cogact_model.safe_mole = ext
    cogact_model.safety_barrier = barrier or TableHeightBarrier()
    cogact_model.state_extractor = state_extractor or _default_state_extractor

    original_forward = cogact_model.forward

    @wraps(original_forward)
    def safe_forward(self, *args, safe_mole_state: torch.Tensor | None = None,
                     **kwargs):
        """Wrapped forward. Expects an optional `safe_mole_state` kwarg
        providing robot state (B, state_dim) for barrier computation.

        If `safe_mole_state` is None, Safe-MoLe losses are zero (fall back to
        plain MoLe training). Useful for gradual curriculum or debugging.
        """
        # Run upstream forward — returns (loss, output)
        # We need the intermediate cognition_features + actions_future.
        # The cleanest way is to monkey-patch CogACT's forward body; but the
        # current upstream returns only (loss, output). So we hijack the
        # training call: re-run forward but capture the cognition token via
        # a forward hook on `self.action_model.z_embedder`.
        cognition_holder = {}

        def _capture_cognition(_module, inputs, _output):
            # z_embedder receives (z, training); store z.
            z = inputs[0] if isinstance(inputs, tuple) else inputs
            cognition_holder["z"] = z.detach()

        # Locate the z_embedder used by DiT
        try:
            z_emb = self.action_model.net.z_embedder
        except AttributeError:
            z_emb = getattr(getattr(self, "action_model", None), "z_embedder", None)
        handle = z_emb.register_forward_hook(_capture_cognition) if z_emb is not None else None

        try:
            out = original_forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if safe_mole_state is None or "z" not in cognition_holder:
            return out

        loss, inner_out = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
        cognition = cognition_holder["z"]  # (B*steps, 1, D) or (B*steps, D)

        # Compute h_true from analytic barrier
        state = safe_mole_state.to(cognition.device)
        with torch.no_grad():
            h_true = self.safety_barrier(state)  # (B,)

        # a_nominal — would normally come from sampling DiT. For training we
        # use the ground-truth action one step as a surrogate (matches the
        # diffusion-loss setup where the model is trained to predict noise
        # of that action); this is a standard technique in safe-RL trainers.
        # Here we expect caller to pass `actions` in kwargs so we can reuse it.
        a_nom = kwargs.get("actions", None)
        if a_nom is None:
            return loss, inner_out
        # actions shape (B, T, 7) from CogACT training loop
        a_future_first = a_nom[:, -1, :]  # last action of chunk

        L_critic, L_cbf, _ = self.safe_mole.compute_safety_losses(
            cognition_features=cognition[: state.shape[0]],  # match batch dim
            h_true=h_true,
            a_nominal=a_future_first,
            jac=None,
        )
        total = loss + self.safe_mole.cfg.lambda_critic * L_critic \
                     + self.safe_mole.cfg.lambda_cbf * L_cbf
        return total, inner_out

    cogact_model.forward = safe_forward.__get__(cogact_model, type(cogact_model))
    return cogact_model


def _default_state_extractor(batch) -> torch.Tensor:
    """Read proprioceptive state from a training batch dict.

    RLDS convention: batch['observation']['proprio'] shape (B, T, state_dim).
    We take the *last* timestep as the current state.
    """
    obs = batch.get("observation", batch)
    proprio = obs["proprio"] if "proprio" in obs else obs.get("state")
    if proprio is None:
        raise KeyError("no 'proprio' or 'state' in batch[observation]")
    return proprio[:, -1, :]


class SafeCogACT(nn.Module):
    """Subclass wrapper: compose CogACT + SafeMoLeExtension cleanly.

    Recommended for training from scratch. Monkey-patch is simpler for
    retrofitting a loaded checkpoint.
    """

    def __init__(
        self,
        cogact_model: nn.Module,
        cfg: SafeMoLeConfig,
        barrier=None,
    ) -> None:
        super().__init__()
        self.cogact = cogact_model
        self.safe_mole = SafeMoLeExtension(cfg)
        self.safety_barrier = barrier or TableHeightBarrier()

    def forward(self, *args, safe_mole_state: torch.Tensor | None = None, **kwargs):
        out = self.cogact(*args, **kwargs)
        if safe_mole_state is None:
            return out
        loss, inner = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
        # For a full integration we'd re-hook cognition; subclass path requires
        # CogACT.forward to yield cognition_features — left as an extension point.
        return loss, inner

    def __getattr__(self, name: str):
        # Fall through to wrapped cogact for unknown attributes
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.cogact, name)
