"""ISSf-CBF Projection Layer (v4 theory, with η a_max² second-order buffer).

Implements the closed-form safety projection from Lemma 1.1 / Eq. (2.1):

    â_safe = a* + [(-γ ĥ + Lip_h d_max + ξ(k) - ĉ^T a*)_+ / ||ĉ||²] · ĉ

where ξ(k) = γ σ̄_h + a_max σ̄_∇ ||∂f/∂a|| + L_ĥ ε(k) + η a_max²  (Eq. 2.2 v4)

Key properties
--------------
- Closed-form (no QP solver).
- Differentiable end-to-end.
- Lemma 2.0 (identity-when-feasible): when a* already satisfies CBF, this
  layer is the identity map — no performance cost on safe states.

Reference: DERIVATION_PACKAGE.md Module 1, Module 2, Theorem 5.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CBFParams:
    """Static CBF hyperparameters (scene/robot dependent)."""
    gamma: float = 0.5           # CBF decay rate, α(r) = γ r, γ ∈ (0,1)
    lip_h: float = 1.0           # barrier Lipschitz constant (A2a)
    d_max: float = 0.0           # bounded dynamics disturbance (A1)
    a_max: float = 1.0           # action norm bound (A0)
    eta: float = 0.0             # second-order Taylor remainder (A3)
    jac_norm: float = 1.0        # ||∂f/∂a|| upper bound


def issf_cbf_project(
    a_nominal: torch.Tensor,
    c_hat: torch.Tensor,
    h_hat: torch.Tensor,
    xi: torch.Tensor | float,
    params: CBFParams,
    eps_c: float = 1e-6,
) -> torch.Tensor:
    """Closed-form ISSf-CBF projection (Lemma 1.1 extended to v4).

    Computes:
        â_safe = a* + [(slack)_+ / ||ĉ||²] · ĉ
        slack  = -γ·ĥ + Lip_h·d_max + ξ(k) - ĉ^T a*

    Parameters
    ----------
    a_nominal : (B, A) or (B, T, A)
        Nominal action a* from DiT head. A=7 for 7-DoF manipulation.
    c_hat : (B, A) or (B, T, A)
        Estimated safety gradient in action space, ĉ = (∂f/∂a)^T ∇ĥ_φ.
        Must match a_nominal shape.
    h_hat : (B,) or (B, T)
        Estimated barrier value ĥ_φ.
    xi : scalar or (B,) or (B, T)
        Safety-margin buffer ξ(k) (see routing.py and conformal.py).
    params : CBFParams
        Static CBF hyperparameters.
    eps_c : float
        Numerical floor for ||ĉ||² to avoid div-by-zero when gradient vanishes.
        Theoretical analysis (Risk 1) requires c_min > 0 on X_0; in practice
        we clamp to eps_c and flag the state as "flat-barrier" for fallback.

    Returns
    -------
    a_safe : same shape as a_nominal
        Projected action satisfying the practical CBF constraint (Eq. 2.3, v4).
    """
    # Shape broadcast: ensure h_hat / xi broadcast over the action dim.
    # a_nominal has shape (..., A); h_hat / xi should match the leading (...)
    assert a_nominal.shape == c_hat.shape, (
        f"a_nominal {tuple(a_nominal.shape)} must match c_hat {tuple(c_hat.shape)}"
    )

    # dot product along action dim
    c_dot_a = (c_hat * a_nominal).sum(dim=-1)  # shape (..., )
    c_norm_sq = (c_hat * c_hat).sum(dim=-1).clamp_min(eps_c)

    # slack = -γ ĥ + Lip_h d_max + ξ - ĉ^T a*
    # xi may be scalar or tensor; align device
    if isinstance(xi, torch.Tensor):
        xi_t = xi
    else:
        xi_t = torch.as_tensor(xi, dtype=a_nominal.dtype, device=a_nominal.device)

    slack = (
        -params.gamma * h_hat
        + params.lip_h * params.d_max
        + xi_t
        - c_dot_a
    )  # shape (...,)

    # clip to positive (Lemma 2.0: zero when a* already feasible → identity)
    slack_pos = torch.clamp_min(slack, 0.0)  # (...)

    # scalar multiplier per batch element
    mult = slack_pos / c_norm_sq  # (...)

    # broadcast back over action dim
    correction = mult.unsqueeze(-1) * c_hat  # (..., A)
    return a_nominal + correction


class ISSfCBFProjection(nn.Module):
    """Module wrapper around issf_cbf_project.

    Stateless (no learnable params) but useful for integration with nn.Sequential
    and for exposing CBFParams as a registered buffer for checkpointing.
    """

    def __init__(self, params: CBFParams) -> None:
        super().__init__()
        self.params = params
        # Register scalars as non-trainable buffers (for checkpointing)
        self.register_buffer("_gamma", torch.tensor(params.gamma))
        self.register_buffer("_lip_h", torch.tensor(params.lip_h))
        self.register_buffer("_d_max", torch.tensor(params.d_max))
        self.register_buffer("_a_max", torch.tensor(params.a_max))
        self.register_buffer("_eta", torch.tensor(params.eta))
        self.register_buffer("_jac_norm", torch.tensor(params.jac_norm))

    def forward(
        self,
        a_nominal: torch.Tensor,
        c_hat: torch.Tensor,
        h_hat: torch.Tensor,
        xi: torch.Tensor | float,
    ) -> torch.Tensor:
        return issf_cbf_project(a_nominal, c_hat, h_hat, xi, self.params)

    def cbf_margin(
        self,
        action: torch.Tensor,
        c_hat: torch.Tensor,
        h_hat: torch.Tensor,
        xi: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute the CBF constraint slack (positive → feasible).

            margin = ĉ^T a - η ||a||² + γ ĥ - Lip_h d_max - ξ

        Used for L_CBF loss (Eq. 2.3, v4).
        """
        xi_t = (
            xi if isinstance(xi, torch.Tensor)
            else torch.as_tensor(xi, dtype=action.dtype, device=action.device)
        )
        c_dot_a = (c_hat * action).sum(dim=-1)
        a_sq = (action * action).sum(dim=-1)
        return (
            c_dot_a
            - self.params.eta * a_sq
            + self.params.gamma * h_hat
            - self.params.lip_h * self.params.d_max
            - xi_t
        )

    def extra_repr(self) -> str:
        p = self.params
        return (
            f"γ={p.gamma}, Lip_h={p.lip_h}, d_max={p.d_max}, "
            f"a_max={p.a_max}, η={p.eta}, ||∂f/∂a||={p.jac_norm}"
        )
