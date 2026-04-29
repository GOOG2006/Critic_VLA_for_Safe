"""Safety-Aware STAR Routing — k*(ρ̂_t) budget selector (v4 theory).

Given current safety margin ρ̂_t = ĥ_φ(f) and conformal quantiles σ̄_h, σ̄_∇,
this module computes the minimum layer budget k*(ρ_t) satisfying the practical
CBF feasibility condition (Eq. 2.3, v4):

    ξ(k) = γ σ̄_h + a_max σ̄_∇ ||∂f/∂a|| + L_ĥ ε(k) + η a_max²  ≤  ρ̂_t

The constant-part depends on global conformal quantiles and never on k (v3
convention). Only L_ĥ · ε(k) depends on k through A6 (nested skip).

Fallback: if feasibility set is empty, returns k=K (full network) and flags
a "retreat" signal for the execution layer.

Reference: DERIVATION_PACKAGE.md Module 4, Theorem 2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BudgetConfig:
    """Configuration for k*(ρ̂_t) selector."""
    K: int                                # total LLM layers (MoLe: 32)
    gamma: float = 0.5
    sigma_h_bar: float = 0.1              # σ̄_h (conformal)
    sigma_grad_bar: float = 0.1           # σ̄_∇ (conformal)
    jac_norm: float = 1.0                 # ||∂f/∂a||
    a_max: float = 1.0
    eta: float = 0.0
    lip_h_hat: float = 1.0                # L_ĥ (feature-space Lipschitz of critic, A4)
    k_min: int = 1                        # lower bound on active layers


class SafetyAwareBudget:
    """Selects k*(ρ̂) given pre-computed ε(k) profile from calibration."""

    def __init__(
        self,
        eps_profile: np.ndarray | torch.Tensor,
        cfg: BudgetConfig,
    ) -> None:
        """Parameters
        ----------
        eps_profile : array of shape (K+1,)
            ε(k) for k ∈ {0, 1, ..., K}. Must be non-increasing, ε(K)=0 (A6).
        cfg : BudgetConfig
        """
        if isinstance(eps_profile, torch.Tensor):
            eps_profile = eps_profile.detach().cpu().numpy()
        eps = np.asarray(eps_profile, dtype=np.float64)
        if eps.shape[0] != cfg.K + 1:
            raise ValueError(
                f"eps_profile length {eps.shape[0]} != cfg.K+1 ({cfg.K+1})"
            )
        # Validate A6: non-increasing + ε(K)=0
        if eps[cfg.K] > 1e-8:
            raise ValueError(
                f"A6 requires ε(K)=0, got ε({cfg.K})={eps[cfg.K]:.4g}"
            )
        diffs = np.diff(eps)
        if np.any(diffs > 1e-6):
            # allow tiny numerical slack but warn
            idx = int(np.argmax(diffs))
            raise ValueError(
                f"A6 requires monotone-nonincreasing ε(k); violation at k={idx} "
                f"(ε{idx}={eps[idx]:.4g}, ε{idx+1}={eps[idx+1]:.4g})"
            )

        self.eps = eps                    # shape (K+1,)
        self.cfg = cfg
        # Pre-compute ξ(k) table once (constants + L_ĥ ε(k) + η a_max²)
        self.xi_table = self._compute_xi_table()

    def _compute_xi_table(self) -> np.ndarray:
        c = self.cfg
        const = (
            c.gamma * c.sigma_h_bar
            + c.a_max * c.sigma_grad_bar * c.jac_norm
            + c.eta * c.a_max ** 2
        )
        return const + c.lip_h_hat * self.eps       # (K+1,)

    def xi(self, k: int | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return ξ(k)."""
        if isinstance(k, torch.Tensor):
            idx = k.long().clamp(0, self.cfg.K)
            xi_arr = torch.as_tensor(self.xi_table, device=k.device, dtype=torch.float32)
            return xi_arr[idx]
        k_arr = np.asarray(k)
        return self.xi_table[np.clip(k_arr, 0, self.cfg.K)]

    def k_star(self, rho_hat: float | np.ndarray | torch.Tensor) -> int | np.ndarray | torch.Tensor:
        """Minimum k such that ξ(k) ≤ ρ̂_t (definition 4.1).

        Fallback convention: if no such k exists, return K (full network).

        Accepts scalar, numpy array, or torch tensor; returns same type.
        """
        if isinstance(rho_hat, torch.Tensor):
            return self._k_star_tensor(rho_hat)

        rho_arr = np.atleast_1d(np.asarray(rho_hat, dtype=np.float64))
        # For each rho, find min k in {k_min,...,K} with xi[k] <= rho, else K
        xi = self.xi_table  # (K+1,)
        K = self.cfg.K
        k_min = self.cfg.k_min
        # xi is non-increasing in k, so searchsorted-style: find smallest k where xi[k] <= rho
        result = np.full(rho_arr.shape, K, dtype=np.int64)
        # xi[::-1] would be non-decreasing: use reversed search
        # but direct linear scan is fine for K=32
        for i, rho in enumerate(rho_arr):
            feasible_ks = np.where(xi[k_min:K+1] <= rho)[0]
            if feasible_ks.size == 0:
                result[i] = K  # fallback
            else:
                result[i] = int(feasible_ks[0] + k_min)
        return result[0] if np.isscalar(rho_hat) else result

    def _k_star_tensor(self, rho_hat: torch.Tensor) -> torch.Tensor:
        """Vectorized k* on GPU."""
        # xi: (K+1,) → broadcast over batch
        device = rho_hat.device
        xi = torch.as_tensor(self.xi_table, device=device, dtype=rho_hat.dtype)
        # feasible[b, k] = True if xi[k] <= rho_hat[b] (for k in [k_min, K])
        ks = torch.arange(self.cfg.k_min, self.cfg.K + 1, device=device)
        xi_ks = xi[ks]  # (K-k_min+1,)
        # rho_hat (B,) vs xi_ks (K-k_min+1,) → (B, K-k_min+1)
        feas = xi_ks.unsqueeze(0) <= rho_hat.unsqueeze(1)
        # smallest k feasible per row; fallback K if none
        # argmax on bool returns first True if any, else 0; use where to detect no-feasible
        any_feas = feas.any(dim=1)
        first_feas_idx = feas.float().argmax(dim=1)  # index into ks[]
        k_star = ks[first_feas_idx]                    # (B,)
        k_star = torch.where(any_feas, k_star, torch.full_like(k_star, self.cfg.K))
        return k_star

    def is_feasible(self, rho_hat: float | torch.Tensor) -> bool | torch.Tensor:
        """Whether the feasibility set is non-empty for this rho."""
        if isinstance(rho_hat, torch.Tensor):
            return (self.xi_table[-1] <= rho_hat.detach().cpu().item())  # xi(K) = min
        return float(self.xi_table[-1]) <= float(rho_hat)

    def __repr__(self) -> str:
        c = self.cfg
        return (
            f"SafetyAwareBudget(K={c.K}, γ={c.gamma}, σ̄_h={c.sigma_h_bar:.3g}, "
            f"σ̄_∇={c.sigma_grad_bar:.3g}, L_ĥ={c.lip_h_hat:.3g}, "
            f"ξ range=[{self.xi_table.min():.3g}, {self.xi_table.max():.3g}])"
        )
