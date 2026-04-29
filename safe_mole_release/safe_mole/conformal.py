"""Split-conformal calibration for safety critic (Module 3, v4 convention).

We calibrate:
  1. σ̂_h(k) per layer budget k, using pessimistic score s_i = ĥ_φ - h.
     Guarantee: Pr[h ≥ ĥ_φ - σ̂_h(k)] ≥ 1-δ per budget k (A5 direction).

  2. σ̂_∇(k) per layer budget, using g_i = ||∇h - ∇ĥ_φ||.
     Guarantee: Pr[||∇h - ∇ĥ_φ|| ≤ σ̂_∇(k)] ≥ 1-δ per budget k.

Then form **global worst-case** quantile (v3 convention):
    σ̄_h = max_k σ̂_h(k),  σ̄_∇ = max_k σ̂_∇(k)

This gives Theorem 2 monotonicity trivially (ξ(k) = const + L_ĥ·ε(k)) while
preserving per-k coverage by the superset argument.

Reference: DERIVATION_PACKAGE.md Module 3, A5.
"""
from __future__ import annotations
from math import ceil
import numpy as np


def split_conformal_quantile(
    scores: np.ndarray,
    delta: float,
) -> float:
    """Standard split-conformal upper quantile.

    Given n i.i.d. calibration scores s_1, ..., s_n and a fresh test score s,
    the quantile σ̂ = s_((⌈(1-δ)(n+1)⌉)) satisfies:
        Pr[s_test ≤ σ̂] ≥ 1 - δ

    under exchangeability of calibration and test distribution (A8).

    Parameters
    ----------
    scores : (n,) numpy array of nonconformity scores
    delta : miscoverage level, 0 < delta < 1

    Returns
    -------
    quantile : float
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n < 1:
        raise ValueError("calibration set must be non-empty")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    # rank = ⌈(1-δ)(n+1)⌉, clamped to [1, n] (1-indexed)
    rank = min(max(ceil((1 - delta) * (n + 1)), 1), n)
    return float(s[rank - 1])


def global_worst_case_quantile(
    per_k_quantiles: np.ndarray,
) -> float:
    """v3 convention: σ̄ = max_k σ̂(k).

    The resulting scalar satisfies per-k coverage (superset argument,
    Theorem 2 / R3 fix) and makes ξ(k) provably monotone.
    """
    return float(np.max(per_k_quantiles))


def calibrate_critic(
    h_true: np.ndarray,
    h_pred: np.ndarray,
    delta: float = 0.01,
) -> float:
    """Calibrate σ̂_h for a single layer budget.

    Uses the v3 pessimistic score direction s_i = ĥ_φ(x_i) - h(x_i).
    Guarantees Pr[h(x) ≥ ĥ_φ(x) - σ̂_h] ≥ 1 - δ (A5 direction).

    Parameters
    ----------
    h_true : (n,) ground-truth barrier values on calibration set
    h_pred : (n,) predicted barrier values on calibration set
    delta : per-step miscoverage level

    Returns
    -------
    sigma_h : float
    """
    scores = np.asarray(h_pred, dtype=np.float64) - np.asarray(h_true, dtype=np.float64)
    return split_conformal_quantile(scores, delta)


def calibrate_gradient(
    grad_err_norm: np.ndarray,
    delta: float = 0.01,
) -> float:
    """Calibrate σ̂_∇ given ||∇h - ∇ĥ_φ|| scores per calibration point."""
    return split_conformal_quantile(np.asarray(grad_err_norm, dtype=np.float64), delta)


def calibrate_full(
    h_true_per_k: dict[int, np.ndarray],
    h_pred_per_k: dict[int, np.ndarray],
    grad_err_per_k: dict[int, np.ndarray],
    delta: float = 0.01,
) -> dict[str, float | np.ndarray]:
    """Full calibration pipeline.

    For each routing budget k in {1, ..., K}, compute σ̂_h(k) and σ̂_∇(k),
    then take global worst-case (v3 convention).

    Parameters
    ----------
    h_true_per_k : dict k -> (n_k,) ground-truth h on calibration rollouts
        with routing budget k
    h_pred_per_k : dict k -> (n_k,) critic predictions
    grad_err_per_k : dict k -> (n_k,) gradient error norms
    delta : miscoverage level

    Returns
    -------
    {
        'sigma_h_per_k': (K,) per-budget value quantile,
        'sigma_grad_per_k': (K,) per-budget gradient quantile,
        'sigma_h_bar': float global max,
        'sigma_grad_bar': float global max,
    }
    """
    ks = sorted(h_true_per_k.keys())
    sh = np.array([
        calibrate_critic(h_true_per_k[k], h_pred_per_k[k], delta) for k in ks
    ])
    sg = np.array([
        calibrate_gradient(grad_err_per_k[k], delta) for k in ks
    ])
    return {
        "ks": np.array(ks),
        "sigma_h_per_k": sh,
        "sigma_grad_per_k": sg,
        "sigma_h_bar": float(np.max(sh)),
        "sigma_grad_bar": float(np.max(sg)),
    }


def empirical_coverage(
    h_true: np.ndarray,
    h_pred: np.ndarray,
    sigma_h: float,
) -> float:
    """Empirical fraction of test points where h_true ≥ h_pred - sigma_h.

    Under A5 + exchangeability this should be ≥ 1 - δ.
    Used for Experiment #4 (conformal calibration curve).
    """
    h_true = np.asarray(h_true, dtype=np.float64)
    h_pred = np.asarray(h_pred, dtype=np.float64)
    return float(np.mean(h_true >= h_pred - sigma_h))
