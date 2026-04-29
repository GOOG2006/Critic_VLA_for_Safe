"""Safe-MoLe integration wrapper over MoLe-VLA's CogACT model.

Design:
    - Minimal intrusion into upstream `vla/cogactvla.py`. We attach the safety
      critic + CBF projection as a sibling module via `SafeMoLeExtension`, and
      expose two hooks:

        1. `compute_safety_losses(cognition_features, h_true, a_nominal, jac)`
           returns L_critic and L_CBF to be added to CogACT's training loss.

        2. `project_action(a_nominal, cognition_features, jac)`
           returns a_safe for inference-time CBF-filtered rollouts.

    - The extension itself is a plain nn.Module — can be wrapped by FSDP/DDP
      the same way the rest of CogACT is.

Reference: DERIVATION_PACKAGE.md v4 Module 2, §5.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .critic import SafetyCriticHead
from .projection import ISSfCBFProjection, CBFParams, issf_cbf_project


@dataclass
class SafeMoLeConfig:
    """Hyperparameters for the Safe-MoLe extension."""
    feature_dim: int = 4096
    critic_hidden: int = 256

    gamma: float = 0.5
    lip_h: float = 1.0
    d_max: float = 0.01
    a_max: float = 1.0
    eta: float = 0.0
    jac_norm: float = 1.0

    lambda_critic: float = 0.5
    lambda_cbf: float = 0.5
    lambda_grad: float = 0.1

    sigma_h_bar: float = 0.1
    sigma_grad_bar: float = 0.1
    lip_h_hat: float = 1.0
    eps_k: float = 0.0

    spectral_norm_critic: bool = True

    def cbf_params(self) -> CBFParams:
        return CBFParams(
            gamma=self.gamma, lip_h=self.lip_h, d_max=self.d_max,
            a_max=self.a_max, eta=self.eta, jac_norm=self.jac_norm,
        )

    def xi(self, eps_k: float | None = None) -> float:
        e = self.eps_k if eps_k is None else eps_k
        return (
            self.gamma * self.sigma_h_bar
            + self.a_max * self.sigma_grad_bar * self.jac_norm
            + self.lip_h_hat * e
            + self.eta * self.a_max ** 2
        )


class SafeMoLeExtension(nn.Module):
    """Plug-in module: critic + projection. Attaches alongside CogACT.

    Typical use inside CogACT.forward (after cognition_features extracted):
        L_critic, L_cbf, aux = ext.compute_safety_losses(
            cognition_features, h_true, a_nominal=actions_future, jac=jac)
        total = L_task + bw*L_lb + kw*L_cog + ext.cfg.lambda_critic*L_critic \\
                + ext.cfg.lambda_cbf*L_cbf
    """

    def __init__(self, cfg: SafeMoLeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.critic = SafetyCriticHead(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.critic_hidden,
            use_spectral_norm=cfg.spectral_norm_critic,
        )
        self.projection = ISSfCBFProjection(cfg.cbf_params())

    def critic_loss(self, h_pred: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(h_pred, h_true)

    def gradient_loss(self, grad_pred: torch.Tensor, grad_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(grad_pred, grad_true)

    def cbf_loss(
        self,
        a_nominal: torch.Tensor,
        c_hat: torch.Tensor,
        h_hat: torch.Tensor,
        eps_k: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        """Hinge on practical CBF constraint (2.3) v4 — includes η·||a||²."""
        e = float(eps_k) if isinstance(eps_k, torch.Tensor) else eps_k
        xi = self.cfg.xi(e)
        c_dot_a = (c_hat * a_nominal).sum(dim=-1)
        a_sq = (a_nominal * a_nominal).sum(dim=-1)
        violation = (
            -self.cfg.gamma * h_hat
            + self.cfg.lip_h * self.cfg.d_max
            + xi
            + self.cfg.eta * a_sq
            - c_dot_a
        )
        return torch.clamp_min(violation, 0.0).mean()

    def compute_safety_losses(
        self,
        cognition_features: torch.Tensor,
        h_true: torch.Tensor | None,
        a_nominal: torch.Tensor,
        jac: torch.Tensor | None = None,
        grad_h_true: torch.Tensor | None = None,
        eps_k: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute L_critic and L_CBF jointly."""
        feats = cognition_features.squeeze(1) if cognition_features.dim() == 3 else cognition_features
        h_hat, grad_hat = self.critic.value_and_feature_grad(feats, create_graph=True)

        device = feats.device
        if h_true is not None:
            L_critic = self.critic_loss(h_hat, h_true.to(device, dtype=h_hat.dtype))
        else:
            L_critic = torch.zeros((), device=device)

        if grad_h_true is not None and self.cfg.lambda_grad > 0:
            L_critic = L_critic + self.cfg.lambda_grad * self.gradient_loss(
                grad_hat, grad_h_true.to(device, dtype=grad_hat.dtype))

        if jac is not None:
            c_hat = torch.einsum("bad,bd->ba", jac.to(device, dtype=grad_hat.dtype), grad_hat)
        else:
            A = a_nominal.shape[-1]
            c_hat = grad_hat[..., :A] * self.cfg.jac_norm

        a_step = a_nominal[..., 0, :] if a_nominal.dim() >= 3 else a_nominal
        if c_hat.shape != a_step.shape:
            raise ValueError(f"c_hat {tuple(c_hat.shape)} vs a_step {tuple(a_step.shape)} mismatch")

        L_cbf = self.cbf_loss(a_step, c_hat, h_hat, eps_k)

        aux = {
            "h_hat": h_hat.detach(),
            "grad_hat_norm": grad_hat.detach().norm(dim=-1),
            "c_hat": c_hat.detach(),
            "cbf_margin": self.projection.cbf_margin(
                a_step, c_hat, h_hat, self.cfg.xi(eps_k)).detach(),
        }
        return L_critic, L_cbf, aux

    @torch.no_grad()
    def project_action(
        self,
        a_nominal: torch.Tensor,
        cognition_features: torch.Tensor,
        jac: torch.Tensor | None = None,
        eps_k: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Inference-time CBF projection."""
        feats = cognition_features.squeeze(1) if cognition_features.dim() == 3 else cognition_features
        with torch.enable_grad():
            h_hat, grad_hat = self.critic.value_and_feature_grad(feats, create_graph=False)

        A = a_nominal.shape[-1]
        if jac is not None:
            c_hat = torch.einsum("bad,bd->ba", jac.to(a_nominal.device, dtype=grad_hat.dtype),
                                 grad_hat)
        else:
            c_hat = grad_hat[..., :A] * self.cfg.jac_norm

        if a_nominal.dim() == 3:
            B, T, _ = a_nominal.shape
            c_hat_b = c_hat.unsqueeze(1).expand(B, T, A)
            h_hat_b = h_hat.unsqueeze(1).expand(B, T)
        else:
            c_hat_b = c_hat
            h_hat_b = h_hat

        xi = self.cfg.xi(eps_k)
        a_safe = issf_cbf_project(a_nominal, c_hat_b, h_hat_b, xi, self.cfg.cbf_params())
        displacement = (a_safe - a_nominal).norm(dim=-1)
        violated = displacement > 1e-8
        aux = {
            "h_hat": h_hat.detach(),
            "c_hat": c_hat.detach(),
            "displacement": displacement.detach(),
            "violation_mask": violated.detach(),
            "violation_rate": violated.float().mean().item(),
        }
        return a_safe, aux

    def update_conformal_quantiles(self, sigma_h_bar: float, sigma_grad_bar: float) -> None:
        self.cfg.sigma_h_bar = float(sigma_h_bar)
        self.cfg.sigma_grad_bar = float(sigma_grad_bar)

    def extra_repr(self) -> str:
        c = self.cfg
        return f"λ_critic={c.lambda_critic}, λ_cbf={c.lambda_cbf}, γ={c.gamma}, ξ(k=0)={c.xi():.4f}"
