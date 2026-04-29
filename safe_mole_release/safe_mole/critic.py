"""Safety Critic Head ĥ_φ — estimates barrier h(x) from cognition token.

Maps MoLe's cognition token e^c ∈ R^d (from CogKD) to:
  - ĥ_φ(f): scalar estimate of h(x)
  - ∇ĥ_φ(f): gradient w.r.t. feature (autograd-differentiable)

Lipschitz property (A4) enforced via spectral normalization.

Reference: DERIVATION_PACKAGE.md §A4–A5, Module 2.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SafetyCriticHead(nn.Module):
    """2-layer MLP predicting a scalar safety barrier ĥ from cognition features.

    Parameters
    ----------
    feature_dim : int
        Dimension of cognition token (MoLe uses 4096 for LLaMA2-7B).
    hidden_dim : int, default 256
        MLP hidden width.
    use_spectral_norm : bool, default True
        Enforces Lipschitz bound for A4. Disable only for ablation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        l1 = nn.Linear(feature_dim, hidden_dim)
        l2 = nn.Linear(hidden_dim, 1)
        if use_spectral_norm:
            l1 = spectral_norm(l1)
            l2 = spectral_norm(l2)

        self.net = nn.Sequential(
            l1,
            nn.GELU(),
            l2,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute ĥ_φ(features).

        Parameters
        ----------
        features : (..., feature_dim) tensor
            Cognition features (batched; last dim is feature_dim).

        Returns
        -------
        h_hat : (...,) tensor
            Scalar barrier estimates, one per batch element.
        """
        return self.net(features).squeeze(-1)

    def value_and_feature_grad(
        self,
        features: torch.Tensor,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (ĥ_φ, ∇_feature ĥ_φ) jointly via autograd.

        Needed at inference time to form estimated safety gradient
          ĉ(x) = (∂f/∂a)^T · ∇_x ĥ_φ
        via chain rule; the ∂f^{(s)}/∂x factor comes from the feature extractor.

        Parameters
        ----------
        features : (B, D) tensor
            Cognition features, requires_grad=False at input (we enable internally).
        create_graph : bool, default False
            Pass True during training to allow double backward (for higher-order
            gradient regularization). Keep False for pure inference.

        Returns
        -------
        h_hat : (B,) tensor
        grad : (B, D) tensor — ∂ĥ_φ / ∂features
        """
        features = features.detach().requires_grad_(True)
        h_hat = self.forward(features)
        grad_outputs = torch.ones_like(h_hat)
        (grad,) = torch.autograd.grad(
            outputs=h_hat,
            inputs=features,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        return h_hat, grad

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}"
