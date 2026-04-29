"""Safety barrier functions h(x) for RLBench / real-robot manipulation.

Two families:
  1. Analytic barriers (no scene geometry needed) — e.g., workspace limits,
     height-above-table, force magnitude. Differentiable, cheap.
  2. SDF barriers (signed-distance from scene meshes) — for collision avoidance
     in cluttered scenes. Requires the RLBench CoppeliaSim scene to be parsed.

For the first training smoke-tests we use (1). For the real safety experiments
we extend to (2) via RLBench scene introspection.

Each barrier exposes:
    h(x) : (...) tensor
    grad_h_x(x) : (..., state_dim) tensor   [optional; autograd fallback]

Reference: DERIVATION_PACKAGE.md A2 (h Lipschitz + smooth).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import torch
import torch.nn as nn


@dataclass
class TableHeightBarrier:
    """h(x) = z_ee - z_min  — positive above table, negative below.

    Simple analytic barrier for the "don't drop through the table" constraint.
    Trivially Lipschitz (Lip_h = 1), smooth, differentiable.

    Expects state[..., z_index] to hold end-effector z coordinate.

    Parameters
    ----------
    z_index : int, default 2
        Index of z-coordinate in proprioceptive state vector (convention: XYZ).
    z_min : float, default 0.752
        RLBench default table height (m). Can be shifted up for margin.
    """
    z_index: int = 2
    z_min: float = 0.752

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.z_index] - self.z_min

    def grad_state(self, state: torch.Tensor) -> torch.Tensor:
        """∂h/∂state — exactly one-hot at z_index."""
        g = torch.zeros_like(state)
        g[..., self.z_index] = 1.0
        return g


@dataclass
class PickAboveBarrier:
    """GRAPE-style stage-1 collision barrier for pick-and-place LIBERO tasks.

    h(x) = z_ee - z_target_obj   positive = EE above target object (safe approach),
                                 negative = EE punches into/through target obj.

    Only meaningful during stage-1 (approach before grasp). Callers should gate
    enforcement on an is_stage1 flag derived from obj lift height vs. initial.

    Takes full state = concat(eef_pos[3], obj_pos[3]) so grad is one-hot diff.
    """
    eef_z_idx: int = 2
    obj_z_idx: int = 5

    def __call__(self, state):
        return state[..., self.eef_z_idx] - state[..., self.obj_z_idx]

    def grad_state(self, state):
        import torch
        g = torch.zeros_like(state)
        g[..., self.eef_z_idx] = 1.0
        g[..., self.obj_z_idx] = -1.0
        return g


@dataclass
class WorkspaceBoxBarrier:
    """h(x) = min_i {workspace_max[i] - x[i], x[i] - workspace_min[i]}

    Soft "stay inside box" constraint. Smoothed via log-sum-exp for differentiability.

    Parameters
    ----------
    lo, hi : (state_dim,) float tensor
        Box limits.
    beta : float, default 20.0
        Smoothness parameter for soft-min (larger = closer to true min).
    xyz_slice : slice
        Which indices of state are the XYZ position.
    """
    lo: torch.Tensor
    hi: torch.Tensor
    beta: float = 20.0
    xyz_slice: slice = field(default_factory=lambda: slice(0, 3))

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        xyz = state[..., self.xyz_slice]
        # gap per dimension (positive inside, negative outside)
        upper_gap = self.hi.to(state) - xyz
        lower_gap = xyz - self.lo.to(state)
        gaps = torch.cat([upper_gap, lower_gap], dim=-1)  # (..., 2*d)
        # soft-min ≈ -(1/β) log Σ exp(-β gap_i)
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * gaps, dim=-1)


def compute_analytic_h_and_grad(
    state: torch.Tensor,
    barrier: Callable[[torch.Tensor], torch.Tensor],
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic computation of h(state) and ∇_state h via autograd.

    For analytic barriers this is more convenient than writing hand-crafted
    gradients. Outputs:
        h : (...) tensor
        grad : same shape as state

    Use when the barrier has no closed-form gradient accessor.
    """
    s = state.detach().requires_grad_(True)
    h = barrier(s)
    grad = torch.autograd.grad(
        outputs=h,
        inputs=s,
        grad_outputs=torch.ones_like(h),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return h.detach(), grad.detach()


class BarrierRegistry:
    """Registry of barriers by name — used to configure training runs."""

    @staticmethod
    def build(name: str, **kwargs) -> Callable:
        if name == "table_height":
            return TableHeightBarrier(**kwargs)
        if name == "workspace_box":
            # Require lo / hi as kwargs
            lo = torch.as_tensor(kwargs.pop("lo"), dtype=torch.float32)
            hi = torch.as_tensor(kwargs.pop("hi"), dtype=torch.float32)
            return WorkspaceBoxBarrier(lo=lo, hi=hi, **kwargs)
        raise ValueError(f"unknown barrier '{name}'")
