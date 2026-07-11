"""losses.py — auxiliary objectives for the O1-Anti stack."""

import torch
import torch.nn.functional as F


def load_balance_loss(usage: torch.Tensor) -> torch.Tensor:
    """usage: (M,) mean soft routing mass per module. Zero when uniform.

    Scaled variance: M * sum((u - 1/M)^2), matching the spec's sigma^2(usage)
    up to a constant so the coefficient is size-independent.
    """
    M = usage.shape[0]
    return M * ((usage - 1.0 / M) ** 2).sum()


def state_continuity_loss(s: torch.Tensor) -> torch.Tensor:
    """s: (B, T, d_state) liquid trajectory → mean ||ds/dt||^2."""
    if s.shape[1] < 2:
        return s.new_zeros(())
    return (s[:, 1:] - s[:, :-1]).pow(2).mean()


def flow_matching_loss(field, z1: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Linear-interpolant flow matching for the skeleton generator.

    z_t = (1-t) z0 + t z1 with z0 ~ N(0,I); z1 is the target skeleton (a
    learned latent from SkeletonEncoder, detached). Regression target is the
    constant velocity z1 - z0.
    """
    z1 = z1.detach()
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)
    zt = (1.0 - t)[:, None, None] * z0 + t[:, None, None] * z1
    v = field.velocity(zt, t, ctx)
    return F.mse_loss(v, z1 - z0)
