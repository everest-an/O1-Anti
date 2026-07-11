"""
module_graph.py — Context-routed Neural Module Graph (pillar 2: compute).

There is no fixed layer stack. A context encoder summarizes the input once;
a global router then commits to an ordered path of `path_len` modules out of
a library of `n_modules`. Only modules on the path run — activation ratio is
path_len / n_modules by construction.

Routing is trained with straight-through Gumbel-softmax: execution is hard
(dispatch each sequence to exactly one module per slot) while the router
still receives gradients through the straight-through scalar gate. A
load-balance penalty (usage variance) prevents module collapse.

New capabilities can be added by growing the library and finetuning only the
router — old modules stay frozen (incremental learning for free).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig
from .nla import NeuralLiquidAdjacency


class ContextEncoder(nn.Module):
    """Whole-input summary → context embedding c (one shot, before routing)."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ctx),
            nn.GELU(),
            nn.Linear(cfg.d_ctx, cfg.d_ctx),
        )

    def forward(self, h: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h: (B, T, d), pad_mask: (B, T) True=real token → (B, d_ctx)."""
        if pad_mask is not None:
            w = pad_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)
        return self.net(pooled)


class GlobalRouter(nn.Module):
    """One slot-wise categorical choice over the module library per path step."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        self.slot_logits = nn.Linear(cfg.d_ctx, cfg.path_len * cfg.n_modules)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ctx: (B, d_ctx) → (weights: (B, L, M) straight-through one-hot,
        usage: (M,) mean soft usage for the load-balance loss)."""
        B = ctx.shape[0]
        logits = self.slot_logits(ctx).view(B, self.cfg.path_len, self.cfg.n_modules)
        if self.training:
            w = F.gumbel_softmax(logits, tau=self.cfg.router_tau, hard=True, dim=-1)
            usage = F.softmax(logits / self.cfg.router_tau, dim=-1).mean(dim=(0, 1))
        else:
            idx = logits.argmax(dim=-1)
            w = F.one_hot(idx, self.cfg.n_modules).to(logits.dtype)
            usage = w.mean(dim=(0, 1))
        return w, usage


class NeuralModule(nn.Module):
    """One library entry: pre-norm NLA mixing + FFN, both residual."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.nla = NeuralLiquidAdjacency(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_out, s) — s is the liquid trajectory for regularization."""
        mixed, s = self.nla(self.norm1(h))
        h = h + mixed
        return h + self.ffn(self.norm2(h)), s


class ModuleLibrary(nn.Module):
    """Executes the routed path with hard per-sequence dispatch."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        self.modules_list = nn.ModuleList(NeuralModule(cfg) for _ in range(cfg.n_modules))

    def forward(self, h: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, T, d), weights: (B, L, M) straight-through one-hot.

        Returns (h_out, continuity) where continuity is the mean liquid
        state-continuity penalty over all executed modules.
        """
        from .losses import state_continuity_loss

        cont = h.new_zeros(())
        n_exec = 0
        for slot in range(self.cfg.path_len):
            w = weights[:, slot, :]                       # (B, M)
            idx = w.argmax(dim=-1)                        # (B,)
            out = h
            for m in idx.unique().tolist():
                rows = (idx == m).nonzero(as_tuple=True)[0]
                h_m, s_m = self.modules_list[m](h[rows])
                cont = cont + state_continuity_loss(s_m)
                n_exec += 1
                # straight-through scalar keeps router gradient alive
                g = w[rows, m].reshape(-1, 1, 1)
                out = out.index_copy(0, rows, h[rows] + g * (h_m - h[rows]))
            h = out
        return h, cont / max(n_exec, 1)
