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


# ---------------------------------------------------------------------------
# Token-level (fine-grained) routing — MoE-FFN on a dense NLA backbone.
# The global ModuleLibrary above routes one path per whole input, which is too
# coarse for language modeling. Here NLA stays dense (coherent sequence mixing,
# pillar 1) and only the FFN is routed per token to one of n_modules experts
# (pillar 2, fine-grained). This is the standard MoE-Transformer split.
# ---------------------------------------------------------------------------
class MoEFeedForward(nn.Module):
    """Per-token top-e routed FFN experts with a load-balance signal."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_modules)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                          nn.Linear(cfg.d_ff, cfg.d_model))
            for _ in range(cfg.n_modules)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, T, d) → (out, usage (E,)). Top-e experts per token, combined
        by renormalized softmax gate weights; unselected experts don't run."""
        B, T, d = h.shape
        E, top_e = self.cfg.n_modules, self.cfg.moe_top_e
        flat = h.reshape(-1, d)                               # (N, d)
        probs = F.softmax(self.gate(flat), dim=-1)            # (N, E)
        topw, topi = probs.topk(top_e, dim=-1)               # (N, top_e)
        topw = topw / topw.sum(dim=-1, keepdim=True)          # renormalize

        out = torch.zeros_like(flat)
        for m in range(E):
            sel = (topi == m)                                # (N, top_e) bool
            tok = sel.any(dim=-1)                            # (N,) token uses expert m
            if not tok.any():
                continue
            rows = tok.nonzero(as_tuple=True)[0]
            w = (topw * sel).sum(dim=-1)[rows].unsqueeze(-1)  # this token's weight on m
            out[rows] += w * self.experts[m](flat[rows])
        # load-balance: fraction of tokens routed to each expert × mean gate prob
        frac = torch.zeros(E, device=h.device)
        frac.scatter_add_(0, topi.reshape(-1), torch.ones_like(topi.reshape(-1), dtype=h.dtype))
        frac = frac / frac.sum().clamp_min(1.0)
        usage = 0.5 * (frac + probs.mean(dim=0))             # blend count + prob mass
        return out.reshape(B, T, d), usage


class MoEBlock(nn.Module):
    """Dense NLA (pillar 1) + per-token routed MoE-FFN (pillar 2)."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.nla = NeuralLiquidAdjacency(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoEFeedForward(cfg)

    def forward(self, h: torch.Tensor):
        mixed, s = self.nla(self.norm1(h))
        h = h + mixed
        ffn, usage = self.moe(self.norm2(h))
        return h + ffn, s, usage


class TokenMoETrunk(nn.Module):
    """Stack of MoE blocks — the fine-grained-routing trunk for language modeling."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg.n_layers))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: (B, T, d) → (h_out, usage (E,) averaged over layers, continuity)."""
        from .losses import state_continuity_loss

        usage = h.new_zeros(self.cfg.n_modules)
        cont = h.new_zeros(())
        for blk in self.blocks:
            h, s, u = blk(h)
            usage = usage + u
            cont = cont + state_continuity_loss(s)
        n = len(self.blocks)
        return h, usage / n, cont / n
