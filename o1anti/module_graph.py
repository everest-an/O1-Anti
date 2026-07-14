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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig
from .losses import state_continuity_loss
from .nla import NeuralLiquidAdjacency


class RoPECausalAttention(nn.Module):
    """Causal multi-head self-attention with rotary position embeddings.

    Used as the attention mixer in a hybrid TokenMoETrunk. Unlike NLA — whose
    liquid state scan encodes position implicitly — plain softmax attention is
    permutation-equivariant and needs explicit positional grounding, so RoPE is
    baked in (rotating q,k) rather than relying on an external pos embedding.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0, "head dim must be even for RoPE"
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, T, dh) → rotary-embedded (LLaMA half-split convention)."""
        T, dh = x.shape[-2], x.shape[-1]
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh))
        ang = torch.outer(pos, inv)                       # (T, dh/2)
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1)   # (T, dh)
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
        x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
        xr = torch.cat([-x2, x1], dim=-1)
        return x * cos + xr * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)     # (B, H, T, dh)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = self._rope(q), self._rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, d))


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
        by renormalized softmax gate weights; unselected experts don't run.

        Selection uses noisy logits at train time (Shazeer 2017) so exploration
        keeps every expert occasionally winning a token; the COMBINE weights and
        the load-balance `usage` signal come from the clean softmax so the
        gradient is unbiased."""
        B, T, d = h.shape
        E, top_e = self.cfg.n_modules, self.cfg.moe_top_e
        flat = h.reshape(-1, d)                               # (N, d)
        logits = self.gate(flat)                              # (N, E)
        probs = F.softmax(logits, dim=-1)                     # clean gate (grad + balance)

        sel_logits = logits
        if self.training and self.cfg.moe_noise > 0:
            sel_logits = logits + (self.cfg.moe_noise / E) * torch.randn_like(logits)
        topi = sel_logits.topk(top_e, dim=-1).indices        # (N, top_e) — which experts
        # combine weights from the CLEAN softmax, renormalized over the chosen set
        topw = probs.gather(1, topi)
        topw = topw / topw.sum(dim=-1, keepdim=True)          # (N, top_e)

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
    """One token-trunk block: a sequence mixer + a per-token routed MoE-FFN.

    mixer="nla" (default) uses Neural Liquid Adjacency (pillar 1, cheap
    long-range, O(n·d_c) cache). mixer="attn" uses full causal multi-head
    attention (dense many-relations mixing, full KV cache) — hybrid trunks
    interleave the two (Jamba/Zamba pattern). Attention layers return s=None so
    the trunk skips them in the liquid state-continuity penalty."""

    def __init__(self, cfg: O1AntiConfig, mixer: str = "nla"):
        super().__init__()
        self.mixer_kind = mixer
        self.norm1 = nn.LayerNorm(cfg.d_model)
        if mixer == "attn":
            self.attn = RoPECausalAttention(cfg.d_model, cfg.nla_heads)
        else:
            self.nla = NeuralLiquidAdjacency(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoEFeedForward(cfg)

    def forward(self, h: torch.Tensor):
        x = self.norm1(h)
        if self.mixer_kind == "attn":
            mixed = self.attn(x)                          # RoPE causal attention
            s = None
        else:
            mixed, s = self.nla(x)
        h = h + mixed
        ffn, usage = self.moe(self.norm2(h))
        return h + ffn, s, usage


class TokenMoETrunk(nn.Module):
    """Stack of MoE blocks — the fine-grained-routing trunk for language
    modeling. With cfg.hybrid_attn_every > 0, every N-th block mixes with full
    attention instead of NLA (hybrid), the rest use NLA."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        blocks = []
        for i in range(cfg.n_layers):
            is_attn = cfg.hybrid_attn_every > 0 and (i + 1) % cfg.hybrid_attn_every == 0
            blocks.append(MoEBlock(cfg, mixer="attn" if is_attn else "nla"))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: (B, T, d) → (h_out, usage (E,) averaged over layers, continuity)."""
        usage = h.new_zeros(self.cfg.n_modules)
        cont = h.new_zeros(())
        n_nla = 0
        for blk in self.blocks:
            h, s, u = blk(h)
            usage = usage + u
            if s is not None:                    # attention layers have no liquid state
                cont = cont + state_continuity_loss(s)
                n_nla += 1
        return h, usage / len(self.blocks), cont / max(n_nla, 1)
