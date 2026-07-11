"""
generation.py — Liquid state-transition generation (pillar 3: latency).

Two-stage, non-autoregressive:

  Stage 1  SkeletonGenerator — a flow-matching vector field v(z, t; ctx)
           integrated with a handful of Euler steps maps noise straight to a
           continuous "semantic skeleton" (skel_len << seq_len vectors).
           Training target: linear-interpolant flow matching against
           skeletons pooled from the ground-truth sequence.

  Stage 2  ParallelDecoder — bidirectional blocks with cross-attention into
           the skeleton emit ALL tokens at once, refined with a fixed number
           of mask-predict rounds (re-mask lowest-confidence positions and
           re-decode).

Cost model for a length-n generation: ode_steps + decode_iters forward
passes total, instead of n autoregressive steps.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0,1] → (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class SkeletonGenerator(nn.Module):
    """Flow-matching field over the skeleton; few-step Euler at inference."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.field = nn.Sequential(
            nn.Linear(d + d + cfg.d_ctx, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def velocity(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """z: (B, L_skel, d), t: (B,), ctx: (B, d_ctx) → (B, L_skel, d)."""
        L = z.shape[1]
        temb = timestep_embedding(t, self.cfg.d_model).unsqueeze(1).expand(-1, L, -1)
        c = ctx.unsqueeze(1).expand(-1, L, -1)
        return self.field(torch.cat([z, temb, c], dim=-1))

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Integrate dz/dt = v(z, t; ctx) from noise (t=0) to skeleton (t=1)."""
        B = ctx.shape[0]
        z = torch.randn(
            B, self.cfg.skel_len, self.cfg.d_model,
            device=ctx.device, dtype=ctx.dtype, generator=generator,
        )
        n = self.cfg.ode_steps
        dt = 1.0 / n
        for i in range(n):
            t = torch.full((B,), i * dt, device=ctx.device, dtype=ctx.dtype)
            z = z + dt * self.velocity(z, t, ctx)
        return z

    @staticmethod
    def pool_skeleton(h: torch.Tensor, skel_len: int) -> torch.Tensor:
        """Ground-truth skeleton: adaptive average pool of the target
        sequence representation. h: (B, T, d) → (B, skel_len, d)."""
        return F.adaptive_avg_pool1d(h.transpose(1, 2), skel_len).transpose(1, 2)


class DecoderBlock(nn.Module):
    """Bidirectional self-attention + cross-attention into the skeleton."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        d, nh = cfg.d_model, cfg.n_dec_heads
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, d)
        )

    def forward(self, h: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        h = h + self.self_attn(x, x, x, need_weights=False)[0]
        x = self.norm2(h)
        h = h + self.cross_attn(x, skel, skel, need_weights=False)[0]
        return h + self.ffn(self.norm3(h))


class ParallelDecoder(nn.Module):
    """Emit every position at once, conditioned on the skeleton."""

    def __init__(self, cfg: O1AntiConfig, embed: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.embed = embed                       # shared with the LM trunk
        self.mask_emb = nn.Parameter(torch.zeros(cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(cfg.max_seq_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_dec_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = embed.weight          # weight tying

    def logits(self, tokens: torch.Tensor, masked: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) ids, masked: (B, T) bool (True → use [MASK]),
        skel: (B, L_skel, d) → (B, T, vocab)."""
        h = self.embed(tokens)
        h = torch.where(masked.unsqueeze(-1), self.mask_emb.expand_as(h), h)
        h = h + self.pos_emb[: h.shape[1]]
        for blk in self.blocks:
            h = blk(h, skel)
        return self.head(self.norm(h))

    @torch.no_grad()
    def mask_predict(self, skel: torch.Tensor, length: int) -> torch.Tensor:
        """Fixed-round mask-predict decoding → (B, length) token ids."""
        B = skel.shape[0]
        device = skel.device
        tokens = torch.zeros(B, length, dtype=torch.long, device=device)
        masked = torch.ones(B, length, dtype=torch.bool, device=device)
        iters = self.cfg.decode_iters
        for it in range(iters):
            probs = F.softmax(self.logits(tokens, masked, skel), dim=-1)
            conf, pred = probs.max(dim=-1)
            tokens = torch.where(masked, pred, tokens)
            conf = torch.where(masked, conf, torch.ones_like(conf))
            if it == iters - 1:
                break
            # re-mask the lowest-confidence fraction, annealed to zero
            n_mask = int(length * (1.0 - (it + 1) / iters))
            if n_mask == 0:
                break
            remask_idx = conf.argsort(dim=-1)[:, :n_mask]
            masked = torch.zeros_like(masked).scatter(1, remask_idx, True)
        return tokens
