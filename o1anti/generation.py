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


class SkeletonEncoder(nn.Module):
    """Learned semantic skeleton: a bidirectional encoder over the target
    sequence, mean-pooled into `skel_len` latent slots. Trained jointly with
    the decoder through the reconstruction loss, so the compression is
    invertible — unlike a raw average-pool of embeddings. This is the flow
    generator's regression target (in latent space, à la latent diffusion)."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.pos = nn.Parameter(torch.zeros(cfg.max_seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d, cfg.n_dec_heads, cfg.d_ff, batch_first=True)
            for _ in range(cfg.n_dec_layers)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h_target: torch.Tensor) -> torch.Tensor:
        """h_target: (B, T, d) → (B, skel_len, d) learned latents."""
        h = h_target + self.pos[: h_target.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return F.adaptive_avg_pool1d(h.transpose(1, 2), self.cfg.skel_len).transpose(1, 2)


class SkeletonGenerator(nn.Module):
    """Flow-matching field over the skeleton — a small transformer that
    self-attends across skeleton slots and cross-attends into the encoded
    prompt memory, conditioned on the diffusion time. Conditioning on the
    full prompt (not a single pooled vector) is what lets the generated
    skeleton actually track the input."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.slot_pos = nn.Parameter(torch.zeros(cfg.skel_len, d))
        nn.init.normal_(self.slot_pos, std=0.02)
        self.t_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_dec_layers))
        self.out = nn.Linear(d, d)

    def velocity(self, z: torch.Tensor, t: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """z: (B, L_skel, d), t: (B,), mem: (B, T_prompt, d) → (B, L_skel, d)."""
        temb = self.t_proj(timestep_embedding(t, self.cfg.d_model)).unsqueeze(1)
        h = z + self.slot_pos[: z.shape[1]] + temb
        for blk in self.blocks:
            h = blk(h, mem)                       # self-attn + cross-attn into prompt
        return self.out(h)

    @torch.no_grad()
    def sample(self, mem: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Integrate dz/dt = v(z, t; mem) from noise (t=0) to skeleton (t=1)."""
        B = mem.shape[0]
        z = torch.randn(
            B, self.cfg.skel_len, self.cfg.d_model,
            device=mem.device, dtype=mem.dtype, generator=generator,
        )
        n = self.cfg.ode_steps
        dt = 1.0 / n
        for i in range(n):
            t = torch.full((B,), i * dt, device=mem.device, dtype=mem.dtype)
            z = z + dt * self.velocity(z, t, mem)
        return z


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
        # positional addressing INTO the skeleton, so output position i can
        # align to its skeleton slot during cross-attention.
        self.skel_pos = nn.Parameter(torch.zeros(cfg.skel_len, cfg.d_model))
        nn.init.normal_(self.skel_pos, std=0.02)
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
        skel = skel + self.skel_pos[: skel.shape[1]]      # positional keys
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
