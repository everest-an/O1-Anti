"""
generation.py — Liquid state-transition generation (pillar 3: latency).

Two-stage, non-autoregressive. A length-n sequence costs (stage-1 + decode_iters)
forward passes instead of n autoregressive steps.

  Stage 1  produce a "semantic skeleton" (skel_len << n latent vectors) from the
           prompt. Three interchangeable generators:
             SkeletonPrior (regress)  — deterministic prompt→skeleton, 1 pass.
             SkeletonGenerator (flow) — flow-matching neural ODE, ode_steps.
             VectorQuantizer + SkeletonPrior (discrete) — VQ codes, 1 pass.
           SkeletonEncoder produces the training-time target skeleton from the
           ground-truth sequence (the stage-2 teacher).

  Stage 2  ParallelDecoder — bidirectional blocks cross-attend into the skeleton
           and emit ALL tokens at once, refined with a fixed number of
           mask-predict rounds (re-mask lowest-confidence positions, re-decode).

CRITICAL: position embeddings here use O(1) init (cfg.pos_emb_std ≈ 1), NOT the
usual 0.02. In mask-predict's first round every position is masked, so the query
is only mask_emb + pos; a tiny pos makes all queries identical and cross-attention
cannot address positions, collapsing decode to random. Output queries and
skeleton keys also SHARE the position embedding for a diagonal alignment prior.
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
        nn.init.normal_(self.pos, std=cfg.pos_emb_std)
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


class VectorQuantizer(nn.Module):
    """Product-quantized (cosine, L2-normalized) codebook with straight-through
    gradients. Turns the continuous skeleton into `skel_len` discrete codes, so
    stage-1 generation becomes a parallel classification problem instead of
    continuous flow matching.

    d_model is split into `vq_groups` independent subvectors, each with its own
    codebook_size-entry codebook (product quantization / PQ). Combinatorial code
    space is codebook_size^vq_groups while total codebook params stay
    codebook_size*d_model — same footprint as one codebook. This fixes the
    fidelity cap of a single codebook (E9): with vq_groups=1 the nearest code was
    only cos-sim~0.63 to the target on real data, capping decode accuracy at
    0.63; splitting into groups quantizes each low-dimensional subvector far more
    precisely (curse-of-dimensionality effect on nearest-neighbour matching).

    Each subvector and its codebook are L2-normalized before nearest-neighbour
    lookup (quantize on the unit sphere) — the standard fix for codebook
    collapse at small scale (ViT-VQGAN), unaffected by grouping."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.beta = cfg.vq_beta
        self.G = cfg.vq_groups
        self.dg = cfg.d_model // self.G
        self.codebooks = nn.ModuleList(
            nn.Embedding(cfg.codebook_size, self.dg) for _ in range(self.G)
        )
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, std=1.0)

    def forward(self, z: torch.Tensor):
        """z: (B, L, d) → (q_st: (B, L, d), codes: (B, L, G), vq_loss)."""
        B, L, d = z.shape
        z_g = F.normalize(z.view(B, L, self.G, self.dg), dim=-1)   # (B,L,G,dg)
        q_parts, code_parts, vq_loss = [], [], 0.0
        for g in range(self.G):
            cb = F.normalize(self.codebooks[g].weight, dim=-1)     # (K, dg)
            zg = z_g[:, :, g, :]                                    # (B,L,dg)
            sim = torch.einsum("bld,kd->blk", zg, cb)
            codes = sim.argmax(dim=-1)                              # (B,L)
            q = cb[codes]                                           # (B,L,dg)
            vq_loss = vq_loss + F.mse_loss(q, zg.detach()) + self.beta * F.mse_loss(zg, q.detach())
            q_parts.append(zg + (q - zg).detach())                  # straight-through
            code_parts.append(codes)
        q_st = torch.cat(q_parts, dim=-1)                          # (B, L, d), unit-norm per group
        codes = torch.stack(code_parts, dim=-1)                    # (B, L, G)
        return q_st, codes, vq_loss / self.G

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, L, G) → (B, L, d) unit-norm-per-group codebook vectors."""
        parts = [F.normalize(self.codebooks[g].weight, dim=-1)[codes[..., g]]
                 for g in range(self.G)]
        return torch.cat(parts, dim=-1)


class SkeletonPrior(nn.Module):
    """Parallel prior: learned slot queries cross-attend into the prompt memory
    and predict the skeleton in ONE pass, replacing the flow-matching ODE.

    out_dim = codebook_size → discrete code logits (discrete mode);
    out_dim = d_model       → continuous skeleton regression (regress mode)."""

    def __init__(self, cfg: O1AntiConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.slots = nn.Parameter(torch.zeros(cfg.skel_len, d))
        nn.init.normal_(self.slots, std=cfg.pos_emb_std)
        # prompt memory needs position info so slots can address prompt order
        self.mem_pos = nn.Parameter(torch.zeros(cfg.max_seq_len, d))
        nn.init.normal_(self.mem_pos, std=cfg.pos_emb_std)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_dec_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out_dim)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """mem: (B, T_prompt, d) → (B, skel_len, out_dim)."""
        mem = mem + self.mem_pos[: mem.shape[1]]
        h = self.slots.unsqueeze(0).expand(mem.shape[0], -1, -1)
        for blk in self.blocks:
            h = blk(h, mem)
        return self.head(self.norm(h))


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
        nn.init.normal_(self.slot_pos, std=cfg.pos_emb_std)
        self.mem_pos = nn.Parameter(torch.zeros(cfg.max_seq_len, d))
        nn.init.normal_(self.mem_pos, std=cfg.pos_emb_std)
        self.t_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_dec_layers))
        self.out = nn.Linear(d, d)

    def velocity(self, z: torch.Tensor, t: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """z: (B, L_skel, d), t: (B,), mem: (B, T_prompt, d) → (B, L_skel, d)."""
        temb = self.t_proj(timestep_embedding(t, self.cfg.d_model)).unsqueeze(1)
        mem = mem + self.mem_pos[: mem.shape[1]]
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
        nn.init.normal_(self.pos_emb, std=cfg.pos_emb_std)
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
        # SHARE the position embedding with the skeleton keys: query at output
        # position i and key at skeleton slot i then share a strong positional
        # component, giving cross-attention a diagonal alignment prior.
        skel = skel + self.pos_emb[: skel.shape[1]]
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
