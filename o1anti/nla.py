"""
nla.py — Neural Liquid Adjacency (pillar 1: memory).

Replaces softmax attention + KV cache. Per position we cache only a compressed
state c_j = W_c x_j with d_c << d_model; routing keys and values are both
derived from c_j, so the inference cache is O(n * d_c) instead of the
O(n * 2 * d_model) KV cache.

A liquid global state s_t (input-dependent gated recurrence, parallel-scan
trainable) summarizes the whole prefix; it shapes both the routing query and
an output gate, giving the "adjacency" its liquid, context-dependent nature.

Each token connects to at most `top_k` past positions, chosen by content
scores (with optional Gumbel exploration noise at train time). Training runs
the score matrix densely (compute is O(n^2) but memory-light); a block-sparse
kernel is the scaling roadmap. Inference `step()` is exactly consistent with
the parallel forward.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig


class LiquidStateScan(nn.Module):
    """s_t = a_t * s_{t-1} + b_t with input-dependent diagonal gates.

    a_t = sigmoid(W_a x_t)  in (0,1),  b_t = (1 - a_t) * tanh(W_u x_t).
    Trained with a Hillis-Steele associative scan (log2(T) steps, no division,
    numerically stable); decoded with the exact same recurrence one step at a
    time.
    """

    def __init__(self, d_in: int, d_state: int):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_in, 2 * d_state)
        # bias decay gates toward "remember" at init
        nn.init.constant_(self.in_proj.bias[:d_state], 2.0)

    def _gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.in_proj(x)
        a = torch.sigmoid(ab[..., : self.d_state])
        b = (1.0 - a) * torch.tanh(ab[..., self.d_state :])
        return a, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) → s: (B, T, d_state), s_0 assumed 0."""
        a, b = self._gates(x)
        A, Bv = a, b
        offset = 1
        T = x.shape[1]
        while offset < T:
            A_prev = F.pad(A, (0, 0, offset, 0), value=1.0)[:, :T]
            B_prev = F.pad(Bv, (0, 0, offset, 0), value=0.0)[:, :T]
            Bv = A * B_prev + Bv
            A = A * A_prev
            offset *= 2
        return Bv

    def step(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (B, d_in), s_prev: (B, d_state) → s_t: (B, d_state)."""
        a, b = self._gates(x_t)
        return a * s_prev + b


class NeuralLiquidAdjacency(nn.Module):
    """Top-K dynamic sparse aggregation over compressed position states."""

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        d, dc, ds = cfg.d_model, cfg.d_c, cfg.d_state
        self.compress = nn.Linear(d, dc)            # c_j — the only cached tensor
        self.key = nn.Linear(dc, dc, bias=False)    # routing key from c_j
        self.value = nn.Linear(dc, d, bias=False)   # value from c_j
        self.query = nn.Linear(d + ds, dc)          # routing query from (x_t, s_t)
        self.gate = nn.Linear(ds, d)                # liquid output gate
        self.out = nn.Linear(d, d)
        self.state = LiquidStateScan(d, ds)
        self.scale = 1.0 / math.sqrt(dc)

    # ------------------------------------------------------------------ train
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d) → (out: (B, T, d), s: (B, T, d_state))."""
        B, T, _ = x.shape
        K = min(self.cfg.top_k, T)

        s = self.state(x)                                    # (B, T, ds)
        c = self.compress(x)                                 # (B, T, dc)
        k = self.key(c)                                      # (B, T, dc)
        v = self.value(c)                                    # (B, T, d)
        q = self.query(torch.cat([x, s], dim=-1))            # (B, T, dc)

        scores = torch.einsum("btd,bjd->btj", q, k) * self.scale
        if self.training and self.cfg.nla_route_noise > 0:
            u = torch.rand_like(scores).clamp_min(1e-9)
            scores = scores - self.cfg.nla_route_noise * torch.log(-torch.log(u))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        topv, topi = scores.topk(K, dim=-1)                  # (B, T, K)
        sparse = torch.full_like(scores, float("-inf")).scatter(-1, topi, topv)
        alpha = torch.nan_to_num(F.softmax(sparse, dim=-1))  # rows with < K valid slots
        if self.training:
            # straight-through: execute sparse, but let gradients flow through
            # the dense softmax so non-selected positions keep a learning signal
            dense = torch.nan_to_num(F.softmax(scores, dim=-1))
            alpha = alpha.detach() + dense - dense.detach()

        agg = torch.einsum("btj,bjd->btd", alpha, v)
        out = self.out(agg * torch.sigmoid(self.gate(s)))
        return out, s

    # -------------------------------------------------------------- inference
    def init_cache(self, batch: int, device=None, dtype=None) -> Dict[str, torch.Tensor]:
        return {
            "c": torch.zeros(batch, 0, self.cfg.d_c, device=device, dtype=dtype),
            "s": torch.zeros(batch, self.cfg.d_state, device=device, dtype=dtype),
        }

    def step(self, x_t: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        """x_t: (B, d). Mutates cache; returns out_t: (B, d).

        Cache holds c (B, t, d_c) and s (B, d_state) only — no KV pairs.
        """
        s = self.state.step(x_t, cache["s"])
        c_t = self.compress(x_t)
        cache["c"] = torch.cat([cache["c"], c_t.unsqueeze(1)], dim=1)
        cache["s"] = s

        c = cache["c"]                                       # (B, t, dc)
        k = self.key(c)
        v = self.value(c)
        q = self.query(torch.cat([x_t, s], dim=-1))          # (B, dc)

        scores = torch.einsum("bd,bjd->bj", q, k) * self.scale
        K = min(self.cfg.top_k, c.shape[1])
        topv, topi = scores.topk(K, dim=-1)
        alpha = F.softmax(topv, dim=-1)                      # (B, K)
        v_sel = v.gather(1, topi.unsqueeze(-1).expand(-1, -1, v.shape[-1]))
        agg = (alpha.unsqueeze(-1) * v_sel).sum(dim=1)       # (B, d)
        return self.out(agg * torch.sigmoid(self.gate(s)))

    @staticmethod
    def cache_bytes_per_token(cfg: O1AntiConfig, dtype_bytes: int = 2) -> int:
        """Inference memory per token per layer, vs 2*d_model for a KV cache."""
        return cfg.d_c * dtype_bytes
