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
scores (with optional Gumbel exploration noise at train time). By default
training runs the score matrix densely (exact, O(n^2)). Setting
`cfg.nla_block_size > 0` switches to a two-stage block-sparse path that is
sub-quadratic (O(T^1.5) at bs≈sqrt(T)) and closely approximates the exact
top-K (see NeuralLiquidAdjacency for details). Inference `step()` is exactly
consistent with the parallel forward.
"""

import math
from typing import Dict, Tuple

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
    """Top-K dynamic sparse aggregation over compressed position states.

    Scope note: the primary win is the *inference cache* — O(n·d_c) compressed
    states instead of an O(n·d_model) KV cache (see cache_bytes_per_token).

    Training compute has two paths (cfg.nla_block_size):
      = 0 (default): exact O(n²) — forward() materializes the full (T, T) score
          matrix before the top-K. Used by all validated results.
      > 0: block-sparse two-stage top-K, sub-quadratic. Queries first score
          against per-block summary keys (coarse, O(T·T/bs)), keep the top
          `nla_cand_blocks` blocks (own block always in), then do fine top-K only
          within those blocks (O(T·cand·bs)). At bs≈sqrt(T) the total is
          O(T^1.5). Causality is exact (fine stage applies an absolute-position
          mask); only WHICH neighbours are considered is approximate — measured
          cosine to the exact path is ~0.96–0.99 at T≤512, and the score-op
          reduction grows with T (≈1.4× at T=100, ≈4.3× at T=512).
    """

    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        d, dc, ds = cfg.d_model, cfg.d_c, cfg.d_state
        self.H = cfg.nla_heads
        self.d_v = d // self.H                       # per-head value dim
        self.d_k = dc                                # per-head key/query dim
        self.compress = nn.Linear(d, dc)             # c_j — the only cached tensor
        self.key = nn.Linear(dc, self.H * self.d_k, bias=False)   # H routing keys from c_j
        self.value = nn.Linear(dc, d, bias=False)                # H values (concat = d) from c_j
        self.query = nn.Linear(d + ds, self.H * self.d_k)        # H routing queries from (x_t, s_t)
        self.gate = nn.Linear(ds, d)                 # liquid output gate
        self.out = nn.Linear(d, d)
        self.state = LiquidStateScan(d, ds)
        self.scale = 1.0 / math.sqrt(self.d_k)

    # ------------------------------------------------------------------ train
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d) → (out: (B, T, d), s: (B, T, d_state)). Multi-head: each
        head routes to its own top-K neighbours over the shared cached c_j."""
        B, T, _ = x.shape
        H, dk, dv = self.H, self.d_k, self.d_v

        s = self.state(x)                                    # (B, T, ds)
        c = self.compress(x)                                 # (B, T, dc)
        k = self.key(c).view(B, T, H, dk).transpose(1, 2)    # (B, H, T, dk)
        v = self.value(c).view(B, T, H, dv).transpose(1, 2)  # (B, H, T, dv)
        q = self.query(torch.cat([x, s], dim=-1)).view(B, T, H, dk).transpose(1, 2)

        bs = self.cfg.nla_block_size
        if bs and T > 2 * bs:
            agg = self._aggregate_blocksparse(q, k, v)       # O(T^1.5) approx
        else:
            agg = self._aggregate_dense(q, k, v)             # O(T²) exact

        agg = agg.transpose(1, 2).reshape(B, T, H * dv)      # (B, T, d)
        out = self.out(agg * torch.sigmoid(self.gate(s)))
        return out, s

    def _route_noise(self, scores: torch.Tensor) -> torch.Tensor:
        if self.training and self.cfg.nla_route_noise > 0:
            u = torch.rand_like(scores).clamp_min(1e-9)
            scores = scores - self.cfg.nla_route_noise * torch.log(-torch.log(u))
        return scores

    def _aggregate_dense(self, q, k, v):
        """Exact O(T²): full score matrix, causal top-K, straight-through."""
        B, H, T, dk = q.shape
        K = min(self.cfg.top_k, T)
        scores = torch.einsum("bhtd,bhjd->bhtj", q, k) * self.scale   # (B, H, T, T)
        scores = self._route_noise(scores)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        topv, topi = scores.topk(K, dim=-1)                  # (B, H, T, K) — per head
        sparse = torch.full_like(scores, float("-inf")).scatter(-1, topi, topv)
        alpha = torch.nan_to_num(F.softmax(sparse, dim=-1))
        if self.training:
            # straight-through: sparse forward, dense-softmax gradient so
            # non-selected positions still learn (per head).
            dense = torch.nan_to_num(F.softmax(scores, dim=-1))
            alpha = alpha.detach() + dense - dense.detach()
        return torch.einsum("bhtj,bhjd->bhtd", alpha, v)     # (B, H, T, dv)

    def _aggregate_blocksparse(self, q, k, v):
        """Sub-quadratic two-stage top-K. Blocks of size bs; each query picks
        `cand` candidate blocks by block-summary score (coarse, O(T·T/bs)), then
        does fine top-K only inside those blocks (O(T·cand·bs)). Own block is
        always a candidate (locality/self). Exact absolute-position causal mask
        is applied at the fine stage, so causality is preserved exactly; only
        WHICH neighbours are considered is approximated."""
        B, H, T, dk = q.shape
        dv = v.shape[-1]
        bs = self.cfg.nla_block_size
        nb = (T + bs - 1) // bs
        Tp = nb * bs                                         # padded length
        pad = Tp - T
        device = q.device

        kf = F.pad(k, (0, 0, 0, pad))                        # (B,H,Tp,dk)
        vf = F.pad(v, (0, 0, 0, pad))
        # block-summary keys: mean of real keys per block (padded slots -> 0,
        # divided by real count so padding doesn't dilute the mean)
        kb = kf.view(B, H, nb, bs, dk)
        real = torch.ones(Tp, device=device)
        real[T:] = 0.0
        cnt = real.view(nb, bs).sum(-1).clamp_min(1.0)       # (nb,)
        bk = kb.sum(3) / cnt.view(1, 1, nb, 1)               # (B,H,nb,dk)

        # coarse block scores + causal block mask (query block qb sees blocks<=qb)
        bsc = torch.einsum("bhtd,bhnd->bhtn", q, bk) * self.scale   # (B,H,T,nb)
        qb = (torch.arange(T, device=device) // bs)          # (T,) query's block
        blk = torch.arange(nb, device=device)
        block_ok = blk.view(1, nb) <= qb.view(T, 1)          # (T,nb) causal blocks
        bsc = bsc.masked_fill(~block_ok.view(1, 1, T, nb), float("-inf"))
        # force own block to always be selected: bump its score to +inf
        own = F.one_hot(qb, nb).bool().view(1, 1, T, nb)
        bsc = bsc.masked_fill(own, float("inf"))

        cand = min(self.cfg.nla_cand_blocks, nb)
        cblocks = bsc.topk(cand, dim=-1).indices             # (B,H,T,cand) block ids

        # expand candidate blocks -> candidate positions, then gather fine k/v
        C = cand * bs
        posidx = (cblocks.unsqueeze(-1) * bs
                  + torch.arange(bs, device=device).view(1, 1, 1, 1, bs))  # (B,H,T,cand,bs)
        posidx = posidx.reshape(B, H, T, C)                  # (B,H,T,C) positions in [0,Tp)

        BH = B * H
        kf_flat = kf.reshape(BH, Tp, dk)
        vf_flat = vf.reshape(BH, Tp, dv)
        idx_flat = posidx.reshape(BH, T * C)
        kg = kf_flat.gather(1, idx_flat.unsqueeze(-1).expand(BH, T * C, dk)).view(B, H, T, C, dk)
        vg = vf_flat.gather(1, idx_flat.unsqueeze(-1).expand(BH, T * C, dv)).view(B, H, T, C, dv)

        # fine scores over gathered candidates
        fsc = (q.unsqueeze(3) * kg).sum(-1) * self.scale     # (B,H,T,C)
        fsc = self._route_noise(fsc)
        # exact causal mask on gathered absolute positions (also masks padding,
        # since padded positions have absolute index >= T > any query t)
        t_abs = torch.arange(T, device=device).view(1, 1, T, 1)
        fsc = fsc.masked_fill(posidx.view(B, H, T, C) > t_abs, float("-inf"))

        K = min(self.cfg.top_k, C)
        topv, topi = fsc.topk(K, dim=-1)                     # (B,H,T,K)
        sparse = torch.full_like(fsc, float("-inf")).scatter(-1, topi, topv)
        alpha = torch.nan_to_num(F.softmax(sparse, dim=-1))
        if self.training:
            dense = torch.nan_to_num(F.softmax(fsc, dim=-1))
            alpha = alpha.detach() + dense - dense.detach()
        return torch.einsum("bhtc,bhtcd->bhtd", alpha, vg)   # (B,H,T,dv)

    # -------------------------------------------------------------- inference
    def init_cache(self, batch: int, device=None, dtype=None) -> Dict[str, torch.Tensor]:
        return {
            "c": torch.zeros(batch, 0, self.cfg.d_c, device=device, dtype=dtype),
            "s": torch.zeros(batch, self.cfg.d_state, device=device, dtype=dtype),
        }

    def step(self, x_t: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        """x_t: (B, d). Mutates cache; returns out_t: (B, d).

        Cache holds c (B, t, d_c) and s (B, d_state) only — no KV pairs, and its
        size is independent of the head count.
        """
        B = x_t.shape[0]
        H, dk, dv = self.H, self.d_k, self.d_v
        s = self.state.step(x_t, cache["s"])
        c_t = self.compress(x_t)
        cache["c"] = torch.cat([cache["c"], c_t.unsqueeze(1)], dim=1)
        cache["s"] = s

        c = cache["c"]                                       # (B, t, dc)
        t = c.shape[1]
        k = self.key(c).view(B, t, H, dk).transpose(1, 2)    # (B, H, t, dk)
        v = self.value(c).view(B, t, H, dv).transpose(1, 2)  # (B, H, t, dv)
        q = self.query(torch.cat([x_t, s], dim=-1)).view(B, H, dk)   # (B, H, dk)

        scores = torch.einsum("bhd,bhjd->bhj", q, k) * self.scale    # (B, H, t)
        K = min(self.cfg.top_k, t)
        topv, topi = scores.topk(K, dim=-1)                  # (B, H, K)
        alpha = F.softmax(topv, dim=-1)                      # (B, H, K)
        v_sel = torch.gather(v, 2, topi.unsqueeze(-1).expand(-1, -1, -1, dv))  # (B,H,K,dv)
        agg = (alpha.unsqueeze(-1) * v_sel).sum(dim=2)       # (B, H, dv)
        agg = agg.reshape(B, H * dv)                         # (B, d)
        return self.out(agg * torch.sigmoid(self.gate(s)))

    @staticmethod
    def cache_bytes_per_token(cfg: O1AntiConfig, dtype_bytes: int = 2) -> int:
        """Inference memory per token per layer, vs 2*d_model for a KV cache."""
        return cfg.d_c * dtype_bytes
