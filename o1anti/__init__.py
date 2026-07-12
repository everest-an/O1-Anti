"""
o1anti — Anti-Transformer architecture (MTLNN v2 blueprint).

Three pillars, each attacking one Transformer cost root:

  1. nla.py          — Neural Liquid Adjacency: replaces O(n^2) attention +
                       KV cache with top-K dynamic sparse connections over
                       compressed per-position states (O(n*d_c) memory).
                       Multi-head (cfg.nla_heads) for attention-style mixing;
                       cache size stays independent of head count.
  2. module_graph.py — Neural Module Graph, two routing granularities
                       (cfg.routing_granularity):
                         "global" — one module path (NeuralModule, NLA+FFN)
                                    per whole input (ContextEncoder + Global-
                                    Router + ModuleLibrary); only path_len of
                                    n_modules run per input.
                         "token"  — dense NLA backbone + per-token routed
                                    MoE-FFN (TokenMoETrunk/MoEBlock/
                                    MoEFeedForward); finer-grained, used for
                                    language modeling.
  3. generation.py   — Liquid state-transition generation, non-autoregressive.
                       Stage 1 (cfg.skeleton_mode) produces a compact semantic
                       skeleton: "regress" (deterministic prior, default),
                       "flow" (flow-matching neural ODE), or "discrete"
                       (product-quantized VQ codes). Stage 2 (ParallelDecoder)
                       emits all tokens at once via mask-predict.
"""

from .config import O1AntiConfig
from .nla import NeuralLiquidAdjacency, LiquidStateScan
from .module_graph import (
    ContextEncoder,
    GlobalRouter,
    ModuleLibrary,
    NeuralModule,
    MoEBlock,
    MoEFeedForward,
    TokenMoETrunk,
)
from .generation import (
    SkeletonEncoder,
    SkeletonGenerator,
    SkeletonPrior,
    VectorQuantizer,
    ParallelDecoder,
)
from .model import O1AntiModel
from .losses import load_balance_loss, state_continuity_loss, flow_matching_loss

__all__ = [
    "O1AntiConfig",
    "NeuralLiquidAdjacency",
    "LiquidStateScan",
    "ContextEncoder",
    "GlobalRouter",
    "ModuleLibrary",
    "NeuralModule",
    "MoEBlock",
    "MoEFeedForward",
    "TokenMoETrunk",
    "SkeletonEncoder",
    "SkeletonGenerator",
    "SkeletonPrior",
    "VectorQuantizer",
    "ParallelDecoder",
    "O1AntiModel",
    "load_balance_loss",
    "state_continuity_loss",
    "flow_matching_loss",
]
