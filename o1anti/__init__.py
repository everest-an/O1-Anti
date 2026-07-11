"""
o1anti — Anti-Transformer architecture (MTLNN v2 blueprint).

Three pillars, each attacking one Transformer cost root:

  1. nla.py          — Neural Liquid Adjacency: replaces O(n^2) attention +
                       KV cache with top-K dynamic sparse connections over
                       compressed per-position states (O(n*d_c) memory).
  2. module_graph.py — Context-routed Neural Module Graph: a global router
                       picks one module path per input, so only a small
                       fraction of total parameters is ever activated.
  3. generation.py   — Liquid state-transition generation: flow-matching
                       neural-ODE semantic skeleton + parallel mask-predict
                       decoding, replacing token-by-token autoregression.
"""

from .config import O1AntiConfig
from .nla import NeuralLiquidAdjacency, LiquidStateScan
from .module_graph import ContextEncoder, GlobalRouter, ModuleLibrary, NeuralModule
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
