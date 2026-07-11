"""
model.py — full O1-Anti assembly.

Understanding path (also the causal LM used for pretraining):
    tokens → embed → ContextEncoder → GlobalRouter → routed module path
           → norm → tied LM head

Generation path (two-stage, non-autoregressive):
    prompt → trunk → ctx → SkeletonGenerator (few-step ODE from noise)
           → ParallelDecoder (mask-predict, fixed rounds) → tokens
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig
from .generation import ParallelDecoder, SkeletonGenerator
from .losses import flow_matching_loss, load_balance_loss
from .module_graph import ContextEncoder, GlobalRouter, ModuleLibrary


@dataclass
class O1AntiOutput:
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ctx: Optional[torch.Tensor] = None


class O1AntiModel(nn.Module):
    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.context = ContextEncoder(cfg)
        self.router = GlobalRouter(cfg)
        self.library = ModuleLibrary(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.skeleton = SkeletonGenerator(cfg)
        self.decoder = ParallelDecoder(cfg, self.embed)

    # ------------------------------------------------------------------ trunk
    def encode(self, input_ids: torch.Tensor):
        """→ (hidden (B,T,d), ctx (B,d_ctx), usage (M,), continuity scalar)."""
        h = self.embed(input_ids)
        ctx = self.context(h)
        w, usage = self.router(ctx)
        h, cont = self.library(h, w)
        return self.norm(h), ctx, usage, cont

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> O1AntiOutput:
        """Causal LM objective (labels = input_ids for standard next-token)."""
        h, ctx, usage, cont = self.encode(input_ids)
        logits = self.lm_head(h)
        out = O1AntiOutput(logits=logits, ctx=ctx)
        if labels is not None:
            lm = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.cfg.vocab_size),
                labels[:, 1:].reshape(-1),
            )
            aux = (
                self.cfg.load_balance_coef * load_balance_loss(usage)
                + self.cfg.state_continuity_coef * cont
            )
            out.lm_loss, out.aux_loss, out.loss = lm, aux, lm + aux
        return out

    # ------------------------------------------------------- generation train
    def generation_loss(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Stage-1 flow matching + stage-2 masked parallel decoding, both
        conditioned on the prompt context (teacher skeleton for stage 2)."""
        with torch.no_grad():
            h_p = self.embed(prompt_ids)
        ctx = self.context(h_p)

        h_t = self.embed(target_ids)
        fm = flow_matching_loss(self.skeleton, h_t, ctx, self.cfg.skel_len)

        skel_gt = SkeletonGenerator.pool_skeleton(h_t, self.cfg.skel_len).detach()
        masked = torch.rand_like(target_ids, dtype=torch.float) < 0.5
        masked |= ~masked.any(dim=-1, keepdim=True)  # at least one mask per row
        logits = self.decoder.logits(target_ids, masked, skel_gt)
        dec = F.cross_entropy(
            logits[masked].reshape(-1, self.cfg.vocab_size),
            target_ids[masked].reshape(-1),
        )
        return fm + dec

    # ---------------------------------------------------------------- sampler
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        length: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Non-autoregressive: ode_steps + decode_iters passes, not `length`."""
        h, ctx, _, _ = self.encode(prompt_ids)
        skel = self.skeleton.sample(ctx, generator=generator)
        return self.decoder.mask_predict(skel, length)

    def num_parameters(self, active_only: bool = False) -> int:
        if not active_only:
            return sum(p.numel() for p in self.parameters())
        per_module = sum(p.numel() for p in self.library.modules_list[0].parameters())
        total = sum(p.numel() for p in self.parameters())
        all_modules = per_module * self.cfg.n_modules
        return total - all_modules + per_module * self.cfg.path_len
