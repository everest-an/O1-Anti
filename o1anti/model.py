"""
model.py — full O1-Anti assembly.

Understanding path (also the causal LM used for pretraining):
    tokens → embed → ContextEncoder → GlobalRouter → routed module path
           → norm → tied LM head

Generation path (two-stage, non-autoregressive):
    prompt → embed → stage-1 skeleton generator → ParallelDecoder
           → (mask-predict, fixed rounds) → tokens

    Stage-1 generator depends on cfg.skeleton_mode:
      "regress"  (default) — SkeletonPrior regresses prompt→skeleton, 1 pass.
      "flow"               — SkeletonGenerator flow-matching ODE, ode_steps.
      "discrete"           — VectorQuantizer codes + SkeletonPrior, 1 pass.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import O1AntiConfig
from .generation import (
    ParallelDecoder,
    SkeletonEncoder,
    SkeletonGenerator,
    SkeletonPrior,
    VectorQuantizer,
)
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
        self.skel_encoder = SkeletonEncoder(cfg)
        self.decoder = ParallelDecoder(cfg, self.embed)
        if cfg.skeleton_mode == "regress":
            self.prior = SkeletonPrior(cfg, out_dim=cfg.d_model)
        elif cfg.skeleton_mode == "discrete":
            self.vq = VectorQuantizer(cfg)
            self.prior = SkeletonPrior(cfg, out_dim=cfg.codebook_size)
        else:
            self.skeleton = SkeletonGenerator(cfg)

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
        """Stage-1 skeleton generation + stage-2 masked parallel decoding.

        The stage-1 term depends on cfg.skeleton_mode:
          "regress"  (default): SkeletonEncoder makes a target skeleton; the
                     prior regresses prompt→skeleton (MSE); the decoder trains on
                     a noise-perturbed skeleton so it tolerates regression error.
          "discrete": encoder → VQ codes; the prior predicts codes from the
                     prompt (cross-entropy); decoder reconstructs from the codes.
          "flow":    flow-matching a continuous latent skeleton from noise.
        Stage-2 is always CMLM cross-entropy over the masked positions.
        """
        mem = self.embed(prompt_ids)                     # prompt memory (B,T,d)
        h_t = self.embed(target_ids)
        skel = self.skel_encoder(h_t)                    # learned latent skeleton

        if self.cfg.skeleton_mode == "regress":
            pred = self.prior(mem)                        # (B, skel_len, d)
            stage1 = F.mse_loss(pred, skel.detach())
            # robustness: decode from a noise-perturbed skeleton so the decoder
            # tolerates the prior's regression error at inference. σ scales with
            # each slot's std so it matches the skeleton's dynamic range.
            std = skel.detach().std(dim=-1, keepdim=True)
            skel = skel + self.cfg.skel_noise * std * torch.randn_like(skel)
        elif self.cfg.skeleton_mode == "discrete":
            skel, codes, stage1 = self.vq(skel)          # quantize + VQ loss
            prior_logits = self.prior(mem)               # (B, skel_len, K)
            stage1 = stage1 + F.cross_entropy(
                prior_logits.reshape(-1, self.cfg.codebook_size),
                codes.detach().reshape(-1),
            )
        else:
            stage1 = flow_matching_loss(self.skeleton, skel, mem)

        # CMLM masking: per-row mask ratio ~ U(0,1] so the decoder sees the
        # fully-masked regime it starts inference from, not just a fixed 50%.
        ratio = torch.rand(target_ids.shape[0], 1, device=target_ids.device)
        masked = torch.rand_like(target_ids, dtype=torch.float) < ratio
        masked |= ~masked.any(dim=-1, keepdim=True)  # at least one mask per row

        logits = self.decoder.logits(target_ids, masked, skel)
        dec = F.cross_entropy(
            logits[masked].reshape(-1, self.cfg.vocab_size),
            target_ids[masked].reshape(-1),
        )
        return stage1 + dec

    # ---------------------------------------------------------------- sampler
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        length: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Non-autoregressive. discrete: 1 (prior) + decode_iters passes;
        flow: ode_steps + decode_iters passes. Never `length` passes."""
        mem = self.embed(prompt_ids)
        if self.cfg.skeleton_mode == "regress":
            skel = self.prior(mem)                        # (B, skel_len, d)
        elif self.cfg.skeleton_mode == "discrete":
            codes = self.prior(mem).argmax(dim=-1)       # (B, skel_len)
            skel = self.vq.embed_codes(codes)
        else:
            skel = self.skeleton.sample(mem, generator=generator)
        return self.decoder.mask_predict(skel, length)

    def num_parameters(self, active_only: bool = False) -> int:
        if not active_only:
            return sum(p.numel() for p in self.parameters())
        per_module = sum(p.numel() for p in self.library.modules_list[0].parameters())
        total = sum(p.numel() for p in self.parameters())
        all_modules = per_module * self.cfg.n_modules
        return total - all_modules + per_module * self.cfg.path_len
