"""
p3_parallel_decode.py — P3 go/no-go: parallel decode vs autoregressive.

Question: can the two-stage liquid generator (few-step ODE skeleton +
fixed-round mask-predict) reconstruct a target sequence at a quality
comparable to token-by-token decoding, while using a constant number of
forward passes instead of one-per-token?

Task: conditional sequence reconstruction. Given a prompt, the target is a
deterministic function of it (a fixed permutation + shift), so quality is
measurable by exact token match. We compare:

  autoregressive : a small causal LM decodes left-to-right, `length` passes.
  parallel       : O1Anti SkeletonGenerator + ParallelDecoder,
                   ode_steps + decode_iters passes total.

Reported: token accuracy for both, and the pass-count / wall-clock speedup.
Because both models are tiny the wall-clock ratio understates the asymptotic
win (which scales with `length`); the pass-count ratio is the honest metric.

Run:  python experiments/p3_parallel_decode.py --steps 1500 --length 48
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import O1AntiConfig
from o1anti.generation import (
    ParallelDecoder,
    SkeletonEncoder,
    SkeletonGenerator,
    SkeletonPrior,
    VectorQuantizer,
)
from o1anti.module_graph import ContextEncoder

VOCAB = 32


def make_pair(batch, prompt_len, length, device, perm):
    """target is a deterministic, learnable function of the prompt: the prompt
    tiled to `length` then shifted by a fixed per-block constant. Both decoders
    can solve this, so accuracy isolates the decode *mechanism*, not task luck."""
    prompt = torch.randint(2, VOCAB, (batch, prompt_len), device=device)
    idx = torch.arange(length, device=device) % prompt_len
    gathered = prompt[:, idx]                              # tile prompt → length
    block_shift = (torch.arange(length, device=device) // prompt_len) % 3
    target = 2 + (gathered - 2 + block_shift) % (VOCAB - 2)
    return prompt, target


# ----------------------------------------------------------------- baseline
class ARDecoder(nn.Module):
    def __init__(self, cfg, prompt_len, length):
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(prompt_len + length, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d, cfg.n_dec_heads, cfg.d_ff, batch_first=True)
            for _ in range(cfg.n_dec_layers)
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.embed.weight
        self.prompt_len = prompt_len

    def forward(self, seq):
        h = self.embed(seq) + self.pos[: seq.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(seq.shape[1], device=seq.device)
        for b in self.blocks:
            h = b(h, src_mask=mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, prompt, length):
        seq = prompt.clone()
        for _ in range(length):
            nxt = self(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, self.prompt_len :]


# --------------------------------------------------------------- o1anti gen
class ParallelGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB, cfg.d_model)
        self.skel_encoder = SkeletonEncoder(cfg)
        self.decoder = ParallelDecoder(cfg, self.embed)
        if cfg.skeleton_mode == "regress":
            self.prior = SkeletonPrior(cfg, out_dim=cfg.d_model)
        elif cfg.skeleton_mode == "discrete":
            self.vq = VectorQuantizer(cfg)
            self.prior = SkeletonPrior(cfg, out_dim=cfg.vq_groups * cfg.codebook_size)
        else:
            self.skeleton = SkeletonGenerator(cfg)

    def loss(self, prompt, target):
        mem = self.embed(prompt)
        skel = self.skel_encoder(self.embed(target))
        if self.cfg.skeleton_mode == "regress":
            stage1 = F.mse_loss(self.prior(mem), skel.detach())
            std = skel.detach().std(dim=-1, keepdim=True)   # noise-robust decoder
            skel = skel + self.cfg.skel_noise * std * torch.randn_like(skel)
        elif self.cfg.skeleton_mode == "discrete":
            skel, codes, stage1 = self.vq(skel)          # codes: (B, L, G)
            G, K = self.cfg.vq_groups, self.cfg.codebook_size
            logits = self.prior(mem).view(*codes.shape[:2], G, K)
            stage1 = stage1 + F.cross_entropy(logits.reshape(-1, K), codes.detach().reshape(-1))
        else:
            from o1anti.losses import flow_matching_loss
            stage1 = flow_matching_loss(self.skeleton, skel, mem)
        ratio = torch.rand(target.shape[0], 1, device=target.device)  # CMLM masking
        masked = torch.rand(target.shape, device=target.device) < ratio
        masked |= ~masked.any(dim=-1, keepdim=True)
        dec = F.cross_entropy(self.decoder.logits(target, masked, skel)[masked], target[masked])
        return stage1 + dec

    @torch.no_grad()
    def generate(self, prompt, length):
        mem = self.embed(prompt)
        if self.cfg.skeleton_mode == "regress":
            skel = self.prior(mem)
        elif self.cfg.skeleton_mode == "discrete":
            G, K = self.cfg.vq_groups, self.cfg.codebook_size
            logits = self.prior(mem).view(mem.shape[0], self.cfg.skel_len, G, K)
            skel = self.vq.embed_codes(logits.argmax(dim=-1))
        else:
            skel = self.skeleton.sample(mem)
        return self.decoder.mask_predict(skel, length)

    @torch.no_grad()
    def generate_teacher_skeleton(self, target, length):
        """Diagnostic: decode from the encoded GT skeleton to isolate stage 2."""
        skel_gt = self.skel_encoder(self.embed(target))
        if getattr(self.cfg, "skeleton_mode", "regress") == "discrete":
            skel_gt, _, _ = self.vq(skel_gt)
        return self.decoder.mask_predict(skel_gt, length)


def acc(pred, target):
    return (pred == target).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--prompt_len", type=int, default=12)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--skel_len", type=int, default=0, help="0 = length//3")
    ap.add_argument("--skeleton_mode", choices=["regress", "discrete", "flow"], default="regress")
    ap.add_argument("--ode_steps", type=int, default=8)
    ap.add_argument("--decode_iters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-min", type=float, default=None,
                    help="fail if end-to-end generated acc < this (regression anchor)")
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    perm = torch.randperm(max(args.length, args.prompt_len), device=device)
    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=args.prompt_len + args.length,
        skel_len=args.skel_len or max(args.length // 3, 4), ode_steps=args.ode_steps,
        decode_iters=args.decode_iters, n_dec_layers=2, n_dec_heads=4,
        d_ff=2 * args.d_model, d_c=24, d_state=48, skeleton_mode=args.skeleton_mode,
    )
    print(f"P3 parallel-decode | device={device} length={args.length} "
          f"mode={args.skeleton_mode} decode_iters={args.decode_iters}")

    # --- autoregressive baseline ---
    torch.manual_seed(args.seed)
    ar = ARDecoder(cfg, args.prompt_len, args.length).to(device)
    opt = torch.optim.AdamW(ar.parameters(), lr=2e-3)
    for step in range(args.steps):
        prompt, target = make_pair(args.batch, args.prompt_len, args.length, device, perm)
        seq = torch.cat([prompt, target], dim=1)
        logits = ar(seq)[:, args.prompt_len - 1 : -1]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(args.steps // 5, 1) == 0:
            print(f"  [ar] step {step+1}/{args.steps} loss {loss.item():.4f}")

    # --- parallel generator ---
    torch.manual_seed(args.seed)
    pg = ParallelGen(cfg).to(device)
    opt = torch.optim.AdamW(pg.parameters(), lr=2e-3)
    for step in range(args.steps):
        prompt, target = make_pair(args.batch, args.prompt_len, args.length, device, perm)
        loss = pg.loss(prompt, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(args.steps // 5, 1) == 0:
            print(f"  [parallel] step {step+1}/{args.steps} loss {loss.item():.4f}")

    # --- eval ---
    ar.eval(); pg.eval()
    prompt, target = make_pair(256, args.prompt_len, args.length, device, perm)
    with torch.no_grad():
        t0 = time.time(); ar_pred = ar.generate(prompt, args.length); ar_t = time.time() - t0
        t0 = time.time(); pg_pred = pg.generate(prompt, args.length); pg_t = time.time() - t0
        pg_teacher = pg.generate_teacher_skeleton(target, args.length)

    ar_passes = args.length
    stage1_passes = 1 if args.skeleton_mode == "discrete" else args.ode_steps
    pg_passes = stage1_passes + args.decode_iters
    print(f"\n{'variant':<22}{'tok acc':>9}{'passes':>9}{'wall ms':>10}")
    print(f"{'autoregressive':<22}{acc(ar_pred, target):>9.3f}{ar_passes:>9}{ar_t*1000:>10.1f}")
    print(f"{'parallel (generated)':<22}{acc(pg_pred, target):>9.3f}{pg_passes:>9}{pg_t*1000:>10.1f}")
    print(f"{'parallel (GT skeleton)':<22}{acc(pg_teacher, target):>9.3f}{args.decode_iters:>9}{'—':>10}")
    ar_acc = acc(ar_pred, target)
    gen_acc = acc(pg_pred, target)
    teach_acc = acc(pg_teacher, target)
    stage2 = "PROVEN" if teach_acc >= 0.9 * ar_acc else "partial"
    s1_name = {"regress": "deterministic prior", "discrete": "VQ code prior",
               "flow": "flow-matching"}[args.skeleton_mode]
    stage1 = "PROVEN" if gen_acc >= 0.9 * ar_acc else (
        "ok" if gen_acc >= 0.7 * ar_acc else "OPEN — generated skeleton far from encoder latent")
    print(f"\npass-count speedup: {ar_passes/pg_passes:.1f}x   "
          f"wall-clock speedup: {ar_t/max(pg_t,1e-6):.1f}x")
    print("\nverdict:")
    print(f"  end-to-end (generated skeleton): {gen_acc:.3f} in {pg_passes} passes "
          f"vs AR {ar_acc:.3f} in {ar_passes} passes -> {stage1} "
          f"({ar_passes/pg_passes:.0f}x fewer passes)")
    print(f"  stage 1 ({s1_name}): generated {gen_acc:.3f} vs GT-skeleton ceiling {teach_acc:.3f}")
    print(f"  stage 2 (parallel decoder): {teach_acc:.3f} in {args.decode_iters} "
          f"passes -> {stage2}")

    if args.assert_min is not None:
        assert gen_acc >= args.assert_min, (
            f"end-to-end generated acc {gen_acc:.3f} < {args.assert_min}")
        assert pg_passes < ar_passes, f"parallel passes {pg_passes} !< AR {ar_passes}"
        print(f"\n[assert] generated acc {gen_acc:.3f} >= {args.assert_min} and "
              f"{pg_passes} < {ar_passes} passes  OK")


if __name__ == "__main__":
    main()
