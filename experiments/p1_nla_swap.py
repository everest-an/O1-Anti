"""
p1_nla_swap.py — P1 go/no-go: Neural Liquid Adjacency vs dense attention.

Two matched 2-block causal LMs are trained on the selective-copy task
(input: N random tokens, separator, then the model must reproduce them):

  baseline : dense softmax attention (nn.MultiheadAttention, causal) + FFN
  nla      : NeuralLiquidAdjacency + FFN  (same d_model / d_ff / depth)

Reported: held-out recall token accuracy, final loss, parameter counts, and
the analytic inference-cache footprint per token per layer (KV vs c_j).

Run:  python experiments/p1_nla_swap.py --steps 400 --seq 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import NeuralLiquidAdjacency, O1AntiConfig

VOCAB = 32          # tokens 2..31 are content; 0=pad, 1=separator
SEP = 1


def make_batch(batch: int, n_copy: int, device) -> torch.Tensor:
    """[content x n_copy, SEP, content x n_copy] — recall across the SEP."""
    content = torch.randint(2, VOCAB, (batch, n_copy), device=device)
    sep = torch.full((batch, 1), SEP, device=device, dtype=torch.long)
    return torch.cat([content, sep, content], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, h):
        x = self.norm1(h)
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        h = h + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        return h + self.ffn(self.norm2(h))


class NLABlock(nn.Module):
    def __init__(self, cfg: O1AntiConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d)
        self.nla = NeuralLiquidAdjacency(cfg)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, d))

    def forward(self, h):
        mixed, _ = self.nla(self.norm1(h))
        h = h + mixed
        return h + self.ffn(self.norm2(h))


class TinyLM(nn.Module):
    def __init__(self, kind: str, cfg: O1AntiConfig, n_blocks: int = 2):
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(cfg.max_seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        blk = (lambda: NLABlock(cfg)) if kind == "nla" else (lambda: DenseBlock(d, cfg.d_ff))
        self.blocks = nn.ModuleList(blk() for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        h = self.embed(ids) + self.pos[: ids.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def run(kind: str, args, device) -> dict:
    torch.manual_seed(args.seed)
    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=2 * args.seq + 2,
        d_c=args.d_c, d_state=64, top_k=args.top_k, d_ff=4 * args.d_model,
        nla_route_noise=args.route_noise,
    )
    model = TinyLM(kind, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n_copy = args.seq
    t0 = time.time()
    for step in range(args.steps):
        ids = make_batch(args.batch, n_copy, device)
        logits = model(ids)
        # loss only on the recall half (predict token t+1 from t)
        tgt = ids[:, n_copy + 1 :]                 # content after the SEP
        pred = logits[:, n_copy : n_copy + n_copy] # positions SEP .. end-1
        loss = F.cross_entropy(pred.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 5, 1) == 0:
            print(f"  [{kind}] step {step+1}/{args.steps}  loss {loss.item():.4f}")
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        ids = make_batch(256, n_copy, device)
        logits = model(ids)
        tgt = ids[:, n_copy + 1 :]
        pred = logits[:, n_copy : n_copy + n_copy].argmax(-1)
        acc = (pred == tgt).float().mean().item()
        loss = F.cross_entropy(
            logits[:, n_copy : n_copy + n_copy].reshape(-1, VOCAB), tgt.reshape(-1)
        ).item()

    cache = (
        cfg.d_c * 2 if kind == "nla" else 2 * args.d_model * 2  # fp16 bytes/token/layer
    )
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "acc": acc, "loss": loss, "train_s": train_s, "cache_B_per_tok": cache,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=32, help="content tokens to recall")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_c", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--route_noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"P1 NLA-swap | device={args.device} seq={args.seq} steps={args.steps}")
    results = {kind: run(kind, args, args.device) for kind in ("baseline", "nla")}

    print(f"\n{'variant':<10}{'params':>9}{'recall acc':>12}{'loss':>9}"
          f"{'train s':>9}{'cache B/tok/layer':>19}")
    for kind, r in results.items():
        print(f"{kind:<10}{r['params']:>9,}{r['acc']:>12.3f}{r['loss']:>9.4f}"
              f"{r['train_s']:>9.1f}{r['cache_B_per_tok']:>19}")
    ratio = results["baseline"]["cache_B_per_tok"] / results["nla"]["cache_B_per_tok"]
    print(f"\ninference cache reduction: {ratio:.1f}x "
          f"(go/no-go: NLA recall acc within a few points of baseline)")


if __name__ == "__main__":
    main()
