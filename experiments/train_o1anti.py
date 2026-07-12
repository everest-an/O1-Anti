"""
train_o1anti.py — E8: real-text language modeling (pillars 1+2 on English).

P1–P4 validated the architecture on synthetic tasks. E8 is the first real-text
evidence: byte-level causal language modeling on WikiText-2, comparing the
O1-Anti trunk (context-routed modules with NLA) against a matched dense
Transformer, reported as validation bits-per-byte (BPB).

This trains the *understanding* path (`O1AntiModel.forward`, pillars 1+2). Only
the LM-relevant parameters participate; the generation stack (pillar 3) is not
built here (`skeleton_mode` is irrelevant to LM). The dense baseline is sized so
its parameter count matches the O1-Anti trunk's *active* LM params, so BPB is a
fair like-for-like at equal compute.

Byte-level (vocab 256) keeps the dependency surface to torch alone at eval time;
`datasets` is used only to load the cached WikiText corpus (or pass --data FILE).

Run (CPU-feasible):
  python experiments/train_o1anti.py --steps 1500 --seq 128 --d_model 128
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import O1AntiConfig, O1AntiModel

VOCAB = 256  # byte-level


# --------------------------------------------------------------------- data
def load_text(source: str) -> tuple[str, str]:
    """Return (train_text, val_text). source='wikitext' uses the cached corpus;
    otherwise it is treated as a path to a UTF-8 text file (90/10 split)."""
    if source == "wikitext":
        from datasets import load_dataset
        tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        va = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        join = lambda ds: "".join(t for t in ds["text"] if t.strip())
        return join(tr), join(va)
    text = Path(source).read_text(encoding="utf-8")
    cut = int(len(text) * 0.9)
    return text[:cut], text[cut:]


def to_bytes(text: str, device) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long, device=device)


def get_batch(data: torch.Tensor, batch: int, seq: int, gen: torch.Generator):
    ix = torch.randint(0, data.numel() - seq - 1, (batch,), generator=gen, device=data.device)
    x = torch.stack([data[i : i + seq] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + seq] for i in ix])
    return x, y


# ---------------------------------------------------------- dense baseline
class DenseLM(nn.Module):
    """Standard causal Transformer LM, matched by width/depth to O1-Anti active."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, seq: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Parameter(torch.zeros(seq, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, y=None):
        h = self.embed(x) + self.pos[: x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        for b in self.blocks:
            h = b(h, src_mask=mask)
        logits = self.head(self.norm(h))
        loss = None if y is None else F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        return logits, loss


# ------------------------------------------------------------ param count
def lm_active_params(model, cfg) -> int:
    """Params actually used by the LM forward path: exclude the generation
    stack (pillar 3, unused in forward) and count only path_len active modules.
    Dedupes the tied embed/head weight."""
    gen_mods = [getattr(model, n, None) for n in
                ("skel_encoder", "decoder", "prior", "vq", "skeleton")]
    gen_ids = {id(p) for m in gen_mods if m is not None for p in m.parameters()}
    kept = {id(p): p for p in model.parameters() if id(p) not in gen_ids}  # dedup tied
    total = sum(p.numel() for p in kept.values())
    per_module = sum(p.numel() for p in model.library.modules_list[0].parameters())
    return total - per_module * (cfg.n_modules - cfg.path_len)  # only active modules


# ------------------------------------------------------------------- train
def evaluate(model, data, batch, seq, iters, device, is_o1anti):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    tot = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, batch, seq, gen)
            if is_o1anti:
                out = model(x, labels=y)
                tot += out.lm_loss.item()
            else:
                tot += model(x, y)[1].item()
    model.train()
    return tot / iters


def train(model, data_tr, data_va, args, device, is_o1anti):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(data_tr, args.batch, args.seq, gen)
        if is_o1anti:
            loss = model(x, labels=y).loss
        else:
            loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(args.steps // 6, 1) == 0:
            val = evaluate(model, data_va, args.batch, args.seq, 20, device, is_o1anti)
            print(f"  step {step+1}/{args.steps}  train {loss.item():.3f}  "
                  f"val nats {val:.3f}  bpb {val/math.log(2):.3f}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext", help="'wikitext' or a text file path")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_modules", type=int, default=6)
    ap.add_argument("--path_len", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-max-bpb", type=float, default=None)
    ap.add_argument("--skip-dense", action="store_true", help="train only O1-Anti")
    args = ap.parse_args()
    device = args.device

    print(f"E8 real-text LM | device={device} data={args.data} seq={args.seq} "
          f"d_model={args.d_model} steps={args.steps}")
    tr_txt, va_txt = load_text(args.data)
    data_tr, data_va = to_bytes(tr_txt, device), to_bytes(va_txt, device)
    print(f"corpus: {data_tr.numel():,} train bytes / {data_va.numel():,} val bytes")

    # --- O1-Anti trunk ---
    torch.manual_seed(args.seed)
    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=args.seq,
        n_modules=args.n_modules, path_len=args.path_len, top_k=args.top_k,
        d_ff=4 * args.d_model, d_c=args.d_model // 4, d_state=args.d_model // 2,
        d_ctx=args.d_model, skeleton_mode="regress",
    )
    o1 = O1AntiModel(cfg).to(device)
    o1_active = lm_active_params(o1, cfg)
    print(f"\n[O1-Anti] active LM params ~{o1_active:,} "
          f"(path {args.path_len}/{args.n_modules} modules; generation stack excluded)")
    o1_time = train(o1, data_tr, data_va, args, device, is_o1anti=True)
    o1_bpb = evaluate(o1, data_va, args.batch, args.seq, 40, device, True) / math.log(2)

    results = {"O1-Anti": (o1_bpb, o1_active, o1_time)}

    # --- matched dense Transformer ---
    if not args.skip_dense:
        # size depth so dense params ≈ O1-Anti active LM params
        d = args.d_model
        per_layer = 4 * d * d + 2 * d * (4 * d)  # attn(qkvo) + ffn, rough
        n_layers = max(2, round((o1_active - VOCAB * d) / per_layer))
        torch.manual_seed(args.seed)
        dense = DenseLM(d, n_layers, n_heads=4, d_ff=4 * d, seq=args.seq).to(device)
        dpar = sum(p.numel() for p in dense.parameters())
        print(f"\n[Dense] {n_layers} layers, {dpar:,} params (matched to O1-Anti active)")
        d_time = train(dense, data_tr, data_va, args, device, is_o1anti=False)
        d_bpb = evaluate(dense, data_va, args.batch, args.seq, 40, device, False) / math.log(2)
        results["Dense"] = (d_bpb, dpar, d_time)

    print(f"\n{'model':<10}{'val bits/byte':>15}{'params':>14}{'train s':>10}")
    for name, (bpb, par, t) in results.items():
        print(f"{name:<10}{bpb:>15.3f}{par:>14,}{t:>10.1f}")
    if "Dense" in results:
        gap = (results['O1-Anti'][0] - results['Dense'][0]) / results['Dense'][0] * 100
        print(f"\nO1-Anti vs dense: {gap:+.1f}% BPB at matched active params "
              f"({'competitive' if abs(gap) < 15 else 'gap'} on real text)")

    if args.assert_max_bpb is not None:
        assert o1_bpb <= args.assert_max_bpb, f"O1-Anti val BPB {o1_bpb:.3f} > {args.assert_max_bpb}"
        print(f"\n[assert] O1-Anti val BPB {o1_bpb:.3f} <= {args.assert_max_bpb}  OK")


if __name__ == "__main__":
    main()
