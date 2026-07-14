"""
e11_mqar_vs_ssm.py — the decisive test: NLA vs a real SSM (Mamba) on precise
associative recall, the task that separates "retrieval-capable" from
"state-blurred" architectures.

Motivation. NLA's claimed unique niche is: a small O(n·d_c) inference cache (like
an SSM, unlike attention's O(n·d_model) KV) BUT with precise content retrieval
(like attention, unlike an SSM). Multi-Query Associative Recall (MQAR) is the
canonical probe for exactly this axis — the Zoology / Based line (Arora et al.
2023-24) showed SSMs like Mamba degrade on associative recall as the number of
key→value pairs grows past what their fixed recurrent state can hold, while
attention holds. If NLA is genuinely "SSM-cheap but attention-precise", it should
track attention here and beat Mamba, and the gap should widen with more pairs.

Task (MQAR). Sequence = D pairs [k1 v1 k2 v2 ... kD vD] then Q query keys, each a
key that appeared, immediately followed by its value: [... kq1 vq1 kq2 vq2 ...].
The model is scored ONLY on predicting each vqi from the position of kqi — pure
recall of the value a key was paired with earlier. Keys are distinct within a
sequence; values may repeat. Exact-match accuracy on held-out sequences.

Archs (matched parameter budget, reported):
  attention : dense causal Transformer (upper bound).
  nla       : Neural Liquid Adjacency (ours).
  mamba     : HF MambaForCausalLM, a real selective-SSM SOTA baseline (sequential
              CPU fallback when the CUDA kernel is absent — correct, slower).

Sweep --pairs to stress recall capacity. Prediction: attention & NLA hold as
pairs grow; Mamba degrades.

Run:  python experiments/e11_mqar_vs_ssm.py --steps 4000 --pairs 8 16 32 --seeds 0 1
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import NeuralLiquidAdjacency, O1AntiConfig

# vocab layout: [0]=pad, keys in [1, 1+NK), values in [1+NK, 1+NK+NV)
NK = 64      # key vocabulary
NV = 64      # value vocabulary
KEY0 = 1
VAL0 = 1 + NK
VOCAB = 1 + NK + NV


def make_batch(batch, pairs, queries, device, gen):
    """MQAR sequence + a boolean mask of the value positions to score."""
    assert pairs <= NK, f"pairs={pairs} exceeds key vocabulary NK={NK}"
    seq_len = 2 * pairs + 2 * queries
    seq = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    score = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    for b in range(batch):
        keys = (KEY0 + torch.randperm(NK, generator=gen)[:pairs])           # distinct keys
        vals = VAL0 + torch.randint(0, NV, (pairs,), generator=gen)          # values (may repeat)
        # pair section
        seq[b, 0:2 * pairs:2] = keys
        seq[b, 1:2 * pairs:2] = vals
        # query section: sample `queries` of the pairs, present key then value
        qsel = torch.randint(0, pairs, (queries,), generator=gen)
        qpos = 2 * pairs
        seq[b, qpos + 0::2] = keys[qsel]
        seq[b, qpos + 1::2] = vals[qsel]
        # score the value token predicted from its query-key position
        score[b, qpos + 0::2] = True                                        # predict next = value
    return seq, score


# ------------------------------------------------------------ our two archs
class DenseBlock(nn.Module):
    def __init__(self, d, d_ff, heads=4):
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
    def __init__(self, cfg):
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
    def __init__(self, kind, cfg, max_len, n_blocks):
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(max_len, d))
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


def build_mamba(hidden, layers, state, device):
    from transformers import MambaConfig, MambaForCausalLM
    cfg = MambaConfig(vocab_size=VOCAB, hidden_size=hidden, num_hidden_layers=layers,
                      state_size=state)
    return MambaForCausalLM(cfg).to(device)


def logits_of(model, kind, ids):
    if kind == "mamba":
        return model(input_ids=ids).logits
    return model(ids)


# ------------------------------------------------------------------- train
def run(kind, pairs, args, device, seed):
    torch.manual_seed(seed)
    dgen = torch.Generator().manual_seed(seed + 1000)
    seq_len = 2 * pairs + 2 * args.queries

    if kind == "mamba":
        model = build_mamba(args.mamba_hidden, args.mamba_layers, args.mamba_state, device)
    else:
        cfg = O1AntiConfig(vocab_size=VOCAB, d_model=args.d_model, max_seq_len=seq_len,
                           d_c=args.d_c, d_state=64, top_k=args.top_k, d_ff=4 * args.d_model,
                           nla_heads=args.nla_heads)
        model = TinyLM(kind, cfg, max_len=seq_len, n_blocks=args.n_blocks).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        seq, score = make_batch(args.batch, pairs, args.queries, device, dgen)
        logits = logits_of(model, kind, seq)
        # predict token t+1 from t; score only the value positions
        pred = logits[:, :-1]
        tgt = seq[:, 1:]
        m = score[:, :-1]
        loss = F.cross_entropy(pred[m], tgt[m])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(args.steps // 8, 1) == 0:
            with torch.no_grad():
                vs, vsc = make_batch(256, pairs, args.queries, device, dgen)
                vp = logits_of(model, kind, vs)[:, :-1].argmax(-1)
                vm = vsc[:, :-1]
                vacc = (vp[vm] == vs[:, 1:][vm]).float().mean().item()
            print(f"    [{kind} P={pairs} seed={seed}] step {step+1}/{args.steps} "
                  f"loss {loss.item():.4f} acc {vacc:.3f}", flush=True)
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        seq, score = make_batch(512, pairs, args.queries, device, dgen)
        logits = logits_of(model, kind, seq)
        pred = logits[:, :-1].argmax(-1)
        tgt = seq[:, 1:]
        m = score[:, :-1]
        acc = (pred[m] == tgt[m]).float().mean().item()
    return {"acc": acc, "train_s": train_s,
            "params": sum(p.numel() for p in model.parameters())}


def mean_std(v):
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--archs", nargs="+", default=["attention", "nla", "mamba"])
    # our-arch sizing. NOTE: n_blocks>=4 & d_model>=128 are REQUIRED for a valid
    # experiment — with n_blocks=2/d_model=96 even attention (the upper bound)
    # cannot form induction heads and stalls at ~0.15 on MQAR, so the comparison
    # is meaningless. At 4 blocks / d_model 128, attention solves pairs=8 to
    # 1.000 in ~1000 steps (verified), establishing a valid ceiling.
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--d_c", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--nla_heads", type=int, default=4)
    # mamba sizing (its layer is ~half a Transformer block, so more layers to match)
    ap.add_argument("--mamba_hidden", type=int, default=128)
    ap.add_argument("--mamba_layers", type=int, default=6)
    ap.add_argument("--mamba_state", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)  # 1e-3 (not 2e-3) is stabler here
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    print(f"E11 MQAR: NLA vs SSM | device={device} pairs={args.pairs} seeds={args.seeds} "
          f"archs={args.archs}")
    rows = {}                                   # pairs -> {arch: [accs]}
    pcount = {}
    for P in args.pairs:
        print(f"\n--- {P} key-value pairs (seq_len {2*P+2*args.queries}) ---")
        rows[P] = {a: [] for a in args.archs}
        for seed in args.seeds:
            for a in args.archs:
                r = run(a, P, args, device, seed)
                rows[P][a].append(r["acc"]); pcount[a] = r["params"]
            line = "  ".join(f"{a}={rows[P][a][-1]:.3f}" for a in args.archs)
            print(f"    seed={seed}: {line}")

    hdr = "pairs".rjust(6) + "".join(f"{a+' (mean±std)':>22}" for a in args.archs)
    print("\n" + hdr)
    for P in args.pairs:
        cells = ""
        for a in args.archs:
            m, s = mean_std(rows[P][a])
            cells += f"{f'{m:.3f} ± {s:.3f}':>22}"
        print(f"{P:>6}{cells}")
    print("\nparams: " + "  ".join(f"{a}={pcount[a]:,}" for a in args.archs))
    print("\ngo/no-go: if NLA tracks attention and beats Mamba as pairs grow, NLA has "
          "the retrieval precision an SSM's fixed state lacks — its distinctive niche "
          "(SSM-cheap cache + attention-precise recall) is real.")


if __name__ == "__main__":
    main()
