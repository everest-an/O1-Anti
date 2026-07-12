"""
e10_real_needle.py — showcasing NLA's actual strength: sparse long-range
retrieval EMBEDDED IN REAL TEXT, at increasing context length.

E8 measured O1-Anti on generic byte-level language modeling and found a robust
~22% BPB gap vs dense (documented, four ablations, not closed). P1 separately
showed NLA MATCHES dense attention on synthetic selective-copy while caching 8x
less. Those two results are not in tension — generic LM rewards dense many-
relations mixing (NLA's weak spot); sparse fact-retrieval rewards exactly what
NLA is built for. This experiment checks whether that P1 advantage survives when
the "haystack" is real English text (not synthetic tokens) and the context is
LONGER than E8's LM probe — the setting where the architecture's value
proposition (an O(n*d_c) cache, not O(n*d_model)) actually matters.

Task: needle-in-real-haystack. A real WikiText passage of length `ctx_len` bytes
is used as filler; a synthetic key/value fact is inserted at a random depth
("[KEY:xxxxx]=yyy"); the same key is repeated as a query at the end, and the
model must predict the 3-digit value — exact match, scored on held-out
haystacks/keys/positions.

Reported per context length: retrieval accuracy (O1-Anti vs dense, matched
active params) and the analytic inference-cache size (bytes/token/layer) for
each. The claim under test: accuracy holds (or nearly holds) as context grows,
while the cache-size gap (already 8x at P1's toy scale) stays constant or grows
in O1-Anti's favor as ctx_len increases.

CAUTION — this task shows threshold/grokking-like convergence within a fixed
step budget: loss sits on a high plateau, then drops sharply at an unpredictable
step (observed: sometimes at step ~3000, sometimes never within 4500). A
single seed can therefore land on either side of that threshold by luck, giving
a misleading comparison (one early single-seed run showed NLA winning 2 of 3
lengths by a wide margin but LOSING the third). Use --seeds with 2+ values to
average this out; a single seed is a preliminary signal only, not a result.

Run:  python experiments/e10_real_needle.py --steps 6000 --lengths 128 256 512 --seeds 0 1
"""

import argparse
import os
import string
import sys
import time
from pathlib import Path

# Force offline mode BEFORE any datasets import — see train_o1anti.py for why:
# load_dataset() can hang for a very long time on a network round-trip even
# when the corpus is already cached (observed: >75 min stuck, ~0s CPU used).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import NeuralLiquidAdjacency, O1AntiConfig

VOCAB = 256  # byte-level, so real text and synthetic markers share one vocab
KEY_ALPHABET = string.ascii_uppercase + string.digits


# --------------------------------------------------------------------- data
def load_corpus() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return "".join(t for t in ds["text"] if t.strip())


def make_batch(corpus: str, batch: int, ctx_len: int, ans_len: int, gen: torch.Generator, device):
    """Real-text haystack with an inserted [KEY:xxxxx]=yyy fact, queried at the
    end. Returns (seq, ans_start) — ans_start is the position where the model
    must start predicting the `ans_len`-digit value."""
    key_len = 5
    fact_len = 2 + key_len + 1 + ans_len + 1  # "[K" + key + "]=" + digits (roughly)
    query_len = 2 + key_len + 2               # "[K" + key + "]="
    min_ctx = query_len + ans_len + 2 * fact_len + 1  # room for both fact and query+value
    assert ctx_len >= min_ctx, (
        f"ctx_len={ctx_len} too short to fit a fact + query + value "
        f"(need >= {min_ctx} for ans_len={ans_len})"
    )
    seqs = torch.zeros(batch, ctx_len, dtype=torch.long, device=device)
    ans_start = ctx_len - ans_len  # query+value always end-aligned
    for b in range(batch):
        start = torch.randint(0, len(corpus) - ctx_len, (1,), generator=gen).item()
        haystack = list(corpus[start : start + ctx_len].encode("utf-8", errors="ignore"))
        haystack = (haystack + [32] * ctx_len)[:ctx_len]  # pad with spaces if short

        key = "".join(KEY_ALPHABET[i] for i in
                       torch.randint(0, len(KEY_ALPHABET), (key_len,), generator=gen).tolist())
        value = "".join(str(d) for d in
                         torch.randint(0, 10, (ans_len,), generator=gen).tolist())
        fact = f"[K{key}]={value}"
        query = f"[K{key}]="

        depth = torch.randint(fact_len, ctx_len - query_len - ans_len - fact_len,
                              (1,), generator=gen).item()
        fact_b = list(fact.encode())
        haystack[depth : depth + len(fact_b)] = fact_b

        query_b = list(query.encode())
        haystack[ans_start - len(query_b) : ans_start] = query_b
        value_b = list(value.encode())
        haystack[ans_start : ans_start + ans_len] = value_b

        seqs[b] = torch.tensor(haystack[:ctx_len], dtype=torch.long, device=device)
    return seqs, ans_start


# ------------------------------------------------------------------- models
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
    def __init__(self, kind, cfg, max_len, n_blocks=2):
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


# ------------------------------------------------------------------- train
def run(kind, corpus, ctx_len, args, device, seed):
    torch.manual_seed(seed)
    ans_len = args.ans_len
    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=ctx_len,
        d_c=args.d_c, d_state=64, top_k=args.top_k, d_ff=4 * args.d_model,
    )
    model = TinyLM(kind, cfg, max_len=ctx_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for step in range(args.steps):
        seq, ans_start = make_batch(corpus, args.batch, ctx_len, ans_len, gen, device)
        logits = model(seq)
        pred = logits[:, ans_start - 1 : ctx_len - 1]      # predict each value digit
        tgt = seq[:, ans_start:]
        loss = F.cross_entropy(pred.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"    [{kind} L={ctx_len} seed={seed}] step {step+1}/{args.steps}  loss {loss.item():.4f}")
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        seq, ans_start = make_batch(corpus, 200, ctx_len, ans_len, gen, device)
        logits = model(seq)
        pred = logits[:, ans_start - 1 : ctx_len - 1].argmax(-1)
        tgt = seq[:, ans_start:]
        exact = (pred == tgt).all(dim=-1).float().mean().item()   # whole value correct
        digit_acc = (pred == tgt).float().mean().item()

    cache = cfg.d_c * 2 if kind == "nla" else 2 * args.d_model * 2  # fp16 B/tok/layer
    return {"exact": exact, "digit_acc": digit_acc, "train_s": train_s,
            "cache_B_per_tok": cache, "params": sum(p.numel() for p in model.parameters())}


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--ans_len", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_c", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="train+eval each (kind, length) once per seed and report mean+std "
                         "(this task shows threshold-like convergence — single-seed "
                         "results are noisy; use multiple seeds for a trustworthy signal)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-min", type=float, default=None,
                    help="fail if NLA mean exact-match < this at ANY length")
    args = ap.parse_args()
    device = args.device

    print(f"E10 real-text needle retrieval | device={device} lengths={args.lengths} "
          f"seeds={args.seeds}")
    corpus = load_corpus()
    print(f"corpus: {len(corpus):,} chars")

    rows = []
    for L in args.lengths:
        print(f"\n--- context length {L} ---")
        d_exacts, n_exacts, d_cache, n_cache = [], [], None, None
        for seed in args.seeds:
            d = run("dense", corpus, L, args, device, seed)
            n = run("nla", corpus, L, args, device, seed)
            d_exacts.append(d["exact"])
            n_exacts.append(n["exact"])
            d_cache, n_cache = d["cache_B_per_tok"], n["cache_B_per_tok"]
            print(f"    seed={seed}: dense={d['exact']:.3f}  nla={n['exact']:.3f}")
        rows.append((L, d_exacts, n_exacts, d_cache, n_cache))

    print(f"\n{'ctx_len':>8}{'dense exact (mean±std)':>26}{'nla exact (mean±std)':>24}"
          f"{'cache ratio':>13}")
    for L, d_exacts, n_exacts, d_cache, n_cache in rows:
        dm, ds = mean_std(d_exacts)
        nm, ns = mean_std(n_exacts)
        ratio = d_cache / n_cache
        print(f"{L:>8}{f'{dm:.3f} ± {ds:.3f}':>26}{f'{nm:.3f} ± {ns:.3f}':>24}{ratio:>12.1f}x")

    print(f"\n({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''} per cell — "
          f"this task shows threshold-like convergence within a fixed step budget, "
          f"so single-seed numbers are noisy; std shown above is the honest spread.)")
    print("\ngo/no-go: NLA mean exact-match tracks dense across context lengths while "
          "caching a constant multiple less per token — the memory win holds on "
          "real text and does not erode as context grows.")

    if args.assert_min is not None:
        worst = min(mean_std(n_exacts)[0] for _, _, n_exacts, _, _ in rows)
        assert worst >= args.assert_min, f"NLA worst mean exact-match {worst:.3f} < {args.assert_min}"
        print(f"\n[assert] NLA mean exact-match >= {args.assert_min} at every length  OK")


if __name__ == "__main__":
    main()
