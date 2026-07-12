"""
p2_module_routing.py — P2 go/no-go: Neural Module Graph vs dense stack.

Question: can a context-routed sparse module path (only `path_len` of
`n_modules` modules run per input) match a dense stack that runs *every*
module, while activating far fewer parameters?

Task: sequence classification by "regime". Each sequence is generated under
one of R latent regimes (different token-transition rules); the model must
route to the right specialist and predict the regime label from the pooled
representation. This is exactly the setting the module graph is built for —
distinct input types that benefit from distinct computation paths.

  dense : runs all n_modules modules in a fixed stack, then classifies.
  routed: ContextEncoder → GlobalRouter picks path_len modules → classifies.

Reported: accuracy, total vs active params, activation ratio, and per-module
usage entropy (did the router actually specialize, or collapse?).

Run:  python experiments/p2_module_routing.py --steps 800 --regimes 4
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import O1AntiConfig
from o1anti.module_graph import ContextEncoder, GlobalRouter, ModuleLibrary, NeuralModule

VOCAB = 32


def make_regime_batch(batch: int, seq: int, n_regimes: int, device):
    """Each sequence follows one regime's Markov-ish rule; label = regime id."""
    labels = torch.randint(0, n_regimes, (batch,), device=device)
    ids = torch.zeros(batch, seq, dtype=torch.long, device=device)
    ids[:, 0] = torch.randint(2, VOCAB, (batch,), device=device)
    for b in range(batch):
        r = labels[b].item()
        step = r + 1                      # regime-specific stride
        for t in range(1, seq):
            ids[b, t] = 2 + (ids[b, t - 1] - 2 + step) % (VOCAB - 2)
    return ids, labels


class DenseStack(nn.Module):
    """Runs every module in a fixed stack (upper bound, full compute)."""

    def __init__(self, cfg: O1AntiConfig, n_regimes: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, cfg.d_model)
        self.modules_list = nn.ModuleList(NeuralModule(cfg) for _ in range(cfg.n_modules))
        self.cls = nn.Linear(cfg.d_model, n_regimes)

    def forward(self, ids):
        h = self.embed(ids)
        for m in self.modules_list:
            h, _ = m(h)
        return self.cls(h.mean(dim=1))

    def active_params(self):
        return sum(p.numel() for p in self.parameters())


class RoutedGraph(nn.Module):
    """ContextEncoder → GlobalRouter → path_len modules only."""

    def __init__(self, cfg: O1AntiConfig, n_regimes: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB, cfg.d_model)
        self.context = ContextEncoder(cfg)
        self.router = GlobalRouter(cfg)
        self.library = ModuleLibrary(cfg)
        self.cls = nn.Linear(cfg.d_model, n_regimes)
        self.last_usage = None

    def forward(self, ids):
        h = self.embed(ids)
        ctx = self.context(h)
        w, usage = self.router(ctx)
        self.last_usage = usage
        h, _ = self.library(h, w)
        return self.cls(h.mean(dim=1))

    def active_params(self):
        shared = (
            sum(p.numel() for p in self.embed.parameters())
            + sum(p.numel() for p in self.context.parameters())
            + sum(p.numel() for p in self.router.parameters())
            + sum(p.numel() for p in self.cls.parameters())
        )
        per_module = sum(p.numel() for p in self.library.modules_list[0].parameters())
        return shared + per_module * self.cfg.path_len


def train_eval(model, args, device, is_routed):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(args.steps):
        ids, labels = make_regime_batch(args.batch, args.seq, args.regimes, device)
        logits = model(ids)
        loss = F.cross_entropy(logits, labels)
        if is_routed:
            from o1anti.losses import load_balance_loss
            loss = loss + model.cfg.load_balance_coef * load_balance_loss(model.last_usage)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 5, 1) == 0:
            print(f"  step {step+1}/{args.steps}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        ids, labels = make_regime_batch(512, args.seq, args.regimes, device)
        acc = (model(ids).argmax(-1) == labels).float().mean().item()
        usage = None
        if is_routed:
            model(ids)
            usage = model.last_usage.clone()
    return acc, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--regimes", type=int, default=4)
    ap.add_argument("--n_modules", type=int, default=8)
    ap.add_argument("--path_len", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-min", type=float, default=None,
                    help="fail if routed acc < this (regression anchor)")
    ap.add_argument("--assert-max-activation", type=float, default=0.5,
                    help="fail if activation ratio exceeds this")
    args = ap.parse_args()

    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=args.seq,
        n_modules=args.n_modules, path_len=args.path_len, d_ctx=64,
        d_ff=2 * args.d_model, d_c=24, d_state=48, top_k=6,
    )
    print(f"P2 module-routing | device={args.device} regimes={args.regimes} "
          f"n_modules={args.n_modules} path_len={args.path_len}")

    torch.manual_seed(args.seed)
    print("[dense stack]")
    dense = DenseStack(cfg, args.regimes).to(args.device)
    d_acc, _ = train_eval(dense, args, args.device, is_routed=False)

    torch.manual_seed(args.seed)
    print("[routed graph]")
    routed = RoutedGraph(cfg, args.regimes).to(args.device)
    r_acc, usage = train_eval(routed, args, args.device, is_routed=True)

    d_total = sum(p.numel() for p in dense.parameters())
    r_total = sum(p.numel() for p in routed.parameters())
    r_active = routed.active_params()
    ratio = r_active / r_total
    ent = -(usage * (usage + 1e-9).log()).sum().item()
    max_ent = torch.log(torch.tensor(float(args.n_modules))).item()

    print(f"\n{'variant':<14}{'acc':>7}{'total params':>14}{'active params':>15}{'activation':>12}")
    print(f"{'dense stack':<14}{d_acc:>7.3f}{d_total:>14,}{dense.active_params():>15,}{'100%':>12}")
    print(f"{'routed graph':<14}{r_acc:>7.3f}{r_total:>14,}{r_active:>15,}{ratio*100:>11.1f}%")
    print(f"\nrouter usage: {[round(u, 3) for u in usage.tolist()]}")
    print(f"usage entropy {ent:.3f} / {max_ent:.3f} max "
          f"({'specialized/balanced' if ent > 0.5 * max_ent else 'near-collapse'})")
    print(f"\ngo/no-go: routed acc within a few points of dense at "
          f"{ratio*100:.0f}% activation → compute win is real")

    if args.assert_min is not None:
        assert r_acc >= args.assert_min, f"routed acc {r_acc:.3f} < {args.assert_min}"
        assert ratio <= args.assert_max_activation, (
            f"activation {ratio:.3f} > {args.assert_max_activation}")
        print(f"\n[assert] routed acc {r_acc:.3f} >= {args.assert_min} and "
              f"activation {ratio:.3f} <= {args.assert_max_activation}  OK")


if __name__ == "__main__":
    main()
