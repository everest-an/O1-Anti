"""
e11b_grok_diagnostic.py — is NLA's MQAR-recall instability FIXABLE?

E11 established that NLA *can* solve pairs=8 MQAR to 1.000 (its recall mechanism
is sufficient) but does so UNRELIABLY: bimodal grok/no-grok — some seeds jump to
~1.0 (late, ~step 6000), others stay stuck at ~0.15 for the whole budget. That
signature is optimization/grokking instability, not a capacity wall.

This diagnostic sweeps the strongest grokking levers (weight decay; and, if
requested, more seeds/steps) on NLA at pairs=8, and reports the GROK RATE
(fraction of seeds reaching acc>0.9) and median grok step per setting. Attention
(always groks by ~step 1250) and Mamba are omitted — this is purely "what makes
NLA grok reliably".

Verdict logic:
  - if some weight_decay makes grok rate ~1.0 (all seeds), NLA recall is FIXABLE
    (a training-stability fix), and the niche claim can be revisited.
  - if no setting lifts the grok rate, the instability is deeper and the niche
    claim stays refuted.

Run:  python experiments/e11b_grok_diagnostic.py --wds 0.01 0.1 1.0 --seeds 0 1 2 3
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.e11_mqar_vs_ssm import run  # reuse the exact training loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # fixed valid-capacity config (attention solves MQAR here)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--d_c", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--nla_heads", type=int, default=4)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"E11b grok diagnostic | NLA pairs={args.pairs} steps={args.steps} "
          f"device={args.device} wds={args.wds} seeds={args.seeds}", flush=True)

    rows = []
    for wd in args.wds:
        args.weight_decay = wd
        accs, grok_steps = [], []
        for seed in args.seeds:
            r = run("nla", args.pairs, args, args.device, seed)
            accs.append(r["acc"])
            grok_steps.append(r["grok_step"])
            gs = r["grok_step"] if r["grok_step"] is not None else "-"
            print(f"  wd={wd} seed={seed}: acc={r['acc']:.3f} grok_step={gs}", flush=True)
        n_grok = sum(1 for a in accs if a > 0.9)
        gsteps = [g for g in grok_steps if g is not None]
        med = sorted(gsteps)[len(gsteps) // 2] if gsteps else None
        rows.append((wd, n_grok, len(accs), sum(accs) / len(accs), med))

    print(f"\n{'weight_decay':>13}{'grok_rate':>12}{'mean_acc':>10}{'median_grok_step':>18}")
    for wd, ng, n, ma, med in rows:
        print(f"{wd:>13}{f'{ng}/{n}':>12}{ma:>10.3f}{str(med):>18}")
    best = max(rows, key=lambda r: (r[1], r[3]))
    verdict = ("FIXABLE — a weight_decay setting groks all/most seeds"
               if best[1] >= max(1, int(0.75 * best[2]))
               else "STILL UNSTABLE — no swept setting made grokking reliable")
    print(f"\nverdict: {verdict} (best wd={best[0]}, grok {best[1]}/{best[2]})")


if __name__ == "__main__":
    main()
