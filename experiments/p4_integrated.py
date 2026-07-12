"""
p4_integrated.py — P4 go/no-go: all three pillars in ONE model.

P1/P2/P3 each validated a pillar in isolation. P4 checks they compose: the real
`O1AntiModel` conditions generation on the routed module trunk (pillar 2, with
NLA from pillar 1 inside each module) and decodes non-autoregressively (pillar 3)
— trained end-to-end with a single objective.

Task: the P3 reconstruction task (prompt tiled + block shift → target), but run
through the full model's `generation_loss` / `generate`, not a standalone decoder.

Reported: end-to-end generated token accuracy, the module-graph activation ratio
(proving pillar 2 is really sparse, not collapsed to a dense path), and the
forward-pass count vs an autoregressive model of the same width.

Run:  python experiments/p4_integrated.py --steps 2000 --length 48
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import O1AntiConfig, O1AntiModel

VOCAB = 32


def make_pair(batch, prompt_len, length, device):
    """target = prompt tiled to `length`, block-shifted — deterministic & learnable."""
    prompt = torch.randint(2, VOCAB, (batch, prompt_len), device=device)
    idx = torch.arange(length, device=device) % prompt_len
    gathered = prompt[:, idx]
    block_shift = (torch.arange(length, device=device) // prompt_len) % 3
    target = 2 + (gathered - 2 + block_shift) % (VOCAB - 2)
    return prompt, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--prompt_len", type=int, default=12)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_modules", type=int, default=6)
    ap.add_argument("--path_len", type=int, default=2)
    ap.add_argument("--decode_iters", type=int, default=6)
    ap.add_argument("--skeleton_mode", choices=["regress", "flow", "discrete"], default="regress")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-min", type=float, default=None)
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    cfg = O1AntiConfig(
        vocab_size=VOCAB, d_model=args.d_model, max_seq_len=args.prompt_len + args.length,
        n_modules=args.n_modules, path_len=args.path_len, d_ctx=64,
        d_ff=2 * args.d_model, d_c=24, d_state=48, top_k=8,
        skel_len=args.length, decode_iters=args.decode_iters,
        n_dec_layers=2, n_dec_heads=4, skeleton_mode=args.skeleton_mode,
    )
    model = O1AntiModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    print(f"P4 integrated | device={device} length={args.length} "
          f"n_modules={args.n_modules} path_len={args.path_len} mode={args.skeleton_mode}")
    model.train()
    for step in range(args.steps):
        prompt, target = make_pair(args.batch, args.prompt_len, args.length, device)
        loss = model.generation_loss(prompt, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 5, 1) == 0:
            print(f"  step {step+1}/{args.steps}  loss {loss.item():.4f}")

    model.eval()
    prompt, target = make_pair(256, args.prompt_len, args.length, device)
    with torch.no_grad():
        gen = model.generate(prompt, args.length)
        acc = (gen == target).float().mean().item()
        # measure the actual module activation ratio on eval inputs
        _, _, usage, _ = model.encode(prompt)
    active_modules = int((usage > 1e-6).sum().item())
    activation = model.num_parameters(active_only=True) / model.num_parameters()
    stage1_passes = {"regress": 1, "discrete": 1, "flow": cfg.ode_steps}[args.skeleton_mode]
    passes = stage1_passes + args.decode_iters

    print(f"\n{'metric':<34}{'value':>16}")
    print(f"{'end-to-end generated tok-acc':<34}{acc:>16.3f}")
    print(f"{'module path len / library':<34}{f'{args.path_len} / {args.n_modules}':>16}")
    print(f"{'param activation ratio':<34}{activation*100:>15.1f}%")
    print(f"{'forward passes (vs AR = length)':<34}{f'{passes} vs {args.length}':>16}")
    print(f"{'total / active params':<34}"
          f"{f'{model.num_parameters():,} / {model.num_parameters(active_only=True):,}':>16}")
    print(f"\ngo/no-go: all three pillars in one model — generated {acc:.3f} at "
          f"{activation*100:.0f}% activation, {passes} passes (AR needs {args.length}).")

    if args.assert_min is not None:
        assert acc >= args.assert_min, f"generated acc {acc:.3f} < {args.assert_min}"
        assert activation < 1.0, "module graph not sparse (activation >= 100%)"
        assert passes < args.length, "not fewer passes than autoregressive"
        print(f"\n[assert] acc {acc:.3f} >= {args.assert_min}, activation "
              f"{activation*100:.0f}% < 100%, {passes} < {args.length} passes  OK")


if __name__ == "__main__":
    main()
