"""Shape, causality, and train/inference-consistency tests for o1anti."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1anti import (
    LiquidStateScan,
    NeuralLiquidAdjacency,
    O1AntiConfig,
    O1AntiModel,
    ParallelDecoder,
    SkeletonGenerator,
)
from o1anti.losses import load_balance_loss

CFG = O1AntiConfig(
    vocab_size=64, d_model=32, max_seq_len=64, d_c=8, d_state=16, top_k=4,
    n_modules=4, path_len=2, d_ctx=16, d_ff=64, skel_len=4, ode_steps=4,
    decode_iters=3, n_dec_layers=1, n_dec_heads=2,
)


def test_liquid_scan_matches_stepwise():
    torch.manual_seed(0)
    scan = LiquidStateScan(8, 16).eval()
    x = torch.randn(2, 33, 8)  # non power of two on purpose
    s_par = scan(x)
    s = torch.zeros(2, 16)
    for t in range(x.shape[1]):
        s = scan.step(x[:, t], s)
        assert torch.allclose(s_par[:, t], s, atol=1e-5), f"diverged at t={t}"


def test_nla_causality():
    torch.manual_seed(0)
    nla = NeuralLiquidAdjacency(CFG).eval()
    x = torch.randn(1, 20, CFG.d_model)
    out1, _ = nla(x)
    x2 = x.clone()
    x2[:, 15:] += 10.0  # perturb the future
    out2, _ = nla(x2)
    assert torch.allclose(out1[:, :15], out2[:, :15], atol=1e-5)


def test_nla_step_matches_parallel():
    torch.manual_seed(0)
    nla = NeuralLiquidAdjacency(CFG).eval()
    x = torch.randn(2, 12, CFG.d_model)
    with torch.no_grad():
        out_par, _ = nla(x)
        cache = nla.init_cache(2)
        for t in range(12):
            out_t = nla.step(x[:, t], cache)
            assert torch.allclose(out_par[:, t], out_t, atol=1e-4), f"t={t}"
    assert cache["c"].shape == (2, 12, CFG.d_c)


def test_nla_cache_smaller_than_kv():
    kv_bytes = 2 * CFG.d_model * 2  # K and V, fp16
    assert NeuralLiquidAdjacency.cache_bytes_per_token(CFG) * 4 <= kv_bytes


def test_router_one_hot_and_dispatch():
    torch.manual_seed(0)
    model = O1AntiModel(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (3, 10))
    h, ctx, usage, cont = model.encode(ids)
    assert h.shape == (3, 10, CFG.d_model)
    assert usage.shape == (CFG.n_modules,)
    assert torch.isfinite(cont)
    assert load_balance_loss(torch.full((4,), 0.25)).abs() < 1e-8


def test_lm_forward_backward():
    torch.manual_seed(0)
    model = O1AntiModel(CFG).train()
    ids = torch.randint(0, CFG.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out.loss.backward()
    grads = [p.grad for p in model.router.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "router got no gradient"


def test_generation_loss_and_sample():
    torch.manual_seed(0)
    model = O1AntiModel(CFG).train()
    prompt = torch.randint(0, CFG.vocab_size, (2, 8))
    target = torch.randint(0, CFG.vocab_size, (2, 24))
    loss = model.generation_loss(prompt, target)
    assert torch.isfinite(loss)
    loss.backward()

    model.eval()
    gen = torch.Generator().manual_seed(0)
    toks = model.generate(prompt, length=24, generator=gen)
    assert toks.shape == (2, 24)
    assert toks.max() < CFG.vocab_size and toks.min() >= 0


@pytest.mark.parametrize("mode", ["regress", "flow", "discrete"])
def test_all_skeleton_modes_train_and_sample(mode):
    """Every stage-1 skeleton_mode must build, backprop, and sample. Only
    'regress' was exercised before; flow/discrete were experiment-only."""
    import dataclasses

    cfg = dataclasses.replace(CFG, skeleton_mode=mode)
    torch.manual_seed(0)
    model = O1AntiModel(cfg).train()
    prompt = torch.randint(0, cfg.vocab_size, (2, 8))
    target = torch.randint(0, cfg.vocab_size, (2, 20))
    loss = model.generation_loss(prompt, target)
    assert torch.isfinite(loss)
    loss.backward()
    # stage-1 params must receive gradient
    stage1 = model.prior if mode != "flow" else model.skeleton
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage1.parameters())

    model.eval()
    toks = model.generate(prompt, length=20)
    assert toks.shape == (2, 20)
    assert toks.max() < cfg.vocab_size and toks.min() >= 0


def test_skeleton_encoder_and_ode_shapes():
    from o1anti.generation import SkeletonEncoder

    h = torch.randn(2, 30, CFG.d_model)
    enc = SkeletonEncoder(CFG).eval()
    z1 = enc(h)
    assert z1.shape == (2, CFG.skel_len, CFG.d_model)
    # generator now conditions on the full prompt memory (B, T_prompt, d)
    sg = SkeletonGenerator(CFG).eval()
    mem = torch.randn(2, 8, CFG.d_model)
    z = sg.sample(mem)
    assert z.shape == (2, CFG.skel_len, CFG.d_model)


def test_active_params_fraction():
    model = O1AntiModel(CFG)
    total = model.num_parameters()
    active = model.num_parameters(active_only=True)
    assert active < total


def test_parallel_decoder_reconstructs_aligned_skeleton():
    """The P3 alignment guarantee: given a per-position 'answer' skeleton, the
    parallel decoder must reconstruct it in one pass. This is what breaks if the
    position-embedding scale regresses back to ~0.02 (all masked queries become
    identical and cross-attention can't tell positions apart)."""
    from o1anti.generation import ParallelDecoder

    torch.manual_seed(0)
    cfg = O1AntiConfig(vocab_size=24, d_model=48, max_seq_len=40, skel_len=24,
                       decode_iters=4, n_dec_layers=2, n_dec_heads=4, d_ff=96)
    embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
    skel_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
    dec = ParallelDecoder(cfg, embed)
    # dedupe: dec.head.weight is tied to embed.weight, so the naive concat would
    # hand AdamW the same tensor twice (a warning today, an error in future torch).
    params = {id(p): p for m in (embed, skel_embed, dec) for p in m.parameters()}
    opt = torch.optim.AdamW(list(params.values()), lr=3e-3)
    L = 24
    for _ in range(400):
        t = torch.randint(0, cfg.vocab_size, (32, L))
        skel = skel_embed(t)                       # skeleton = per-position answer
        r = torch.rand(32, 1)
        m = torch.rand(32, L) < r
        m |= ~m.any(dim=-1, keepdim=True)
        loss = F.cross_entropy(dec.logits(t, m, skel)[m], t[m])
        opt.zero_grad(); loss.backward(); opt.step()
    t = torch.randint(0, cfg.vocab_size, (64, L))
    acc = (dec.mask_predict(skel_embed(t), L) == t).float().mean().item()
    assert acc > 0.9, f"aligned-skeleton reconstruction collapsed to {acc:.3f}"
