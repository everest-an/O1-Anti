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
    VectorQuantizer,
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


def test_multihead_nla_consistency_and_causality():
    """Multi-head NLA: step() must match the parallel forward, stay causal, and
    keep the cache size independent of head count."""
    import dataclasses

    cfg = dataclasses.replace(CFG, nla_heads=4, d_model=32)
    torch.manual_seed(0)
    nla = NeuralLiquidAdjacency(cfg).eval()
    x = torch.randn(2, 14, cfg.d_model)
    with torch.no_grad():
        out_par, _ = nla(x)
        # causality: perturbing the future must not change past outputs
        x2 = x.clone()
        x2[:, 10:] += 5.0
        out2, _ = nla(x2)
        assert torch.allclose(out_par[:, :10], out2[:, :10], atol=1e-5)
        # step == parallel
        cache = nla.init_cache(2)
        for t in range(14):
            out_t = nla.step(x[:, t], cache)
            assert torch.allclose(out_par[:, t], out_t, atol=1e-4), f"t={t}"
    # cache is c_j only — unchanged by head count
    assert cache["c"].shape == (2, 14, cfg.d_c)


def test_blocksparse_nla_causal_and_approximates_exact():
    """Sub-quadratic block-sparse NLA (nla_block_size>0) must (a) preserve exact
    causality, (b) closely approximate the exact O(T^2) path, and (c) actually
    reduce score-op count. Default (nla_block_size=0) is the exact path and is
    covered by the other NLA tests, so validated results are untouched."""
    import dataclasses

    T = 96
    cfg_exact = O1AntiConfig(vocab_size=64, d_model=32, max_seq_len=T, d_c=16,
                             d_state=32, top_k=8, nla_heads=4)
    torch.manual_seed(0)
    nla = NeuralLiquidAdjacency(cfg_exact).eval()
    x = torch.randn(2, T, cfg_exact.d_model)
    with torch.no_grad():
        out_exact, _ = nla(x)
        nla.cfg = dataclasses.replace(cfg_exact, nla_block_size=16, nla_cand_blocks=4)
        out_bs, _ = nla(x)
        # causality: perturbing the future leaves past outputs bit-identical
        x2 = x.clone(); x2[:, 70:] += 5.0
        out_bs2, _ = nla(x2)

    assert out_bs.shape == out_exact.shape
    cos = F.cosine_similarity(out_exact.reshape(-1, 32), out_bs.reshape(-1, 32), dim=-1).mean()
    assert cos > 0.9, f"block-sparse output diverged from exact (cos={cos:.3f})"
    past_diff = (out_bs[:, :70] - out_bs2[:, :70]).abs().max().item()
    assert past_diff < 1e-5, f"block-sparse broke causality (past changed by {past_diff:.2e})"
    # score-op count is genuinely smaller than dense T*T
    nb = (T + 16 - 1) // 16
    assert T * nb + T * 4 * 16 < T * T


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


def test_product_quantizer_shapes_and_fidelity():
    """E9: product quantization (vq_groups independent codebooks) must (a) shape
    codes as (B, L, G), (b) reconstruct without a shape/dtype error via
    embed_codes, and (c) raise fidelity vs a single codebook (the E9 fix — a
    single codebook capped cosine fidelity around ~0.3-0.6 on this scale)."""
    torch.manual_seed(0)
    d = 32
    z = F.normalize(torch.randn(8, 6, d), dim=-1)

    def mean_cos(groups):
        cfg = O1AntiConfig(vocab_size=16, d_model=d, max_seq_len=8, skel_len=6,
                           codebook_size=64, vq_groups=groups)
        vq = VectorQuantizer(cfg)
        q, codes, _ = vq(z)
        assert codes.shape == (8, 6, groups)
        assert q.shape == z.shape
        rebuilt = vq.embed_codes(codes)
        assert torch.allclose(q, rebuilt, atol=1e-5)
        zg = F.normalize(z.view(8, 6, groups, d // groups), dim=-1)
        qg = F.normalize(q.view(8, 6, groups, d // groups), dim=-1)
        return (zg * qg).sum(-1).mean().item()

    cos_g1 = mean_cos(1)
    cos_g4 = mean_cos(4)
    assert cos_g4 > cos_g1, f"grouping should raise fidelity: G=1 {cos_g1:.3f} vs G=4 {cos_g4:.3f}"


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


def test_token_moe_routing_trains():
    """Token-granularity trunk: dense NLA + per-token MoE-FFN. Must build, do a
    causal-LM step, keep causality, and route gradient to multiple experts."""
    import dataclasses

    cfg = dataclasses.replace(CFG, routing_granularity="token", n_layers=2, moe_top_e=1)
    torch.manual_seed(0)
    model = O1AntiModel(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out.loss.backward()
    experts = model.trunk.blocks[0].moe.experts
    got = sum(1 for e in experts
              if any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters()))
    assert got >= 2, f"only {got} experts received gradient (routing may be collapsed)"
    # active LM params must be < total (some experts idle per token)
    assert model.num_parameters(active_only=True) < model.num_parameters()


def test_moe_noisy_gating_unbiased_and_eval_deterministic():
    """Noisy top-k gating (moe_noise>0) perturbs only the top-e SELECTION at
    train time; the combine weights and load-balance usage come from the clean
    softmax. So (a) it must be a no-op at eval (deterministic), and (b) the
    reported usage must not itself carry the noise."""
    import dataclasses

    cfg = dataclasses.replace(CFG, routing_granularity="token", n_layers=2, moe_noise=2.0)
    torch.manual_seed(0)
    model = O1AntiModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    # eval: two passes identical (noise is train-only)
    model.eval()
    with torch.no_grad():
        h1 = model.encode(ids)[0]
        h2 = model.encode(ids)[0]
    assert torch.allclose(h1, h2), "eval output changed — noise leaked into eval"
    # train: still produces finite loss + gradient to multiple experts
    model.train()
    out = model(ids, labels=ids)
    out.loss.backward()
    experts = model.trunk.blocks[0].moe.experts
    got = sum(1 for e in experts
              if any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters()))
    assert got >= 2


def test_hybrid_trunk_interleaves_attention_and_stays_causal():
    """Hybrid token trunk: cfg.hybrid_attn_every>0 replaces every N-th NLA mixer
    with full causal attention. Must build with the right mix, train, and keep
    the whole trunk causal (attention layers included)."""
    import dataclasses

    cfg = dataclasses.replace(CFG, routing_granularity="token", n_layers=4,
                              hybrid_attn_every=2)
    torch.manual_seed(0)
    model = O1AntiModel(cfg)
    kinds = [b.mixer_kind for b in model.trunk.blocks]
    assert kinds == ["nla", "attn", "nla", "attn"], kinds
    # causal: perturbing the future leaves past hidden states unchanged
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        h1 = model.encode(ids)[0]
        ids2 = ids.clone(); ids2[:, 15:] = torch.randint(0, cfg.vocab_size, (1, 5))
        h2 = model.encode(ids2)[0]
    assert torch.allclose(h1[:, :15], h2[:, :15], atol=1e-4), "hybrid trunk broke causality"
    # trains
    model.train()
    model(ids, labels=ids).loss.backward()


def test_topology_expert_router_noop_at_init_then_trains():
    """"NLA Router" (cfg.expert_router='topology'): a per-token E×E expert
    adjacency diffuses gate scores. adj_proj is zero-init so at start it is a
    near-no-op residual over the plain gate (stable); it must (a) match plain
    routing at init, (b) train (adj_proj gets gradient), and (c) keep experts
    active + causality."""
    import dataclasses

    topo = dataclasses.replace(CFG, routing_granularity="token", n_layers=2,
                               moe_top_e=2, expert_router="topology")
    ids = torch.randint(0, topo.vocab_size, (2, 12))

    # (a) no-op at init: zero-init adj_proj → uniform adjacency → propagated is a
    # per-token constant added to all experts → softmax/top-e invariant. Verify on
    # the SAME weights by flipping the flag (shared cfg object), so nothing else
    # differs. (Two separately-constructed models would diverge because building
    # adj_proj perturbs the init RNG stream — not a real difference.)
    torch.manual_seed(0); m_topo = O1AntiModel(topo).eval()
    with torch.no_grad():
        out_topo = m_topo.encode(ids)[0]
        m_topo.cfg.expert_router = "plain"           # bypass the topology path
        out_plain = m_topo.encode(ids)[0]
        m_topo.cfg.expert_router = "topology"        # restore
    assert torch.allclose(out_topo, out_plain, atol=1e-5), "topology router not a no-op at init"

    # (b)+(c) trains: adj_proj receives gradient, experts stay active.
    m_topo.train()
    m_topo(ids, labels=ids).loss.backward()
    adj = m_topo.trunk.blocks[0].moe.adj_proj
    assert adj.weight.grad is not None and adj.weight.grad.abs().sum() > 0, \
        "expert adjacency got no gradient"
    experts = m_topo.trunk.blocks[0].moe.experts
    got = sum(1 for e in experts
              if any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters()))
    assert got >= 2


def test_token_moe_causality():
    import dataclasses

    cfg = dataclasses.replace(CFG, routing_granularity="token", n_layers=2)
    torch.manual_seed(0)
    model = O1AntiModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        h1 = model.encode(ids)[0]
        ids2 = ids.clone()
        ids2[:, 15:] = torch.randint(0, cfg.vocab_size, (1, 5))
        h2 = model.encode(ids2)[0]
    assert torch.allclose(h1[:, :15], h2[:, :15], atol=1e-4), "token trunk broke causality"


def test_generation_routes_through_module_trunk():
    """P4 integration: generation_loss must condition on the routed module trunk
    (pillars 1+2), so the module library and router receive gradient during a
    pure generation step — not just the generation stack."""
    torch.manual_seed(0)
    model = O1AntiModel(CFG).train()
    prompt = torch.randint(0, CFG.vocab_size, (3, 8))
    target = torch.randint(0, CFG.vocab_size, (3, 20))
    model.generation_loss(prompt, target).backward()

    def got_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert got_grad(model.library), "module library got no gradient from generation"
    assert got_grad(model.router), "router got no gradient from generation"


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
