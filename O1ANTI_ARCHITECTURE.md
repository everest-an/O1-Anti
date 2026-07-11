# O1-Anti — the Anti-Transformer (MTLNN v2)

O1-Anti is an independent architecture line under the MT-LNN project. Where the
main O1 repo is a *microtubule-inspired liquid* language model, **O1-Anti attacks
the three cost roots of the Transformer directly**, one pillar each:

| Transformer cost root | O1-Anti pillar | Module |
|---|---|---|
| O(n²) attention + KV cache → **memory wall** | Neural Liquid Adjacency (top-K dynamic sparse links over compressed states) | [`o1anti/nla.py`](o1anti/nla.py) |
| 100% dense activation → **compute waste** | Context-routed Neural Module Graph (one path per input) | [`o1anti/module_graph.py`](o1anti/module_graph.py) |
| Serial token decode → **latency wall** | Liquid state-transition generation (few-step ODE skeleton + parallel decode) | [`o1anti/generation.py`](o1anti/generation.py) |

This is a research prototype at toy scale (hundreds of K params). It exists to
give **go/no-go signals** on each pillar cheaply, before any large-scale run.

## Pillar 1 — Neural Liquid Adjacency (memory)

Instead of caching K and V (`2·d_model` per token per layer), each position
caches only a compressed state `c_j = W_c x_j` with `d_c ≪ d_model`. Routing
keys and values are both derived from `c_j`, so the **inference cache is
`O(n·d_c)`**. A parallel-scan liquid state `s_t` summarizes the whole prefix and
gates the output, making the adjacency context-dependent ("liquid"). Each token
attends to at most `top_k` past positions chosen by content score.

Training uses a **straight-through estimator**: the forward pass aggregates the
hard top-K neighbors, but gradients flow through the dense softmax so
non-selected positions still receive a learning signal — this was the difference
between the module failing to learn and matching dense attention (see P1 below).

`step()` is exactly consistent with the parallel `forward()` (unit-tested).

## Pillar 2 — Neural Module Graph (compute)

No fixed layer stack. A `ContextEncoder` summarizes the input once; a
`GlobalRouter` commits to an ordered path of `path_len` modules out of a library
of `n_modules`. **Only modules on the path run** — activation ratio is
`path_len / n_modules`. Routing is trained with straight-through Gumbel-softmax
plus a load-balance penalty (usage variance) to prevent module collapse. New
capabilities = grow the library, finetune only the router.

## Pillar 3 — Liquid state-transition generation (latency)

Two stages, both non-autoregressive:

1. `SkeletonGenerator` — a flow-matching vector field integrated with a few
   Euler steps maps noise → a continuous "semantic skeleton" (`skel_len ≪ n`).
2. `ParallelDecoder` — bidirectional blocks cross-attend into the skeleton and
   emit **all tokens at once**, refined with a fixed number of mask-predict
   rounds.

Cost for a length-`n` generation: `ode_steps + decode_iters` forward passes,
not `n`.

## Empirical status

### P1 — NLA vs dense attention (selective copy, recall across a separator)

Matched 2-block LMs, `d_model=128`, 1500 steps, CPU:

| Variant | Params | Recall acc | Final loss | Cache B/tok/layer (fp16) |
|---|---:|---:|---:|---:|
| Dense attention | 409,344 | **1.000** | 0.0009 | 512 |
| **Neural Liquid Adjacency** | 390,784 | **1.000** | 0.0000 | **64 (8× smaller)** |

**Go.** NLA reaches full recall matching dense attention while caching 8× less
per token. Reproduce: `python experiments/p1_nla_swap.py --steps 1500 --seq 32`.

### P2 — Module routing vs dense stack (regime classification)

4 latent regimes, `n_modules=8`, `path_len=2`, 800 steps, CPU:

| Variant | Acc | Total params | Active params | Activation |
|---|---:|---:|---:|---:|
| Dense stack (runs all 8) | 1.000 | 559,876 | 559,876 | 100% |
| **Routed graph (runs 2)** | **1.000** | 571,284 | **153,972** | **27%** |

Router usage entropy 1.94 / 2.08 max — modules specialized, no collapse.

**Go.** Routed path matches the dense stack at 27% activation.
Reproduce: `python experiments/p2_module_routing.py --steps 800 --regimes 4`.

### P3 — Parallel decode vs autoregressive (sequence reconstruction, len 48)

| Path | Tok acc | Forward passes |
|---|---:|---:|
| Autoregressive baseline | 1.000 | 48 |
| **Parallel decoder, faithful skeleton** | **0.999** | **8 (6× fewer)** |
| Parallel, *generated* skeleton (end-to-end) | 0.07 | 8 |

**Partial — stage 2 proven, stage 1 open.** The parallel mask-predict decoder
reconstructs a length-48 sequence at 0.999 accuracy in a fixed 8 passes vs 48
autoregressive steps — the **latency mechanism works**. Two fixes were decisive:
(a) learned positional keys into the skeleton for cross-attention alignment, and
(b) CMLM-style uniform mask-ratio training so the decoder sees the fully-masked
regime it starts inference from. The remaining gap is **stage 1**: flow-matching
a continuous latent skeleton from noise doesn't yet converge at toy scale
(generated skeleton ≠ encoder latent), so the end-to-end generated path is still
near random. This is the blueprint's flagged hardest pillar and the next research
target (consistency/rectified-flow training, larger scale, or a discrete
skeleton). Reproduce: `python experiments/p3_parallel_decode.py --steps 2000
--length 48 --skel_len 48 --decode_iters 8`.

## Layout

```
o1anti/
  config.py        # single O1AntiConfig dataclass
  nla.py           # pillar 1 — NeuralLiquidAdjacency, LiquidStateScan
  module_graph.py  # pillar 2 — ContextEncoder, GlobalRouter, ModuleLibrary
  generation.py    # pillar 3 — SkeletonGenerator, ParallelDecoder
  losses.py        # load balance, state continuity, flow matching
  model.py         # O1AntiModel — LM path + generation path
experiments/
  p1_nla_swap.py        # P1 — NLA vs dense attention
  p2_module_routing.py  # P2 — routed graph vs dense stack
  p3_parallel_decode.py # P3 — parallel decode vs autoregressive
tests/
  test_o1anti.py   # shapes, causality, train/inference consistency
```

## Quick start

```bash
python -m pytest tests/test_o1anti.py -q      # 9 tests
python experiments/p1_nla_swap.py --steps 1500 --seq 32
python experiments/p2_module_routing.py --steps 800 --regimes 4
python experiments/p3_parallel_decode.py --steps 2000 --length 48 --skel_len 48 --decode_iters 8
```

Requires `torch` (tested on 2.5.1). No custom CUDA kernels — the block-sparse
gather/scatter for NLA is on the scaling roadmap, not needed at prototype scale.
