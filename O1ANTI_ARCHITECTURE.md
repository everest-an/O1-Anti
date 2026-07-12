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

Full end-to-end (prompt → skeleton → parallel decode), 2500 steps, CPU:

| Path | Tok acc | Forward passes |
|---|---:|---:|
| Autoregressive baseline | 1.000 | 48 |
| **Parallel, end-to-end (regress skeleton)** | **1.000** | **14 (3.4× fewer)** |
| Parallel, faithful skeleton (stage-2 ceiling) | 1.000 | 6 |

**Go.** The full non-autoregressive pipeline reaches AR-equal accuracy (1.000)
generating all 48 tokens in **14 passes vs 48** — a 3.4× pass-count and 3.3×
wall-clock speedup. Stage 1 (`regress`) and stage 2 both proven; the other
skeleton modes also work end-to-end (`flow` 0.998, `discrete` 0.63 — see below).

Getting here took finding one decisive bug and two design fixes:

- **Position-embedding scale (the bug).** Init was `std=0.02`. When every
  position is masked (mask-predict's first round), each query is just
  `mask_emb + pos`, so a tiny `pos` makes all queries identical and
  cross-attention can't tell positions apart — parallel decode collapsed to
  ~random *even when handed the exact answer as the skeleton*. Setting
  `pos_emb_std≈1.0` fixed it: aligned reconstruction jumped 0.06 → 1.000 in a
  single pass. Locked in by `test_parallel_decoder_reconstructs_aligned_skeleton`.
- **Shared positional keys.** Output queries and skeleton keys share the same
  position embedding, giving cross-attention a diagonal alignment prior.
- **CMLM uniform mask-ratio training** so the decoder trains on the fully-masked
  regime it starts inference from.

Three interchangeable **stage-1** skeleton generators (`--skeleton_mode`):

| Mode | Mechanism | Stage-1 passes | Generated acc |
|---|---|---:|---:|
| `regress` (default) | deterministic prior (prompt→skeleton MSE + noise-robust decode) | 1 | **1.000** |
| `flow` | flow-matching neural ODE (stochastic) | `ode_steps` | 0.998 |
| `discrete` | VQ codebook + parallel code prior | 1 | 0.63 |

`regress` is best when the target is (near-)deterministic in the prompt; `flow`
is the stochastic/diverse-sampling path; `discrete` needs residual/product VQ to
raise codebook capacity (current single-code VQ caps fidelity). Reproduce:
`python experiments/p3_parallel_decode.py --steps 2500 --length 48 --skel_len 48
--decode_iters 6 --skeleton_mode regress`.

### P4 — All three pillars in one model (integration)

P1–P3 validate each pillar in isolation. P4 checks they **compose**: the real
`O1AntiModel` conditions generation on the routed module trunk (pillar 2, with
NLA from pillar 1 inside each module) and decodes non-autoregressively (pillar 3),
trained end-to-end with one objective. Same reconstruction task, len 48, 2000
steps, CPU:

| Metric | Value |
|---|---:|
| End-to-end generated tok-acc | **1.000** |
| Module path / library | 2 / 6 (pillar 2 live & sparse) |
| Param activation ratio | **73.8%** (< 100%) |
| Forward passes vs autoregressive | **7 vs 48** |

**Go.** All three pillars run together in a single trained model: it generates at
full accuracy, activates 74% of parameters (the module graph stays sparse — 2 of 6
modules per input; the always-on generation stack is the rest), and needs 7 passes
where autoregression needs 48. `test_generation_routes_through_module_trunk` locks
in that the module library and router receive gradient during a pure generation
step. Reproduce: `python experiments/p4_integrated.py --steps 2000 --length 48`.

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
  p4_integrated.py      # P4 — all three pillars in one model
tests/
  test_o1anti.py   # 14 tests: shapes, causality, consistency, all modes, P4
```

## Quick start

```bash
python -m pytest tests/test_o1anti.py -q      # 14 tests
python experiments/p1_nla_swap.py --steps 1500 --seq 32
python experiments/p2_module_routing.py --steps 800 --regimes 4
python experiments/p3_parallel_decode.py --steps 2000 --length 48 --skel_len 48 --decode_iters 8
python experiments/p4_integrated.py --steps 2000 --length 48
```

Regression anchors (fail the run if a pillar's headline metric drops):

```bash
python experiments/p2_module_routing.py --steps 800 --regimes 4 --assert-min 0.9
python experiments/p3_parallel_decode.py --steps 2000 --length 48 --skel_len 48 \
       --decode_iters 8 --skeleton_mode regress --assert-min 0.9
```

Requires `torch` (tested on 2.5.1). No custom CUDA kernels — the block-sparse
gather/scatter for NLA is on the scaling roadmap, not needed at prototype scale.
