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

**Training compute.** The exact path (default) scores densely — O(n²) — so the
memory win (O(n·d_c) cache) is real but training FLOPs are not sub-quadratic.
`cfg.nla_block_size > 0` enables a **two-stage block-sparse** path: queries pick
`nla_cand_blocks` candidate blocks by block-summary score (coarse, own block
always kept), then run fine top-K only inside those blocks. At block size ≈√T
this is **O(T^1.5)**, causality stays exact (an absolute-position mask on the
fine stage), and it approximates the exact top-K well — cosine to the exact
output ≈0.96–0.99 up to T=512, with the score-op reduction growing from ≈1.4×
(T=100) to ≈4.3× (T=512). It is opt-in and approximate, so all validated
results below use the exact path.

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
wall-clock speedup. Stage 1 (`regress`) and stage 2 both proven; all three
skeleton modes now reach ~1.000 end-to-end (`flow` 0.998, `discrete` 1.000 after
the E9 fix below).

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
| `discrete` | product-quantized VQ codes + parallel code prior | 1 | **1.000** (E9) |

`regress` is best when the target is (near-)deterministic in the prompt; `flow`
is the stochastic/diverse-sampling path; `discrete` gives a genuinely discrete
latent (useful for downstream discrete-token pipelines) with no quality loss
after E9.

**E9 — product quantization fixed `discrete` (0.63 → 1.000).** The single
codebook (`vq_groups=1`) capped nearest-neighbour fidelity at cos-sim≈0.6 —
curse-of-dimensionality on `d_model`-dim nearest-neighbour search with a
codebook_size=256 codebook. Fix: split `d_model` into `vq_groups=4` independent
subvectors, each with its own 256-entry codebook (product quantization). Same
total codebook parameters (`codebook_size × d_model`), but combinatorial code
space grows to `codebook_size^vq_groups` and each low-dimensional
nearest-neighbour match is far more precise. Measured fidelity (untrained,
random target): mean cos-sim 0.29 (G=1) → 0.55 (G=4) → 0.72 (G=8). End-to-end
P3 discrete-mode result: **1.000 generated accuracy in 7 passes vs AR's 48**,
matching `regress`/`flow`. Locked in by
`test_product_quantizer_shapes_and_fidelity`.

Reproduce: `python experiments/p3_parallel_decode.py --steps 2500 --length 48
--skel_len 48 --decode_iters 6 --skeleton_mode regress` (or `--skeleton_mode
discrete` for the E9 result).

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

### E8 — Real-text language modeling (first non-synthetic evidence)

P1–P4 run on synthetic tasks. E8 is the first real-text check: byte-level causal
language modeling on WikiText-2, comparing the O1-Anti trunk (context-routed NLA
modules) against a dense Transformer **matched to O1-Anti's active LM params**
(generation stack excluded, tied weights deduped, only `path_len` modules
counted), reported as validation bits-per-byte (BPB).

WikiText-2, byte-level, seq 128, d_model 128, 1500 steps, CPU (10.9M train bytes):

| Model | Val bits/byte | Active params | Train wall-clock |
|---|---:|---:|---:|
| Dense Transformer (3 layers) | **2.87** | 644K | 190 s |
| **O1-Anti trunk** (NLA + 3/6 modules) | 3.56 | 603K | 439 s |

**Gap, not GO — the honest finding.** At matched active params the O1-Anti trunk
lands **+23.9% BPB behind** a dense Transformer on generic English LM. Both are
still improving at step 1500 (dense faster), so neither is converged, but the
gap is real and consistent throughout training. This contrasts sharply with the
synthetic tasks (P1–P4), where O1-Anti *matched* dense — so the gap is specific
to broad language modeling, not a universal deficit.

**We then tried to close it with four ablations** (byte-level, seq 128, d_model
128, 1500 steps unless noted, same protocol):

| Variant | Val BPB | vs dense | Lever tested |
|---|---:|---:|---|
| Dense Transformer | **2.87** | — | baseline |
| O1-Anti, **global** routing | 3.56 | +23.9% | — |
| O1-Anti, **token** MoE routing | 3.51 | +22.1% | routing granularity |
| O1-Anti, token + **full-dim, full-K NLA** (d_c=128, top_k=128) | 3.70¹ | — | compression + sparsity |
| O1-Anti, token + **4-head NLA** | 3.52 | +22.6% | operator head count |

¹ 1000 steps (vs 3.67 for compressed NLA at that step), and it tracked *worse*
throughout — extra capacity didn't help even discounting undertraining.

**Conclusion — the gap is robust; none of the obvious levers move it.** Token-level
routing, full-dimension/full-density NLA, and multi-head NLA each closed
essentially none of the ~22% gap. (We predicted multi-head would help — it did
not; reported here rather than buried.) The ~22% deficit vs a matched dense
Transformer is stable across every routing/NLA knob we varied.

The honest read: O1-Anti's departures from dense attention — values routed through
a small compressed state, top-K sparse aggregation, liquid gating — **collectively
trade generic-LM quality for the memory and compute savings**, and that trade
appears intrinsic to the design at this scale rather than a tunable hyperparameter.
This is consistent with the pillar results: NLA *matches* dense on synthetic
sparse-retrieval (P1, its home turf) but lags on the dense, local, many-relations
mixing generic LM rewards. The architecture's real value proposition is therefore
**efficiency where sparse long-range structure dominates** (the O(n·d_c) cache and
sparse compute are genuine and preserved), not parity with dense attention on
broad language modeling. Closing the LM gap, if possible, needs a change more
fundamental than the knobs tested here — likely to the value/compression path
(which is exactly what buys the memory win, so the trade may be irreducible).

Reproduce: `python experiments/train_o1anti.py --steps 1500 --seq 128 --d_model
128` (global); add `--routing token`, `--nla_heads 4`, or `--d_c 128 --top_k 128`
for the ablations.

### E10 — Sparse retrieval on real text (NLA's actual home turf)

E8 measured generic LM — NLA's documented weak spot. E10 tests the setting P1
already validated NLA for (sparse content-based retrieval), but embedded in real
WikiText passages instead of synthetic tokens, swept across context length. A
synthetic `[KEY:xxxxx]=yy` fact is inserted at a random depth in a real-text
haystack and queried at the end; scored by exact value match.

**This task shows threshold/grokking-like convergence**, not smooth learning:
loss sits on a high plateau, then drops sharply at an unpredictable step. A
first single-seed run was consequently noisy — NLA won big at two lengths but
lost at the third — and was NOT reported as a result; it was reported as
"inconclusive, needs multiple seeds" (see git history for the honest negative
version). Rerun with 2 seeds per cell, 6000 steps (the budget that reliably
converges in this regime), batch 24, d_model 96:

| ctx_len | steps | dense exact (mean ± std) | NLA exact (mean ± std) | cache ratio |
|---:|---:|---:|---:|---:|
| 128 | 6000 | 0.265 ± 0.040 | **0.748 ± 0.192** | 6.0× |
| 256 | 6000 | 0.117 ± 0.042 | 0.073 ± 0.043 | 6.0× |
| 256 | 12000 | 0.172 ± 0.022 | 0.438 ± **0.413** | 6.0× |
| 512 | 6000 | 0.035 ± 0.005 | **0.837 ± 0.123** | 6.0× |

**Go at L=128 and L=512; L=256 is bimodal (grok-or-not).** At the two lengths
where NLA reliably crosses its convergence threshold, it beats dense by a wide
margin **in both seeds independently** — not single-seed luck — while caching 6×
less per token. L=256 is the honest asterisk: at 6000 steps NLA looked simply
undertrained (0.073); extending to 12000 steps let **one** seed grok
spectacularly (0.850, loss dropping at step ~9000) while the **other** never
crossed the threshold (0.025) — hence the huge ±0.413 spread. So at L=256 NLA's
*ceiling* clearly exceeds dense (0.85 vs 0.15–0.20), but whether it groks within
budget is seed-dependent and unreliable; more seeds would be needed to estimate
the grok probability. We report this rather than cherry-picking the 0.850 seed.

Dense, by contrast, plateaus at ~0.15–0.20 across L=256 regardless of budget
(loss flat from step 3000 to 12000) — it isn't grokking-limited, it just tops
out lower.

This result and E8 are not in tension: E8 stresses NLA's weak spot (dense,
many-relations LM mixing); E10 stresses its strength (sparse, content-addressed
retrieval), where its ceiling is far higher than dense — though at some lengths
its optimization is a threshold/grokking process that a fixed CPU-scale budget
doesn't always reach. Reproduce: `python experiments/e10_real_needle.py --steps
6000 --lengths 128 256 512 --ans_len 2 --batch 24 --d_model 96 --seeds 0 1`
(add `--steps 12000 --lengths 256` for the extended L=256 probe).

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
  train_o1anti.py       # E8 — real-text byte-level LM, BPB vs dense
  e10_real_needle.py    # E10 — real-text sparse retrieval, multi-seed
tests/
  test_o1anti.py   # 18 tests: shapes, causality, consistency, all modes, P4, multi-head NLA, PQ
```

## Quick start

```bash
python -m pytest tests/test_o1anti.py -q      # 18 tests
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

Requires `torch` (tested on 2.5.1). No custom CUDA kernels. A pure-PyTorch
sub-quadratic training path for NLA is available (`cfg.nla_block_size > 0`,
two-stage block-sparse, O(T^1.5)); a fused block-sparse CUDA kernel would push
it further but is not needed at prototype scale.
