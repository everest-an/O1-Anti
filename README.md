<div align="center">

# O1-Anti

## The Anti-Transformer

**Attack the three cost roots of the Transformer — one architectural pillar each.**

[![MIT License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-18_passing-brightgreen?style=for-the-badge)](tests/test_o1anti.py)

Dynamic sparse memory · Context-routed sparse compute · Non-autoregressive generation

</div>

---

O1-Anti is a research architecture that replaces the three most expensive parts
of a Transformer, instead of tuning them. Each pillar targets one cost root, and
each is validated by a cheap, reproducible go/no-go experiment at toy scale
(hundreds of K params, CPU-runnable).

| Transformer cost root | O1-Anti pillar | Module |
|---|---|---|
| O(n²) attention + KV cache → **memory wall** | Neural Liquid Adjacency | [`o1anti/nla.py`](o1anti/nla.py) |
| 100% dense activation → **compute waste** | Context-routed Neural Module Graph | [`o1anti/module_graph.py`](o1anti/module_graph.py) |
| Serial token decode → **latency wall** | Liquid state-transition generation | [`o1anti/generation.py`](o1anti/generation.py) |

Full design and derivations: [`O1ANTI_ARCHITECTURE.md`](O1ANTI_ARCHITECTURE.md).

## Validation status (toy scale, CPU)

| Pillar | Experiment | Result | Verdict |
|---|---|---|---|
| **P1** memory | NLA vs dense attention (selective copy) | 100% recall matching dense attention, **8× smaller** per-token cache | **GO** |
| **P2** compute | Routed graph vs dense stack (regime classification) | 100% acc at **27% activation**, balanced module usage | **GO** |
| **P3** latency | Parallel decode vs autoregressive (len-48 reconstruction) | AR-equal 1.000 in **14 passes vs 48** (3.4× fewer) | **GO** |
| **P4** integration | All three pillars in one trained model | 1.000 generated at **74% activation, 7 passes vs 48** | **GO** |
| **E8** real text | Byte-level LM on WikiText-2 vs matched dense | **+22% BPB behind** dense (3.51 vs 2.87) | **GAP** |
| **E10** real text | Needle retrieval in real WikiText, 2 seeds/length | **RETRACTED** — the NLA "win" was a weight-decay artifact; at `wd=0.1` (both models regularized) it's a wash (L=128 dense 0.74 vs NLA 0.67, both bimodal) | **INCONCLUSIVE** |
| **E11** recall | MQAR: NLA vs attention vs Mamba (SSM) | at `wd=0.1` NLA **1.000 = attention** (pairs 8, 16, multi-seed) — *matches* the ceiling at lower cache; Mamba comparison not yet clean (was `wd=0.01`) | **GO (parity)** |

The four synthetic go/no-go tests (P1–P4) all pass, and the pillars **compose** in
one end-to-end model. The real-text and recall probes are more sobering, and a
**weight-decay confound** runs through them (documented honestly rather than
buried):

- **E8 (generic LM):** NLA trails a matched dense Transformer by ~22% BPB, and
  four ablations fail to close it. A real, intrinsic limit at this scale.
- **E10 (real-text retrieval): RETRACTED.** The earlier "NLA beats dense" result
  was run at the torch-default `weight_decay=0.01`. Re-running with **both** models
  properly regularized (`wd=0.1`) erases the gap — L=128 becomes a bimodal wash
  (dense 0.74 vs NLA 0.67), L=512 both fail to grok in budget. NLA's apparent
  real-text retrieval advantage does **not** survive a fair comparison.
- **E11 (associative recall, MQAR):** the one clean positive. At `wd=0.1` NLA
  reliably reaches **attention-level** recall (1.000 at pairs 8 and 16, multiple
  seeds, cross-validated) — it *matches* the ceiling, at a fraction of the cache.
  It does not *exceed* attention; the value is parity-quality at lower cache cost.
  (The Mamba comparison is not yet clean — Mamba was at `wd=0.01`; needs a GPU
  re-run.)

**Honest current read:** the strongest supported claim is *NLA matches attention
on synthetic content-recall at ~6–8× smaller cache* (E11 + the analytic cache
win). The real-text retrieval advantage (E10) did not hold up; generic LM (E8) is
a real weakness. This is a narrower, efficiency-parity story — not a demonstrated
quality advantage on real tasks. Full analysis in `O1ANTI_ARCHITECTURE.md`
§ E8/E10/E11; honest positioning in `O1ANTI_POSITIONING.md`.

> **Positioning.** The honest, evidence-based product thesis — *a
> memory-efficient architecture for retrieval- and long-context-heavy
> workloads*, not a general-purpose Transformer/DeepSeek replacement — and an
> evidence-gated roadmap are in [`O1ANTI_POSITIONING.md`](O1ANTI_POSITIONING.md).
> Mechanisms proposed but not validated (expert-topology routing — ablated, no
> benefit; weight-VQ — untested; GWT bottleneck — no measured win) are tracked in
> [`DEV_PLAN_REALITY_MAP.md`](DEV_PLAN_REALITY_MAP.md).

### P1 — Neural Liquid Adjacency (memory)

Instead of caching K and V (`2·d_model` per token per layer), each position
caches only a compressed state `c_j = W_c x_j` with `d_c ≪ d_model`; routing keys
and values both derive from `c_j`, so the **inference cache is `O(n·d_c)`**. A
parallel-scan liquid state `s_t` summarizes the prefix and gates the output,
making the adjacency context-dependent ("liquid"). Each token attends to at most
`top_k` past positions by content score. A straight-through estimator keeps the
forward pass sparse while gradients flow through a dense softmax.

| Variant | Params | Recall acc | Cache B/tok/layer |
|---|---:|---:|---:|
| Dense attention | 409K | 1.000 | 512 |
| **Neural Liquid Adjacency** | 391K | 1.000 | **64 (8×)** |

### P2 — Neural Module Graph (compute)

No fixed layer stack. A context encoder summarizes the input once; a global
router commits to an ordered path of `path_len` modules out of a library of
`n_modules`. **Only modules on the path run** — activation ratio is
`path_len / n_modules`. Straight-through Gumbel-softmax routing plus a
load-balance penalty prevents module collapse. New capability = grow the library,
finetune only the router.

| Variant | Acc | Active params | Activation |
|---|---:|---:|---:|
| Dense stack (runs all 8) | 1.000 | 559,876 | 100% |
| **Routed graph (runs 2)** | 1.000 | **153,972** | **27%** |

### P3 — Liquid state-transition generation (latency)

Two non-autoregressive stages: a stage-1 generator produces a compact "semantic
skeleton" from the prompt, and a stage-2 parallel decoder cross-attends into it
to emit **all tokens at once**, refined over a fixed number of mask-predict
rounds. A length-`n` generation costs `stage1 + decode_iters` passes, not `n`.

| Path | Tok acc | Forward passes |
|---|---:|---:|
| Autoregressive baseline | 1.000 | 48 |
| **Parallel, end-to-end** | 1.000 | **14 (3.4× fewer)** |

Three interchangeable stage-1 skeleton generators (`--skeleton_mode`), all now at
parity: `regress` (deterministic, 1.000), `flow` (flow-matching neural ODE,
0.998), `discrete` (product-quantized VQ codes, 1.000 — fixed from an earlier
0.63 by splitting the codebook into independent subvector groups, see
`O1ANTI_ARCHITECTURE.md` § E9).

### P4 — All three pillars in one model

The full `O1AntiModel` conditions generation on the routed module trunk (pillars
1+2) and decodes non-autoregressively (pillar 3), trained end-to-end. It generates
at **1.000** accuracy while activating **74%** of parameters (the module graph
stays sparse — 2 of 6 modules per input) in **7 passes vs 48** for autoregression.
The pillars compose without interference.

> The decisive fix was a subtle one: position embeddings in the generation stack
> must use O(1) init, not the usual 0.02. In mask-predict's first round every
> position is masked, so a tiny position signal makes all queries identical and
> cross-attention cannot address positions. See `O1ANTI_ARCHITECTURE.md` § P3.

## Quick start

```bash
pip install torch numpy
python -m pytest tests/test_o1anti.py -q                                   # 18 tests

python experiments/p1_nla_swap.py --steps 1500 --seq 32                     # P1
python experiments/p2_module_routing.py --steps 800 --regimes 4            # P2
python experiments/p3_parallel_decode.py --steps 2500 --length 48 \
       --skel_len 48 --decode_iters 6 --skeleton_mode regress             # P3
python experiments/p4_integrated.py --steps 2000 --length 48               # P4
```

Minimal end-to-end use:

```python
import torch
from o1anti import O1AntiConfig, O1AntiModel

cfg = O1AntiConfig(vocab_size=256, d_model=128, n_modules=8, path_len=4)
model = O1AntiModel(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 64))
out = model(ids, labels=ids)          # causal-LM path (understanding)
out.loss.backward()

prompt = torch.randint(0, cfg.vocab_size, (2, 16))
tokens = model.generate(prompt, length=48)   # non-autoregressive generation
```

## Layout

```
o1anti/
  config.py        # single O1AntiConfig dataclass
  nla.py           # pillar 1 — NeuralLiquidAdjacency, LiquidStateScan
  module_graph.py  # pillar 2 — ContextEncoder, GlobalRouter, ModuleLibrary
  generation.py    # pillar 3 — SkeletonEncoder/Generator/Prior, VQ, ParallelDecoder
  losses.py        # load balance, state continuity, flow matching
  model.py         # O1AntiModel — LM path + generation path
experiments/       # P1–P4 go/no-go harnesses (p4_integrated = all pillars)
tests/             # 18 tests: shapes, causality, consistency, all modes, P4, multi-head NLA, PQ
```

---

## Origin: MT-LNN (inspiration, not load-bearing)

O1-Anti's "liquid" ideas descend from **MT-LNN** (Microtubule-Inspired Liquid
Neural Network), the project's first architecture line — a closed-form
liquid-time-constant network with a microtubule-inspired protofilament structure
that replaces the Transformer FFN. The liquid state-space recurrence in
`o1anti/nla.py` is a direct descendant of MT-LNN's LTC layer, repurposed here to
drive sparse routing rather than to replace the FFN.

The biological framing (13 protofilaments, GTP-cap gating, anesthesia-response
hooks, IIT-Φ) is **inspiration for the priors, not a scientific claim** — it
motivated the architecture but is not what the benchmarks measure. MT-LNN's
own empirical results, papers, pitch decks, and training pipelines remain in
this repository (`BENCHMARKS.md`, `mt_lnn_arxiv.pdf`, `PRD.md`, `SPEC.md`,
`train.py`, `train_llama_mt_adapter.py`, and the figures/decks). Where any doc
disagrees on numbers, the benchmark tables are canonical.

O1-Anti is the forward-looking line: same liquid intuition, but pushed to the
architectural level as a systematic answer to Transformer cost, with every claim
backed by a reproducible go/no-go experiment.

## License

MIT — see [LICENSE](LICENSE).
