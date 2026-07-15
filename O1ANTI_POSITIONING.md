# O1-Anti — Evidence-Based Positioning

A repositioning of the "160-200B MoE, DeepSeek-class" development plan, grounded
in what the experiments actually showed (see `O1ANTI_ARCHITECTURE.md` for the raw
results, `DEV_PLAN_REALITY_MAP.md` for the plan-vs-repo mechanism mapping). It
keeps the ambition but points it where the evidence points.

## 1. Validation scorecard (what we can and can't claim)

| Claim | Verdict | Evidence |
|---|---|---|
| NLA is a **memory-efficient sequence mixer** (O(n·d_c) cache, ~6–8× smaller than KV) | ✅ **holds** | analytic + P1 |
| NLA does **attention-precise content recall** | ✅ **holds** (at `wd=0.1`) | E11 MQAR: 1.000 = attention at pairs 8/16, beats Mamba 0.81 |
| NLA **beats dense on real-text sparse retrieval** | ✅ **holds** (2/3 lengths) | E10: L=128/512, 6× cache |
| NLA matches dense on **generic language modeling** | ❌ **fails** | E8: −22% BPB, robust to 5 ablations |
| "NLA Router" (expert-topology MoE routing) is a differentiator | ❌ **no benefit** | 3-seed ablation: 3.366 vs 3.371, < noise, +2.6% params |
| Residual-VQ **weight** compression works | ❓ **untested** | (E9 validated VQ as a *latent* codec, not weights) |
| GWT bottleneck is an efficiency win | ❌ **no measured win** | relocated to "inspiration" in MT-LNN |
| 160B / 1–3T tokens / <100万 RMB / MMLU>65% | ❌ **no scaling evidence** | — |

**One-line reading:** the *sequence-mixer* NLA is a genuine, validated asset —
**efficient long-range retrieval**. The plan's other three differentiators are
either disproven (expert-router), untested (weight-VQ), or non-load-bearing
(GWT). And the plan *misuses* the one thing that works (NLA-the-mixer) as an
expert-router, which it is not.

## 2. The honest product thesis

Not "an anti-Transformer that beats DeepSeek on everything." The evidence
supports something narrower and real:

> **A memory-efficient architecture for retrieval- and long-context-heavy
> workloads**, where the KV-cache is the bottleneck and the task is dominated by
> *finding and copying the right information* rather than dense many-relations
> reasoning.

Where it wins (evidence-backed): long-document QA, retrieval-augmented
generation, agent/long-memory contexts, needle-in-haystack — anywhere the cache
cost dominates and recall precision matters. Where it doesn't (evidence-backed):
generic open-domain language modeling, where dense attention's many-relations
mixing is worth its cost (E8's −22% is real and unclosed).

This is a **defensible niche**, not a defeat. "Efficient recall at SSM-cache
cost" is exactly the gap the Based/Zoology line cares about, and NLA clears it.

## 3. Evidence-gated roadmap (replaces the 160B leap)

Each stage must pass before the next is funded. No parameter/cost/metric target
is quoted until a measured scaling curve exists.

- **Stage 0 — close the recall proof (cheap, needs a little GPU).**
  Finish E11: pairs=32 + a clean Mamba comparison at `wd=0.1`; re-test E10 L=256
  at `wd=0.1` (likely the same grokking fix). Deliverable: NLA's retrieval
  advantage stated with full pair/length scaling and a real-SSM baseline. Cost:
  a few GPU-hours.

- **Stage 1 — a *hybrid* small model on a retrieval task suite (1 GPU, days).**
  NLA layers for cheap long-range + a few dense-attention layers for the mixing
  NLA lacks (the Jamba/Zamba pattern is already in the repo). Train ~100–300M on
  long-context/retrieval benchmarks (needle, long-doc QA) vs a dense baseline and
  a Mamba baseline, matched params. Deliverable: does the hybrid keep NLA's cache
  win while closing the generic-LM gap? Standard MoE FFN — no expert-router,
  no weight-VQ (both unsupported).

- **Stage 2 — measure a scaling curve (multi-GPU, weeks).**
  Only after Stage 1 is positive: train 3 sizes, fit the loss-vs-params/tokens
  trend on the *target* (retrieval-heavy) distribution. Deliverable: the first
  honest basis for any size/token/cost projection.

- **Stage 3 — scale to the size the curve justifies (not a preset 160B).**

Optional, independent of the above: if MoE *serving cost* is a goal, test
weight-VQ as a standalone compression study (quality-loss vs compression-ratio) —
but as a deployment optimization, not an architecture differentiator.

## 4. Cost/scale — honest version

The plan's <100万 RMB / 160B / MMLU>65% figures have **no evidentiary basis** and
should not inform any financial commitment. Replace with:

- **Now:** unknown. We have toy-scale (hundreds of K params, CPU) go/no-go
  signals only. Any B-scale number today is a guess.
- **Earnable after Stage 2:** a real cost model, derived from the measured
  scaling curve, including team + infrastructure + data pipeline — not GPU rent
  alone. Until then, treat all large-scale numbers as hypotheses.

## 5. What to drop, and why

- **"NLA Router" / expert-topology routing** — ablated, no benefit. Use a
  standard MoE gate.
- **GWT bottleneck as a load-bearing efficiency mechanism** — no measured win;
  keep only as inspiration unless it earns a falsifiable metric.
- **The "three innovations" differentiation framing** — two don't hold, one is
  untested. The differentiation is NLA-the-mixer's retrieval efficiency, full stop.
- **Preset 160B / cost / MMLU targets** — replace with the scaling curve from the
  roadmap.

## 6. What to keep and build on

- **NLA as an efficient retrieval mixer** (the validated core).
- **The hybrid trunk** (NLA + occasional dense attention) — the honest answer to
  E8's generic-LM gap, already implemented.
- **Standard, proven MoE** for the FFN — sparsity without the unsupported extras.
- **The non-autoregressive generation stack** (P3/E9) — a separate validated
  capability, useful where latency matters.
- **The `wd=0.1` grokking lesson** — bake it into defaults for recall training.
