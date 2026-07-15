# O1-Anti Dev-Plan Reality Map

A grounding document for the "160-200B MoE, DeepSeek-class" development plan. Its
purpose is narrow and honest: for every mechanism the plan names, state **what it
actually is in this codebase**, **what we have and haven't verified**, and **what
would have to be true** before the plan's use of it is justified. Written before
the decisive E11 result is in, so nothing here assumes the direction is validated.

Status legend: ✅ verified at toy scale · ⚠️ partial / conditional · ❌ not tested ·
🔬 under test now.

---

## The naming problem

The plan borrows three names from this repo — **NLA**, **residual/product VQ**,
**GWT bottleneck** — but attaches each to a *different* mechanism than the one we
built and measured. Before any of it goes into a 160B design, the names must be
pinned to real, testable mechanisms. Otherwise "engineering the innovations in"
is really "introducing three untested components under familiar labels."

### 1. "NLA Router" (plan) ≠ Neural Liquid Adjacency (repo)

| | Plan's "NLA Router" | Repo's NLA (`o1anti/nla.py`) |
|---|---|---|
| What it is | A router that adds **dynamic topology between MoE experts** — expert-to-expert adjacency that reshapes routing scores per input | A **sequence mixer** that replaces softmax attention: top-K content routing over compressed per-token states `c_j`, with a liquid state scan |
| Operates over | The expert set (a routing prior) | The token sequence (an attention replacement) |
| What we measured | ❌ nothing — this mechanism does not exist in the repo | ✅ P1 (matches dense attention on selective copy, 8× smaller cache); ⚠️ E8 (−22% on generic LM, robust to 5 ablations); ✅ E10 (beats dense on real-text retrieval at 2/3 lengths); ✅ E11 (attention-level MQAR recall at `wd=0.1`, pairs 8/16, beats Mamba) |

**Consequence.** The plan's headline "创新点1" is a brand-new idea that shares only
a name with the thing we have evidence for. It needs its own ablation from zero
(Task #3), and NLA's existing evidence does **not** transfer to it. If the plan
wants the *sequence-mixer* NLA instead (the thing with evidence), that is an
attention replacement, not a router — a different architectural slot entirely.

### 2. "Residual VQ for expert-weight compression" (plan) ≠ product-VQ skeleton codec (repo, E9)

| | Plan's use | Repo's VQ (`VectorQuantizer`, E9) |
|---|---|---|
| Quantizes | **Expert FFN weights** (a model-compression scheme) | The generation **skeleton latent** (an activation/latent codec in the non-autoregressive decoder) |
| Goal | Shrink expert params 30-50%, <2% quality loss | Make the discrete skeleton mode reach parity (fixed 0.63 → 1.000) |
| Evidence | ❌ never tested on weights | ✅ works as a latent codec at toy scale |

**Consequence.** Weight quantization and latent quantization are different problems
with different failure modes (weight VQ interacts with training dynamics, gradient
flow through the codebook, and per-expert redundancy that we have no data on).
"再压缩30~50%，精度损失<2%" is an **unverified target**, not a result — it must be
earned by an ablation, not assumed.

### 3. "GWT bottleneck" (plan, efficiency mechanism) — was "inspiration, not load-bearing"

In the MT-LNN line, the Global-Workspace / GWT framing was explicitly **relocated
to "inspiration, not load-bearing"** after a rigor review (recorded in the M1
project history). The plan reintroduces it as a concrete **cross-layer
communication-reduction** mechanism. That may be a fine idea — but it currently
has **no measured efficiency win**. Rule for inclusion: it must come with a
falsifiable metric (e.g. "reduces cross-layer bytes by X% at ≤Y% quality cost on a
1B ablation") or it stays out of the load-bearing design.

---

## What the accumulated evidence actually supports (as of this writing)

- **NLA is a memory-efficient sequence mixer with a real niche in sparse
  retrieval** (P1, E10), and a **real, robust weakness on generic LM** (E8, −22%,
  not closed by 5 ablations incl. token routing, full-dim/full-K, multi-head,
  hybrid-attention). Honest positioning so far: *specialist, not a general
  Transformer replacement.*
- **The generation stack (P3/E9) works at toy scale** — non-autoregressive decode
  at ~3× fewer passes, three skeleton modes at parity.
- **The pillars compose** in one small model (P4).
- ✅ **RESOLVED (E11): NLA has reliable attention-precise recall.** The
  load-bearing question — does NLA recall like attention rather than blur like an
  SSM — is now answered **yes**, with a caveat about *how* we learned it. At the
  default `weight_decay=0.01` NLA looked refuted (bimodal grok, collapse at
  pairs=16); a grokking diagnostic showed the cause was the hyperparameter. At
  `wd=0.1`, NLA reaches **1.000 = attention** on MQAR at pairs 8 and 16 (multiple
  seeds, GPU+CPU cross-validated) and **beats Mamba's 0.81**. pairs=32 is
  inconclusive (grokking cost explodes with pairs; attention doesn't grok it in
  15k CPU steps either) and needs a GPU-scale budget. Net: the *sequence-mixer*
  NLA's niche — SSM-cheap cache + attention-precise recall — is **supported**.
  (This does NOT transfer to the plan's "NLA Router"; see §1.)

## What must be true before the 160B plan is more than aspiration

1. ✅ **E11 — DONE, positive.** NLA reaches attention recall (pairs 8, 16) and
   beats Mamba, at `wd=0.1`. Remaining: pairs=32 + clean Mamba-at-`wd=0.1`
   comparison on GPU (pending compute). The core "efficient + precise-retrieval"
   thesis holds for the sequence-mixer. *(Task #1 → #2)*
2. **1B ablation still required** — E11 validates the sequence-mixer NLA, NOT the
   plan's *new* mechanisms (expert-topology router, weight-VQ). Those still need
   their own from-zero ablation vs a standard MoE at 1B, same data/budget; NLA's
   recall win does not transfer to them. *(Task #3)*
3. **A scaling curve** — measured 1B→(few B) trend, before any 160B / token-budget
   / cost / MMLU target is quoted as anything but a guess. *(Task #5)*

## Cost/scale caveat (flagged, not endorsed)

The plan's <100万 RMB for a 160B MoE on 1-3T tokens, "4 people to start", and
MMLU>65% are **not backed by any scaling evidence we hold**. Before these inform
any real financial commitment, they must be replaced by figures derived from the
1B scaling run and include full team + infrastructure cost, not GPU rent alone.
This is a high-consequence, outward-facing set of claims — treat as hypotheses.
