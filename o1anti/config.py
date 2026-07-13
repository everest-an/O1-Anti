"""config.py — single dataclass config for the O1-Anti (MTLNN v2) stack."""

from dataclasses import dataclass


@dataclass
class O1AntiConfig:
    # --- core dims ---
    vocab_size: int = 256
    d_model: int = 128
    max_seq_len: int = 512

    # --- pillar 1: Neural Liquid Adjacency ---
    d_c: int = 32           # compressed per-position state (the only thing cached)
    d_state: int = 64       # liquid global state s_t dimension (m in the spec)
    top_k: int = 8          # neighbors per token
    nla_heads: int = 1      # NLA routing/value heads. Cache stays O(n·d_c) (keys
                            # and values are projected per-head from the shared
                            # cached c_j), but H heads give attention-style
                            # multi-relation mixing. d_model must be divisible by H.
    # Sub-quadratic training path (block-sparse two-stage top-K). 0 = exact dense
    # O(T²) scoring (default, matches all validated results). >0 = block size:
    # queries first pick nla_cand_blocks candidate blocks by block-summary score
    # (O(T·T/bs)), then do fine top-K only within those blocks (O(T·cand·bs)).
    # With bs≈√T this is O(T^1.5). Approximate — the picked neighbours can differ
    # from the exact top-K, so it is an inference/training-speed lever, opt-in.
    nla_block_size: int = 0
    nla_cand_blocks: int = 4    # candidate blocks kept per query (own block always in)
    # Gumbel exploration noise on the routing scores (train only). Kept at 0:
    # the straight-through estimator in NeuralLiquidAdjacency.forward already
    # feeds gradients to non-selected positions, and an ablation (P1) shows any
    # noise > 0 now DESTROYS learning (0.045 vs 0.995 recall). Left as a knob
    # only for research; do not enable without re-checking that ablation.
    nla_route_noise: float = 0.0

    # --- pillar 2: Neural Module Graph ---
    n_modules: int = 8      # module library size (M) / MoE expert count
    path_len: int = 4       # modules executed per input (L) — "global" routing
    d_ctx: int = 64         # context embedding dim
    router_tau: float = 1.0         # gumbel-softmax temperature
    load_balance_coef: float = 0.01
    d_ff: int = 256         # FFN width inside a module / MoE expert
    # routing_granularity:
    #   "global" — one module path per whole input (context-routed). Validated
    #              on the P2 classification task; too coarse for language modeling.
    #   "token"  — dense NLA backbone (pillar 1) + per-TOKEN routed MoE-FFN
    #              (pillar 2, fine-grained). n_layers blocks; each token picks
    #              moe_top_e of n_modules FFN experts. Better for real LM.
    routing_granularity: str = "global"
    n_layers: int = 4       # trunk depth for token-MoE routing
    moe_top_e: int = 1      # experts activated per token (top-e)
    # Hybrid trunk (token routing only). 0 = every layer mixes with NLA (default,
    # matches validated results). N>0 = every N-th layer (i where (i+1)%N==0) uses
    # FULL causal attention instead of NLA, the rest use NLA — the Jamba/Zamba
    # "interleave attention with a cheap mixer" pattern. Motivation: E8 showed
    # all-NLA trades ~22% LM quality vs dense and 4 ablations couldn't close it;
    # a few full-attention layers restore the dense many-relations mixing while
    # NLA layers keep the cheap long-range path. Attention layers cache full KV
    # (the memory win applies only to the NLA layers).
    hybrid_attn_every: int = 0
    # Noisy top-k gating (Shazeer et al. 2017). Train-only Gaussian noise on the
    # gate logits before top-e selection, std = moe_noise / n_modules — a
    # standard defense against expert collapse (a dead expert never runs, so only
    # the load-balance loss can resurrect it; noise lets it occasionally win a
    # token and get a real gradient). Combine weights + load-balance usage still
    # come from the clean softmax, so it's unbiased and eval is deterministic.
    # EMPIRICAL NOTE: at prototype scale the plain gating (moe_noise=0) already
    # keeps all experts balanced — a collapse diagnostic showed 8/8 experts
    # active at MAX entropy with noise both OFF and ON (the load_balance_coef
    # penalty suffices). So this is available hardening for larger/harder runs,
    # NOT a fix for an observed collapse. Default 0 (exact, matches validated E8).
    moe_noise: float = 0.0

    # --- pillar 3: liquid state-transition generation ---
    skel_len: int = 16      # semantic skeleton length (L_skel)
    ode_steps: int = 8      # Euler steps for the flow-matching ODE (continuous mode)
    decode_iters: int = 3   # mask-predict refinement rounds
    n_dec_layers: int = 2   # parallel decoder depth
    n_dec_heads: int = 4
    # Position-embedding init scale for the generation stack. MUST be O(1), not
    # 0.02: when every position is masked the query is just mask_emb + pos, so a
    # tiny pos makes all queries identical and cross-attention cannot tell
    # positions apart (parallel decode collapses to ~random). std≈1 fixes it.
    pos_emb_std: float = 1.0
    # skeleton_mode:
    #   "regress"  (deterministic prompt→skeleton predictor, 1 pass, default) —
    #              works when the target is (near-)deterministic in the prompt.
    #   "flow"     (flow-matching neural ODE, ode_steps passes) — stochastic.
    #   "discrete" (VQ codebook + parallel code prior, 1 pass) — needs
    #              residual/product VQ to raise capacity; research variant.
    skeleton_mode: str = "regress"
    codebook_size: int = 256   # entries per VQ codebook (discrete mode)
    vq_beta: float = 0.25      # commitment loss weight (discrete mode)
    # Product quantization: split d_model into vq_groups independent subvectors,
    # each with its OWN codebook_size-entry codebook. Combinatorial code space is
    # codebook_size^vq_groups while total codebook params stay codebook_size*d_model
    # (same as a single codebook). Fixes the single-codebook fidelity cap (E9 —
    # was capped at cos-sim~0.63 / 0.63 decode acc with vq_groups=1).
    vq_groups: int = 4
    skel_noise: float = 0.5    # regress mode: Gaussian noise (× per-slot std)
                               # added to the skeleton in decoder training so the
                               # decoder tolerates the prior's regression error.

    # --- regularization ---
    state_continuity_coef: float = 1e-4
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even (time embedding)"
        assert self.top_k >= 1
        assert self.path_len <= self.n_modules or self.n_modules > 0
        assert self.d_model % self.nla_heads == 0, "d_model must divide nla_heads"
        assert self.d_model % self.vq_groups == 0, "d_model must divide vq_groups"
