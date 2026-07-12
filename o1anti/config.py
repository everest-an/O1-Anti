"""config.py — single dataclass config for the O1-Anti (MTLNN v2) stack."""

from dataclasses import dataclass, field


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
