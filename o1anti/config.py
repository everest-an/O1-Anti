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
    # Gumbel exploration noise on the routing scores (train only). Kept at 0:
    # the straight-through estimator in NeuralLiquidAdjacency.forward already
    # feeds gradients to non-selected positions, and an ablation (P1) shows any
    # noise > 0 now DESTROYS learning (0.045 vs 0.995 recall). Left as a knob
    # only for research; do not enable without re-checking that ablation.
    nla_route_noise: float = 0.0

    # --- pillar 2: Neural Module Graph ---
    n_modules: int = 8      # module library size (M)
    path_len: int = 4       # modules executed per input (L)
    d_ctx: int = 64         # context embedding dim
    router_tau: float = 1.0         # gumbel-softmax temperature
    load_balance_coef: float = 0.01
    d_ff: int = 256         # FFN width inside a module

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
    codebook_size: int = 256   # VQ codebook entries (discrete mode)
    vq_beta: float = 0.25      # commitment loss weight (discrete mode)
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
