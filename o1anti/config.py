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
    nla_route_noise: float = 1.0   # gumbel noise scale on routing scores (train only)

    # --- pillar 2: Neural Module Graph ---
    n_modules: int = 8      # module library size (M)
    path_len: int = 4       # modules executed per input (L)
    d_ctx: int = 64         # context embedding dim
    router_tau: float = 1.0         # gumbel-softmax temperature
    load_balance_coef: float = 0.01
    d_ff: int = 256         # FFN width inside a module

    # --- pillar 3: liquid state-transition generation ---
    skel_len: int = 16      # semantic skeleton length (L_skel)
    ode_steps: int = 8      # Euler steps for the flow-matching ODE
    decode_iters: int = 3   # mask-predict refinement rounds
    n_dec_layers: int = 2   # parallel decoder depth
    n_dec_heads: int = 4

    # --- regularization ---
    state_continuity_coef: float = 1e-4
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even (time embedding)"
        assert self.top_k >= 1
        assert self.path_len <= self.n_modules or self.n_modules > 0
