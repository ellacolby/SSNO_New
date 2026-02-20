import math
import torch
import torch.nn as nn


class NAOLayer(nn.Module):
    """
    Single iterative attention layer of the Nonlocal Attention Operator (NAO).

    Implements one step of the update (Eq. 9 of the paper):

        J_t = Attn(J_{t-1} ; θ_t)  J_{t-1}  +  J_{t-1}

    where the attention matrix is

        Attn[J ; θ_t] = σ( J W_{Qt} W_{Kt}^T J^T / √d_k )

    and σ is the *linear (identity) activation* as recommended in the
    paper (following Yu et al. 2024, Cao et al. 2021, Lu et al. 2025).
    This means the N × N kernel is computed without softmax, giving a
    linear attention map that is bilinear in the spatial features.

    A LayerNorm is applied after the residual addition to stabilise
    training in the absence of a normalising activation.

    Args:
        d_model (int): Feature dimension of J at this layer.
        d_k     (int): Dimension of query/key projections.
    """

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        """
        Args:
            J: (batch, N, d_model)   — current NAO state.

        Returns:
            J_new: (batch, N, d_model)  — updated state (residual + norm).
        """
        Q = self.W_Q(J)   # (batch, N, d_k)
        K = self.W_K(J)   # (batch, N, d_k)

        # Linear (non-softmax) N × N attention matrix: σ(Q K^T / √d_k)
        # where σ = identity  (Eq. 10)
        A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)   # (batch, N, N)

        # Nonlocal aggregation with residual
        J_new = A @ J + J   # (batch, N, d_model)
        return self.norm(J_new)


class NAO(nn.Module):
    """
    Nonlocal Attention Operator (NAO) — the *implicit BDF step* of ASNO.

    Given the temporal extrapolation H_{m+1} (from the Transformer Encoder)
    and the forcing field F_{m+1}, NAO approximates the solution to the
    implicit nonlinear static equation (BDF implicit step, Eq. 4):

        X̃_{m+1}  −  X_{m+1}  +  Δt · β · F( (m+1)Δt, X_{m+1} )  =  0

    by learning a data-driven nonlocal kernel operator (Eq. 12):

        X^out(y)  =  ∫ K[H, F](y, z)  F(z)  dz

    The kernel K ∈ R^{N × N} is constructed from two attention matrices
    (Eq. 11):

        K  =  W_{P,h} · σ( Q_h K_h^T / √d_k )
           +  W_{P,f} · σ( Q_f K_h^T / √d_k )

    where
        Q_h, K_h  come from J_T  (encodes both H and F after T layers),
        Q_f       comes from F    (direct forcing attention),
        σ = identity (linear activation),
        W_{P,h}, W_{P,f}  are learnable scalar mixing weights.

    Architecture
    ------------
    1. Project cat(H, F)  →  J_0  ∈  R^{N × d_model}.
    2. Apply T NAOLayer steps (iterative attention with residual).
    3. Compute K ∈ R^{N × N} via two attention cross/self-products.
    4. Output  X^out = K @ F,  then a linear projection to d_out.

    Args:
        d_h       (int): Feature dimension of H per spatial point.
        d_f       (int): Feature dimension of F per spatial point.
        d_k       (int): Query/key dimension for attention.
        d_model   (int): Internal model dimension after input projection.
        n_layers  (int): Number of iterative attention layers (T in paper).
        d_out     (int): Output feature dimension per spatial point.
    """

    def __init__(
        self,
        d_h: int,
        d_f: int,
        d_k: int,
        d_model: int,
        n_layers: int,
        d_out: int,
    ):
        super().__init__()
        self.d_k = d_k

        # Project concatenated (H ∥ F) to internal model dimension
        self.input_proj = nn.Linear(d_h + d_f, d_model)

        # T iterative attention layers  (Eq. 9-10)
        self.attn_layers = nn.ModuleList(
            [NAOLayer(d_model, d_k) for _ in range(n_layers)]
        )

        # Kernel weights — J_T branch  (W_{Q_{T+1}}, W_{K_{T+1}})
        self.W_Q_h = nn.Linear(d_model, d_k, bias=False)
        self.W_K_h = nn.Linear(d_model, d_k, bias=False)

        # Kernel weights — F branch  (separate Q projection for forcing)
        self.W_Q_f = nn.Linear(d_f, d_k, bias=False)

        # Learnable scalar mixing weights W_{P,h} and W_{P,f}  (Eq. 11)
        self.W_Ph = nn.Parameter(torch.ones(1))
        self.W_Pf = nn.Parameter(torch.ones(1))

        # Output projection: maps d_f → d_out after kernel application
        self.output_proj = nn.Linear(d_f, d_out)

    def forward(self, H: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch, N, d_h)   — temporal extrapolation from TE (H_{m+1}
                                    reshaped to per-spatial-point features).
            F: (batch, N, d_f)   — forcing field at the next time step.

        Returns:
            X_out: (batch, N, d_out)  — predicted next state X_{m+1}.
        """
        # ── Step 1: form J_0 = (H_{1:d}, F_{1:d})  ──────────────────────────
        J = torch.cat([H, F], dim=-1)   # (batch, N, d_h + d_f)
        J = self.input_proj(J)          # (batch, N, d_model)

        # ── Step 2: T iterative attention steps  ─────────────────────────────
        for layer in self.attn_layers:
            J = layer(J)                # (batch, N, d_model)

        # ── Step 3: build the nonlocal kernel K  (Eq. 11)  ───────────────────
        # H-branch: self-attention of J_T
        Q_h = self.W_Q_h(J)    # (batch, N, d_k)
        K_h = self.W_K_h(J)    # (batch, N, d_k)
        A_h = (Q_h @ K_h.transpose(-1, -2)) / math.sqrt(self.d_k)  # (batch, N, N)

        # F-branch: cross-attention — F queries against J keys
        Q_f = self.W_Q_f(F)    # (batch, N, d_k)
        A_f = (Q_f @ K_h.transpose(-1, -2)) / math.sqrt(self.d_k)  # (batch, N, N)

        # Combined kernel with learned scalar weights W_{P,h}, W_{P,f}
        K_kernel = self.W_Ph * A_h + self.W_Pf * A_f   # (batch, N, N)

        # ── Step 4: apply kernel  X^out = ∫ K(y,z) F(z) dz  (Eq. 12)  ──────
        # Discretised as a matrix-vector product over the N spatial points
        X_out = K_kernel @ F            # (batch, N, d_f)
        X_out = self.output_proj(X_out) # (batch, N, d_out)

        return X_out
