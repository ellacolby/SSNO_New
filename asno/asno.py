import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .nao import NAO


class ASNO(nn.Module):
    """
    Attention-based Spatio-Temporal Neural Operator (ASNO).

    ASNO learns to predict the next state of a spatio-temporal physical
    system from a window of past states and the next-step forcing field.
    Its design is inspired by the implicit-explicit (IMEX) decomposition
    of the Backward Differentiation Formula (BDF):

        Explicit step  →  Transformer Encoder (TE)  →  temporal extrapolation
        Implicit step  →  Nonlocal Attention Operator (NAO) →  spatial correction

    Full forward prediction:

        X^out_{m+1}  =  ASNO( X_m, X_{m-1}, …, X_{m-n+1},  F_{m+1} )
                     =  NAO(  TE( X_m, …, X_{m-n+1} ),       F_{m+1} )

    Spatial layout
    --------------
    Every state snapshot is represented as a field over N_spatial nodes,
    each carrying d_field scalar values:

        X_m  ∈  R^{N_spatial × d_field}

    For PDE benchmarks N_spatial is the number of grid points (e.g. 441
    for a 21×21 Darcy mesh).  For the Lorenz ODE N_spatial = 3 (the three
    state variables treated as "spatial" nodes) and d_field = 1.

    The TE flattens each snapshot to R^{d_state} = R^{N_spatial × d_field}
    to form the temporal token sequence; the NAO then operates over the
    N_spatial dimension with per-node features.

    Args:
        N_spatial   (int): Number of spatial nodes.
        d_field     (int): State features per spatial node.
        d_f         (int): Forcing features per spatial node.
        d_embed     (int): TE embedding dimension.
        n_heads     (int): TE attention heads.
        n_layers_te (int): TE encoder depth.
        n_steps     (int): History window length n (BDF order).
        d_k         (int): NAO query/key dimension.
        d_model_nao (int): NAO internal model dimension.
        n_layers_nao(int): Number of iterative NAO attention layers (T).
        dropout     (float): Dropout in the TE.
    """

    def __init__(
        self,
        N_spatial: int,
        d_field: int,
        d_f: int,
        d_embed: int = 128,
        n_heads: int = 4,
        n_layers_te: int = 2,
        n_steps: int = 5,
        d_k: int = 64,
        d_model_nao: int = 128,
        n_layers_nao: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N_spatial = N_spatial
        self.d_field = d_field
        self.d_f = d_f
        self.n_steps = n_steps

        d_state = N_spatial * d_field   # flattened state dimension for TE

        # ── Explicit BDF step: Transformer Encoder ────────────────────────────
        self.te = TransformerEncoder(
            d_state=d_state,
            d_embed=d_embed,
            n_heads=n_heads,
            n_layers=n_layers_te,
            n_steps=n_steps,
            dropout=dropout,
        )

        # ── Implicit BDF step: Nonlocal Attention Operator ────────────────────
        # The TE output H is reshaped to (batch, N_spatial, d_field) before
        # being fed to NAO, so d_h = d_field.
        self.nao = NAO(
            d_h=d_field,
            d_f=d_f,
            d_k=d_k,
            d_model=d_model_nao,
            n_layers=n_layers_nao,
            d_out=d_field,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        f_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step prediction.

        Args:
            x_seq:  (batch, n_steps, N_spatial, d_field)  or
                    (batch, n_steps, d_state)
                    Past state sequence ordered oldest-to-newest.
            f_next: (batch, N_spatial, d_f)
                    Forcing field at the next time step (m+1).

        Returns:
            X_out: (batch, N_spatial, d_field)
                   Predicted state at time step m+1.
        """
        batch = x_seq.shape[0]

        # Flatten spatial dims if needed: (..., N_spatial, d_field) → (..., d_state)
        x_flat = x_seq.reshape(batch, self.n_steps, -1)   # (batch, n_steps, d_state)

        # ── Explicit step: temporal extrapolation ─────────────────────────────
        H_flat = self.te(x_flat)                            # (batch, d_state)
        H = H_flat.view(batch, self.N_spatial, self.d_field)  # (batch, N, d_field)

        # ── Implicit step: spatial correction via NAO ─────────────────────────
        X_out = self.nao(H, f_next)   # (batch, N_spatial, d_field)

        return X_out

    def rollout(
        self,
        x_init_seq: torch.Tensor,
        f_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive rollout for long-horizon prediction.

        Each predicted state is fed back as input for the next step,
        mirroring the autoregressive integration scheme described in the
        paper (Section 4).  Errors accumulate over successive rollouts;
        ASNO's separable architecture is designed to keep this growth small.

        Args:
            x_init_seq: (batch, n_steps, N_spatial, d_field)
                        Seed history (the first n_steps ground-truth states).
            f_seq:      (batch, T_future, N_spatial, d_f)
                        Forcing fields for every future time step.

        Returns:
            preds: (batch, T_future, N_spatial, d_field)
                   Predicted states for all T_future steps.
        """
        preds = []
        x_window = x_init_seq   # (batch, n_steps, N_spatial, d_field)

        for t in range(f_seq.shape[1]):
            f_t = f_seq[:, t]                        # (batch, N_spatial, d_f)
            x_next = self.forward(x_window, f_t)     # (batch, N_spatial, d_field)
            preds.append(x_next.unsqueeze(1))

            # Slide the history window: drop oldest, append new prediction
            x_window = torch.cat(
                [x_window[:, 1:], x_next.unsqueeze(1)], dim=1
            )

        return torch.cat(preds, dim=1)   # (batch, T_future, N_spatial, d_field)
