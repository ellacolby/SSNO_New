import torch
import torch.nn as nn

from .stu_encoder import STUEncoder
from .sfo_operator import SFOOperator


class ASNO(nn.Module):
    """
    Attention-based Spatio-Temporal Neural Operator (ASNO).

    ASNO learns to predict the next state of a spatio-temporal physical
    system from a window of past states and the next-step forcing field.
    Its design is inspired by the implicit-explicit (IMEX) decomposition
    of the Backward Differentiation Formula (BDF):

        Explicit step  →  STU Encoder              →  temporal extrapolation
        Implicit step  →  SFO Operator             →  spatial correction

    Full forward prediction:

        X^out_{m+1}  =  ASNO( X_m, X_{m-1}, …, X_{m-n+1},  F_{m+1} )
                     =  SFO( STU( X_m, …, X_{m-n+1} ),      F_{m+1} )

    Both components use the STU / Spectral Filtering framework:
      - STUEncoder  : per-point spectral filtering over the temporal axis.
                      Each of the N_spatial grid points runs its OWN independent
                      temporal STU over its d_field-dimensional history of n_steps
                      values.  The STU never sees across spatial locations — it
                      only sees one point's time series at a time.
      - SFOOperator : spectral filtering over the spatial axis.  After the
                      temporal step produces one extrapolated value per point,
                      the SFO mixes information across all N_spatial locations
                      via separable 2D spectral convolution.

    Per-point temporal processing
    -----------------------------
    Rather than flattening each snapshot to a d_state = N_spatial × d_field
    vector (which scrambles spatial structure into an unordered bag of numbers),
    the temporal step operates on each grid point independently:

        x_seq  (batch, n_steps, N_spatial, d_field)
            ↓  permute + merge batch and space
        (batch × N_spatial, n_steps, d_field)   ← N_spatial independent sequences
            ↓  STUEncoder: each point's time series processed separately
        (batch × N_spatial, d_field)
            ↓  reshape
        H  (batch, N_spatial, d_field)           ← one extrapolation per point

    This means the temporal STU processes d_field-dimensional tokens (= 1 for
    most benchmarks), making it very lightweight.  All spatial interaction is
    deferred entirely to the SFO implicit step.

    Spatial layout
    --------------
    Every state snapshot is represented as a field over N_spatial nodes,
    each carrying d_field scalar values:

        X_m  ∈  R^{N_spatial × d_field}

    For PDE benchmarks N_spatial is the number of grid points (e.g. 441
    for a 21×21 Darcy mesh).

    Args:
        N_spatial         (int): Number of spatial nodes. Must be fixed at
                                 construction (SFO precomputes Hilbert filters).
        d_field           (int): State features per spatial node.
        d_f               (int): Forcing features per spatial node.
        num_filters_te    (int): Spectral filters for the temporal STU encoder.
        num_filters_sfo   (int): Spectral filters (USB rank L) for each SFO
                                 layer. Paper uses L=16–20.
        n_steps           (int): History window length n (BDF order).
        d_model_sfo       (int): Internal channel dimension of the SFO operator.
        n_layers_sfo      (int): Number of SFO layers T (paper uses T=4).
        use_mlp_te        (bool): MLP in the temporal STU encoder.
        use_mlp_sfo       (bool): MLP inside each SFO layer. Default True.
        mlp_hidden_dim_te (int|None): MLP hidden size for temporal encoder.
        mlp_hidden_dim_sfo(int|None): MLP hidden size for SFO layers
                                      (None → 4 * d_model_sfo).
        dropout           (float): Dropout for both components.
        hankel_L          (bool): Single-branch Hankel. Default False.
    """

    def __init__(
        self,
        N_spatial: int,
        d_field: int,
        d_f: int,
        num_filters_te: int = 64,
        num_filters_sfo: int = 20,
        n_steps: int = 5,
        d_model_sfo: int = 128,
        n_layers_sfo: int = 4,
        use_mlp_te: bool = False,
        use_mlp_sfo: bool = True,
        mlp_hidden_dim_te: int | None = None,
        mlp_hidden_dim_sfo: int | None = None,
        dropout: float = 0.1,
        hankel_L: bool = False,
    ):
        super().__init__()
        self.N_spatial = N_spatial
        self.d_field = d_field
        self.d_f = d_f
        self.n_steps = n_steps

        # Per-point temporal: the STU sees one point's d_field-dim time series,
        # not the whole flattened snapshot.  d_state = d_field (e.g. 1), not
        # N_spatial * d_field (e.g. 441).  Much smaller and faster.
        # ── Explicit BDF step: STU Encoder (per-point temporal axis) ──────────
        self.te = STUEncoder(
            d_state=d_field,
            n_steps=n_steps,
            num_filters=num_filters_te,
            use_mlp=use_mlp_te,
            mlp_hidden_dim=mlp_hidden_dim_te,
            dropout=dropout,
            hankel_L=hankel_L,
        )

        # ── Implicit BDF step: SFO Operator (spatial axis) ────────────────────
        # SFO receives cat(H, y_init) as H (d_h = d_field * 2) and F (d_f).
        # Both have d_field features per spatial point → d_h = d_field * 2.
        # grid_h / grid_w are resolved inside SFOOperator if not supplied:
        # perfect-square N_spatial → 2D separable; otherwise 1D fallback.
        self.sfo = SFOOperator(
            d_h=d_field * 2,
            d_f=d_f,
            N_spatial=N_spatial,
            num_filters=num_filters_sfo,
            d_model=d_model_sfo,
            n_layers=n_layers_sfo,
            d_out=d_field,
            use_mlp=use_mlp_sfo,
            mlp_hidden_dim=mlp_hidden_dim_sfo,
            dropout=dropout,
            hankel_L=hankel_L,
        )
        # Expose resolved grid shape for inspection / logging
        self.grid_h = self.sfo.grid_h
        self.grid_w = self.sfo.grid_w

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

        # Ensure 4D: (batch, n_steps, N_spatial, d_field)
        x_4d = x_seq.reshape(batch, self.n_steps, self.N_spatial, self.d_field)

        # ── Explicit step: per-point temporal extrapolation ───────────────────
        # Give every spatial point its own independent temporal sequence.
        # permute → (batch, N_spatial, n_steps, d_field)
        # reshape → (batch × N_spatial, n_steps, d_field)  ← N_spatial separate seqs
        x_per_point = x_4d.permute(0, 2, 1, 3).reshape(
            batch * self.N_spatial, self.n_steps, self.d_field
        )
        H_per_point = self.te(x_per_point)   # (batch × N_spatial, d_field)
        H = H_per_point.reshape(batch, self.N_spatial, self.d_field)  # (batch, N, d_field)

        # ── y_init: most recent observed state ───────────────────────────────
        # x_4d[:, -1] is X_m — the last step in the history window.
        y_init = x_4d[:, -1]   # (batch, N_spatial, d_field)

        # Concatenate TE extrapolation with current state along feature dim
        H_aug = torch.cat([H, y_init], dim=-1)   # (batch, N_spatial, d_field * 2)

        # ── Implicit step: spatial correction via SFO ─────────────────────────
        X_out = self.sfo(H_aug, f_next)   # (batch, N_spatial, d_field)

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
