import math
import torch
import torch.nn as nn
from mini_stu import MiniSTU


class SFOLayer(nn.Module):
    """
    Single separable-2D Spectral Filtering Operator layer.

    Implements the row-then-column factorisation described in Section 3 of
    "Learning PDE Operators via Spectral Filtering" (arXiv:2601.17090).
    The Hilbert eigenvectors used by STU are separable:

        φ_l(x₁, x₂) = φ_l(x₁) · φ_l(x₂)

    so a full 2D spatial convolution decomposes into two independent 1D
    passes — one along grid rows, one along grid columns — without any
    approximation error relative to the true 2D separable kernel.

    Forward pass
    ------------
    Given v  ∈  R^{batch × N × d}  where  N = grid_h × grid_w:

        1. Reshape:   (batch, N, d)  →  (batch, grid_h, grid_w, d)

        2. Row pass   — treat (batch · grid_h) as batch, grid_w as sequence:
               v  ←  v  +  stu_row( LayerNorm_row(v) )

        3. Column pass — transpose so grid_h becomes the sequence dim:
               v  ←  v  +  stu_col( LayerNorm_col(v) )

        4. Reshape back:  (batch, grid_h, grid_w, d)  →  (batch, N, d)

    Each pass has its own LayerNorm and its own MiniSTU with precomputed
    Hilbert filters for the corresponding axis length.  Both passes share the
    same num_filters, d_model, and MLP settings.

    Complexity per layer: O((grid_h + grid_w) · (L·d²  +  d·log(max(H,W))))
    vs O(N²·d) = O((grid_h·grid_w)²·d) for attention.

    Args:
        grid_h      (int): Height of the spatial grid (number of rows).
        grid_w      (int): Width  of the spatial grid (number of columns).
        d_model     (int): Channel dimension d.
        num_filters (int): Number of spectral filters L (USB rank).
                           Paper uses L = 16–20.
        use_mlp     (bool): Enable MiniSTU's internal MLP gating. Default True.
        mlp_hidden_dim (int|None): MLP hidden size.
                           None → 4 * d_model inside MiniSTU.
        dropout     (float): Dropout inside the MLP. Default 0.0.
        hankel_L    (bool): Single-branch Hankel (faster) vs two-branch
                            (more expressive). Default False.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        d_model: int,
        num_filters: int = 20,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        hankel_L: bool = False,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.has_col_pass = grid_h > 1

        _mlp_hidden = mlp_hidden_dim if mlp_hidden_dim is not None else 4 * d_model

        # num_filters must not exceed the axis length (Hankel matrix is seq_len×seq_len)
        nf_row = min(num_filters, grid_w)
        nf_col = min(num_filters, grid_h) if self.has_col_pass else 0

        # ── Row-wise pass: sequence length = grid_w ───────────────────────────
        self.norm_row = nn.LayerNorm(d_model)
        self.stu_row = MiniSTU(
            seq_len=grid_w,
            num_filters=nf_row,
            input_dim=d_model,
            output_dim=d_model,
            use_hankel_L=hankel_L,
            use_mlp=use_mlp,
            mlp_hidden_dim=_mlp_hidden,
            mlp_dropout=dropout,
        )

        # ── Column-wise pass: sequence length = grid_h (skipped if grid_h=1) ─
        if self.has_col_pass:
            self.norm_col = nn.LayerNorm(d_model)
            self.stu_col = MiniSTU(
                seq_len=grid_h,
                num_filters=nf_col,
                input_dim=d_model,
                output_dim=d_model,
                use_hankel_L=hankel_L,
                use_mlp=use_mlp,
                mlp_hidden_dim=_mlp_hidden,
                mlp_dropout=dropout,
            )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (batch, N, d_model)   where  N = grid_h * grid_w

        Returns:
            v_out: (batch, N, d_model)
        """
        batch, N, d = v.shape

        # ── Unflatten to 2D grid ──────────────────────────────────────────────
        v = v.view(batch, self.grid_h, self.grid_w, d)

        # ── Row pass: filter along columns (width axis) ───────────────────────
        v_r = v.reshape(batch * self.grid_h, self.grid_w, d)
        v_r = v_r + self.stu_row(self.norm_row(v_r))
        v = v_r.view(batch, self.grid_h, self.grid_w, d)

        # ── Column pass: filter along rows (height axis) ──────────────────────
        if self.has_col_pass:
            v = v.permute(0, 2, 1, 3)                          # (batch, grid_w, grid_h, d)
            v_c = v.reshape(batch * self.grid_w, self.grid_h, d)
            v_c = v_c + self.stu_col(self.norm_col(v_c))
            v = v_c.view(batch, self.grid_w, self.grid_h, d)
            v = v.permute(0, 2, 1, 3)                          # (batch, grid_h, grid_w, d)

        return v.reshape(batch, N, d)


class SFOOperator(nn.Module):
    """
    Separable-2D Spectral Filtering Operator — drop-in replacement for NAO
    as the *implicit BDF step* of ASNO.

    Architecture (arXiv:2601.17090):

        cat(H, F)  ∈  R^{N × (d_h + d_f)}
            ↓  lifting projection P
        J_0  ∈  R^{N × d_model}
            ↓  T × SFOLayer  (row pass + column pass + residuals)
        J_T  ∈  R^{N × d_model}
            ↓  projection Q
        X_out  ∈  R^{N × d_out}

    Each SFOLayer applies spectral filtering separately along the grid_h
    (row) and grid_w (column) axes, matching the paper's separable-mode
    factorisation.  This preserves 2D spatial neighbourhood structure that
    is lost when all N points are flattened into a single 1D sequence.

    Grid shape auto-detection
    -------------------------
    If grid_h / grid_w are not supplied, SFOOperator tries to infer them:
      - If N_spatial is a perfect square  →  grid_h = grid_w = √N_spatial
      - Otherwise                         →  grid_h = 1, grid_w = N_spatial
        (1D fallback; a warning is printed)

    Args:
        d_h           (int): Feature dim of H per spatial point (= d_field*2).
        d_f           (int): Feature dim of F per spatial point.
        N_spatial     (int): Total spatial nodes  (= grid_h * grid_w).
        grid_h        (int|None): Grid height. None → auto-detect.
        grid_w        (int|None): Grid width.  None → auto-detect.
        num_filters   (int): USB rank L per layer (paper: 16–20).
        d_model       (int): Internal channel dimension.
        n_layers      (int): Number of SFO layers T (paper: 4).
        d_out         (int): Output features per spatial point (= d_field).
        use_mlp       (bool): MLP gating inside each SFO layer. Default True.
        mlp_hidden_dim(int|None): MLP hidden size (None → 4 * d_model).
        dropout       (float): Dropout in SFO layers. Default 0.0.
        hankel_L      (bool): Single-branch Hankel. Default False.
    """

    def __init__(
        self,
        d_h: int,
        d_f: int,
        N_spatial: int,
        grid_h: int | None = None,
        grid_w: int | None = None,
        num_filters: int = 20,
        d_model: int = 128,
        n_layers: int = 4,
        d_out: int = 1,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        hankel_L: bool = False,
    ):
        super().__init__()
        self.N_spatial = N_spatial

        # ── Grid shape resolution ──────────────────────────────────────────────
        if grid_h is None and grid_w is None:
            sqrt_n = math.isqrt(N_spatial)
            if sqrt_n * sqrt_n == N_spatial:
                grid_h = grid_w = sqrt_n
            else:
                # Find the most square-like rectangular factorization
                best = (1, N_spatial)
                for h in range(sqrt_n, 0, -1):
                    if N_spatial % h == 0:
                        best = (h, N_spatial // h)
                        break
                grid_h, grid_w = best
                if grid_h == 1:
                    print(
                        f"[SFOOperator] Warning: N_spatial={N_spatial} has no rectangular "
                        f"factorisation; using 1D filtering (grid_h=1, grid_w={N_spatial}). "
                        f"Column pass will be skipped."
                    )
                else:
                    print(
                        f"[SFOOperator] N_spatial={N_spatial} → rectangular grid "
                        f"{grid_h}×{grid_w} (separable 2D filtering)."
                    )

        self.grid_h = grid_h
        self.grid_w = grid_w

        # ── Lifting: cat(H, F) → J_0  (P in paper notation) ──────────────────
        self.input_proj = nn.Linear(d_h + d_f, d_model)

        # ── T separable-2D SFO layers ─────────────────────────────────────────
        self.sfo_layers = nn.ModuleList([
            SFOLayer(
                grid_h=grid_h,
                grid_w=grid_w,
                d_model=d_model,
                num_filters=num_filters,
                use_mlp=use_mlp,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                hankel_L=hankel_L,
            )
            for _ in range(n_layers)
        ])

        # ── Projection: J_T → X_out  (Q in paper notation) ───────────────────
        self.output_proj = nn.Linear(d_model, d_out)

    def forward(self, H: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch, N_spatial, d_h)  — temporal extrapolation + y_init.
            F: (batch, N_spatial, d_f)  — forcing field at next time step.

        Returns:
            X_out: (batch, N_spatial, d_out)  — predicted next state.
        """
        # ── Lifting ───────────────────────────────────────────────────────────
        J = torch.cat([H, F], dim=-1)   # (batch, N, d_h + d_f)
        J = self.input_proj(J)          # (batch, N, d_model)

        # ── Separable 2D spectral filtering ───────────────────────────────────
        for layer in self.sfo_layers:
            J = layer(J)                # (batch, N, d_model)

        # ── Projection ────────────────────────────────────────────────────────
        return self.output_proj(J)      # (batch, N, d_out)
