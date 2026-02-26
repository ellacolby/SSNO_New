import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder implementing the *explicit BDF step* of ASNO.

    Given a sequence of n past (flattened) states {X_{m-n+1}, ..., X_m},
    this module predicts the latent temporal extrapolation H_{m+1}.  The
    role is analogous to the explicit step of the Backward Differentiation
    Formula (BDF):

        H_{m+1} ≈ -Σ_{k=1}^n  α_k · X_{m-k+1}

    where the BDF coefficients α_k are learned through multi-head
    self-attention rather than being fixed numerically.

    Architecture
    ------------
    1. Linear embedding  :  X_t ∈ R^{d_state}  →  E_t ∈ R^{d_embed}
    2. Positional encoding added to E_t (learnable, matches original code).
    3. L-layer standard Transformer Encoder (bidirectional self-attention +
       feed-forward sub-layers).
    4. Output at the *last temporal position* is projected back to R^{d_state}
       to give H_{m+1} with the same shape as one state snapshot.

    Args:
        d_state  (int): Dimension of the flattened state vector
                        (= N_spatial × d_field).
        d_embed  (int): Internal embedding dimension.
        n_heads  (int): Number of attention heads.
        n_layers (int): Number of Transformer Encoder layers.
        n_steps  (int): Sequence length (history steps n).
        dropout  (float): Dropout probability.
    """

    def __init__(
        self,
        d_state: int,
        d_embed: int,
        n_heads: int,
        n_layers: int,
        n_steps: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_embed = d_embed
        self.n_steps = n_steps

        # W_E in R^{d_state × d_embed}  (Eq. 2 of paper)
        self.embedding = nn.Linear(d_state, d_embed)

        # Learnable positional encoding: shape (1, n_steps, d_embed).
        # Initialised to zeros and trained jointly — matches the original code.
        self.pos_encoding = nn.Parameter(torch.zeros(1, n_steps, d_embed))

        self.dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=4 * d_embed,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Project d_embed → d_state so that H_{m+1} has the same shape as X
        self.output_proj = nn.Linear(d_embed, d_state)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, n_steps, d_state)
                    Sequence ordered oldest-to-newest:
                    column 0 = X_{m-n+1}, last column = X_m.

        Returns:
            H: (batch, d_state)
               Latent temporal extrapolation H_{m+1}.  Reshape to
               (batch, N_spatial, d_field) before passing to NAO.
        """
        # (batch, n_steps, d_state) → (batch, n_steps, d_embed)
        E = self.embedding(x_seq)
        E = self.dropout(E + self.pos_encoding)   # broadcast over batch

        # Bidirectional self-attention over the temporal sequence
        out = self.transformer(E)   # (batch, n_steps, d_embed)

        # The output at the last position captures the "next-step" extrapolation
        # (analogous to BDF using the n most-recent steps to predict step n+1)
        H_embed = out[:, -1, :]     # (batch, d_embed)

        H = self.output_proj(H_embed)   # (batch, d_state)
        return H
