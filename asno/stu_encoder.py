import torch
import torch.nn as nn
from mini_stu import MiniSTU


class STUEncoder(nn.Module):
    """
    Drop-in replacement for TransformerEncoder using the Spectral Transform
    Unit (STU) as the temporal backbone.

    STU replaces multi-head self-attention with spectral filtering via Hankel
    matrix eigenvectors, giving sub-quadratic complexity in sequence length
    while retaining strong long-range temporal modelling.

    Interface is identical to TransformerEncoder:
        forward(x_seq: (batch, n_steps, d_state)) -> H: (batch, d_state)

    MiniSTU outputs the full sequence (batch, n_steps, d_state); we take only
    the last position [:, -1, :] to match the BDF-step extrapolation semantics.

    Args:
        d_state      (int):  Feature dimension per spatial point (= d_field).
                             NOT the full flattened state N_spatial * d_field —
                             the temporal STU operates per-point, so each token
                             is d_field-dimensional (typically 1 scalar).
        n_steps      (int):  History window length (sequence length).
        num_filters  (int):  Number of spectral filters K. Controls capacity
                             (analogous to d_embed in the Transformer version).
                             Typical values: 32–128.
        use_mlp      (bool): Whether to add a small MLP after spectral filtering
                             for extra non-linearity. Default False.
        mlp_hidden_dim (int|None): MLP hidden size. Defaults to 4 * d_state
                             when use_mlp=True and this is None.
        mlp_num_layers (int): Depth of the optional MLP. Default 2.
        dropout      (float): Dropout rate passed to MiniSTU MLP. Default 0.1.
        hankel_L     (bool): Use single-branch Hankel (faster) vs two-branch
                             (more expressive). Default False (two-branch).
    """

    def __init__(
        self,
        d_state: int,
        n_steps: int,
        num_filters: int = 64,
        use_mlp: bool = False,
        mlp_hidden_dim: int | None = None,
        mlp_num_layers: int = 2,
        dropout: float = 0.1,
        hankel_L: bool = False,
    ):
        super().__init__()
        self.d_state = d_state
        self.n_steps = n_steps

        _mlp_hidden = mlp_hidden_dim if mlp_hidden_dim is not None else 4 * d_state

        self.stu = MiniSTU(
            seq_len=n_steps,
            num_filters=num_filters,
            input_dim=d_state,
            output_dim=d_state,
            use_hankel_L=hankel_L,
            use_mlp=use_mlp,
            mlp_hidden_dim=_mlp_hidden,
            mlp_num_layers=mlp_num_layers,
            mlp_dropout=dropout,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, n_steps, d_state)
                    Sequence ordered oldest-to-newest.

        Returns:
            H: (batch, d_state)
               Latent temporal extrapolation H_{m+1}.
        """
        # MiniSTU: (batch, n_steps, d_state) -> (batch, n_steps, d_state)
        out = self.stu(x_seq)

        # Extract the last temporal position, matching TransformerEncoder semantics
        return out[:, -1, :]   # (batch, d_state)
