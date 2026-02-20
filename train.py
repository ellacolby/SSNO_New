"""
Training script for ASNO (Attention-based Spatio-Temporal Neural Operator).

Replace the placeholder dataset section with your actual data loader.
Expected tensor shapes per batch:

    x_seq  : (batch, n_steps, N_spatial, d_field)  — past n_steps states
    f_next : (batch, N_spatial, d_f)               — forcing at next step
    x_next : (batch, N_spatial, d_field)           — ground-truth next state

Usage:
    python train.py

Quick-start configs for the paper's benchmarks:

    Darcy flow  (21×21 grid)    N_spatial=441, d_field=1, d_f=1, n_steps=5
    Navier-Stokes (30×30 grid)  N_spatial=900, d_field=1, d_f=1, n_steps=5
    Lorenz system               N_spatial=3,   d_field=1, d_f=1, n_steps=5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from asno import ASNO


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Problem dimensions ────────────────────────────────────────────────────
    # Adjust these to match your dataset.
    "N_spatial": 441,       # spatial nodes  (e.g. 21×21 for Darcy)
    "d_field": 1,           # state features per spatial node
    "d_f": 1,               # forcing features per spatial node
    "n_steps": 5,           # BDF history length

    # ── Transformer Encoder (explicit / temporal step) ────────────────────────
    "d_embed": 128,         # embedding dimension
    "n_heads": 4,           # attention heads  (d_embed must be divisible)
    "n_layers_te": 2,       # encoder depth

    # ── Nonlocal Attention Operator (implicit / spatial step) ─────────────────
    "d_k": 64,              # query/key dimension
    "d_model_nao": 128,     # NAO internal model dimension
    "n_layers_nao": 3,      # iterative attention layers (T in paper)

    # ── Regularisation ────────────────────────────────────────────────────────
    "dropout": 0.1,

    # ── Optimisation ──────────────────────────────────────────────────────────
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "n_epochs": 100,
    "grad_clip": 1.0,       # max gradient norm (0 = disabled)

    # ── Misc ──────────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "best_asno.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean relative L² loss over the batch, averaged across spatial nodes and
    fields.  Matches the L² error used in neural-operator benchmarks.

        L = mean_batch( ||pred - target||_2  /  (||target||_2 + ε) )
    """
    diff_norm   = torch.norm(pred   - target, dim=(-2, -1))   # (batch,)
    target_norm = torch.norm(target,          dim=(-2, -1))   # (batch,)
    return torch.mean(diff_norm / (target_norm + 1e-8))


def cumulative_l2_error(
    model: ASNO,
    x_init_seq: torch.Tensor,
    f_seq: torch.Tensor,
    x_true_seq: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the cumulative L² error E_T = Σ_{t=1}^T ||X_t^true - X_t^pred||_2
    for a single trajectory, following Eq. (14) of the paper.

    Args:
        x_init_seq:  (1, n_steps, N_spatial, d_field)
        f_seq:       (1, T, N_spatial, d_f)
        x_true_seq:  (1, T, N_spatial, d_field)

    Returns:
        Scalar tensor containing E_T.
    """
    model.eval()
    with torch.no_grad():
        x_init_seq = x_init_seq.to(device)
        f_seq       = f_seq.to(device)
        x_true_seq  = x_true_seq.to(device)

        preds = model.rollout(x_init_seq, f_seq)   # (1, T, N, d)
        errors = torch.norm(
            (preds - x_true_seq).squeeze(0),        # (T, N, d)
            dim=(-2, -1),
        )   # (T,)
        return errors.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model: ASNO,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for x_seq, f_next, x_next in loader:
        x_seq  = x_seq.to(device)
        f_next = f_next.to(device)
        x_next = x_next.to(device)

        optimizer.zero_grad()
        pred = model(x_seq, f_next)
        loss = relative_l2_loss(pred, x_next)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: ASNO,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for x_seq, f_next, x_next in loader:
        x_seq  = x_seq.to(device)
        f_next = f_next.to(device)
        x_next = x_next.to(device)
        pred = model(x_seq, f_next)
        total_loss += relative_l2_loss(pred, x_next).item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = ASNO(
        N_spatial   = cfg["N_spatial"],
        d_field     = cfg["d_field"],
        d_f         = cfg["d_f"],
        d_embed     = cfg["d_embed"],
        n_heads     = cfg["n_heads"],
        n_layers_te = cfg["n_layers_te"],
        n_steps     = cfg["n_steps"],
        d_k         = cfg["d_k"],
        d_model_nao = cfg["d_model_nao"],
        n_layers_nao= cfg["n_layers_nao"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Replace this block with your actual data loading.
    #
    # Expected shapes:
    #   x_seq  : (N_samples, n_steps, N_spatial, d_field)
    #   f_next : (N_samples, N_spatial, d_f)
    #   x_next : (N_samples, N_spatial, d_field)
    #
    # For Darcy flow (paper Section 4.1):
    #   100 profiles × 96 windows × 20 permutations = 192 000 samples
    #   80:20 train/test split
    #   → N_train = 153 600,  N_test = 38 400

    N      = cfg["N_spatial"]
    ns     = cfg["n_steps"]
    df     = cfg["d_field"]
    dff    = cfg["d_f"]
    N_tr, N_te = 1000, 200   # placeholder sizes

    x_seq_tr  = torch.randn(N_tr, ns, N, df)
    f_next_tr = torch.randn(N_tr, N, dff)
    x_next_tr = torch.randn(N_tr, N, df)

    x_seq_te  = torch.randn(N_te, ns, N, df)
    f_next_te = torch.randn(N_te, N, dff)
    x_next_te = torch.randn(N_te, N, df)

    train_loader = DataLoader(
        TensorDataset(x_seq_tr, f_next_tr, x_next_tr),
        batch_size=cfg["batch_size"], shuffle=True, pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_seq_te, f_next_te, x_next_te),
        batch_size=cfg["batch_size"], shuffle=False, pin_memory=True,
    )

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"]
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_test_loss = float("inf")

    for epoch in range(1, cfg["n_epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, cfg["grad_clip"]
        )
        test_loss = eval_epoch(model, test_loader, device)
        scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), cfg["checkpoint_path"])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{cfg['n_epochs']}  |  "
                f"Train {train_loss:.4f}  |  "
                f"Test {test_loss:.4f}  |  "
                f"Best {best_test_loss:.4f}"
            )

    print(f"\nDone. Best test loss: {best_test_loss:.4f}")
    print(f"Checkpoint saved to: {cfg['checkpoint_path']}")


if __name__ == "__main__":
    main()
