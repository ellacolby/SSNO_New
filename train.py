"""
Training script for ASNO (Attention-based Spatio-Temporal Neural Operator).

Replace the placeholder dataset section with your actual data loader.
Expected tensor shapes per batch:

    x_seq  : (batch, n_steps, N_spatial, d_field)  — past n_steps states
    f_next : (batch, N_spatial, d_f)               — forcing at next step
    x_next : (batch, N_spatial, d_field)           — ground-truth next state

Usage:
    python train.py

Benchmark configs (hyperparameters matched to paper's reported parameter counts):

    Darcy flow  (21×21, ~760K params)
        N_spatial=441, d_field=1, d_f=1, n_steps=5
        d_embed=128, n_heads=4, n_layers_te=3
        d_k=32, d_model_nao=256, n_layers_nao=2

    Navier-Stokes (30×30, ~4.66M params)
        N_spatial=900, d_field=1, d_f=1, n_steps=5
        d_embed=384, n_heads=4, n_layers_te=2
        d_k=128, d_model_nao=320, n_layers_nao=4

    Lorenz system (~258K params)
        N_spatial=3, d_field=1, d_f=1, n_steps=5
        d_embed=96, n_heads=2, n_layers_te=2
        d_k=32, d_model_nao=256, n_layers_nao=1
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asno import ASNO
from asno.data import inspect_hdf5, load_pdebench


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
    "n_layers_te": 3,       # encoder depth

    # ── Nonlocal Attention Operator (implicit / spatial step) ─────────────────
    "d_k": 32,              # query/key dimension
    "d_model_nao": 256,     # NAO internal model dimension
    "n_layers_nao": 2,      # iterative attention layers (T in paper)

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
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ASNO on a PDEBench dataset.")
    p.add_argument("--data_dir",  type=str, default="./data",
                   help="Directory containing HDF5 dataset files.")
    p.add_argument("--dataset",   type=str, required=True,
                   help="Filename inside data_dir, e.g. 1D_Burgers_Sols_Nu0.1.hdf5")
    p.add_argument("--n_traj",    type=int, default=200,
                   help="Number of trajectories to load (default: 200).")
    p.add_argument("--spatial_subsample", type=int, default=1,
                   help="Keep every n-th spatial point (default: 1 = no subsampling).")
    p.add_argument("--inspect",   action="store_true",
                   help="Print HDF5 file structure and exit.")
    # Optional CONFIG overrides
    p.add_argument("--n_steps",   type=int,   default=None)
    p.add_argument("--n_epochs",  type=int,   default=None)
    p.add_argument("--batch_size",type=int,   default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    return p.parse_args()


def main():
    args   = _parse_args()
    cfg    = dict(CONFIG)   # copy so we can mutate

    # Apply CLI overrides
    for key in ("n_steps", "n_epochs", "batch_size", "lr", "checkpoint_path"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    data_path = Path(args.data_dir) / args.dataset

    if args.inspect:
        inspect_hdf5(str(data_path))
        return

    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {data_path}")
    train_ds, test_ds, data_info = load_pdebench(
        path             = str(data_path),
        n_steps          = cfg["n_steps"],
        train_frac       = 0.8,
        n_traj           = args.n_traj,
        spatial_subsample= args.spatial_subsample,
    )

    # Override model dimensions to match the loaded data
    cfg["N_spatial"] = data_info["N_spatial"]
    cfg["d_field"]   = data_info["d_field"]
    cfg["d_f"]       = data_info["d_f"]

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,  pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=cfg["batch_size"], shuffle=False, pin_memory=True,
    )

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
