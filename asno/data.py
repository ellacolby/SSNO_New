"""PDEBench HDF5 data loader for ASNO training.

Supports the standard PDEBench dataset formats:

  Seed-based 1D time-dependent (Burgers, Advection, Allen-Cahn, diff-sorp, …)
    HDF5 structure: /seed=XXXXXX/data  shape (N_t, N_x, d)

  2D time-dependent (diff-react, rdb, Allen-Cahn 2D, …)
    HDF5 structure: /u  [+ /v]  shape (N, N_t, Nx, Ny) or (N, N_t, Nx, Ny, d)

  2D steady-state Darcy flow
    HDF5 structure: /nu  shape (N, Nx, Ny)   — permeability (forcing)
                    /u   shape (N, Nx, Ny)   — pressure (solution)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def inspect_hdf5(path: str) -> None:
    """Print the structure and shapes of an HDF5 file to stdout."""
    with h5py.File(path, "r") as f:
        print(f"File : {path}")
        print(f"Keys : {list(f.keys())}\n")
        def _visitor(name, obj):
            pad = "  " * (name.count("/") + 1)
            if isinstance(obj, h5py.Dataset):
                print(f"{pad}/{name}  shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(_visitor)


def load_pdebench(
    path: str,
    n_steps: int = 5,
    train_frac: float = 0.8,
    n_traj: int | None = 200,
    spatial_subsample: int = 1,
) -> tuple[TensorDataset, TensorDataset, dict]:
    """Load a PDEBench HDF5 file and return (train_dataset, test_dataset, info).

    Each dataset yields (x_seq, f_next, x_next) tuples:
        x_seq  : (n_steps, N_spatial, d_field)  — history window
        f_next : (N_spatial, d_f)               — forcing at the next step
        x_next : (N_spatial, d_field)           — ground-truth next state

    For unforced time-dependent PDEs (Burgers, Advection, …) f_next is
    all-zeros with d_f=1.  For Darcy flow f_next is the permeability field.

    The returned ``info`` dict contains:
        N_spatial (int) : spatial nodes after subsampling
        d_field   (int) : state channels
        d_f       (int) : forcing channels

    Args:
        path             : path to the .hdf5 / .h5 file
        n_steps          : history window length
        train_frac       : fraction of windows used for training
        n_traj           : max trajectories to load (None = all)
        spatial_subsample: keep every n-th spatial point (1 = no subsampling)
    """
    u, f = _load_raw(path, n_traj, spatial_subsample)
    is_steady = (u.ndim == 3)   # (N, N_spatial, d) for Darcy, 4-D otherwise

    if is_steady:
        x_seqs, f_nexts, x_nexts = _windows_steady(u, f, n_steps)
    else:
        x_seqs, f_nexts, x_nexts = _windows_timedep(u, f, n_steps)

    N         = x_seqs.shape[0]
    n_train   = int(N * train_frac)
    N_spatial = x_seqs.shape[2]
    d_field   = x_seqs.shape[3]
    d_f       = f_nexts.shape[2]

    def _ds(a, b, c):
        return TensorDataset(
            torch.from_numpy(a).float(),
            torch.from_numpy(b).float(),
            torch.from_numpy(c).float(),
        )

    train_ds = _ds(x_seqs[:n_train],  f_nexts[:n_train],  x_nexts[:n_train])
    test_ds  = _ds(x_seqs[n_train:],  f_nexts[n_train:],  x_nexts[n_train:])
    info     = dict(N_spatial=N_spatial, d_field=d_field, d_f=d_f)

    print(f"Loaded {N:,} windows  ({n_train:,} train / {N - n_train:,} test)")
    print(f"  x_seq  : (n_steps={n_steps}, N_spatial={N_spatial}, d_field={d_field})")
    print(f"  f_next : (N_spatial={N_spatial}, d_f={d_f})")
    print(f"  x_next : (N_spatial={N_spatial}, d_field={d_field})")

    return train_ds, test_ds, info


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(
    path: str,
    n_traj: int | None,
    spatial_subsample: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (u, f) arrays from a PDEBench HDF5 file.

    u shape:
        time-dependent  → (N_traj, N_t, N_spatial, d_field)
        steady-state    → (N_traj, N_spatial, d_field)
    f shape: same leading dims as u, or None
    """
    with h5py.File(path, "r") as hf:
        keys = set(hf.keys())

        # ── Seed-based: most 1D PDEBench datasets ─────────────────────────────
        seed_keys = sorted(k for k in keys if k.startswith("seed="))
        if seed_keys:
            if n_traj is not None:
                seed_keys = seed_keys[:n_traj]
            arrays = [hf[sk]["data"][:] for sk in seed_keys]
            u = np.stack(arrays, axis=0)       # (N_traj, N_t, N_x, ...)
            if u.ndim == 3:
                u = u[..., None]               # ensure channel dim
            u = u[:, :, ::spatial_subsample, :]
            return u, None

        # ── 2D Darcy: nu (permeability) + u (pressure) ────────────────────────
        if "nu" in keys and "u" in keys:
            sl = slice(None, n_traj)
            nu = hf["nu"][sl]                  # (N, Nx, Ny)
            u  = hf["u"][sl]
            if u.ndim == 3:
                nu, u = nu[..., None], u[..., None]
            N, Nx, Ny, d = u.shape
            u  = u.reshape(N, Nx * Ny, d)[:, ::spatial_subsample, :]
            nu = nu.reshape(N, Nx * Ny, d)[:, ::spatial_subsample, :]
            return u, nu                       # steady-state: (N, N_spatial, d)

        # ── Multi-field time-dependent: u + v (diff-react, rdb) ───────────────
        if "u" in keys and "v" in keys:
            sl = slice(None, n_traj)
            u  = hf["u"][sl]
            v  = hf["v"][sl]
            uv = np.stack([u, v], axis=-1)     # (..., 2)
            if uv.ndim == 4:                   # 1D: (N, N_t, Nx, 2)
                pass
            elif uv.ndim == 5:                 # 2D: (N, N_t, Nx, Ny, 2)
                N, T, Nx, Ny, d = uv.shape
                uv = uv.reshape(N, T, Nx * Ny, d)
            uv = uv[:, :, ::spatial_subsample, :]
            return uv, None

        # ── Generic: single solution field ────────────────────────────────────
        for key in ("u", "tensor", "data"):
            if key in keys:
                arr = hf[key][slice(None, n_traj)]
                if arr.ndim == 3:              # (N, N_t, N_x) → add channel
                    arr = arr[..., None]
                elif arr.ndim == 5:            # 2D: (N, N_t, Nx, Ny, d)
                    N, T, Nx, Ny, d = arr.shape
                    arr = arr.reshape(N, T, Nx * Ny, d)
                arr = arr[:, :, ::spatial_subsample, :]
                return arr, None

        raise ValueError(
            f"Unrecognized HDF5 structure in '{path}'.\n"
            f"Root keys: {sorted(keys)}\n"
            "Run `from asno.data import inspect_hdf5; inspect_hdf5('<path>')` "
            "to see the full file structure."
        )


def _windows_timedep(
    u: np.ndarray,
    f: np.ndarray | None,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised sliding-window extraction from (N_traj, N_t, N_spatial, d)."""
    N_traj, N_t, N_spatial, d = u.shape
    T_win   = N_t - n_steps                         # windows per trajectory

    # win_idx[w, s] = w + s  →  index into time axis
    win_idx = np.arange(n_steps)[None, :] + np.arange(T_win)[:, None]

    # u[:, win_idx] → (N_traj, T_win, n_steps, N_spatial, d)
    x_seqs  = u[:, win_idx].reshape(-1, n_steps, N_spatial, d).astype(np.float32)
    x_nexts = u[:, n_steps:].reshape(-1, N_spatial, d).astype(np.float32)

    if f is not None:
        f_nexts = f[:, n_steps:].reshape(-1, N_spatial, d).astype(np.float32)
    else:
        f_nexts = np.zeros((N_traj * T_win, N_spatial, 1), dtype=np.float32)

    return x_seqs, f_nexts, x_nexts


def _windows_steady(
    u: np.ndarray,
    f: np.ndarray | None,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady-state (Darcy): no time dimension.

    x_seq is all-zero history; f_next is the permeability (nu) field.
    """
    N, N_spatial, d = u.shape
    x_seqs  = np.zeros((N, n_steps, N_spatial, d), dtype=np.float32)
    x_nexts = u.astype(np.float32)
    f_nexts = (f.astype(np.float32) if f is not None
               else np.zeros((N, N_spatial, 1), dtype=np.float32))
    return x_seqs, f_nexts, x_nexts
