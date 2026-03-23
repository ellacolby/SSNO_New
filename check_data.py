import h5py, numpy as np
with h5py.File("/root/data/combined_N250_k8.h5", "r") as f:
    def check(name, obj):
        if isinstance(obj, h5py.Dataset):
            d = obj[:]
            print(f"{name}: shape={d.shape}, NaN={np.isnan(d).any()}, Inf={np.isinf(d).any()}, min={d.min():.3f}, max={d.max():.3f}")
    f.visititems(check)
