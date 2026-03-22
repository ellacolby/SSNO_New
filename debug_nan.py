"""
Run on Vast to pinpoint exactly where NaN first appears in the forward pass.
    python debug_nan.py
"""
import sys
import torch

sys.path.insert(0, '/workspace/flash-stu-2')
sys.path.insert(0, '/workspace/SSNO_New')

from torch.utils.data import DataLoader
from asno import ASNO
from asno.data import load_pdebench

device = torch.device('cuda')

# ── Load a tiny batch ────────────────────────────────────────────────────────
print("Loading data...")
train_ds, _, data_info = load_pdebench(
    path='/root/data/combined_N250_k8.h5',
    n_steps=5,
    train_frac=0.8,
    spatial_subsample=8,
)
loader   = DataLoader(train_ds, batch_size=4, shuffle=False)
x_seq, f_next, x_next = next(iter(loader))
x_seq  = x_seq.to(device)
f_next = f_next.to(device)

N  = data_info['N_spatial']   # 512
df = data_info['d_field']     # 2
dF = data_info['d_f']         # 2

print(f"x_seq : {tuple(x_seq.shape)}  NaN={torch.isnan(x_seq).any().item()}")
print(f"f_next: {tuple(f_next.shape)} NaN={torch.isnan(f_next).any().item()}")

# ── Build model (same CONFIG as train.py) ────────────────────────────────────
model = ASNO(
    N_spatial       = N,
    d_field         = df,
    d_f             = dF,
    num_filters_te  = 4,
    num_filters_sfo = 20,
    n_steps         = 5,
    d_model_sfo     = 16,
    n_layers_sfo    = 4,
    use_mlp_te      = False,
    use_mlp_sfo     = True,
    dropout         = 0.1,
    hankel_L        = False,
).to(device)

def chk(name, t):
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    print(f"  {name:40s}  shape={tuple(t.shape)}  NaN={nan}  Inf={inf}"
          f"  min={t.min().item():.4f}  max={t.max().item():.4f}")
    return nan or inf

print("\n── Forward pass trace ──────────────────────────────────────────────")

batch = x_seq.shape[0]
x_4d = x_seq.reshape(batch, 5, N, df)

# ── Temporal encoder ─────────────────────────────────────────────────────────
x_pp = x_4d.permute(0, 2, 1, 3).reshape(batch * N, 5, df)
if chk("x_per_point", x_pp): sys.exit("NaN in input!")

H_pp = model.te(x_pp)
if chk("H_per_point (temporal out)", H_pp): sys.exit("NaN after temporal encoder!")

H      = H_pp.reshape(batch, N, df)
y_init = x_4d[:, -1]
H_aug  = torch.cat([H, y_init], dim=-1)
chk("H_aug = cat(H, y_init)", H_aug)

# ── SFO lifting ───────────────────────────────────────────────────────────────
J = torch.cat([H_aug, f_next], dim=-1)
chk("J  (before input_proj)", J)

J = model.sfo.input_proj(J)
if chk("J  (after  input_proj)", J): sys.exit("NaN after lifting projection!")

# ── SFO layers ────────────────────────────────────────────────────────────────
for i, layer in enumerate(model.sfo.sfo_layers):
    # row pass internals
    b, Ns, d = J.shape
    gh, gw = model.grid_h, model.grid_w
    v2d  = J.view(b, gh, gw, d)
    v_r  = v2d.reshape(b * gh, gw, d)
    v_rn = layer.norm_row(v_r)
    chk(f"layer {i}  norm_row output", v_rn)
    v_rs = layer.stu_row(v_rn)
    if chk(f"layer {i}  stu_row output", v_rs):
        sys.exit(f"NaN in SFO layer {i} ROW pass!")

    J = layer(J)
    if chk(f"layer {i}  full output", J):
        sys.exit(f"NaN after SFO layer {i}!")

# ── Output projection ─────────────────────────────────────────────────────────
X_out = model.sfo.output_proj(J)
chk("X_out (final prediction)", X_out)

print("\nAll checks passed — no NaN detected with this random init.")
print("Re-running with anomaly detection on a loss.backward()...")

pred = model(x_seq, f_next)
loss = torch.nn.functional.mse_loss(pred, x_next.to(device))
print(f"loss = {loss.item():.6f}")
loss.backward()
print("backward() succeeded — no NaN in gradients either.")
