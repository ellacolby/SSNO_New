import torch
from asno import ASNO
torch.set_grad_enabled(False)

model = ASNO(
    N_spatial=512, d_field=2, d_f=2, n_steps=5,
    num_filters_te=4, num_filters_sfo=20,
    d_model_sfo=16, n_layers_sfo=4,
    use_mlp_sfo=True, hankel_L=False,
).cuda().eval()

layer = model.sfo.sfo_layers[0]
print(f"grid_h={layer.grid_h}, grid_w={layer.grid_w}")
print(f"stu_row filters={layer.stu_row.num_filters}, stu_col filters={layer.stu_col.num_filters}")

J = torch.randn(4, 512, 16, device='cuda')

# row pass
v = J.view(4, 16, 32, 16)
v_r = v.reshape(64, 32, 16)
norm_r = layer.norm_row(v_r)
print(f"norm_row:  NaN={torch.isnan(norm_r).any().item()}  max={norm_r.abs().max():.3f}")
stu_r = layer.stu_row(norm_r)
print(f"stu_row:   NaN={torch.isnan(stu_r).any().item()}  max={stu_r.abs().max():.3f}")
v_r = v_r + stu_r
v = v_r.view(4, 16, 32, 16)
print(f"after row: NaN={torch.isnan(v).any().item()}  max={v.abs().max():.3f}")

# col pass
v = v.permute(0, 2, 1, 3)
v_c = v.reshape(128, 16, 16)
print(f"col input: NaN={torch.isnan(v_c).any().item()}  max={v_c.abs().max():.3f}")
norm_c = layer.norm_col(v_c)
print(f"norm_col:  NaN={torch.isnan(norm_c).any().item()}  max={norm_c.abs().max():.3f}")
stu_c = layer.stu_col(norm_c)
print(f"stu_col:   NaN={torch.isnan(stu_c).any().item()}  max={stu_c.abs().max():.3f}")
