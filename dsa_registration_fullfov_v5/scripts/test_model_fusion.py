#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from dsa_reg.model_fusion import build_registration_residual_fusion
from dsa_reg.model_multitask import build_multitask_outcome_model, multitask_bce_loss

model = build_registration_residual_fusion(8, 4, dropout=0.0)
model.eval()
z2d = torch.randn(3, 8)
zreg = torch.tensor([[float("nan"), 1, 2, 3]] * 3)
out, meta = model(z2d, zreg, torch.zeros(3))
assert torch.equal(out, z2d)
assert float(meta["gate"]) < 0.1
try:
    model(z2d, zreg, torch.zeros(3, 2))
except ValueError:
    pass
else:
    raise AssertionError("qreg shape validation did not fire")
print("MODEL FUSION PASS: qreg=0 exact identity, NaN-safe, shape-checked")

mt = build_multitask_outcome_model(8, 4, dropout=0.0)
outputs = mt(z2d, torch.nan_to_num(zreg), torch.ones(3))
losses = multitask_bce_loss(
    outputs, torch.tensor([0.0, 1.0, 1.0]), torch.tensor([0.0, float("nan"), 1.0]), lambda_rel=0.5
)
assert torch.isfinite(losses["loss"])
assert torch.allclose(losses["loss"], losses["loss_abs"] + 0.5 * losses["loss_rel"])
print("MULTITASK PASS: loss_abs + lambda_rel * loss_rel with missing-label masking")
