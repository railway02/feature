"""Dual-task absolute/relative outcome head on top of quality-gated fusion."""
from __future__ import annotations

from .model_fusion import build_registration_residual_fusion, _torch


def build_multitask_outcome_model(z2d_dim: int, zreg_dim: int, fusion_hidden=128,
                                  head_hidden=64, init_gate_logit=-2.5, dropout=0.1):
    torch, nn = _torch()
    fusion = build_registration_residual_fusion(
        z2d_dim, zreg_dim, hidden=fusion_hidden,
        init_gate_logit=init_gate_logit, dropout=dropout,
    )

    class MultiTaskOutcomeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = fusion
            self.shared = nn.Sequential(
                nn.Linear(z2d_dim, head_hidden), nn.GELU(), nn.Dropout(dropout)
            )
            self.abs_head = nn.Linear(head_hidden, 1)
            self.rel_head = nn.Linear(head_hidden, 1)

        def forward(self, z2d, zreg, qreg):
            zfinal, fusion_meta = self.fusion(z2d, zreg, qreg)
            h = self.shared(zfinal)
            return {
                "logit_abs": self.abs_head(h).squeeze(-1),
                "logit_rel": self.rel_head(h).squeeze(-1),
                "zfinal": zfinal,
                **fusion_meta,
            }

    return MultiTaskOutcomeModel()


def multitask_bce_loss(outputs, y_abs, y_rel, lambda_rel=0.5):
    torch, _ = _torch()
    losses = {}
    mask_abs = torch.isfinite(y_abs)
    mask_rel = torch.isfinite(y_rel)
    if not torch.any(mask_abs):
        raise ValueError("Batch has no finite y_abs labels")
    losses["loss_abs"] = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["logit_abs"][mask_abs], y_abs[mask_abs].float()
    )
    if torch.any(mask_rel):
        losses["loss_rel"] = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["logit_rel"][mask_rel], y_rel[mask_rel].float()
        )
    else:
        losses["loss_rel"] = outputs["logit_rel"].sum() * 0.0
    losses["loss"] = losses["loss_abs"] + float(lambda_rel) * losses["loss_rel"]
    return losses

