#!/usr/bin/env python3
"""Synthetic shape and forward/backward checks for key-fusion models."""
from __future__ import annotations

import numpy as np
import torch

from train_adverse_keyfusion import ClinicalPreprocessor, KeyFusionNet, VARIANTS


def main() -> int:
    rng = np.random.default_rng(42)
    cave = torch.from_numpy(rng.normal(size=(4, 2, 10, 512)).astype(np.float32))
    sea = torch.from_numpy(rng.normal(size=(4, 2, 70, 16, 16)).astype(np.float32))
    clinical_raw = rng.normal(size=(4, 23)).astype(np.float32)
    clinical_raw[0, 0] = np.nan
    preprocessor = ClinicalPreprocessor().fit(clinical_raw)
    clinical = torch.from_numpy(preprocessor.transform(clinical_raw))
    missing = torch.zeros((4, 4), dtype=torch.float32)
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    dimensions = {}
    for variant in VARIANTS:
        model = KeyFusionNet(variant, clinical.shape[1])
        logits = model(cave, sea, clinical, missing)
        assert logits.shape == (4,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        assert torch.isfinite(logits).all()
        dimensions[variant] = sum(parameter.numel() for parameter in model.parameters())
    print({"status": "ok", "parameter_counts": dimensions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

