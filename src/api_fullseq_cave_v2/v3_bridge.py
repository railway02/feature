from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from io_ops import sha256_file


class V3Bridge:
    """Use the already frozen v3 scientific primitives instead of reimplementing them."""

    REQUIRED = (
        "load_config", "normalize_phase", "build_fov_mask",
        "baseline_polarity_enhancement", "build_activity_masks",
        "build_kinetic_maps", "build_filling_features",
    )

    def __init__(self, extractor_path: Path, base_config: Path, override_config: Path):
        spec = importlib.util.spec_from_file_location("api_fullseq_v3_frozen_extractor", extractor_path)
        if spec is None or spec.loader is None:
            raise ImportError(extractor_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in self.REQUIRED:
            if not hasattr(module, name):
                raise AttributeError(f"v3 extractor missing {name}")
        self.module: ModuleType = module
        self.config: dict[str, Any] = module.load_config(base_config, override_config)
        self.provenance = {
            "v3_extractor_path": str(extractor_path.resolve()),
            "v3_extractor_sha256": sha256_file(extractor_path),
            "v3_base_config_path": str(base_config.resolve()),
            "v3_base_config_sha256": sha256_file(base_config),
            "v3_override_config_path": str(override_config.resolve()),
            "v3_override_config_sha256": sha256_file(override_config),
        }

    def preprocess(self, frames):
        normalized, low, high = self.module.normalize_phase(frames, self.config)
        fov, fov_qc = self.module.build_fov_mask(normalized, self.config)
        baseline, enhancement, polarity_qc = self.module.baseline_polarity_enhancement(
            normalized, fov, self.config
        )
        masks, activity_qc, activity = self.module.build_activity_masks(enhancement, fov, self.config)
        return {
            "normalized": normalized,
            "normalization_low": low,
            "normalization_high": high,
            "fov": fov,
            "baseline": baseline,
            "enhancement": enhancement,
            "masks": masks,
            "activity": activity,
            "qc": {**fov_qc, **polarity_qc, **activity_qc},
        }

    def kinetic_and_filling(self, enhancement, indices, fov, active_mask, tdc_peak):
        maps, kinetic_features, valid = self.module.build_kinetic_maps(
            enhancement, indices, active_mask, tdc_peak, self.config
        )
        curves, filling_features, visible = self.module.build_filling_features(
            enhancement, indices, fov, active_mask, maps["peak"], self.config
        )
        return maps, kinetic_features, valid, curves, filling_features, visible
