from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from common import sha256_file


class V3Bridge:
    """Load the already-frozen v3 preprocessing/kinetic primitives with provenance."""

    REQUIRED = (
        "load_config", "normalize_phase", "build_fov_mask",
        "baseline_polarity_enhancement", "build_activity_masks",
        "build_kinetic_maps", "build_filling_features",
    )

    def __init__(
        self,
        extractor_path: Path,
        base_config: Path,
        override_config: Path,
        expected_hashes: dict[str, str] | None = None,
    ):
        for path in (extractor_path, base_config, override_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_hashes = expected_hashes or {}
        actual_hashes = {
            "v3_extractor_sha256": sha256_file(extractor_path),
            "v3_base_config_sha256": sha256_file(base_config),
            "v3_override_config_sha256": sha256_file(override_config),
        }
        for key, expected in expected_hashes.items():
            if expected and actual_hashes.get(key) != expected:
                raise AssertionError(f"{key} mismatch: expected={expected}, actual={actual_hashes.get(key)}")

        module_name = "api_fullseq_v3_frozen_extractor"
        spec = importlib.util.spec_from_file_location(module_name, extractor_path)
        if spec is None or spec.loader is None:
            raise ImportError(extractor_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        for name in self.REQUIRED:
            if not hasattr(module, name):
                raise AttributeError(f"v3 extractor missing required primitive: {name}")
        self.module: ModuleType = module
        self.config: dict[str, Any] = module.load_config(base_config, override_config)
        if self.config.get("execution_limits", {}).get("labels_forbidden") is not True:
            raise AssertionError("v3 labels_forbidden must remain true")
        if self.config.get("execution_limits", {}).get("training_forbidden") is not True:
            raise AssertionError("v3 training_forbidden must remain true")
        self.provenance = {
            "v3_extractor_path": str(extractor_path.resolve()),
            "v3_base_config_path": str(base_config.resolve()),
            "v3_override_config_path": str(override_config.resolve()),
            **actual_hashes,
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
