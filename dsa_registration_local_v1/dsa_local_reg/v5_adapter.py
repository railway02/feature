from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from .common import require_sha256


def validate_v5_registration_core(cfg: dict[str, Any]) -> dict[str, str]:
    """Hash-lock the explicitly permitted V5 low-level components.

    The Jacobian/HEMO contract permits these five files only.  In particular this
    adapter never imports V5's pipeline, global correspondence, outcome, or model
    code, which prevents accidental expansion of the frozen method.
    """
    root = Path(cfg["paths"]["v5_root"])
    allowed = {
        "registration_ants": "registration_ants.py",
        "registration_sitk": "registration_sitk.py",
        "preprocessing": "preprocessing.py",
        "hemodynamics": "hemodynamics.py",
        "features": "features.py",
    }
    return {
        key: require_sha256(root / "dsa_reg" / filename, cfg["locks"].get(f"v5_{key}_sha256", ""), f"V5 {key}")
        for key, filename in allowed.items()
    }


def load_v5_module(cfg: dict[str, Any], filename: str) -> ModuleType:
    """Load a V5 file by explicit path, avoiding ambiguous ``dsa_reg`` imports.

    This adapter is intentionally path-locked.  It does not import V5 pipeline.py or any
    full-FOV component.
    """
    if filename not in {
        "registration_ants.py", "registration_sitk.py", "preprocessing.py",
        "hemodynamics.py", "features.py",
    }:
        raise ValueError(f"Local Reference Jacobian/HEMO may not import V5 module {filename}")
    validate_v5_registration_core(cfg)
    path = Path(cfg["paths"]["v5_root"]) / "dsa_reg" / filename
    name = f"local_reference_v5_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V5 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v5_core_description(cfg: dict[str, Any]) -> dict[str, Any]:
    hashes = validate_v5_registration_core(cfg)
    return {
        "v5_root": str(Path(cfg["paths"]["v5_root"]).resolve()),
        "allowed_modules": [
            "registration_sitk.py", "registration_ants.py", "preprocessing.py",
            "hemodynamics.py", "features.py",
        ],
        "forbidden": ["pipeline.py", "global_correspondence.py", "preprocessing.py"],
        "hashes": hashes,
    }
