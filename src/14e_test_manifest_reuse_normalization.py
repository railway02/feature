#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "14e_build_api_record_v1_all_series_manifests.py"

spec = importlib.util.spec_from_file_location("manifest14e", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(TARGET)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.normalize_manifest_scalar(89) == "89"
assert module.normalize_manifest_scalar(89.0) == "89"
assert module.normalize_manifest_scalar("89.0") == "89"
assert module.normalize_manifest_scalar("89") == "89"
assert module.normalize_manifest_scalar(None) == ""

new_phase = {
    "can_run": True,
    "frame_list_hash": "abc",
    "selected_internal_series": 89,
    "n_contiguous_pairs": 14,
}
old_row = {
    "pre_frame_list_hash": "abc",
    "selected_pre_internal_series": 89.0,
    "n_pre_contiguous_pairs": 14.0,
}
assert module.hash_equal_for_phase(new_phase, old_row, "pre") is True

old_row["selected_pre_internal_series"] = 90.0
assert module.hash_equal_for_phase(new_phase, old_row, "pre") is False

print("[PASS] integer-like internal-series normalization")
print("[PASS] 89 equals 89.0 for reuse matching")
