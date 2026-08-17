#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_json, configure_runtime, load_config, write_success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    manifests = Path(config["paths"]["manifests"])
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    manifest = pd.read_csv(manifests / f"cave_manifest_pred_{args.split.casefold()}.csv", dtype=str, keep_default_na=False)
    root = outputs / "cave_pred_roi_featurebank" / args.split.casefold()
    expected = []
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if str(row.get(f"can_run_{phase}", "")).casefold() == "true":
                expected.append((str(row["patient_id"]), str(row["series_uid"]), phase))
    failures = []
    successes = 0
    for patient_id, series_uid, phase in expected:
        directory = root / patient_id / series_uid / phase
        try:
            success = json.loads((directory / ".SUCCESS.json").read_text(encoding="utf-8"))
            embedding = np.load(directory / "embedding_5120.npy")
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if success.get("roi_branch") != "pred":
                raise AssertionError("roi_branch mismatch")
            if embedding.shape != (5120,) or not np.isfinite(embedding).all():
                raise AssertionError("invalid embedding")
            if metadata.get("frame_list_hash") != success.get("frame_list_hash"):
                raise AssertionError("frame hash mismatch")
            successes += 1
        except Exception as exc:
            failures.append({"patient_id": patient_id, "series_uid": series_uid, "phase": phase, "error": repr(exc)})
    summary = {
        "split": args.split,
        "expected_phases": len(expected),
        "success_phases": successes,
        "failed_phases": len(failures),
        "failure_rate": len(failures) / max(len(expected), 1),
        "failures": failures,
    }
    atomic_json(summary, reports / f"pred_roi_{args.split.casefold()}_feature_audit.json")
    if failures:
        raise RuntimeError(f"Pred ROI-CAVE {args.split} incomplete: {len(failures)} failures")
    write_success(reports / f".PRED_ROI_{args.split.upper()}_FEATURES_SUCCESS", "09_verify_featurebank", config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
