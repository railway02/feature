#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from assets import manifest_phase_rows
from common import atomic_csv, atomic_json, configure_runtime, load_config, sha256_file, stage_logger, write_marker


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); args = parser.parse_args()
    config = load_config(args.config); configure_runtime(config)
    finish = stage_logger("02_build_authoritative_roi_manifest")
    manifests = Path(config["paths"]["manifests"]); reports = Path(config["paths"]["reports"])
    train = manifest_phase_rows(Path(config["train_manifest"])); valid = manifest_phase_rows(Path(config["valid_manifest"]))
    frame = pd.concat([train, valid], ignore_index=True)
    if set(train["patient_id"]) & set(valid["patient_id"]): raise AssertionError("Train/Valid patient overlap")
    if frame["phase_uid"].duplicated().any(): raise AssertionError("Duplicate phase_uid")
    frame["annotation_grade_pre_alignment"] = frame.apply(
        lambda row: "A" if row["segmentation_path"] and row["reference_image_path"] else "B" if row["segmentation_path"] else "C", axis=1
    )
    frame["primary_candidate"] = frame["mask_resolution_status"].isin(["exact", "exact_direct_preferred", "recursive_unique"])
    atomic_csv(frame, manifests / "authoritative_roi_manifest_candidates.csv")
    excluded = frame[~frame["primary_candidate"]].copy()
    excluded["exclusion_reason"] = "mask_resolution:" + excluded["mask_resolution_status"].astype(str)
    atomic_csv(excluded, manifests / "excluded_assets_pre_alignment.csv")
    summary = {
        "rows": len(frame), "patients": int(frame["patient_id"].nunique()),
        "train_rows": int((frame["split"] == "Train").sum()), "valid_rows": int((frame["split"] == "Valid").sum()),
        "pre_rows": int((frame["phase"] == "pre").sum()), "post_rows": int((frame["phase"] == "post").sum()),
        "mask_resolution_status": frame["mask_resolution_status"].value_counts().to_dict(),
        "grade_pre_alignment": frame["annotation_grade_pre_alignment"].value_counts().to_dict(),
        "source_type": frame["source_type"].value_counts().to_dict(),
        "train_manifest_sha256": sha256_file(Path(config["train_manifest"])),
        "valid_manifest_sha256": sha256_file(Path(config["valid_manifest"])),
    }
    atomic_json(summary, reports / "authoritative_manifest_pre_alignment.json")
    write_marker(reports / ".MANIFEST_CANDIDATES_SUCCESS", "02_build_authoritative_roi_manifest", config, {}, summary)
    finish(summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
