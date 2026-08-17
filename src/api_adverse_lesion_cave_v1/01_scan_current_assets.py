#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from assets import load_nifti_mask, phase_from_segmentation_path
from common import atomic_csv, atomic_json, atomic_text, configure_runtime, load_config, normalize_patient_id, sha256_file, stage_logger, write_marker


def read_id_list(path: Path, header: int | None) -> set[str]:
    frame = pd.read_csv(path, header=header, dtype=str, keep_default_na=False)
    return {normalize_patient_id(value) for value in frame.iloc[:, -1] if normalize_patient_id(value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config); configure_runtime(config)
    finish = stage_logger("01_scan_current_assets")
    raw = Path(config["raw_root"]); updated = Path(config["updated_root"])
    manifests = Path(config["paths"]["manifests"]); reports = Path(config["paths"]["reports"])
    train = pd.read_excel(config["train_excel"], dtype=str, keep_default_na=False)
    valid = pd.read_excel(config["valid_excel"], dtype=str, keep_default_na=False)
    split_by_patient = {}
    for split, frame in (("Train", train), ("Valid", valid)):
        for value in frame["病案号"]: split_by_patient[normalize_patient_id(value)] = split
    rows = []
    for source_type, root in (("tiantanDSA", raw), ("updated_10_cases", updated)):
        for path in root.rglob("*.nii.gz"):
            if "segmentation" not in path.name.casefold(): continue
            relative = path.relative_to(root)
            patient_id = next((part for part in relative.parts if re.fullmatch(r"\d+", part)), "")
            if not patient_id: continue
            phase = phase_from_segmentation_path(path)
            image_path = path.parent / "Image.nii.gz"
            row = {
                "source_type": source_type, "source_root": str(root), "patient_id": patient_id,
                "split": split_by_patient.get(patient_id, "Outside"), "relative_path": str(relative),
                "segmentation_path": str(path), "phase_inferred": phase,
                "has_sibling_image": image_path.is_file(), "reference_image_path": str(image_path) if image_path.is_file() else "",
                "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            }
            try:
                _, info = load_nifti_mask(path)
                row.update({
                    "readable": True, "shape": "x".join(map(str, info["shape"])),
                    "raw_shape": "x".join(map(str, info["raw_shape"])),
                    "labels": "|".join(map(str, info["labels"])), "nonzero_pixels": info["nonzero_pixels"],
                    "nifti_error": "",
                })
            except Exception as exc:
                row.update({"readable": False, "shape": "", "raw_shape": "", "labels": "", "nonzero_pixels": 0, "nifti_error": repr(exc)})
            rows.append(row)
    inventory = pd.DataFrame(rows).sort_values(["source_type", "patient_id", "segmentation_path"])
    atomic_csv(inventory, manifests / "physical_asset_inventory.csv")
    atomic_csv(inventory, manifests / "asset_inventory.csv")
    old_lists = {
        "pre_biaozhu": (raw / "有Pre-biaozhu的名单.csv", None),
        "post_biaozhu": (raw / "有Post-biaozhu的名单.csv", None),
        "labelled_snapshot": (raw / "已标注nii.gz名单.csv", 0),
        "unlabelled_snapshot": (raw / "尚未标注的名单163.csv", None),
    }
    audits = []
    actual_patients = set(inventory.loc[inventory["readable"], "patient_id"])
    for name, (path, header) in old_lists.items():
        listed = read_id_list(path, header)
        audits.append({
            "list_name": name, "path": str(path), "sha256": sha256_file(path),
            "listed_patients": len(listed), "with_current_segmentation": len(listed & actual_patients),
            "listed_missing_current_segmentation": len(listed - actual_patients),
            "current_segmentation_not_listed": len(actual_patients - listed),
        })
    atomic_csv(pd.DataFrame(audits), manifests / "stale_list_audit.csv")
    patient_counts = inventory.groupby(["source_type", "phase_inferred"])["patient_id"].nunique().to_dict()
    summary = {
        "version": config["version"], "segmentation_files": len(inventory),
        "readable_segmentation_files": int(inventory["readable"].sum()),
        "unique_patients_any_source": int(inventory["patient_id"].nunique()),
        "official_train_rows": len(train), "official_train_patients": train["病案号"].map(normalize_patient_id).nunique(),
        "official_valid_rows": len(valid), "official_valid_patients": valid["病案号"].map(normalize_patient_id).nunique(),
        "patient_counts_by_source_phase": {"|".join(key): int(value) for key, value in patient_counts.items()},
        "unreadable_files": int((~inventory["readable"]).sum()),
    }
    lines=["# Asset audit","",f"- Segmentation files: `{summary['segmentation_files']}`",f"- Readable segmentation files: `{summary['readable_segmentation_files']}`",f"- Unique patients on disk: `{summary['unique_patients_any_source']}`",f"- Official Train patients: `{summary['official_train_patients']}`",f"- Official Valid patients: `{summary['official_valid_patients']}`",f"- Unreadable files: `{summary['unreadable_files']}`","","Old CSV lists were audited only and did not control inclusion."]
    atomic_text("\n".join(lines)+"\n",reports/"asset_audit.md")
    atomic_json(summary, reports / "patient_and_lesion_count_audit.json")
    write_marker(reports / ".ASSET_SCAN_SUCCESS", "01_scan_current_assets", config, {}, summary)
    finish(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
