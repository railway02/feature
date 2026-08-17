#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, configure_runtime, csv_evidence, file_evidence, load_config, sha256_file, write_success


def select_cave_bbox(roi: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    bbox_source = str(config["roi_cave"]["bbox_source"])
    padding_factor = float(config["roi_cave"]["padding_factor"])
    selected = roi.copy()
    selected["source_expanded_bbox_1p5"] = selected["expanded_bbox"]
    chosen_boxes = []
    paddings = []
    area_ratios = []
    fallback_rows = 0
    shape_cache: dict[str, tuple[int, int]] = {}
    for row in selected.to_dict("records"):
        value = str(row.get(bbox_source, "")).strip()
        if not value:
            value = str(row["expanded_bbox"])
            fallback_rows += 1
        parts = tuple(int(item) for item in value.split("|"))
        if len(parts) != 4:
            raise ValueError(f"Invalid {bbox_source}: {value}")
        x0, y0, x1, y1 = parts
        if x0 >= x1 or y0 >= y1 or (x1 - x0) != (y1 - y0):
            raise AssertionError(f"Invalid square CAVE ROI: {value}")
        first_frame = str(row["frame_paths"]).split("|")[0]
        if first_frame not in shape_cache:
            image = cv2.imread(first_frame, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(first_frame)
            shape_cache[first_frame] = image.shape
        height, width = shape_cache[first_frame]
        padding = (max(0, -x0), max(0, -y0), max(0, x1 - width), max(0, y1 - height))
        chosen_boxes.append(value)
        paddings.append("|".join(str(item) for item in padding))
        area_ratios.append(((x1 - x0) * (y1 - y0)) / float(height * width))
    selected["expanded_bbox"] = chosen_boxes
    selected["crop_padding_factor"] = padding_factor
    selected["crop_padding"] = paddings
    selected["roi_area_ratio"] = area_ratios
    selected["roi_cave_bbox_source"] = bbox_source
    audit = {
        "bbox_source": bbox_source,
        "padding_factor": padding_factor,
        "rows": int(len(selected)),
        "fallback_to_1p5_rows": int(fallback_rows),
        "changed_rows": int((selected["expanded_bbox"] != selected["source_expanded_bbox_1p5"]).sum()),
        "roi_area_ratio_median": float(np.median(area_ratios)),
        "roi_area_ratio_mean": float(np.mean(area_ratios)),
        "roi_area_ratio_max": float(np.max(area_ratios)),
    }
    return selected, audit


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    upstream_reports = Path(config["upstream_v1_reports"])
    upstream_manifests = Path(config["upstream_v1_manifests"])
    upstream_outputs = Path(config["upstream_v1_outputs"])
    fast_manifests = Path(config["paths"]["manifests"])
    reports = Path(config["paths"]["reports"])

    for marker in config["upstream_required_markers"]:
        if not (upstream_reports / marker).is_file():
            raise RuntimeError(f"Required upstream marker missing: {marker}")

    inference_summary = json.loads((upstream_reports / "segmentation_inference_summary.json").read_text(encoding="utf-8"))
    segmentation_gate_passed = bool(inference_summary["segmentation_gate"]["passed"])

    required = {
        "authoritative_roi_manifest_primary": upstream_manifests / "authoritative_roi_manifest_primary.csv",
        "segmentation_dataset_index": upstream_manifests / "segmentation_dataset_index.csv",
        "segmentation_train_oof_predictions": upstream_manifests / "segmentation_train_oof_predictions.csv",
        "segmentation_prediction_index": upstream_manifests / "segmentation_prediction_index.csv",
        "roi_manifest_pred": upstream_manifests / "roi_manifest_pred.csv",
        "cave_manifest_pred_train": upstream_manifests / "cave_manifest_pred_train.csv",
        "cave_manifest_pred_valid": upstream_manifests / "cave_manifest_pred_valid.csv",
    }
    report_files = {
        "alignment_audit": upstream_reports / "alignment_audit.csv",
        "segmentation_metrics": upstream_reports / "segmentation_metrics.csv",
        "segmentation_inference_summary": upstream_reports / "segmentation_inference_summary.json",
        "roi_manifest_summary": upstream_reports / "roi_manifest_summary.json",
        "roi_fallbacks": upstream_reports / "roi_fallbacks.csv",
    }
    provenance = {
        "version": config["version"],
        "upstream_markers": {name: file_evidence(upstream_reports / name) for name in config["upstream_required_markers"]},
        "manifests": {},
        "reports": {},
        "mask_morphology_pred_patient": file_evidence(upstream_outputs / "mask_morphology" / "pred_patient_median.csv")
            if (upstream_outputs / "mask_morphology" / "pred_patient_median.csv").is_file() else None,
    }
    for name, source in required.items():
        evidence = csv_evidence(source)
        target = fast_manifests / source.name
        atomic_copy(source, target)
        if sha256_file(target) != evidence["sha256"]:
            raise AssertionError(f"Copied manifest hash mismatch: {name}")
        evidence["fast_copy_path"] = str(target.resolve())
        provenance["manifests"][name] = evidence
    roi_target = fast_manifests / "roi_manifest_pred.csv"
    roi_source_copy = fast_manifests / "roi_manifest_pred_upstream_1p5.csv"
    atomic_copy(roi_target, roi_source_copy)
    roi_source = pd.read_csv(roi_target, dtype=str, keep_default_na=False)
    roi, roi_selection_audit = select_cave_bbox(roi_source, config)
    atomic_csv(roi, roi_target)
    selected_evidence = csv_evidence(roi_target)
    provenance["manifests"]["roi_manifest_pred"]["upstream_fast_copy_path"] = str(roi_source_copy.resolve())
    provenance["manifests"]["roi_manifest_pred"]["selected_fast_path"] = str(roi_target.resolve())
    provenance["manifests"]["roi_manifest_pred"]["selected_fast_sha256"] = selected_evidence["sha256"]
    provenance["roi_cave_selection"] = roi_selection_audit

    for name, source in report_files.items():
        provenance["reports"][name] = csv_evidence(source) if source.suffix == ".csv" else file_evidence(source)

    roi = pd.read_csv(fast_manifests / "roi_manifest_pred.csv", dtype=str, keep_default_na=False)
    prediction = pd.read_csv(fast_manifests / "segmentation_prediction_index.csv", dtype=str, keep_default_na=False)
    train = prediction[prediction["split"] == "Train"]
    valid = prediction[prediction["split"] == "Valid"]
    if set(train["prediction_kind"]) != {"oof"}:
        raise AssertionError("Train Pred masks are not exclusively OOF")
    if set(valid["prediction_kind"]) != {"full_train_frozen"}:
        raise AssertionError("Valid Pred masks are not exclusively frozen full-Train predictions")
    if set(train["patient_id"]) & set(valid["patient_id"]):
        raise AssertionError("Train/Valid patient overlap")
    if set(roi["phase_uid"]) != set(prediction["phase_uid"]):
        raise AssertionError("Pred ROI/prediction phase mismatch")

    provenance["audit"] = {
        "pred_roi_rows": int(len(roi)),
        "train_prediction_rows": int(len(train)),
        "valid_prediction_rows": int(len(valid)),
        "train_patients": int(train["patient_id"].nunique()),
        "valid_patients": int(valid["patient_id"].nunique()),
        "train_mask_source": "oof",
        "valid_mask_source": "full_train_frozen",
        "segmentation_gate_passed": segmentation_gate_passed,
        "segmentation_gate_controls_execution": False,
        "segmentation_quality_warning": not segmentation_gate_passed,
        "segmentation_gate_detail": inference_summary["segmentation_gate"],
    }
    atomic_json(provenance, reports / "upstream_provenance.json")
    write_success(reports / ".UPSTREAM_IMPORTED_SUCCESS", "01_import_upstream", config, provenance["audit"])
    print(json.dumps(provenance["audit"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
