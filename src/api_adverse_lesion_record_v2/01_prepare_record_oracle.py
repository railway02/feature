#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import atomic_csv, atomic_json, load_config, sha256_file, update_run_manifest


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in str(value).split("|"))
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise ValueError(f"Invalid bbox: {value}")
    return parts


def box_text(value: tuple[int, int, int, int]) -> str:
    return "|".join(str(item) for item in value)


def snapshot(config: dict) -> dict:
    root = Path(config["project_root"])
    sys.modules.pop("assets", None)
    sys.modules.pop("common", None)
    reports = Path(config["paths"]["reports"])
    files = [
        root / "configs/api_adverse_lesion_cave_v1.json",
        root / "configs/api_adverse_lesion_cave_fast_v1.json",
        root / "manifests/api_fullseq_v3_train_all_series_frozen.csv",
        root / "manifests/api_fullseq_v3_valid_all_series_frozen.csv",
        root / "manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_primary.csv",
        root / "reports/api_adverse_lesion_cave_v1/segmentation_oof_training_summary.json",
        root / "reports/api_adverse_lesion_cave_fast_v1/matched_logistic_variants_summary.json",
        root / "reports/api_adverse_lesion_cave_gt_oracle_v1/matched_logistic_variants_summary.json",
        Path(config["checkpoint"]),
        Path(config["fixed_trainer"]),
    ]
    evidence = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence[str(path.resolve())] = {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    if evidence[str(Path(config["checkpoint"]).resolve())]["sha256"] != config["checkpoint_sha256"]:
        raise AssertionError("CAVE checkpoint SHA256 mismatch")
    tree_roots = [
        root / "outputs/api_fullseq_cave_v3_featurebank",
        root / "outputs/api_adverse_lesion_cave_v1",
        root / "outputs/api_adverse_lesion_cave_fast_v1",
        root / "outputs/api_adverse_lesion_cave_gt_oracle_v1",
    ]
    trees = {}
    for path in tree_roots:
        count = 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                count += 1
                total += item.stat().st_size
        trees[str(path.resolve())] = {"files": count, "bytes": total}
    payload = {
        "version": config["version"],
        "status": "v1_engineering_complete_science_gate_failed_read_only",
        "files": evidence,
        "trees": trees,
        "never_overwrite_roots": [str(path.resolve()) for path in tree_roots],
    }
    atomic_json(payload, reports / "V1_BASELINE_SNAPSHOT.json")
    return payload


def mapping_tier(row: pd.Series) -> tuple[str, str]:
    if not str(row["suggested_series_uid"]):
        return "unresolved", f"record_series_{row['mapping_status']}"
    single = (
        str(row["mapping_status"]) == "auto_single_remaining"
        and int(row["patient_valid_series_count"]) == 1
        and int(row["record_count_within_patient"]) == 1
    )
    if single:
        return "exact", ""
    if str(row["mapping_status"]) == "auto_unique_location":
        return "high_confidence", ""
    return "unresolved", f"excluded_mapping_policy_{row['mapping_status']}"


def build_manifests(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = Path(config["record_source"])
    manifests = Path(config["paths"]["manifests"])
    records = []
    suggestions = []
    for split in ("train", "valid"):
        records.append(pd.read_csv(source / f"{split}_record_table.csv", dtype=str, keep_default_na=False))
        suggestions.append(pd.read_csv(source / f"{split}_record_series_suggestions.csv", dtype=str, keep_default_na=False))
    record = pd.concat(records, ignore_index=True)
    suggestion = pd.concat(suggestions, ignore_index=True)
    master = record.merge(suggestion, on=[
        "record_uid", "split", "patient_id", "excel_row_number",
        "record_index_within_patient", "normalized_side", "normalized_location",
    ], how="left", validate="one_to_one")
    tiers = master.apply(mapping_tier, axis=1)
    master["mapping_tier"] = [item[0] for item in tiers]
    master["exclusion_reason"] = [item[1] for item in tiers]
    master["series_uid"] = master["suggested_series_uid"]
    master["followup_rroc"] = pd.to_numeric(master["随访RROC123"], errors="raise").astype(int)
    master["target"] = master["followup_rroc"].isin([2, 3]).astype(int)
    adverse = pd.to_numeric(master["不良转归：1是；0否"], errors="raise").astype(int)
    if not np.array_equal(adverse.to_numpy(), master["target"].to_numpy()):
        raise AssertionError("adverse does not equal followup RROC 2/3")

    roi = pd.read_csv(config["roi_manifest"], dtype=str, keep_default_na=False)
    phase_counts = roi.groupby(["split", "series_uid"])["phase"].agg(
        available_phases=lambda values: "|".join(sorted(set(values))),
        phase_count="nunique",
    ).reset_index()
    master = master.merge(phase_counts, on=["split", "series_uid"], how="left")
    master["phase_count"] = pd.to_numeric(master["phase_count"], errors="coerce").fillna(0).astype(int)
    master["has_pre"] = master["available_phases"].fillna("").str.split("|").map(lambda x: "pre" in x)
    master["has_post"] = master["available_phases"].fillna("").str.split("|").map(lambda x: "post" in x)
    missing_mask = master["mapping_tier"].ne("unresolved") & master["phase_count"].eq(0)
    master.loc[missing_mask, "exclusion_reason"] = "mapped_series_has_no_authoritative_mask"
    master.loc[missing_mask, "mapping_tier"] = "unresolved"
    primary = master["mapping_tier"].isin(["exact", "high_confidence"])
    if config["mapping"]["require_pre_and_post"]:
        incomplete = primary & ~(master["has_pre"] & master["has_post"])
        master.loc[incomplete, "exclusion_reason"] = "primary_oracle_requires_pre_and_post"
        master.loc[incomplete, "primary_oracle_included"] = False
    master["task_included"] = primary
    master["primary_oracle_included"] = primary & master["has_pre"] & master["has_post"]
    atomic_csv(master, manifests / "record_master_manifest.csv")
    task = master[master["task_included"]].copy()
    oracle = master[master["primary_oracle_included"]].copy()

    train_oracle = oracle[oracle["split"] == "Train"].copy()
    folds = StratifiedGroupKFold(
        n_splits=int(config["prediction"]["outer_folds"]), shuffle=True,
        random_state=int(config["prediction"]["seed"]),
    )
    train_oracle["fold"] = 0
    for fold, (_, holdout) in enumerate(folds.split(
        np.zeros(len(train_oracle)), train_oracle["target"].to_numpy(),
        train_oracle["patient_id"].astype(str).to_numpy(),
    ), 1):
        train_oracle.iloc[holdout, train_oracle.columns.get_loc("fold")] = fold
    patient_folds = train_oracle[["patient_id", "fold"]].drop_duplicates()
    if patient_folds["patient_id"].duplicated().any():
        raise AssertionError("Patient assigned to multiple folds")
    oracle = oracle.merge(patient_folds, on="patient_id", how="left")
    oracle.loc[oracle["split"] == "Valid", "fold"] = 0
    oracle["fold"] = oracle["fold"].astype(int)
    atomic_csv(task, manifests / "task_followup_rroc23.csv")
    atomic_csv(oracle, manifests / "oracle_record_manifest.csv")
    atomic_csv(patient_folds.sort_values(["fold", "patient_id"]), manifests / "patient_fold_map.csv")

    phase = oracle[[
        "record_uid", "split", "patient_id", "series_uid", "mapping_tier",
        "target", "fold", "followup_rroc",
    ]].merge(roi, on=["split", "patient_id", "series_uid"], how="left", validate="one_to_many")
    if phase["phase"].isna().any():
        raise AssertionError("Oracle record missing phase mapping")
    if phase.duplicated(["record_uid", "phase"]).any():
        raise AssertionError("Record has duplicate phase mapping")
    atomic_csv(phase, manifests / "segmentation_phase_manifest.csv")

    train_patients = set(oracle.loc[oracle.split == "Train", "patient_id"])
    valid_patients = set(oracle.loc[oracle.split == "Valid", "patient_id"])
    if train_patients & valid_patients:
        raise AssertionError("Train/Valid patient overlap")
    if oracle.groupby(["split", "series_uid"])["record_uid"].nunique().max() != 1:
        raise AssertionError("A series was assigned to multiple records")
    summary = {
        "all_records": int(len(master)),
        "mapping_tier_counts": master["mapping_tier"].value_counts().to_dict(),
        "exclusion_reason_counts": master.loc[~master.primary_oracle_included, "exclusion_reason"].value_counts().to_dict(),
        "task_records": int(len(task)),
        "primary_oracle": {},
        "manifest_gate_passed": True,
    }
    for split in ("Train", "Valid"):
        part = oracle[oracle.split == split]
        summary["primary_oracle"][split] = {
            "records": int(len(part)), "patients": int(part.patient_id.nunique()),
            "positive": int(part.target.sum()), "phases": int((phase.split == split).sum()),
        }
    atomic_json(summary, Path(config["paths"]["reports"]) / "record_manifest_audit.json")
    (Path(config["paths"]["reports"]) / ".RECORD_MANIFEST_PASS").write_text("pass\n")
    return oracle, phase, summary


def freeze_temporal_views(config: dict, phase: pd.DataFrame) -> dict:
    root = Path(config["whole_feature_root"])
    rows = []
    for record in phase.to_dict("records"):
        directory = root / str(record["split"]).casefold() / str(record["patient_id"]) / str(record["series_uid"]) / str(record["phase"]).casefold()
        metadata_path = directory / "metadata.json"
        qc_path = directory / "qc.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if str(metadata["frame_list_hash"]) != str(record["frame_list_hash"]):
            raise AssertionError(f"Frame-list hash mismatch: {metadata_path}")
        rows.append({
            "record_uid": record["record_uid"], "split": record["split"],
            "patient_id": record["patient_id"], "series_uid": record["series_uid"],
            "phase": record["phase"], "frame_list_hash": record["frame_list_hash"],
            "frame_indices": json.dumps(metadata["frame_indices"], separators=(",", ":")),
            "blocks": json.dumps(metadata["blocks"], separators=(",", ":")),
            "whole_metadata_path": str(metadata_path.resolve()),
            "whole_qc_path": str(qc_path.resolve()),
            "polarity": qc["polarity"], "polarity_label": qc["polarity_label"],
            "polarity_margin": qc["polarity_margin"],
            "baseline_start_position": qc["baseline_start_position"],
            "baseline_end_position_exclusive": qc["baseline_end_position_exclusive"],
            "temporal_views_locked": True,
        })
    frozen = pd.DataFrame(rows)
    path = Path(config["paths"]["manifests"]) / "frozen_temporal_view_manifest.csv"
    atomic_csv(frozen, path)
    summary = {"rows": len(frozen), "unique_record_phases": frozen[["record_uid", "phase"]].drop_duplicates().shape[0], "all_locked": True}
    atomic_json(summary, Path(config["paths"]["reports"]) / "frozen_temporal_view_audit.json")
    (Path(config["paths"]["reports"]) / ".TEMPORAL_VIEW_PASS").write_text("pass\n")
    return summary


def build_context_manifests(config: dict, phase: pd.DataFrame) -> dict:
    manifests = Path(config["paths"]["manifests"])
    gt = pd.read_csv(config["gt_roi_manifest"], dtype=str, keep_default_na=False)
    base = phase[["record_uid", "fold", "target"]].merge(
        gt, on=["record_uid"] if "record_uid" in gt.columns else [], how="left"
    ) if "record_uid" in gt.columns else phase[["record_uid", "split", "patient_id", "series_uid", "phase", "fold", "target", "mapping_tier"]].merge(
        gt, on=["split", "patient_id", "series_uid", "phase"], how="left", validate="one_to_one"
    )
    frozen = pd.read_csv(manifests / "frozen_temporal_view_manifest.csv", dtype=str, keep_default_na=False)[["record_uid", "phase", "whole_metadata_path", "whole_qc_path"]]
    base = base.merge(frozen, on=["record_uid", "phase"], how="left", validate="one_to_one")
    if base["roi_mask_path"].eq("").any() or base["roi_mask_path"].isna().any():
        raise AssertionError("Missing GT ROI mask path")
    shape_cache: dict[str, tuple[int, int]] = {}
    outputs = {}
    for fraction in config["oracle"]["context_side_fractions"]:
        result = base.copy()
        boxes, paddings, ratios, sides, overrides = [], [], [], [], []
        for row in result.to_dict("records"):
            first = str(row["frame_paths"]).split("|")[0]
            if first not in shape_cache:
                image = cv2.imread(first, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(first)
                shape_cache[first] = image.shape
            height, width = shape_cache[first]
            mask = cv2.imread(str(row["roi_mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask > 0):
                raise RuntimeError(f"Empty GT mask: {row['roi_mask_path']}")
            ys, xs = np.where(mask > 0)
            cx, cy = float(xs.mean()), float(ys.mean())
            lesion_box = parse_box(row["original_bbox"])
            lesion_side = max(lesion_box[2] - lesion_box[0], lesion_box[3] - lesion_box[1])
            fixed_side = int(round(min(height, width) * float(fraction)))
            minimum_side = int(math.ceil(lesion_side * float(config["oracle"]["minimum_lesion_bbox_factor"])))
            side = max(2, fixed_side, minimum_side)
            x0, y0 = int(math.floor(cx - side / 2)), int(math.floor(cy - side / 2))
            box = (x0, y0, x0 + side, y0 + side)
            padding = (max(0, -x0), max(0, -y0), max(0, box[2] - width), max(0, box[3] - height))
            boxes.append(box_text(box)); paddings.append(box_text(padding)); sides.append(side)
            ratios.append(side * side / float(height * width)); overrides.append(int(minimum_side > fixed_side))
        result["expanded_bbox"] = boxes
        result["crop_padding"] = paddings
        result["roi_area_ratio"] = ratios
        result["context_side_pixels"] = sides
        result["context_side_fraction"] = float(fraction)
        result["large_lesion_override"] = overrides
        result["roi_branch"] = f"gt_context_{int(round(float(fraction) * 100))}"
        result["frame_selection_policy"] = "locked_to_whole_cave_v3"
        result["preprocess_policy"] = "whole_preprocess_then_crop"
        tag = str(int(round(float(fraction) * 100)))
        phase_path = manifests / f"gt_context{tag}_phase_manifest.csv"
        atomic_csv(result, phase_path)
        outputs[tag] = {"phase_manifest": str(phase_path), "splits": {}}
        for split in ("Train", "Valid"):
            source = pd.read_csv(config[f"gt_cave_manifest_{split.casefold()}"], dtype=str, keep_default_na=False)
            selected = set(result.loc[result.split == split, "series_uid"])
            cave = source[source.series_uid.isin(selected)].copy()
            if cave.series_uid.nunique() != len(selected):
                raise AssertionError(f"CAVE manifest missing {split} series for scale {tag}")
            path = manifests / f"cave_manifest_gt_context{tag}_{split.casefold()}.csv"
            atomic_csv(cave, path)
            outputs[tag]["splits"][split] = {"path": str(path), "series": int(len(cave))}
        outputs[tag]["phases"] = int(len(result))
        outputs[tag]["median_area_ratio"] = float(np.median(ratios))
        outputs[tag]["large_lesion_overrides"] = int(sum(overrides))
    atomic_json(outputs, Path(config["paths"]["reports"]) / "gt_context_manifest_audit.json")
    (Path(config["paths"]["reports"]) / ".GT_CONTEXT_MANIFEST_PASS").write_text("pass\n")
    return outputs


def build_label_montages(config: dict, phase: pd.DataFrame, limit: int = 60) -> dict:
    root = Path(config["project_root"])
    sys.modules.pop("assets", None)
    sys.modules.pop("common", None)
    sys.path.insert(0, str(root / "code/api_adverse_lesion_cave_v1"))
    import assets
    if assets.nib is None:
        summary = {"requested": limit, "written": 0, "status": "pending_separate_cave_environment"}
        atomic_json(summary, Path(config["paths"]["reports"]) / "label_montage_audit.json")
        return summary
    output = Path(config["paths"]["reports"]) / "label_montage"
    output.mkdir(parents=True, exist_ok=True)
    sample = phase.sort_values(["split", "phase", "annotation_layout", "record_uid"]).groupby(
        ["split", "phase", "annotation_layout"], group_keys=False
    ).head(max(1, limit // max(1, phase.groupby(["split", "phase", "annotation_layout"]).ngroups)))
    sample = sample.head(limit)
    written = 0
    for row in sample.to_dict("records"):
        raw, _ = assets.load_nifti_mask(Path(row["segmentation_path"]))
        mask = assets.apply_orientation(raw, row["orientation_transform"])
        ref = str(row.get("matched_reference_frame_path", ""))
        image = cv2.imread(ref, cv2.IMREAD_GRAYSCALE) if ref else None
        if image is None:
            frames = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in str(row["frame_paths"]).split("|")]
            frames = [item for item in frames if item is not None]
            image = np.median(np.stack(frames), axis=0).astype(np.uint8)
        if mask.shape != image.shape:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panels = [base.copy()]
        labels = ["image"]
        colors = [(30, 30, 255), (30, 220, 30), (255, 80, 30), (240, 220, 30), (220, 30, 220), (30, 220, 220)]
        combined = base.copy()
        for label, color in zip(range(1, 7), colors):
            region = mask == label
            panel = base.copy()
            panel[region] = (0.35 * panel[region] + 0.65 * np.asarray(color)).astype(np.uint8)
            combined[region] = (0.35 * combined[region] + 0.65 * np.asarray(color)).astype(np.uint8)
            panels.append(panel); labels.append(f"label {label}")
        panels.append(combined); labels.append("combined")
        resized = []
        for panel, label in zip(panels, labels):
            tile = cv2.resize(panel, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            resized.append(tile)
        canvas = np.concatenate(resized, axis=1)
        name = f"{row['split']}_{row['patient_id']}_{row['series_uid']}_{row['phase']}.jpg".replace("/", "_")
        cv2.imwrite(str(output / name), canvas)
        written += 1
    summary = {"requested": limit, "written": written, "directory": str(output)}
    atomic_json(summary, Path(config["paths"]["reports"]) / "label_montage_audit.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    Path(config["paths"]["manifests"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
    snap = snapshot(config)
    oracle, phase, mapping = build_manifests(config)
    temporal = freeze_temporal_views(config, phase)
    contexts = build_context_manifests(config, phase)
    montages = build_label_montages(config, phase)
    payload = {"status": "complete", "snapshot_files": len(snap["files"]), "mapping": mapping, "temporal": temporal, "contexts": contexts, "montages": montages}
    update_run_manifest(config, "prepare_record_oracle", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
