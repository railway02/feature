from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import sha256_file, text_bool
from .preprocessing_adapter import PairRecord, audit_input_paths, load_local_reference_pairs
from .v5_adapter import v5_core_description


MASTER_REQUIRED = {
    "split", "patient_id", "series_uid", "series_id", "pre_frame_paths", "post_frame_paths",
    "pre_frame_list_hash", "post_frame_list_hash", "has_prepost_api", "prepost_frame_paths_exist",
    "png2d_png_key_pre", "png2d_png_key_post", "png2d_image_path_pre", "png2d_image_path_post",
    "png2d_mask_path_pre", "png2d_mask_path_post", "png2d_phase_uid_pre", "png2d_phase_uid_post",
    "png2d_mapping_method_pre", "png2d_mapping_method_post", "has_png2d_prepost",
    "geometry_prepost_eligible", "excel_record_rows", "target_values", "master_status",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _outcome_column(workbook: pd.DataFrame, prefix: str) -> str:
    matches = [str(column) for column in workbook.columns if str(column).strip().startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one outcome column starting {prefix!r}, got {matches}")
    return matches[0]


def _workbook_labels(path: str | Path, split: str, prefix: str) -> dict[int, int]:
    workbook = pd.read_excel(path, dtype=object)
    column = _outcome_column(workbook, prefix)
    labels: dict[int, int] = {}
    for position, value in enumerate(workbook[column].tolist(), start=2):
        try:
            label = int(float(value))
        except (TypeError, ValueError):
            continue
        if label in {0, 1}:
            labels[position] = label
    if not labels:
        raise ValueError(f"{split} workbook has no valid binary outcome labels")
    return labels


def _mapped_labels(cfg: dict[str, Any], split: str) -> dict[str, tuple[int, str]]:
    paths = cfg["paths"]
    label_cfg = cfg["labels"]
    mapping_path = paths[f"{split.casefold()}_record_mapping"]
    workbook_path = paths[f"{split.casefold()}_workbook"]
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    required = {"series_uid", "excel_row_number", "mapping_status"}
    _require_columns(mapping, required, f"{split} record mapping")
    mapping = mapping[mapping["mapping_status"].eq(str(label_cfg["accepted_mapping_status"]))].copy()
    labels_by_row = _workbook_labels(workbook_path, split, str(label_cfg["outcome_column_prefix"]))
    result: dict[str, tuple[int, str]] = {}
    for series_uid, rows in mapping.groupby("series_uid", sort=False):
        if len(rows) != 1:
            continue
        row_number = int(float(rows.iloc[0]["excel_row_number"]))
        if row_number in labels_by_row:
            result[str(series_uid)] = (labels_by_row[row_number], f"mapped_{split.casefold()}_excel_row_{row_number}")
    return result


def _direct_master_label(row: pd.Series, required_record_rows: int) -> int | None:
    try:
        record_rows = int(float(row["excel_record_rows"]))
    except (TypeError, ValueError):
        return None
    value = str(row["target_values"]).strip()
    if record_rows == int(required_record_rows) and value in {"0", "1"}:
        return int(value)
    return None


def _build_task_rows(cfg: dict[str, Any], records: list[PairRecord]) -> tuple[pd.DataFrame, list[str]]:
    paths = cfg["paths"]
    expected = cfg["expected"]
    master = pd.read_csv(paths["ready_task_master"], dtype=str, keep_default_na=False)
    _require_columns(master, MASTER_REQUIRED, "ready_task_master")
    failures: list[str] = []
    if len(master) != int(expected["paired_series"]):
        failures.append(f"ready_task_master_rows={len(master)} expected={expected['paired_series']}")
    if master["series_uid"].duplicated().any():
        failures.append("ready_task_master_duplicate_series_uid")
    counts = master["split"].value_counts().to_dict()
    for split, key in (("Train", "paired_series_train"), ("Valid", "paired_series_valid")):
        if int(counts.get(split, 0)) != int(expected[key]):
            failures.append(f"ready_task_master_{split}_rows={counts.get(split, 0)} expected={expected[key]}")

    pairs_by_uid = {record.series_uid: record for record in records}
    if list(master["series_uid"]) != [record.series_uid for record in records]:
        failures.append("ready_task_master_order_differs_from_temporal_pair_manifest")
    mapped = {split: _mapped_labels(cfg, split) for split in ("Train", "Valid")}
    direct_rows = int(cfg["labels"]["direct_master_requires_excel_record_rows"])
    output: list[dict[str, Any]] = []
    for order, row in master.reset_index(drop=True).iterrows():
        uid = str(row["series_uid"])
        split = str(row["split"])
        pair = pairs_by_uid.get(uid)
        reasons: list[str] = []
        if pair is None:
            reasons.append("series_absent_from_temporal_pair_manifest")
        else:
            if (pair.split, pair.patient_id, pair.series_id) != (split, str(row["patient_id"]), str(row["series_id"])):
                reasons.append("master_pair_identity_mismatch")
            if pair.pre.phase_uid != str(row["png2d_phase_uid_pre"]) or pair.post.phase_uid != str(row["png2d_phase_uid_post"]):
                reasons.append("master_pair_phase_uid_mismatch")
            if "|".join(pair.pre.frame_paths) != str(row["pre_frame_paths"]) or "|".join(pair.post.frame_paths) != str(row["post_frame_paths"]):
                reasons.append("master_pair_frame_paths_mismatch")
        for field in ("series_uid", "patient_id", "png2d_png_key_pre", "png2d_png_key_post", "png2d_phase_uid_pre", "png2d_phase_uid_post"):
            if not str(row[field]).strip():
                reasons.append(f"missing_{field}")
        for field in ("has_prepost_api", "prepost_frame_paths_exist", "has_png2d_prepost", "geometry_prepost_eligible"):
            if not text_bool(row[field]):
                reasons.append(f"not_true_{field}")
        for field in ("png2d_image_path_pre", "png2d_image_path_post", "png2d_mask_path_pre", "png2d_mask_path_post"):
            path = Path(str(row[field]))
            if not path.is_file():
                reasons.append(f"missing_current_2d_or_gt_asset_{field}")

        direct_label = _direct_master_label(row, direct_rows)
        mapped_label = mapped.get(split, {}).get(uid)
        target: int | None = None
        provenance = ""
        if direct_label is not None:
            target, provenance = direct_label, "direct_ready_master_single_record"
            if mapped_label is not None and mapped_label[0] != target:
                reasons.append("direct_master_label_disagrees_with_mapped_excel_label")
        elif mapped_label is not None:
            target, provenance = mapped_label
        else:
            reasons.append("formal_outcome_label_unresolved")
        output.append({
            "task_order": order, "split": split, "series_uid": uid, "patient_id": str(row["patient_id"]),
            "series_id": str(row["series_id"]), "target": target if target is not None else "",
            "label_provenance": provenance, "master_target_values": str(row["target_values"]),
            "excel_record_rows": str(row["excel_record_rows"]),
            "pre_phase_uid": str(row["png2d_phase_uid_pre"]), "post_phase_uid": str(row["png2d_phase_uid_post"]),
            "pre_image_path": str(row["png2d_image_path_pre"]), "post_image_path": str(row["png2d_image_path_post"]),
            "pre_mask_path": str(row["png2d_mask_path_pre"]), "post_mask_path": str(row["png2d_mask_path_post"]),
            "contract_valid": int(not reasons), "failure_reason": "|".join(reasons),
        })
    table = pd.DataFrame(output)
    if not table["contract_valid"].astype(bool).all():
        failures.append(f"invalid_ready_task_rows={int((~table['contract_valid'].astype(bool)).sum())}")
    return table, failures


def _assign_train_folds(task_rows: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    train = task_rows[task_rows["split"].eq("Train")].copy()
    expected_rows = int(cfg["expected"]["paired_series_train"])
    failures: list[str] = []
    if len(train) != expected_rows:
        failures.append(f"train_task_rows={len(train)} expected={expected_rows}")
    if not train["contract_valid"].astype(bool).all() or (train["target"].astype(str).str.strip() == "").any():
        failures.append("cannot_assign_folds_with_invalid_or_unlabeled_train_rows")
        train["fold"] = ""
        return train, failures
    y = pd.to_numeric(train["target"], errors="raise").to_numpy()
    groups = train["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=int(cfg["expected"]["outer_folds"]), shuffle=True, random_state=int(cfg["folds"]["seed"])
    )
    fold = pd.Series(index=train.index, dtype="int64")
    for number, (_, holdout) in enumerate(splitter.split(train, y, groups), start=1):
        fold.iloc[holdout] = number
    train["fold"] = fold.astype(int).to_numpy()
    if train["fold"].isna().any():
        failures.append("incomplete_fold_assignment")
    if train.groupby("patient_id")["fold"].nunique().max() != 1:
        failures.append("patient_crosses_outer_folds")
    labels = set(train["fold"].astype(int))
    expected_folds = set(range(1, int(cfg["expected"]["outer_folds"]) + 1))
    if labels != expected_folds:
        failures.append(f"fold_labels={sorted(labels)} expected={sorted(expected_folds)}")
    return train, failures


def audit_preprocessing_contract(cfg: dict[str, Any], *, verify_input_files: bool = True) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = load_local_reference_pairs(cfg)
    expected = cfg["expected"]
    paths = cfg["paths"]

    train = [record for record in records if record.split == "Train"]
    valid = [record for record in records if record.split == "Valid"]
    if len(train) != int(expected["paired_series_train"]) or len(valid) != int(expected["paired_series_valid"]):
        raise AssertionError("Local Reference pair split count changed")

    file_rows = audit_input_paths(records) if verify_input_files else []
    detail = pd.DataFrame(file_rows)
    if verify_input_files and (
        (~detail["reference_exists"]).any() or (~detail["mask_exists"]).any() or (detail["missing_frame_count"] > 0).any()
    ):
        bad = detail[(~detail["reference_exists"]) | (~detail["mask_exists"]) | (detail["missing_frame_count"] > 0)]
        # Preserve all task rows and report failure below; never filter an affected series.
        file_failure = f"missing_temporal_or_old_roi_assets_for_phases={len(bad)}"
    else:
        file_failure = ""

    task_rows, task_failures = _build_task_rows(cfg, records)
    train_folds, fold_failures = _assign_train_folds(task_rows, cfg)

    mapping_methods: dict[str, int] = {}
    for record in records:
        for phase in (record.pre, record.post):
            mapping_methods[phase.mapping_method] = mapping_methods.get(phase.mapping_method, 0) + 1
    failures = [item for item in [file_failure, *task_failures, *fold_failures] if item]
    summary: dict[str, Any] = {
        "status": "pass" if not failures else "fail",
        "purpose": "Stage A Local Reference independent-local-crop preflight",
        "outcome_policy": "new formal 800/211 task; master-single-record labels or uniquely mapped Excel labels",
        "reference_primary_geometry": cfg["geometry"]["primary_mode"],
        "pair_records": len(records),
        "pair_records_by_split": {"Train": len(train), "Valid": len(valid)},
        "phase_records": len(records) * 2,
        "mapping_method_counts_for_paired_phases": mapping_methods,
        "new_task_contract": {
            "train_rows": int((task_rows["split"] == "Train").sum()),
            "valid_rows": int((task_rows["split"] == "Valid").sum()),
            "all_rows_retained": len(task_rows) == int(expected["paired_series"]),
            "canonical_order_source": "ready_task_master.csv",
            "grouping_unit": cfg["folds"]["grouping_unit"],
            "fold_seed": int(cfg["folds"]["seed"]),
            "train_patient_count": int(train_folds["patient_id"].nunique()),
            "train_fold_counts": train_folds["fold"].value_counts().sort_index().to_dict() if "fold" in train_folds else {},
        },
        "input_files_checked": bool(verify_input_files),
        "input_phase_rows": int(len(detail)) if verify_input_files else 0,
        "source_hashes": {
            "ready_task_master": sha256_file(paths["ready_task_master"]),
            "temporal_pairs": sha256_file(paths["temporal_pairs"]),
            "roi_phase_manifest": sha256_file(paths["roi_phase_manifest"]),
            "train_record_mapping": sha256_file(paths["train_record_mapping"]),
            "valid_record_mapping": sha256_file(paths["valid_record_mapping"]),
            "train_workbook": sha256_file(paths["train_workbook"]),
            "valid_workbook": sha256_file(paths["valid_workbook"]),
        },
        "v5_core": v5_core_description(cfg),
        "failures": failures,
    }
    return summary, detail, task_rows, train_folds
