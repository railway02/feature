#!/usr/bin/env python3
"""Import the audited PNG-to-phase mapping into complete frozen phase coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    mapping_cfg = cfg["mapping"]

    audit_path = manifests / "png2d_input_lock.json"
    if not audit_path.is_file():
        raise RuntimeError(f"missing PNG input audit lock: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "success":
        raise RuntimeError("PNG input audit did not pass")

    source_path = manifests / "source_phase_index_all.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    accepted = pd.read_csv(mapping_cfg["accepted_csv"], dtype=str, keep_default_na=False)
    unresolved = pd.read_csv(mapping_cfg["unresolved_csv"], dtype=str, keep_default_na=False)
    conflict = pd.read_csv(mapping_cfg["conflict_csv"], dtype=str, keep_default_na=False)
    failures: list[str] = []

    if source["phase_uid"].duplicated().any() or source["frame_list_hash"].duplicated().any():
        failures.append("source phase_uid/frame_list_hash is not unique")
    if accepted["phase_uid"].duplicated().any() or accepted["frame_list_hash"].duplicated().any():
        failures.append("accepted phase_uid/frame_list_hash is not unique")
    source_by_uid = source.set_index("phase_uid", drop=False)
    source_uids = set(source["phase_uid"])
    extra = sorted(set(accepted["phase_uid"]) - source_uids)
    if extra:
        failures.append(f"accepted mapping outside frozen source: {extra[:5]}")

    accepted_by_uid: dict[str, dict[str, Any]] = {}
    for row in accepted.to_dict("records"):
        uid = str(row["phase_uid"])
        if uid not in source_by_uid.index:
            continue
        src = source_by_uid.loc[uid]
        exact_fields = (
            "patient_id", "split", "series_uid", "phase",
            "frame_paths", "frame_list_hash", "api_dir",
        )
        mismatches = [
            key for key in exact_fields
            if str(row.get(key, "")) != str(src.get(key, ""))
        ]
        if mismatches:
            failures.append(f"{uid}: source mismatch {mismatches}")
            continue
        accepted_by_uid[uid] = row

    unresolved_by_uid: dict[str, dict[str, Any]] = {}
    for row in unresolved.to_dict("records"):
        uid = str(row.get("phase_uid", ""))
        if uid and uid in source_uids:
            unresolved_by_uid[uid] = row

    rows: list[dict[str, Any]] = []
    for source_row in source.to_dict("records"):
        uid = str(source_row["phase_uid"])
        base = {
            **source_row,
            "phase_mapping_status": "missing",
            "mapping_method": "",
            "mapping_reason": "no_png2d_annotation_for_frozen_phase",
            "mapping_input_csv": str(Path(mapping_cfg["accepted_csv"]).resolve()),
            "mapping_input_csv_sha256": sha256_file(mapping_cfg["accepted_csv"]),
            "png_key": "",
            "reference_image_path": "",
            "reference_sha256": "",
            "mask_path": "",
            "mask_sha256": "",
            "annotation_shape": "",
            "mask_label_values": "",
            "mask_label_pixel_counts": "",
            "mask_nonzero_pixels": "",
            "orientation_transform": "",
            "orientation_status": "none",
            "identity_pearson_correlation": "",
            "identity_gradient_correlation": "",
            "candidate_count": "",
            "annotation_source": "png_2d_gt",
            "segmentation_model_used": 0,
        }
        mapped = accepted_by_uid.get(uid)
        if mapped is not None:
            base.update({
                "phase_mapping_status": "accepted",
                "mapping_method": "png2d_mean_identity_verified",
                "mapping_reason": "",
                "png_key": mapped["png_key"],
                "reference_image_path": mapped["image_path"],
                "reference_sha256": mapped["image_sha256"],
                "mask_path": mapped["mask_path"],
                "mask_sha256": mapped["mask_sha256"],
                "annotation_shape": mapped["mask_shape"],
                "mask_label_values": mapped["mask_label_values"],
                "mask_label_pixel_counts": mapped["mask_label_pixel_counts"],
                "mask_nonzero_pixels": mapped["mask_nonzero_pixels"],
                "orientation_transform": "identity",
                "orientation_status": "identity_mean_verified",
                "identity_pearson_correlation": mapped["identity_pearson_correlation"],
                "identity_gradient_correlation": mapped["identity_gradient_correlation"],
                "candidate_count": mapped["candidate_count"],
            })
        elif uid in unresolved_by_uid:
            item = unresolved_by_uid[uid]
            base.update({
                "phase_mapping_status": "needs_review",
                "mapping_method": "png2d_mean_below_correlation_threshold",
                "mapping_reason": str(item.get("reason", "mapping_unresolved")),
                "png_key": str(item.get("png_key", "")),
                "reference_image_path": str(item.get("image_path", "")),
                "reference_sha256": str(item.get("image_sha256", "")),
                "mask_path": str(item.get("mask_path", "")),
                "mask_sha256": str(item.get("mask_sha256", "")),
                "annotation_shape": str(item.get("mask_shape", "")),
                "mask_label_values": str(item.get("mask_label_values", "")),
                "mask_label_pixel_counts": str(item.get("mask_label_pixel_counts", "")),
                "mask_nonzero_pixels": str(item.get("mask_nonzero_pixels", "")),
                "orientation_transform": "identity",
                "orientation_status": "targeted_8_transforms_no_match",
                "identity_pearson_correlation": str(item.get("identity_pearson_correlation", "")),
                "identity_gradient_correlation": str(item.get("identity_gradient_correlation", "")),
                "candidate_count": str(item.get("candidate_count", "")),
            })
        rows.append(base)

    full = pd.DataFrame(rows)
    if len(accepted_by_uid) != int(mapping_cfg["expected_accepted"]):
        failures.append(f"accepted source mapping count={len(accepted_by_uid)}")
    if len(full) != 2622:
        failures.append(f"source phase coverage count={len(full)}")
    if full["phase_uid"].tolist() != source["phase_uid"].tolist():
        failures.append("full mapping changed source phase order")
    accepted_full = full[full["phase_mapping_status"] == "accepted"]
    review_full = full[full["phase_mapping_status"] == "needs_review"]
    missing_full = full[full["phase_mapping_status"] == "missing"]
    if len(accepted_full) != 2199 or len(review_full) != 3 or len(missing_full) != 420:
        failures.append(
            f"coverage counts accepted/review/missing="
            f"{len(accepted_full)}/{len(review_full)}/{len(missing_full)}"
        )
    if len(conflict):
        failures.append(f"unexpected mapping conflict rows={len(conflict)}")

    atomic_csv(full, manifests / "source_phase_with_mask_map.csv")
    atomic_csv(review_full, reports / "02_png2d_mapping_needs_review.csv")
    atomic_csv(missing_full, reports / "02_source_phases_without_png2d.csv")
    outside_source = unresolved[
        ~unresolved.get("phase_uid", pd.Series("", index=unresolved.index)).isin(source_uids)
    ].copy()
    atomic_csv(outside_source, reports / "02_png2d_outside_frozen_source.csv")

    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "source_phases": len(full),
        "accepted": len(accepted_full),
        "needs_review": len(review_full),
        "missing_png2d": len(missing_full),
        "png_rows_outside_frozen_source": len(outside_source),
        "conflict_rows": len(conflict),
        "mapping_method": "png2d_mean_identity_verified",
        "orientation_policy": mapping_cfg["orientation_policy"],
        "source_phase_with_mask_map": str(manifests / "source_phase_with_mask_map.csv"),
        "source_phase_with_mask_map_sha256": sha256_file(
            manifests / "source_phase_with_mask_map.csv"
        ),
    }
    atomic_json(summary, reports / "02_png2d_mapping_import_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

