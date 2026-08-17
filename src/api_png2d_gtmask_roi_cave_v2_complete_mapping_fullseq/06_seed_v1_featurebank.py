#!/usr/bin/env python3
"""Build a versioned composite featurebank from verified v1 artifacts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


OLD_ROOT = Path(
    "/root/autodl-tmp/aneurysm/outputs/"
    "api_png2d_gtmask_roi_cave_v1_fullseq/cave_local_eligible_featurebank"
)
OLD_MANIFEST_ROOT = Path(
    "/root/autodl-tmp/aneurysm/manifests/"
    "api_png2d_gtmask_roi_cave_v1_fullseq"
)
OLD_ROI = OLD_MANIFEST_ROOT / "roi_phase_manifest_eligible.csv"
OLD_TEMPORAL = OLD_MANIFEST_ROOT / "whole_temporal_views_eligible.csv"
OLD_EXCLUSIONS = OLD_MANIFEST_ROOT / "runtime_feature_exclusions.csv"
OLD_REPORT_ROOT = Path(
    "/root/autodl-tmp/aneurysm/reports/"
    "api_png2d_gtmask_roi_cave_v1_fullseq"
)

SCIENCE_COLUMNS = [
    "frame_paths",
    "frame_list_hash",
    "mask_sha256",
    "effective_mask_array_sha256",
    "effective_mask_shape",
    "mask_resized_to_frame",
    "selected_labels",
    "selected_foreground_pixels",
    "original_bbox",
    "expanded_bbox",
    "fallback_bbox",
    "extended_fallback_bbox",
    "whole_metadata_sha256",
]


def phase_destination(root: Path, row: dict[str, str]) -> Path:
    return (
        root
        / row["split"].casefold()
        / row["patient_id"]
        / row["series_uid"]
        / row["phase"]
    )


def link_file(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        os.symlink(source, destination)
        return "symlink"


def wrap_phase(
    source: Path,
    destination: Path,
    row: dict[str, str],
    cfg: dict[str, Any],
    manifest_sha256: str,
) -> tuple[str, str]:
    if destination.exists():
        success_path = destination / ".SUCCESS.json"
        if success_path.is_file():
            payload = json.loads(success_path.read_text(encoding="utf-8"))
            reuse = payload.get("feature_reuse", {})
            if (
                str(payload.get("phase_uid", "")) == row["phase_uid"]
                and reuse.get("source_pipeline") == "api_png2d_gtmask_roi_cave_v1_fullseq"
            ):
                return "resumed", str(reuse.get("link_mode", ""))
        raise RuntimeError(f"unexpected existing composite phase: {destination}")

    temporary = destination.with_name(destination.name + ".seeding")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    link_modes: set[str] = set()
    for item in source.iterdir():
        if item.name in {".SUCCESS.json", "metadata.json"}:
            continue
        target = temporary / item.name
        if item.is_file():
            link_modes.add(link_file(item, target))
        elif item.is_dir():
            os.symlink(item, target, target_is_directory=True)
            link_modes.add("symlink")

    old_success_path = source / ".SUCCESS.json"
    old_metadata_path = source / "metadata.json"
    old_success = json.loads(old_success_path.read_text(encoding="utf-8"))
    metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    link_mode = "+".join(sorted(link_modes))
    reuse = {
        "source_pipeline": "api_png2d_gtmask_roi_cave_v1_fullseq",
        "source_phase_dir": str(source),
        "source_success_sha256": sha256_file(old_success_path),
        "source_metadata_sha256": sha256_file(old_metadata_path),
        "source_embedding_sha256": sha256_file(source / "embedding_5120.npy"),
        "equivalence_contract": "roi_and_temporal_scientific_fields_exact",
        "link_mode": link_mode,
    }

    roi_meta = metadata.setdefault("roi", {})
    roi_meta["pipeline_version"] = cfg["version"]
    roi_meta["roi_pipeline_version"] = cfg["version"]
    roi_meta["mapping_method"] = row["mapping_method"]
    roi_meta["orientation_status"] = row["orientation_status"]
    roi_meta["manual_confirmation_note"] = row.get(
        "manual_confirmation_note", ""
    )
    metadata["feature_reuse"] = reuse
    metadata["provenance"]["feature_reuse"] = reuse

    success = dict(old_success)
    success["roi_pipeline_version"] = cfg["version"]
    success["manifest_sha256"] = manifest_sha256
    success["mapping_method"] = row["mapping_method"]
    success["orientation_status"] = row["orientation_status"]
    success["feature_reuse"] = reuse
    atomic_json(metadata, temporary / "metadata.json")
    atomic_json(success, temporary / ".SUCCESS.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return "seeded", link_mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    outputs = Path(cfg["paths"]["outputs"])
    reports = Path(cfg["paths"]["reports"])
    feature_root = outputs / "cave_local_eligible_featurebank"
    feature_root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    old_roi = pd.read_csv(OLD_ROI, dtype=str, keep_default_na=False)
    new_roi = pd.read_csv(
        manifests / "roi_phase_manifest_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    old_temporal = pd.read_csv(
        OLD_TEMPORAL, dtype=str, keep_default_na=False
    )
    new_temporal = pd.read_csv(
        manifests / "whole_temporal_views_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    if old_roi["phase_uid"].duplicated().any():
        raise AssertionError("v1 ROI phase_uid is not unique")
    if new_roi["phase_uid"].duplicated().any():
        raise AssertionError("v2 ROI phase_uid is not unique")
    old_uids = set(old_roi["phase_uid"])
    new_uids = set(new_roi["phase_uid"])
    if not old_uids.issubset(new_uids):
        raise AssertionError("v2 lost a v1 eligible phase")

    joined = old_roi[["phase_uid", *SCIENCE_COLUMNS]].merge(
        new_roi[["phase_uid", *SCIENCE_COLUMNS]],
        on="phase_uid",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    mismatches: dict[str, int] = {}
    for column in SCIENCE_COLUMNS:
        count = int((joined[f"{column}_v1"] != joined[f"{column}_v2"]).sum())
        if count:
            mismatches[column] = count
    temporal = old_temporal[
        ["phase_uid", "frame_list_hash", "blocks_json"]
    ].merge(
        new_temporal[["phase_uid", "frame_list_hash", "blocks_json"]],
        on="phase_uid",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    temporal_mismatch = int(
        (
            (temporal["frame_list_hash_v1"] != temporal["frame_list_hash_v2"])
            | (temporal["blocks_json_v1"] != temporal["blocks_json_v2"])
        ).sum()
    )
    if mismatches or temporal_mismatch:
        raise AssertionError(
            f"v1/v2 reuse equivalence failed: {mismatches}, "
            f"temporal={temporal_mismatch}"
        )

    new_by_uid = new_roi.set_index("phase_uid", drop=False)
    manifest_hashes = {
        split: sha256_file(
            manifests
            / f"cave_manifest_local_{split.casefold()}_eligible.csv"
        )
        for split in ("Train", "Valid")
    }
    audit_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for success_path in sorted(OLD_ROOT.rglob(".SUCCESS.json")):
        try:
            payload = json.loads(success_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        uid = str(payload.get("phase_uid", ""))
        if not uid:
            continue
        if uid in seen or uid not in new_by_uid.index:
            raise AssertionError(f"unexpected reused phase: {uid}")
        seen.add(uid)
        row = {
            key: str(value)
            for key, value in new_by_uid.loc[uid].to_dict().items()
        }
        source = success_path.parent
        destination = phase_destination(feature_root, row)
        state, link_mode = wrap_phase(
            source,
            destination,
            row,
            cfg,
            manifest_hashes[row["split"]],
        )
        audit_rows.append({
            "phase_uid": uid,
            "split": row["split"],
            "series_uid": row["series_uid"],
            "phase": row["phase"],
            "reuse_state": state,
            "link_mode": link_mode,
            "source_phase_dir": str(source),
            "destination_phase_dir": str(destination),
            "embedding_sha256": sha256_file(source / "embedding_5120.npy"),
            "roi_scientific_fields_equal": True,
            "temporal_indices_equal": True,
        })

    if len(seen) != 2176:
        raise AssertionError(f"expected 2176 reusable v1 successes, got {len(seen)}")

    archive = reports / "resume_schema_archive"
    archive.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        source = OLD_ROOT / f"feature_schema_{split}.json"
        shutil.copy2(source, archive / f"{split}_v1_reused.json")
    for index, source in enumerate(sorted(
        (OLD_REPORT_ROOT / "resume_schema_archive").glob(
            "train_*_before_extended.json"
        )
    )):
        shutil.copy2(
            source, archive / f"train_v1_before_extended_{index:02d}.json"
        )

    reused_exclusions = pd.read_csv(
        OLD_EXCLUSIONS, dtype=str, keep_default_na=False
    )
    reused_exclusions["reuse_source_pipeline"] = (
        "api_png2d_gtmask_roi_cave_v1_fullseq"
    )
    reused_exclusions["reuse_equivalence_contract"] = (
        "roi_and_temporal_scientific_fields_exact"
    )
    reused_exclusion_path = reports / "06_reused_runtime_exclusions.csv"
    atomic_csv(reused_exclusions, reused_exclusion_path)

    supplement = new_roi[~new_roi["phase_uid"].isin(old_uids)].copy()
    for split in ("Train", "Valid"):
        values = (
            supplement[supplement["split"] == split]["series_uid"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        path = reports / f"supplement_series_{split.casefold()}.txt"
        path.write_text(
            "".join(f"{value}\n" for value in values),
            encoding="utf-8",
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["split", "series_uid", "phase"], kind="stable"
    )
    audit_path = reports / "06_v1_feature_reuse_audit.csv"
    atomic_csv(audit, audit_path)
    summary = {
        "status": "success",
        "v1_roi_phases": len(old_roi),
        "v2_roi_phases": len(new_roi),
        "overlap_roi_phases": len(joined),
        "new_v2_phases": len(supplement),
        "reused_successful_phases": len(seen),
        "reused_runtime_exclusions": len(reused_exclusions),
        "scientific_field_mismatches": mismatches,
        "temporal_mismatches": temporal_mismatch,
        "supplement_by_split": [
            {
                "split": str(split),
                "mapping_method": str(method),
                "phase": str(phase),
                "count": int(count),
            }
            for (split, method, phase), count in supplement.groupby(
                ["split", "mapping_method", "phase"]
            ).size().items()
        ],
        "reuse_audit": str(audit_path),
        "reuse_audit_sha256": sha256_file(audit_path),
        "reused_exclusions": str(reused_exclusion_path),
        "reused_exclusions_sha256": sha256_file(reused_exclusion_path),
        "storage_policy": "hardlink_or_symlink_large_v1_artifacts_with_v2_metadata_wrapper",
    }
    atomic_json(summary, reports / "06_v1_feature_reuse_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
