#!/usr/bin/env python3
"""Finalize explicit runtime exclusions after all three ROI activity tiers fail."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


ACTIVE_PATTERN = re.compile(r"Activity ROI too small:\s*(\d+)\s*pixels")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    outputs = Path(cfg["paths"]["outputs"])
    reports = Path(cfg["paths"]["reports"])
    feature_root = outputs / "cave_local_eligible_featurebank"
    roi = pd.read_csv(
        manifests / "roi_phase_manifest_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )

    successes: set[str] = set()
    for path in feature_root.rglob(".SUCCESS.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        uid = str(payload.get("phase_uid", ""))
        if uid:
            successes.add(uid)
    missing = roi[~roi["phase_uid"].isin(successes)].copy()

    reused_path = reports / "06_reused_runtime_exclusions.csv"
    reused_by_uid: dict[str, dict[str, str]] = {}
    if reused_path.is_file():
        reused_table = pd.read_csv(reused_path, dtype=str, keep_default_na=False)
        if reused_table["phase_uid"].duplicated().any():
            raise AssertionError("duplicate reused runtime exclusion phase_uid")
        reused_by_uid = {
            str(row["phase_uid"]): {key: str(value) for key, value in row.items()}
            for row in reused_table.to_dict("records")
        }

    failure_candidates: dict[str, list[Path]] = {}
    for split in ("Train", "Valid"):
        shard_root = outputs / f"shards_{split.casefold()}_full"
        for path in shard_root.glob("shard_*/_failures/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                uid = f"{payload.get('series_uid', '')}::{payload.get('phase', '')}"
                failure_candidates.setdefault(uid, []).append(path)
            except Exception:
                continue

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for record in missing.to_dict("records"):
        uid = str(record["phase_uid"])
        if uid in reused_by_uid:
            reused = dict(reused_by_uid[uid])
            bbox_checks = {
                "primary_bbox": "expanded_bbox",
                "fallback_bbox": "fallback_bbox",
                "extended_fallback_bbox": "extended_fallback_bbox",
            }
            mismatch = [
                old_key
                for old_key, roi_key in bbox_checks.items()
                if reused.get(old_key, "") != str(record[roi_key])
            ]
            if mismatch:
                failures.append(f"{uid}: reused exclusion ROI mismatch {mismatch}")
                continue
            reused["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
            rows.append(reused)
            continue
        candidates = sorted(
            failure_candidates.get(uid, []),
            key=lambda path: (path.stat().st_mtime, str(path)),
            reverse=True,
        )
        if not candidates:
            failures.append(f"{uid}: no shard failure JSON")
            continue
        chosen = candidates[0]
        payload = json.loads(chosen.read_text(encoding="utf-8"))
        error = str(payload.get("error", ""))
        match = ACTIVE_PATTERN.search(error)
        if match is None:
            failures.append(f"{uid}: latest failure is not activity-too-small: {error}")
            continue
        phase_dir = (
            feature_root
            / str(record["split"]).casefold()
            / str(record["patient_id"])
            / str(record["series_uid"])
            / str(record["phase"])
        )
        if (phase_dir / ".SUCCESS.json").is_file():
            failures.append(f"{uid}: exclusion overlaps a formal SUCCESS")
            continue
        rows.append({
            "phase_uid": uid,
            "split": record["split"],
            "patient_id": record["patient_id"],
            "series_uid": record["series_uid"],
            "phase": record["phase"],
            "stage": "cave_local_extraction_after_extended_fallback",
            "reason": "activity_roi_too_small_after_primary_fallback_extended",
            "decision": "exclude",
            "active_pixels": int(match.group(1)),
            "primary_bbox": record["expanded_bbox"],
            "fallback_bbox": record["fallback_bbox"],
            "extended_fallback_bbox": record["extended_fallback_bbox"],
            "primary_roi_area_ratio": record["roi_area_ratio"],
            "fallback_roi_area_ratio": record["fallback_roi_area_ratio"],
            "extended_fallback_roi_area_ratio": record["extended_fallback_roi_area_ratio"],
            "source_failure_json": str(chosen),
            "source_failure_json_sha256": sha256_file(chosen),
            "error": error,
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    table = pd.DataFrame(rows)
    if len(table) and table["phase_uid"].duplicated().any():
        failures.append("duplicate runtime exclusion phase_uid")
    expected = cfg["expected_png2d_stage1"]
    expected_counts = {
        split: int(expected[key])
        for split, key in (
            ("Train", "runtime_activity_exclusions_train"),
            ("Valid", "runtime_activity_exclusions_valid"),
        )
        if expected.get(key) is not None
    }
    actual_counts = (
        table["split"].value_counts().to_dict() if len(table) else {}
    )
    for split, count in expected_counts.items():
        if int(actual_counts.get(split, 0)) != count:
            failures.append(
                f"{split} runtime exclusions={actual_counts.get(split, 0)} != {count}"
            )
    expected_total = expected.get("runtime_activity_exclusions_total")
    if expected_total is not None and len(table) != int(expected_total):
        failures.append(f"runtime exclusions total={len(table)}")
    if len(successes) + len(table) != len(roi):
        failures.append(
            f"feature closure mismatch: success={len(successes)}, "
            f"excluded={len(table)}, eligible={len(roi)}"
        )

    path = manifests / "runtime_feature_exclusions.csv"
    atomic_csv(table, path)
    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "eligible_phases": len(roi),
        "successful_phases": len(successes),
        "runtime_exclusions": len(table),
        "split_counts": actual_counts,
        "reason": "activity_roi_too_small_after_primary_fallback_extended",
        "runtime_feature_exclusions": str(path),
        "runtime_feature_exclusions_sha256": sha256_file(path),
        "success_and_exclusion_overlap": 0,
        "unknown_missing_phases": max(len(roi) - len(successes) - len(table), 0),
    }
    atomic_json(summary, reports / "08_runtime_exclusion_finalization.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

