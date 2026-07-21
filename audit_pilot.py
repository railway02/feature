#!/usr/bin/env python3
"""Hard engineering audit for the 40-series v3 Train pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOTAL_FORMAL_PAIRS = 54404
PAIR_MAP_KEYS = {
    "residual_u_norm", "residual_v_norm", "residual_mag_norm", "fb_relative",
    "uncertainty_log", "soft_weight", "filling_front", "persistent",
    "washout_front", "hard_valid", "pair_order", "frame_index_t", "frame_index_t1",
}
PHASE_FILES = {
    "selected_frames.csv", "pair_features.csv.gz", "frame_kinetics.csv.gz",
    "temporal_curves.csv.gz", "pair_maps.npz", "masks_and_kinetics.npz",
    "phase_summary.json", "metadata.json", ".SUCCESS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairdata-root", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-projected-cache-gib", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    pairdata = Path(args.pairdata_root).resolve()
    feature_dir = Path(args.feature_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    success = output / ".PILOT_ENGINEERING_SUCCESS"
    audit_json = output / "pilot_engineering_audit.json"
    audit_md = output / "pilot_engineering_audit.md"
    if any(path.exists() for path in (success, audit_json, audit_md)) and not args.overwrite:
        raise FileExistsError("Pilot audit outputs already exist")
    for path in (manifest_path, pairdata / ".SUCCESS", pairdata / "run_summary.json", feature_dir / ".FEATURES_SUCCESS", feature_dir / "audit.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "series_uid": str})
    expected_phases: list[tuple[str, str, str, int]] = []
    expected_pairs = 0
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if as_bool(row[f"can_run_{phase}"]):
                count = int(row[f"n_{phase}_contiguous_pairs"])
                expected_phases.append((str(row["patient_id"]), str(row["series_uid"]), phase, count))
                expected_pairs += count

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    total_cache_bytes = 0
    total_observed_pairs = 0
    for patient_id, series_uid, phase, expected_count in expected_phases:
        root = pairdata / patient_id / series_uid / phase
        missing = sorted(PHASE_FILES - {path.name for path in root.iterdir()}) if root.is_dir() else sorted(PHASE_FILES)
        if missing:
            failures.append(f"{patient_id}/{series_uid}/{phase}: missing {missing}")
            continue
        pair = pd.read_csv(root / "pair_features.csv.gz")
        if len(pair) != expected_count:
            failures.append(f"{patient_id}/{series_uid}/{phase}: pairs expected={expected_count} actual={len(pair)}")
        numeric = pair.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if np.isinf(numeric).any():
            failures.append(f"{patient_id}/{series_uid}/{phase}: infinity in pair table")
        with np.load(root / "pair_maps.npz", allow_pickle=False) as maps:
            missing_keys = sorted(PAIR_MAP_KEYS - set(maps.files))
            if missing_keys:
                failures.append(f"{patient_id}/{series_uid}/{phase}: pair-map keys missing {missing_keys}")
            for key in PAIR_MAP_KEYS:
                if key not in maps.files:
                    continue
                if key in {"pair_order", "frame_index_t", "frame_index_t1"}:
                    if maps[key].shape != (len(pair),):
                        failures.append(f"{patient_id}/{series_uid}/{phase}: {key} shape={maps[key].shape}")
                elif maps[key].shape[0] != len(pair) or maps[key].ndim != 3:
                    failures.append(f"{patient_id}/{series_uid}/{phase}: {key} shape={maps[key].shape}")
        summary = json.loads((root / "phase_summary.json").read_text(encoding="utf-8"))
        qc = summary.get("qc_features", {})
        polarity = summary.get("polarity", {})
        activity = summary.get("activity_qc", {})
        row = {
            "patient_id": patient_id, "series_uid": series_uid, "phase": phase,
            "pairs": len(pair),
            "hard_valid_ratio_median": float(pd.to_numeric(pair.get("hard_valid_ratio_fov"), errors="coerce").median()),
            "fb_relative_median": float(pd.to_numeric(pair.get("fb_relative_mean"), errors="coerce").median()),
            "uncertainty_log_median": float(pd.to_numeric(pair.get("uncertainty_log_mean"), errors="coerce").median()),
            "polarity_ambiguous": int(bool(polarity.get("polarity_ambiguous", False))),
            "vessel_fallback": int(bool(activity.get("vessel_fallback_to_active", False))),
            "active_ratio_fov": activity.get("active_ratio_fov"),
            "vessel_ratio_fov": activity.get("vessel_ratio_fov"),
            "visualization_dir": str(root / "visualizations"),
        }
        rows.append(row)
        total_observed_pairs += len(pair)
        total_cache_bytes += (root / "pair_maps.npz").stat().st_size + (root / "masks_and_kinetics.npz").stat().st_size

    phase_audit = pd.DataFrame(rows)
    if not phase_audit.empty:
        phase_audit["review_priority"] = (
            100 * phase_audit["polarity_ambiguous"]
            + 50 * phase_audit["vessel_fallback"]
            + 10 * (1.0 - phase_audit["hard_valid_ratio_median"].fillna(0).clip(0, 1))
        )
        phase_audit = phase_audit.sort_values(
            ["review_priority", "patient_id", "series_uid", "phase"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
    if total_observed_pairs != expected_pairs:
        failures.append(f"root pairs expected={expected_pairs} actual={total_observed_pairs}")
    run = json.loads((pairdata / "run_summary.json").read_text(encoding="utf-8"))
    if int(run.get("processed_pairs", -1)) != expected_pairs:
        failures.append("run_summary processed_pairs mismatch")
    if not run.get("cuda_actually_used") or run.get("cpu_fallback"):
        failures.append("CUDA hard assertion failed")
    if run.get("labels_read") or run.get("model_trained") or run.get("manifest_rescanned"):
        failures.append("forbidden label/training/rescan flag")
    features = json.loads((feature_dir / "audit.json").read_text(encoding="utf-8"))
    if int(features.get("actual", {}).get("pairs", -1)) != expected_pairs:
        failures.append("feature audit pair count mismatch")
    cache_bytes_per_pair = total_cache_bytes / max(total_observed_pairs, 1)
    projected_gib = cache_bytes_per_pair * TOTAL_FORMAL_PAIRS / (1024 ** 3)
    if projected_gib > args.max_projected_cache_gib:
        failures.append(
            f"projected pair-map storage {projected_gib:.2f} GiB exceeds {args.max_projected_cache_gib:.2f} GiB"
        )
    if not phase_audit.empty and not (phase_audit["hard_valid_ratio_median"].fillna(0) > 0).any():
        failures.append("all pilot phases have zero reliable-flow ratio")

    summary = {
        "version": "api_fullseq_v3_pilot_engineering_audit_v1",
        "created_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "series": int(len(manifest)),
        "patients": int(manifest["patient_id"].nunique()),
        "phases": len(expected_phases),
        "pairs": expected_pairs,
        "cache_bytes": total_cache_bytes,
        "cache_bytes_per_pair": cache_bytes_per_pair,
        "projected_all_train_valid_cache_gib": projected_gib,
        "polarity_ambiguous_phases": int(phase_audit["polarity_ambiguous"].sum()) if not phase_audit.empty else 0,
        "vessel_fallback_phases": int(phase_audit["vessel_fallback"].sum()) if not phase_audit.empty else 0,
        "median_hard_valid_ratio": float(phase_audit["hard_valid_ratio_median"].median()) if not phase_audit.empty else None,
        "failures": failures,
        "engineering_pass": not failures,
        "manual_visual_review_required": True,
        "manual_review_instruction": "Inspect representative raw-frame/flow/mask visualizations before Full Train.",
    }
    phase_audit.to_csv(output / "pilot_phase_qc.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_json_atomic(audit_json, summary)
    lines = [
        "# api_fullseq_v3 Pilot engineering audit", "",
        f"- Engineering pass: **{summary['engineering_pass']}**",
        f"- Series / patients / phases / pairs: {summary['series']} / {summary['patients']} / {summary['phases']} / {summary['pairs']}",
        f"- Projected all-data cache: {projected_gib:.2f} GiB",
        f"- Median hard-valid ratio: {summary['median_hard_valid_ratio']}",
        f"- Polarity ambiguous phases: {summary['polarity_ambiguous_phases']}",
        f"- Vessel fallback phases: {summary['vessel_fallback_phases']}", "",
        "## Failures", "",
    ]
    lines.extend([f"- {item}" for item in failures] if failures else ["- None"])
    lines.extend([
        "", "## Manual gate", "",
        "Engineering PASS is not the scientific visual gate. Inspect representative raw frames, vessel/activity masks, and flow overlays before Full Train.",
    ])
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise AssertionError("Pilot engineering audit failed; see " + str(audit_json))
    write_json_atomic(success, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
