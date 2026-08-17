#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file, write_success


REQUIRED_EXCLUSION_COLUMNS = {
    "phase_uid", "split", "stage", "reason", "decision",
}


def load_runtime_exclusions(
    path: Path,
    roi: pd.DataFrame,
    splits: tuple[str, ...],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    failures: list[str] = []
    if not path.is_file():
        return {}, failures

    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_EXCLUSION_COLUMNS - set(table.columns))
    if missing:
        return {}, [f"runtime exclusion 缺少列：{missing}"]

    table = table[table["split"].isin(splits)].copy()
    duplicate = sorted(
        table.loc[table["phase_uid"].duplicated(keep=False), "phase_uid"]
        .astype(str).unique().tolist()
    )
    if duplicate:
        failures.append(f"runtime exclusion 存在重复 phase_uid：{duplicate[:20]}")

    expected_split = dict(zip(
        roi["phase_uid"].astype(str),
        roi["split"].astype(str),
    ))
    records: dict[str, dict[str, str]] = {}
    for row in table.to_dict("records"):
        uid = str(row["phase_uid"])
        split = str(row["split"])
        reason = str(row["reason"]).strip()
        decision = str(row["decision"]).strip().casefold()

        if uid not in expected_split:
            failures.append(f"runtime exclusion 不属于 eligible ROI manifest：{uid}")
            continue
        if expected_split[uid] != split:
            failures.append(
                f"runtime exclusion split 错误：{uid}:{split}!={expected_split[uid]}"
            )
        if decision != "exclude":
            failures.append(f"runtime exclusion decision 不是 exclude：{uid}:{decision}")
        if not reason:
            failures.append(f"runtime exclusion 缺少 reason：{uid}")
        records[uid] = {key: str(value) for key, value in row.items()}
    return records, failures


def phase_dir(feature_root: Path, row: dict[str, Any]) -> Path:
    return (
        feature_root
        / str(row["split"]).casefold()
        / str(row["patient_id"])
        / str(row["series_uid"])
        / str(row["phase"])
    )


def artifact_state(directory: Path) -> tuple[bool, str, str]:
    success_path = directory / ".SUCCESS.json"
    metadata_path = directory / "metadata.json"
    embedding_path = directory / "embedding_5120.npy"
    missing = [
        path.name
        for path in (success_path, metadata_path, embedding_path)
        if not path.is_file()
    ]
    if missing:
        return False, "", "missing:" + ",".join(missing)
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "", f"unreadable_success:{type(exc).__name__}:{exc}"

    actual_hash = sha256_file(embedding_path)
    declared_hash = str(success.get("embedding_sha256", ""))
    if declared_hash and declared_hash != actual_hash:
        return False, actual_hash, "embedding_sha256_mismatch"
    return True, actual_hash, ""


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_phase_atomically(
    source: Path,
    destination: Path,
    quarantine_root: Path,
) -> None:
    if destination.exists():
        valid, _, _ = artifact_state(destination)
        if valid:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_root.mkdir(parents=True, exist_ok=True)
        name = "__".join(destination.parts[-4:])
        shutil.move(
            str(destination),
            str(quarantine_root / f"{name}__{stamp}"),
        )

    temp = destination.with_name(
        destination.name
        + ".consolidating."
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target = temp / relative
        if source_path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            hardlink_or_copy(source_path, target)

    valid, _, reason = artifact_state(temp)
    if not valid:
        shutil.rmtree(temp, ignore_errors=True)
        raise RuntimeError(f"分片 phase 输出不完整：{source}:{reason}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, destination)


def load_candidate_payload(directory: Path) -> dict[str, Any]:
    success_path = directory / ".SUCCESS.json"
    metadata_path = directory / "metadata.json"
    embedding_path = directory / "embedding_5120.npy"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedding = np.load(embedding_path, allow_pickle=False)
    roi_meta = metadata.get("roi", {})
    return {
        "directory": directory,
        "success_path": success_path,
        "success_mtime": float(success_path.stat().st_mtime),
        "success": success,
        "metadata": metadata,
        "embedding": embedding,
        "embedding_hash": sha256_file(embedding_path),
        "provenance": {
            "phase_uid": str(success.get("phase_uid", "")),
            "mask_sha256": str(success.get("mask_sha256", "")),
            "temporal_policy": str(success.get("temporal_policy", "")),
            "used_bbox": str(success.get("used_bbox", "")),
            "fallback_used": str(success.get("fallback_used", "")),
            "feature_schema_sha256": str(
                success.get("feature_schema_sha256", "")
            ),
            "frame_list_hash": str(metadata.get("frame_list_hash", "")),
            "roi_phase_uid": str(roi_meta.get("phase_uid", "")),
            "roi_mask_sha256": str(roi_meta.get("mask_sha256", "")),
            "roi_used_bbox": str(roi_meta.get("used_bbox", "")),
        },
    }


def compare_duplicate_candidates(
    phase_uid: str,
    chosen: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = np.asarray(chosen["embedding"])
    right = np.asarray(candidate["embedding"])
    provenance_equal = chosen["provenance"] == candidate["provenance"]
    shape_equal = left.shape == right.shape
    dtype_equal = left.dtype == right.dtype
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    exact_hash = chosen["embedding_hash"] == candidate["embedding_hash"]

    max_abs = float("nan")
    mean_abs = float("nan")
    rmse = float("nan")
    relative_l2 = float("nan")
    cosine = float("nan")
    numerical_equivalent = False

    if provenance_equal and shape_equal and dtype_equal and finite:
        if exact_hash:
            max_abs = mean_abs = rmse = relative_l2 = 0.0
            cosine = 1.0
            numerical_equivalent = True
        else:
            left64 = left.astype(np.float64, copy=False)
            right64 = right.astype(np.float64, copy=False)
            difference = right64 - left64
            absolute = np.abs(difference)
            max_abs = float(np.max(absolute))
            mean_abs = float(np.mean(absolute))
            rmse = float(np.sqrt(np.mean(difference ** 2)))
            relative_l2 = float(
                np.linalg.norm(difference)
                / max(float(np.linalg.norm(left64)), 1e-12)
            )
            flat_left = left64.ravel()
            flat_right = right64.ravel()
            cosine = float(
                np.dot(flat_left, flat_right)
                / max(
                    float(np.linalg.norm(flat_left) * np.linalg.norm(flat_right)),
                    1e-12,
                )
            )
            # CUDA inference is not guaranteed to be bitwise deterministic.
            # Accept only tiny numeric drift with identical scientific provenance.
            numerical_equivalent = bool(
                relative_l2 <= 5e-5
                and cosine >= 0.99999999
                and max_abs <= 5e-4
            )

    accepted = bool(
        provenance_equal
        and shape_equal
        and dtype_equal
        and finite
        and numerical_equivalent
    )
    return {
        "phase_uid": phase_uid,
        "chosen_directory": str(chosen["directory"]),
        "candidate_directory": str(candidate["directory"]),
        "chosen_success_mtime": chosen["success_mtime"],
        "candidate_success_mtime": candidate["success_mtime"],
        "chosen_embedding_sha256": chosen["embedding_hash"],
        "candidate_embedding_sha256": candidate["embedding_hash"],
        "exact_hash": int(exact_hash),
        "provenance_equal": int(provenance_equal),
        "shape_equal": int(shape_equal),
        "dtype_equal": int(dtype_equal),
        "finite": int(finite),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "accepted_as_numerically_equivalent": int(accepted),
        "resolution": (
            "earliest_success_canonical"
            if accepted
            else "unresolved_conflict"
        ),
    }


def scientific_schema(payload: dict[str, Any]) -> dict[str, Any]:
    # These two fields encode run/split provenance and self-hash.  They are not
    # feature definitions.  All remaining fields must match exactly.
    return {
        key: value
        for key, value in payload.items()
        if key not in {"frozen_config_hash", "schema_sha256"}
    }


def replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def consolidate_schemas(
    feature_root: Path,
    schema_candidates_by_split: dict[str, list[Path]],
    splits: tuple[str, ...],
    reports: Path,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    split_details: dict[str, Any] = {}
    science_payloads: dict[str, dict[str, Any]] = {}

    for split in splits:
        candidates = sorted(
            {path.resolve() for path in schema_candidates_by_split.get(split, [])},
            key=lambda value: str(value),
        )
        if not candidates:
            failures.append(f"{split} 缺少 shard feature_schema.json")
            continue

        byte_groups: dict[str, list[Path]] = defaultdict(list)
        payloads: list[tuple[Path, dict[str, Any]]] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payloads.append((path, payload))
                byte_groups[sha256_file(path)].append(path)
            except Exception as exc:
                failures.append(
                    f"读取 {split} schema 失败：{path}:{type(exc).__name__}:{exc}"
                )

        if len(byte_groups) != 1:
            failures.append(
                f"{split} 内部 feature_schema 不一致：{sorted(byte_groups)}"
            )
            continue
        if not payloads:
            continue

        chosen_path, chosen_payload = payloads[0]
        science = scientific_schema(chosen_payload)
        for other_path, other_payload in payloads[1:]:
            if scientific_schema(other_payload) != science:
                failures.append(
                    f"{split} schema 科学字段不一致：{chosen_path} vs {other_path}"
                )

        split_schema_path = feature_root / f"feature_schema_{split.casefold()}.json"
        replace_file(chosen_path, split_schema_path)
        science_payloads[split] = science
        split_details[split] = {
            "source": str(chosen_path),
            "canonical_split_schema": str(split_schema_path),
            "byte_sha256": sha256_file(chosen_path),
            "schema_sha256": str(chosen_payload.get("schema_sha256", "")),
            "frozen_config_hash": str(
                chosen_payload.get("frozen_config_hash", "")
            ),
            "candidate_count": len(candidates),
        }

    if len(science_payloads) >= 2:
        first_split = next(iter(science_payloads))
        first_science = science_payloads[first_split]
        for split, science in science_payloads.items():
            if science != first_science:
                failures.append(
                    f"Train/Valid feature_schema 科学字段不一致："
                    f"{first_split} vs {split}"
                )

    canonical_split = "Train" if "Train" in split_details else (
        next(iter(split_details)) if split_details else ""
    )
    canonical_path = feature_root / "feature_schema.json"
    if canonical_split:
        replace_file(
            Path(split_details[canonical_split]["canonical_split_schema"]),
            canonical_path,
        )

    audit = {
        "canonical_split": canonical_split,
        "canonical_schema": str(canonical_path),
        "scientific_fields_equal_across_splits": not any(
            "科学字段不一致" in failure for failure in failures
        ),
        "ignored_cross_split_fields": [
            "frozen_config_hash",
            "schema_sha256",
        ],
        "splits": split_details,
    }
    atomic_json(audit, reports / "08_feature_schema_audit.json")
    return audit, failures


def consolidate_from_shards(
    cfg: dict[str, Any],
    roi: pd.DataFrame,
    splits: tuple[str, ...],
    exclusions: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    outputs = Path(cfg["paths"]["outputs"])
    reports = Path(cfg["paths"]["reports"])
    feature_root = outputs / "cave_local_eligible_featurebank"
    feature_root.mkdir(parents=True, exist_ok=True)
    quarantine_root = reports / "08_consolidation_quarantine"

    failures: list[str] = []
    candidates: dict[str, list[Path]] = defaultdict(list)
    schema_candidates_by_split: dict[str, list[Path]] = defaultdict(list)

    for split in splits:
        shard_root = outputs / f"shards_{split.casefold()}_full"
        if not shard_root.is_dir():
            continue
        schema_candidates_by_split[split].extend(
            shard_root.rglob("feature_schema.json")
        )
        for success_path in shard_root.rglob(".SUCCESS.json"):
            try:
                payload = json.loads(success_path.read_text(encoding="utf-8"))
                uid = str(payload.get("phase_uid", ""))
                if uid:
                    candidates[uid].append(success_path.parent)
            except Exception as exc:
                failures.append(
                    f"无法读取 shard SUCCESS：{success_path}:"
                    f"{type(exc).__name__}:{exc}"
                )

    expected_rows = {
        str(row["phase_uid"]): row
        for row in roi[roi["split"].isin(splits)].to_dict("records")
    }
    materialized = 0
    already_present = 0
    duplicate_exact_hash = 0
    duplicate_numeric = 0
    duplicate_audit_rows: list[dict[str, Any]] = []

    for uid, row in expected_rows.items():
        destination = phase_dir(feature_root, row)
        if uid in exclusions:
            if (destination / ".SUCCESS.json").is_file():
                failures.append(f"运行期排除 phase 已存在正式特征：{uid}")
            continue

        valid, _, _ = artifact_state(destination)
        if valid:
            already_present += 1
            continue

        source_paths = candidates.get(uid, [])
        if not source_paths:
            continue

        payloads: list[dict[str, Any]] = []
        for source in source_paths:
            source_valid, _, reason = artifact_state(source)
            if not source_valid:
                failures.append(f"shard phase 无效：{uid}:{source}:{reason}")
                continue
            try:
                payloads.append(load_candidate_payload(source))
            except Exception as exc:
                failures.append(
                    f"读取 shard phase 失败：{uid}:{source}:"
                    f"{type(exc).__name__}:{exc}"
                )

        if not payloads:
            continue

        payloads.sort(
            key=lambda item: (item["success_mtime"], str(item["directory"]))
        )
        chosen = payloads[0]
        conflict = False
        for candidate in payloads[1:]:
            audit_row = compare_duplicate_candidates(uid, chosen, candidate)
            duplicate_audit_rows.append(audit_row)
            if audit_row["exact_hash"]:
                duplicate_exact_hash += 1
            elif audit_row["accepted_as_numerically_equivalent"]:
                duplicate_numeric += 1
            else:
                conflict = True
                failures.append(
                    f"重复 shard 无法判定为数值等价：{uid}:"
                    f"relative_l2={audit_row['relative_l2']}:"
                    f"cosine={audit_row['cosine']}:"
                    f"provenance_equal={audit_row['provenance_equal']}"
                )

        if conflict:
            continue

        try:
            copy_phase_atomically(
                Path(chosen["directory"]),
                destination,
                quarantine_root,
            )
            materialized += 1
        except Exception as exc:
            failures.append(
                f"consolidate phase 失败：{uid}:{type(exc).__name__}:{exc}"
            )

    duplicate_audit_path = reports / "08_duplicate_phase_resolution.csv"
    atomic_csv(pd.DataFrame(duplicate_audit_rows), duplicate_audit_path)

    schema_audit, schema_failures = consolidate_schemas(
        feature_root,
        schema_candidates_by_split,
        splits,
        reports,
    )
    failures.extend(schema_failures)

    return {
        "candidate_phase_uids": len(candidates),
        "materialized_phase_uids": materialized,
        "already_present_phase_uids": already_present,
        "duplicate_exact_hash_pairs": duplicate_exact_hash,
        "duplicate_numerically_equivalent_pairs": duplicate_numeric,
        "duplicate_resolution_audit": str(duplicate_audit_path),
        "schema_audit": schema_audit,
        "split_schema_self_sha256": {
            split: details.get("schema_sha256", "")
            for split, details in schema_audit.get("splits", {}).items()
        },
        "feature_root": str(feature_root),
    }, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid", "All"], default="All")
    parser.add_argument("--no-consolidate-shards", action="store_true")
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
    splits = ("Train", "Valid") if args.split == "All" else (args.split,)

    failures: list[str] = []
    exclusion_path = manifests / "runtime_feature_exclusions.csv"
    exclusions, exclusion_failures = load_runtime_exclusions(
        exclusion_path, roi, splits
    )
    failures.extend(exclusion_failures)

    consolidation = {"disabled": True}
    if not args.no_consolidate_shards:
        consolidation, consolidation_failures = consolidate_from_shards(
            cfg, roi, splits, exclusions
        )
        failures.extend(consolidation_failures)

    index_rows: list[dict[str, object]] = []
    expected_uids: set[str] = set()
    successful_uids: set[str] = set()
    excluded_uids: set[str] = set()

    for split in splits:
        expected = roi[roi["split"] == split].copy()
        for row in expected.to_dict("records"):
            uid = str(row["phase_uid"])
            expected_uids.add(uid)
            directory = phase_dir(feature_root, row)
            success_path = directory / ".SUCCESS.json"
            metadata_path = directory / "metadata.json"
            embedding_path = directory / "embedding_5120.npy"

            status = "ok"
            reason = ""
            success: dict[str, Any] = {}
            metadata: dict[str, Any] = {}

            if uid in exclusions:
                status = "excluded"
                reason = exclusions[uid]["reason"]
                excluded_uids.add(uid)
                if success_path.is_file():
                    failures.append(f"排除 phase 与正式 SUCCESS 重叠：{uid}")
                    status = "invalid"
                    reason = "excluded_success_overlap"
            elif not success_path.is_file():
                status = "missing"
                reason = "missing_success"
                failures.append(f"缺少 phase SUCCESS：{uid}")
            else:
                try:
                    success = json.loads(success_path.read_text(encoding="utf-8"))
                    metadata = (
                        json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.is_file() else {}
                    )
                    if str(success.get("phase_uid", "")) != uid:
                        status, reason = "invalid", "phase_uid_mismatch"
                        failures.append(f"SUCCESS phase_uid 错误：{uid}")
                    elif success.get("temporal_policy") != cfg.get(
                        "temporal", {}
                    ).get("policy", "freeze_whole_indices"):
                        status, reason = "invalid", "temporal_policy_mismatch"
                        failures.append(f"时间策略错误：{uid}")
                    elif (
                        str(success.get("feature_schema_sha256", ""))
                        != str(
                            consolidation.get(
                                "split_schema_self_sha256", {}
                            ).get(split, "")
                        )
                    ):
                        status, reason = "invalid", "feature_schema_mismatch"
                        failures.append(f"feature schema 错误：{uid}")
                    elif str(success.get("mask_sha256", "")) != str(
                        row["mask_sha256"]
                    ):
                        status, reason = "invalid", "mask_sha256_mismatch"
                        failures.append(f"Mask hash 错误：{uid}")
                    elif not metadata_path.is_file():
                        status, reason = "invalid", "missing_metadata"
                        failures.append(f"缺少 metadata：{uid}")
                    elif not embedding_path.is_file():
                        status, reason = "invalid", "missing_embedding"
                        failures.append(f"缺少 embedding_5120.npy：{uid}")
                    else:
                        actual_hash = sha256_file(embedding_path)
                        declared_hash = str(success.get("embedding_sha256", ""))
                        if declared_hash and declared_hash != actual_hash:
                            status, reason = "invalid", "embedding_sha256_mismatch"
                            failures.append(f"embedding hash 错误：{uid}")
                        roi_meta = metadata.get("roi", {})
                        if str(roi_meta.get("phase_uid", "")) != uid:
                            status, reason = (
                                "invalid", "metadata_roi_phase_uid_mismatch"
                            )
                            failures.append(f"metadata ROI phase_uid 错误：{uid}")
                        elif str(roi_meta.get("mask_sha256", "")) != str(
                            row["mask_sha256"]
                        ):
                            status, reason = (
                                "invalid", "metadata_roi_mask_sha256_mismatch"
                            )
                            failures.append(f"metadata ROI Mask hash 错误：{uid}")
                except Exception as exc:
                    status, reason = "invalid", "read_error"
                    failures.append(
                        f"读取 phase 输出失败：{uid}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                if status == "ok":
                    successful_uids.add(uid)

            index_rows.append({
                "phase_uid": uid,
                "split": split,
                "patient_id": row["patient_id"],
                "series_uid": row["series_uid"],
                "phase": row["phase"],
                "status": status,
                "reason": reason,
                "phase_output_dir": str(directory),
                "success_path": str(success_path),
                "metadata_path": str(metadata_path),
                "embedding_path": str(embedding_path),
                "embedding_sha256": success.get("embedding_sha256", ""),
                "fallback_used": success.get("fallback_used", ""),
                "used_bbox": success.get("used_bbox", ""),
                "runtime_exclusion_stage": exclusions.get(uid, {}).get("stage", ""),
                "runtime_exclusion_active_pixels": exclusions.get(uid, {}).get(
                    "active_pixels", ""
                ),
            })

    actual_uids: set[str] = set()
    unreadable_success: list[str] = []
    if feature_root.is_dir():
        for path in feature_root.rglob(".SUCCESS.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                uid = str(payload.get("phase_uid", ""))
                if uid:
                    actual_uids.add(uid)
                else:
                    unreadable_success.append(str(path))
            except Exception:
                unreadable_success.append(str(path))
    if unreadable_success:
        failures.append(
            f"正式 featurebank 有 {len(unreadable_success)} 个无法读取的 SUCCESS"
        )

    extras = sorted(actual_uids - expected_uids)
    if extras:
        failures.append(
            f"featurebank 存在 {len(extras)} 个额外 phase"
        )

    overlap = sorted(successful_uids & excluded_uids)
    if overlap:
        failures.append(f"success 与 runtime exclusion 重叠：{overlap[:20]}")

    closed_uids = successful_uids | excluded_uids
    unknown_missing = sorted(expected_uids - closed_uids)
    if unknown_missing:
        failures.append(
            f"存在 {len(unknown_missing)} 个既无有效特征又无明确排除的 phase"
        )

    schema = feature_root / "feature_schema.json"
    if not schema.is_file():
        failures.append("缺少 feature_schema.json")

    index = pd.DataFrame(index_rows)
    atomic_csv(index, reports / f"08_featurebank_index_{args.split.casefold()}.csv")

    canonical_dir = outputs / "tables" / "local_eligible"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = manifests / "local_phase_coverage_all.csv"
    phase_availability_path = ""
    exclusion_output_path = ""

    if args.split == "All" and coverage_path.is_file():
        coverage = pd.read_csv(
            coverage_path, dtype=str, keep_default_na=False
        )
        status_by_uid = dict(zip(
            index["phase_uid"].astype(str),
            index["status"].astype(str),
        ))
        reason_by_uid = dict(zip(
            index["phase_uid"].astype(str),
            index["reason"].astype(str),
        ))
        availability = coverage.copy()
        availability["extraction_status"] = [
            status_by_uid.get(uid, "not_eligible")
            for uid in availability["phase_uid"].astype(str)
        ]
        availability["runtime_feature_excluded"] = (
            availability["extraction_status"] == "excluded"
        ).astype(int)
        availability["local_feature_available"] = (
            availability["extraction_status"] == "ok"
        ).astype(int)

        reasons: list[str] = []
        for row in availability.to_dict("records"):
            uid = str(row["phase_uid"])
            status = str(row["extraction_status"])
            if status == "ok":
                reasons.append("")
            elif status in {"excluded", "missing", "invalid"}:
                reasons.append(reason_by_uid.get(uid, f"extraction_{status}"))
            else:
                reasons.append(
                    str(row.get("local_exclusion_reason", "")).strip()
                    or status
                )
        availability["feature_exclusion_reason"] = reasons

        report_phase = reports / "08_local_phase_availability.csv"
        canonical_phase = canonical_dir / "local_phase_availability.csv"
        atomic_csv(availability, report_phase)
        atomic_csv(availability, canonical_phase)
        phase_availability_path = str(canonical_phase)

        feature_exclusion = availability[
            availability["local_feature_available"] == 0
        ].copy()
        report_exclusion = reports / "08_local_feature_exclusion.csv"
        canonical_exclusion = canonical_dir / "local_feature_exclusion.csv"
        atomic_csv(feature_exclusion, report_exclusion)
        atomic_csv(feature_exclusion, canonical_exclusion)
        exclusion_output_path = str(canonical_exclusion)

    split_counts: dict[str, dict[str, int]] = {}
    for split in splits:
        group = index[index["split"] == split]
        split_counts[split] = {
            "expected": int(len(group)),
            "ok": int((group["status"] == "ok").sum()),
            "excluded": int((group["status"] == "excluded").sum()),
            "closed": int(group["status"].isin(["ok", "excluded"]).sum()),
            "missing_or_invalid": int(
                group["status"].isin(["missing", "invalid"]).sum()
            ),
            "pre": int((group["phase"] == "pre").sum()),
            "post": int((group["phase"] == "post").sum()),
        }

    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "feature_root": str(feature_root),
        "feature_schema": str(schema),
        "feature_schema_sha256": sha256_file(schema) if schema.is_file() else "",
        "runtime_feature_exclusions": str(exclusion_path),
        "runtime_feature_exclusions_sha256": (
            sha256_file(exclusion_path) if exclusion_path.is_file() else ""
        ),
        "successful_feature_count": len(successful_uids),
        "runtime_exclusion_count": len(excluded_uids),
        "closed_phase_count": len(closed_uids),
        "eligible_phase_count": len(expected_uids),
        "splits": split_counts,
        "extra_phase_uids": extras[:100],
        "unknown_missing_phase_uids": unknown_missing[:100],
        "phase_availability": phase_availability_path,
        "feature_exclusion": exclusion_output_path,
        "consolidation": consolidation,
        "segmentation_model_used": False,
        "local_frames_saved": False,
    }
    atomic_json(
        summary,
        reports / f"08_featurebank_validation_{args.split.casefold()}.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failures:
        raise AssertionError("; ".join(failures[:25]))
    if args.split == "All":
        write_success(
            feature_root / ".ELIGIBLE_FEATUREBANK_SUCCESS.json",
            "eligible_local_cave_featurebank",
            cfg,
            summary,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
