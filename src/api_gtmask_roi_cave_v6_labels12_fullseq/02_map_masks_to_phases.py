#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import atomic_csv, atomic_json, load_config, parse_pipe, sha256_file
from image_match import match_reference
from nifti_io import load_reference_planes, parse_orientations


PRIORITY = {
    "manual": 1000,
    "upstream": 900,
    "path_exact_with_reference": 850,
    "path_exact": 800,
    "reference_match": 700,
    "unique_series_for_phase": 600,
    "unique_phase_for_series": 550,
}


def read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path).expanduser()
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def normalize_path(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return str(Path(value).expanduser().absolute())


def manual_lookup(frame: pd.DataFrame) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_path: dict[str, dict[str, str]] = {}
    by_sha: dict[str, dict[str, str]] = {}
    if frame.empty:
        return by_path, by_sha
    for row in frame.to_dict("records"):
        if str(row.get("enabled", "1")).strip().casefold() in {"0", "false", "no"}:
            continue
        path = normalize_path(str(row.get("mask_path", "")).strip())
        sha = str(row.get("mask_sha256", "")).strip()
        if path:
            by_path[path] = row
        if sha:
            by_sha[sha] = row
    return by_path, by_sha


def upstream_lookup(frame: pd.DataFrame) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_path: dict[str, dict[str, str]] = {}
    by_sha: dict[str, dict[str, str]] = {}
    if frame.empty:
        return by_path, by_sha
    for row in frame.to_dict("records"):
        path = str(row.get("mask_path", row.get("segmentation_path", ""))).strip()
        sha = str(row.get("mask_sha256", row.get("segmentation_sha256", ""))).strip()
        if path:
            by_path[normalize_path(path)] = row
        if sha:
            by_sha[sha] = row
    return by_path, by_sha


def candidate_rows(patient: pd.DataFrame, phase_hint: str, series_hint: str) -> list[dict[str, Any]]:
    selected = patient
    if series_hint:
        restricted = selected[selected["series_uid"].astype(str) == series_hint]
        if not restricted.empty:
            selected = restricted
    if phase_hint in {"pre", "post"}:
        restricted = selected[selected["phase"].astype(str) == phase_hint]
        if not restricted.empty:
            selected = restricted
    rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        rows.append({
            "phase_uid": str(row["phase_uid"]),
            "series_uid": str(row["series_uid"]),
            "phase": str(row["phase"]),
            "frame_paths": parse_pipe(row["frame_paths"]),
        })
    return rows


def resolve_override_phase(row: dict[str, str], index: pd.DataFrame) -> tuple[str, dict[str, str] | None]:
    """Resolve a manual/upstream override row to a source phase.

    Try the row's own ``phase_uid`` first, then ``series_uid::phase``. The
    upstream authoritative manifest stores an opaque hash in ``phase_uid``
    which never matches the ``series_uid::phase`` keys of this pipeline, so
    the constructed fallback is what actually resolves those rows.
    """
    candidates: list[str] = []
    phase_uid = str(row.get("phase_uid", "")).strip()
    if phase_uid:
        candidates.append(phase_uid)
    series_uid = str(row.get("series_uid", "")).strip()
    phase = str(row.get("phase", "")).strip().casefold()
    if series_uid and phase in {"pre", "post"}:
        constructed = f"{series_uid}::{phase}"
        if constructed not in candidates:
            candidates.append(constructed)
    for uid in candidates:
        phase_row = exact_phase(index, uid)
        if phase_row is not None:
            return uid, phase_row
    return candidates[0] if candidates else "", None


def base_result(item: dict[str, str]) -> dict[str, Any]:
    return {
        **item,
        "phase_uid": "",
        "series_uid": "",
        "phase": "",
        "orientation_transform": "identity",
        "reference_plane": "",
        "matched_frame_path": "",
        "matched_frame_index": "",
        "match_score": 0.0,
        "runner_up_score": 0.0,
        "score_margin": 0.0,
        "mapping_method": "",
        "mapping_priority": 0,
        "mapping_status": "unresolved",
        "mapping_reason": "",
    }


def exact_phase(index: pd.DataFrame, phase_uid: str) -> dict[str, str] | None:
    rows = index[index["phase_uid"].astype(str) == phase_uid]
    if len(rows) != 1:
        return None
    return rows.iloc[0].to_dict()


def accepted_result(result: dict[str, Any], phase_row: dict[str, str], method: str, priority: int, reason: str, **extra: Any) -> dict[str, Any]:
    result.update({
        "phase_uid": str(phase_row["phase_uid"]),
        "series_uid": str(phase_row["series_uid"]),
        "phase": str(phase_row["phase"]),
        "mapping_method": method,
        "mapping_priority": priority,
        "mapping_status": "accepted",
        "mapping_reason": reason,
        **extra,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-reference-masks", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    phase_index = pd.read_csv(manifests / "source_phase_index_all.csv", dtype=str, keep_default_na=False)
    inventory = pd.read_csv(manifests / "mask_inventory.csv", dtype=str, keep_default_na=False)
    by_patient = {pid: group.copy() for pid, group in phase_index.groupby("patient_id", sort=False)}

    mapping_cfg = cfg.get("mapping", {})
    transforms = parse_orientations(mapping_cfg.get("orientation_transforms"))
    threshold = float(mapping_cfg.get("reference_accept_score", 0.82))
    margin_threshold = float(mapping_cfg.get("reference_accept_margin", 0.005))
    downsample = int(mapping_cfg.get("match_downsample", 128))
    manual_frame = read_optional_csv(mapping_cfg.get("manual_mapping_csv"))
    manual_by_path, manual_by_sha = manual_lookup(manual_frame)
    upstream_path = mapping_cfg.get("upstream_roi_manifest")
    upstream_frame = read_optional_csv(upstream_path)
    upstream_by_path, upstream_by_sha = upstream_lookup(upstream_frame)

    frame_cache: dict[tuple[str, int], tuple[Any, Any]] = {}
    results: list[dict[str, Any]] = []
    candidate_edges: list[dict[str, Any]] = []
    reference_count = 0
    active_patient = None

    for item in inventory.to_dict("records"):
        patient_key = str(item["patient_id"])
        if patient_key != active_patient:
            # A global 128x128 intensity+gradient cache for all ~57k frames can consume gigabytes.
            # Mapping candidates never cross patients, so a per-patient cache is sufficient.
            frame_cache.clear()
            active_patient = patient_key
        result = base_result(item)
        patient = by_patient.get(patient_key)
        if patient is None or patient.empty:
            result["mapping_reason"] = "patient_not_in_source_phase_index"
            results.append(result)
            continue

        mask_path_key = normalize_path(str(item["mask_path"]))
        mask_sha = str(item["mask_sha256"])

        override = manual_by_path.get(mask_path_key) or manual_by_sha.get(mask_sha)
        if override:
            uid, phase_row = resolve_override_phase(override, phase_index)
            if phase_row is None:
                result["mapping_status"] = "invalid_manual"
                result["mapping_reason"] = f"manual_phase_uid_not_found:{uid}"
            else:
                result = accepted_result(
                    result, phase_row, "manual", PRIORITY["manual"], "manual_mapping",
                    orientation_transform=str(override.get("orientation_transform", "identity") or "identity"),
                    match_score=1.0,
                )
            results.append(result)
            continue

        upstream = upstream_by_path.get(mask_path_key) or upstream_by_sha.get(mask_sha)
        if upstream:
            uid, phase_row = resolve_override_phase(upstream, phase_index)
            if phase_row is not None:
                result = accepted_result(
                    result, phase_row, "upstream", PRIORITY["upstream"], "existing_authoritative_mapping",
                    orientation_transform=str(upstream.get("orientation_transform", "identity") or "identity"),
                    match_score=1.0,
                )
                results.append(result)
                continue

        phase_hint = str(item.get("phase_hint", "")).casefold()
        series_hint = str(item.get("series_hint_uid", "")).strip()
        reference = str(item.get("reference_image_path", "")).strip()

        exact = None
        if series_hint and phase_hint in {"pre", "post"}:
            exact = exact_phase(phase_index, f"{series_hint}::{phase_hint}")

        if reference and (args.max_reference_masks is None or reference_count < args.max_reference_masks):
            reference_count += 1
            candidates = candidate_rows(patient, phase_hint, series_hint)
            if not candidates:
                result["mapping_reason"] = "no_reference_candidates_after_hints"
                results.append(result)
                continue
            try:
                planes = load_reference_planes(reference)
                matches = match_reference(planes, candidates, transforms, downsample=downsample, frame_cache=frame_cache)
            except Exception as exc:
                result["mapping_status"] = "reference_error"
                result["mapping_reason"] = f"{type(exc).__name__}:{exc}"
                results.append(result)
                continue
            if not matches:
                result["mapping_reason"] = "no_reference_match_results"
                results.append(result)
                continue

            # 每个 phase 只保留最佳匹配，便于审计所有候选边。
            best_by_phase: dict[str, Any] = {}
            for match in matches:
                uid = f"{match.series_uid}::{match.phase}"
                if uid not in best_by_phase:
                    best_by_phase[uid] = match
            ranked = sorted(best_by_phase.values(), key=lambda m: (-m.score, m.series_uid, m.phase))
            for rank, match in enumerate(ranked[:20], start=1):
                candidate_edges.append({
                    "mask_path": item["mask_path"], "mask_sha256": mask_sha,
                    "patient_id": item["patient_id"], "candidate_rank": rank,
                    "phase_uid": f"{match.series_uid}::{match.phase}",
                    "series_uid": match.series_uid, "phase": match.phase,
                    "score": match.score, "intensity_score": match.intensity_score,
                    "gradient_score": match.gradient_score, "orientation_transform": match.transform,
                    "reference_plane": match.plane_name, "matched_frame_path": match.frame_path,
                    "matched_frame_index": match.frame_index,
                })
            top = ranked[0]
            runner = ranked[1] if len(ranked) > 1 else None
            runner_score = float(runner.score) if runner else 0.0
            margin = float(top.score) - runner_score
            one_candidate = len(ranked) == 1
            accepted = float(top.score) >= threshold and (one_candidate or margin >= margin_threshold)
            phase_row = exact_phase(phase_index, f"{top.series_uid}::{top.phase}")
            if not accepted and exact is not None:
                # 精确 series 路径 + 已声明 phase 的证据强于参考帧分数；
                # 分数不达标时退回纯路径映射，同时保留参考帧分数供审计。
                result = accepted_result(
                    result, exact, "path_exact", PRIORITY["path_exact"],
                    "mask_path_inside_exact_series_and_phase_declared;reference_below_threshold",
                    orientation_transform=top.transform, reference_plane=top.plane_name,
                    matched_frame_path=top.frame_path, matched_frame_index=top.frame_index,
                    match_score=float(top.score), runner_up_score=runner_score, score_margin=margin,
                )
                results.append(result)
                continue
            method = "path_exact_with_reference" if exact is not None else "reference_match"
            priority = PRIORITY[method]
            result.update({
                "phase_uid": f"{top.series_uid}::{top.phase}",
                "series_uid": top.series_uid,
                "phase": top.phase,
                "orientation_transform": top.transform,
                "reference_plane": top.plane_name,
                "matched_frame_path": top.frame_path,
                "matched_frame_index": top.frame_index,
                "match_score": float(top.score),
                "runner_up_score": runner_score,
                "score_margin": margin,
                "mapping_method": method,
                "mapping_priority": priority,
                "mapping_status": "accepted" if accepted and phase_row is not None else "needs_review",
                "mapping_reason": "reference_score_pass" if accepted else "reference_score_or_margin_below_threshold",
            })
            results.append(result)
            continue

        if exact is not None:
            result = accepted_result(
                result, exact, "path_exact", PRIORITY["path_exact"],
                "mask_path_inside_exact_series_and_phase_declared", match_score=1.0,
            )
            results.append(result)
            continue

        if phase_hint in {"pre", "post"}:
            candidates = patient[patient["phase"].astype(str) == phase_hint]
            if len(candidates) == 1:
                result = accepted_result(
                    result, candidates.iloc[0].to_dict(), "unique_series_for_phase",
                    PRIORITY["unique_series_for_phase"], "patient_has_one_series_for_declared_phase", match_score=0.75,
                )
                results.append(result)
                continue

        if series_hint:
            candidates = patient[patient["series_uid"].astype(str) == series_hint]
            if len(candidates) == 1:
                result = accepted_result(
                    result, candidates.iloc[0].to_dict(), "unique_phase_for_series",
                    PRIORITY["unique_phase_for_series"], "series_has_one_runnable_phase", match_score=0.70,
                )
                results.append(result)
                continue

        result["mapping_reason"] = "ambiguous_mask_without_sufficient_path_or_reference_information"
        results.append(result)

    mapped = pd.DataFrame(results)
    atomic_csv(mapped, manifests / "mask_mapping_attempts.csv")
    atomic_csv(pd.DataFrame(candidate_edges), manifests / "mask_reference_candidate_edges.csv")

    accepted_attempts = mapped[mapped["mapping_status"] == "accepted"].copy()
    primary_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    for phase_uid, group in accepted_attempts.groupby("phase_uid", sort=False):
        # 相同内容的多个文件副本只算一个候选。
        dedup = group.sort_values(["mapping_priority", "match_score", "mask_path"], ascending=[False, False, True])
        dedup = dedup.drop_duplicates("mask_sha256", keep="first")
        top = dedup.iloc[0]
        tied = dedup[
            (pd.to_numeric(dedup["mapping_priority"]) == float(top["mapping_priority"]))
            & ((pd.to_numeric(dedup["match_score"]) - float(top["match_score"])).abs() < 1e-9)
        ]
        if len(tied) > 1 and tied["mask_sha256"].nunique() > 1:
            for row in dedup.to_dict("records"):
                row["final_mapping_status"] = "conflict_same_priority"
                conflict_rows.append(row)
            continue
        row = top.to_dict()
        row["final_mapping_status"] = "accepted_primary"
        primary_rows.append(row)
        for other in dedup.iloc[1:].to_dict("records"):
            other["final_mapping_status"] = "shadowed"
            shadow_rows.append(other)

    primary = pd.DataFrame(primary_rows)
    primary_by_phase = {str(row["phase_uid"]): row for row in primary.to_dict("records")}
    full_rows: list[dict[str, Any]] = []
    for phase in phase_index.to_dict("records"):
        mapped_row = primary_by_phase.get(str(phase["phase_uid"]))
        if mapped_row:
            full_rows.append({**phase, **mapped_row, "phase_mapping_status": "accepted"})
        else:
            attempts = mapped[mapped["phase_uid"].astype(str) == str(phase["phase_uid"])] if "phase_uid" in mapped else pd.DataFrame()
            reason = "no_mask_candidate"
            status = "missing"
            if not attempts.empty:
                statuses = sorted(set(attempts["mapping_status"].astype(str)))
                reason = "|".join(statuses)
                status = "needs_review"
            if any(str(row.get("phase_uid", "")) == str(phase["phase_uid"]) for row in conflict_rows):
                reason = "conflicting_masks"
                status = "conflict"
            full_rows.append({
                **phase,
                "mask_path": "", "mask_sha256": "", "reference_image_path": "",
                "storage_layout": "", "orientation_transform": "", "mapping_method": "",
                "mapping_priority": "", "match_score": "", "runner_up_score": "", "score_margin": "",
                "phase_mapping_status": status, "mapping_reason": reason,
            })

    full_map = pd.DataFrame(full_rows)
    atomic_csv(primary, manifests / "mask_phase_map_primary.csv")
    atomic_csv(full_map, manifests / "source_phase_with_mask_map.csv")
    atomic_csv(pd.DataFrame(conflict_rows), reports / "02_mask_mapping_conflicts.csv")
    unresolved_attempts = mapped[mapped["mapping_status"] != "accepted"].copy()
    atomic_csv(pd.concat([unresolved_attempts, pd.DataFrame(shadow_rows)], ignore_index=True), reports / "02_mask_mapping_unresolved_or_shadowed.csv")
    gaps = full_map[full_map["phase_mapping_status"] != "accepted"].copy()
    atomic_csv(gaps, reports / "02_source_phase_mapping_gaps.csv")

    summary = {
        "source_phases": int(len(phase_index)),
        "mask_inventory": int(len(inventory)),
        "accepted_source_phases": int((full_map["phase_mapping_status"] == "accepted").sum()),
        "missing_source_phases": int((full_map["phase_mapping_status"] == "missing").sum()),
        "needs_review_source_phases": int((full_map["phase_mapping_status"] == "needs_review").sum()),
        "conflict_source_phases": int((full_map["phase_mapping_status"] == "conflict").sum()),
        "accepted_pre": int(((full_map["phase"] == "pre") & (full_map["phase_mapping_status"] == "accepted")).sum()),
        "accepted_post": int(((full_map["phase"] == "post") & (full_map["phase_mapping_status"] == "accepted")).sum()),
        "method_counts": primary["mapping_method"].value_counts().to_dict() if not primary.empty else {},
        "unused_or_unresolved_mask_paths": int((mapped["mapping_status"] != "accepted").sum()),
        "upstream_manifest": str(upstream_path or ""),
        "upstream_manifest_sha256": sha256_file(upstream_path) if upstream_path and Path(upstream_path).is_file() else "",
    }
    atomic_json(summary, reports / "02_mask_mapping_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
