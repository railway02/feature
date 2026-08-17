from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common import as_bool, hash_lines, parse_pipe_ints, parse_pipe_strings, sha256_file

JPEG_SUFFIXES = {".jpg", ".jpeg"}
PARAMETER_TOKENS = ("CBF", "CBV", "MTT", "TTP")
REQUIRED_COLUMNS = {
    "patient_id", "split", "source_type", "series_uid", "series_id",
    "selected_for_extraction", "candidate_valid", "selected_candidate",
    "can_run_pre", "can_run_post", "pre_frame_indices", "post_frame_indices",
    "pre_frame_paths", "post_frame_paths", "pre_frame_list_hash", "post_frame_list_hash",
}


@dataclass(frozen=True)
class PhasePlan:
    patient_id: str
    split: str
    source_type: str
    series_uid: str
    series_id: str
    phase: str
    frame_paths: tuple[str, ...]
    frame_indices: tuple[int, ...]
    frame_list_hash: str
    manifest_sha256: str
    raw_row: dict[str, Any]


@dataclass(frozen=True)
class ManifestBundle:
    frame: pd.DataFrame
    plans: tuple[PhasePlan, ...]
    summary: dict[str, Any]


def contiguous_blocks(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    blocks = [[indices[0]]]
    for value in indices[1:]:
        if value == blocks[-1][-1] + 1:
            blocks[-1].append(value)
        else:
            blocks.append([value])
    return blocks


def expected_pair_count(indices: list[int]) -> int:
    return sum(max(len(block) - 1, 0) for block in contiguous_blocks(indices))


def _expected_int(row: pd.Series, key: str) -> int | None:
    if key not in row.index or pd.isna(row[key]):
        return None
    return int(pd.to_numeric(row[key], errors="raise"))


def _validate_phase(row: pd.Series, phase: str, verify_files: bool, manifest_hash: str) -> PhasePlan | None:
    can_run = as_bool(row[f"can_run_{phase}"])
    paths = parse_pipe_strings(row[f"{phase}_frame_paths"])
    indices = parse_pipe_ints(row[f"{phase}_frame_indices"])
    stored_hash = "" if pd.isna(row[f"{phase}_frame_list_hash"]) else str(row[f"{phase}_frame_list_hash"])
    expected_frames = _expected_int(row, f"n_{phase}_frames")
    expected_pairs = _expected_int(row, f"n_{phase}_contiguous_pairs")

    if not can_run:
        if paths or indices or (expected_frames not in {None, 0}) or (expected_pairs not in {None, 0}):
            raise AssertionError(f"{row.series_uid} {phase}: non-runnable phase contains frames")
        return None
    if len(paths) != len(indices) or len(paths) < 1:
        raise AssertionError(f"{row.series_uid} {phase}: invalid path/index lengths")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise AssertionError(f"{row.series_uid} {phase}: frame indices not sorted/unique")
    if expected_frames is not None and expected_frames != len(indices):
        raise AssertionError(f"{row.series_uid} {phase}: n_frames mismatch")
    if expected_pairs is not None and expected_pairs != expected_pair_count(indices):
        raise AssertionError(f"{row.series_uid} {phase}: contiguous-pair count mismatch")
    if hash_lines(paths) != stored_hash:
        raise AssertionError(f"{row.series_uid} {phase}: frame-list hash mismatch")
    for item in paths:
        path = Path(item)
        if path.suffix.casefold() not in JPEG_SUFFIXES:
            raise AssertionError(f"{row.series_uid} {phase}: non-JPEG selected: {path}")
        if any(token in path.name.upper() for token in PARAMETER_TOKENS):
            raise AssertionError(f"{row.series_uid} {phase}: parameter map selected: {path}")
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
    return PhasePlan(
        patient_id=str(row["patient_id"]), split=str(row["split"]),
        source_type=str(row.get("source_type", "")), series_uid=str(row["series_uid"]),
        series_id=str(row.get("series_id", "")), phase=phase,
        frame_paths=tuple(paths), frame_indices=tuple(indices), frame_list_hash=stored_hash,
        manifest_sha256=manifest_hash, raw_row={str(k): v for k, v in row.to_dict().items()},
    )


def load_manifest(
    path: Path,
    expected_split: str | None = None,
    verify_files: bool = True,
    expected_counts: dict[str, int] | None = None,
    max_series: int | None = None,
    requested_series_uids: set[str] | None = None,
) -> ManifestBundle:
    manifest_hash = sha256_file(path)
    frame = pd.read_csv(path, dtype={"patient_id": str, "series_uid": str}).copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise AssertionError(f"{path}: empty manifest")
    if frame["series_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate series_uid")
    if expected_split is not None and not (
        frame["split"].astype(str).str.casefold() == expected_split.casefold()
    ).all():
        raise AssertionError(f"{path}: split isolation failure")
    for key in ("selected_for_extraction", "candidate_valid", "selected_candidate"):
        if not frame[key].map(as_bool).all():
            bad = frame.loc[~frame[key].map(as_bool), "series_uid"].head().tolist()
            raise AssertionError(f"{path}: {key} contains false rows: {bad}")
    if requested_series_uids:
        before = set(frame["series_uid"].astype(str))
        missing_ids = sorted(requested_series_uids - before)
        if missing_ids:
            raise KeyError(f"Requested series absent: {missing_ids}")
        frame = frame[frame["series_uid"].astype(str).isin(requested_series_uids)].copy()
    frame = frame.sort_values(["patient_id", "series_uid"]).reset_index(drop=True)
    if max_series is not None:
        frame = frame.head(max_series).copy()

    plans: list[PhasePlan] = []
    for _, row in frame.iterrows():
        for phase in ("pre", "post"):
            plan = _validate_phase(row, phase, verify_files, manifest_hash)
            if plan is not None:
                plans.append(plan)
    summary = {
        "manifest": str(path.resolve()), "manifest_sha256": manifest_hash,
        "series": int(frame["series_uid"].nunique()),
        "patients": int(frame["patient_id"].nunique()),
        "pre": int(sum(plan.phase == "pre" for plan in plans)),
        "post": int(sum(plan.phase == "post" for plan in plans)),
        "phases": len(plans),
        "frame_gaps": int(sum(
            sum(b - a > 1 for a, b in zip(plan.frame_indices[:-1], plan.frame_indices[1:]))
            for plan in plans
        )),
    }
    if expected_counts is not None and max_series is None and not requested_series_uids:
        actual = {key: int(summary[key]) for key in expected_counts}
        if actual != expected_counts:
            raise AssertionError(f"Manifest hard-count mismatch: expected={expected_counts}, actual={actual}")
    return ManifestBundle(frame=frame, plans=tuple(plans), summary=summary)
