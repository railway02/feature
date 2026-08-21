from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .common import parse_pipe, require_sha256, text_bool
from .local_geometry import BBox, source_spacing_from_manifest


PAIR_REQUIRED = {
    "split", "patient_id", "series_uid", "series_id", "pre_phase_uid", "post_phase_uid",
    "pre_png_key", "post_png_key", "pre_reference_image_path", "post_reference_image_path",
    "pre_mask_path", "post_mask_path", "pre_frame_paths", "post_frame_paths",
    "pre_n_frames", "post_n_frames", "pre_frame_list_hash", "post_frame_list_hash",
    "pre_mapping_method", "post_mapping_method", "pre_mapping_score", "post_mapping_score",
}

ROI_REQUIRED = {
    "phase_uid", "split", "patient_id", "series_uid", "series_id", "phase", "frame_paths",
    "frame_list_hash", "n_frames", "frame_height", "frame_width", "png_key",
    "reference_image_path", "mask_path", "mapping_method", "mapping_score",
    "orientation_transform", "orientation_status", "effective_mask_array_sha256",
    "mask_resized_to_frame", "resize_scale_x", "resize_scale_y", "original_bbox",
    "expanded_bbox", "padding_left", "padding_top", "padding_right", "padding_bottom", "roi_side",
}


@dataclass(frozen=True)
class PhaseRecord:
    phase_uid: str
    split: str
    patient_id: str
    series_uid: str
    series_id: str
    phase: str
    png_key: str
    reference_image_path: Path
    mask_path: Path
    frame_paths: tuple[str, ...]
    frame_list_hash: str
    frame_height: int
    frame_width: int
    mapping_method: str
    mapping_score: float
    orientation_transform: str
    orientation_status: str
    effective_mask_array_sha256: str
    mask_resized_to_frame: bool
    resize_scale_x: float
    resize_scale_y: float
    original_bbox: BBox
    expanded_bbox: BBox
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    roi_side: int

    @property
    def canvas_shape_yx(self) -> tuple[int, int]:
        return self.frame_height, self.frame_width

    @property
    def source_spacing_xy(self) -> tuple[float, float]:
        return source_spacing_from_manifest(self.resize_scale_x, self.resize_scale_y)


@dataclass(frozen=True)
class PairRecord:
    split: str
    patient_id: str
    series_uid: str
    series_id: str
    pre: PhaseRecord
    post: PhaseRecord


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _phase_from_row(row: dict[str, Any]) -> PhaseRecord:
    frame_paths = parse_pipe(row["frame_paths"])
    declared = int(float(row["n_frames"]))
    if declared != len(frame_paths):
        raise ValueError(f"{row['phase_uid']}: n_frames={declared} but paths={len(frame_paths)}")
    phase = str(row["phase"]).strip().casefold()
    if phase not in {"pre", "post"}:
        raise ValueError(f"{row['phase_uid']}: invalid phase {phase!r}")
    orientation = str(row["orientation_transform"]).strip()
    if orientation != "identity":
        raise ValueError(f"{row['phase_uid']}: unsupported non-identity orientation {orientation!r}")
    return PhaseRecord(
        phase_uid=str(row["phase_uid"]), split=str(row["split"]), patient_id=str(row["patient_id"]),
        series_uid=str(row["series_uid"]), series_id=str(row["series_id"]), phase=phase,
        png_key=str(row["png_key"]), reference_image_path=Path(str(row["reference_image_path"])),
        mask_path=Path(str(row["mask_path"])), frame_paths=frame_paths,
        frame_list_hash=str(row["frame_list_hash"]), frame_height=int(float(row["frame_height"])),
        frame_width=int(float(row["frame_width"])), mapping_method=str(row["mapping_method"]),
        mapping_score=float(row["mapping_score"]), orientation_transform=orientation,
        orientation_status=str(row["orientation_status"]),
        effective_mask_array_sha256=str(row["effective_mask_array_sha256"]),
        mask_resized_to_frame=text_bool(row["mask_resized_to_frame"]),
        resize_scale_x=float(row["resize_scale_x"]), resize_scale_y=float(row["resize_scale_y"]),
        original_bbox=BBox.from_text(str(row["original_bbox"])),
        expanded_bbox=BBox.from_text(str(row["expanded_bbox"])),
        padding_left=int(float(row["padding_left"])), padding_top=int(float(row["padding_top"])),
        padding_right=int(float(row["padding_right"])), padding_bottom=int(float(row["padding_bottom"])),
        roi_side=int(float(row["roi_side"])),
    )


def _check_pair_row(pair_row: dict[str, Any], phase: PhaseRecord, prefix: str) -> None:
    expected_uid = str(pair_row[f"{prefix}_phase_uid"])
    expected_paths = parse_pipe(pair_row[f"{prefix}_frame_paths"])
    if phase.phase_uid != expected_uid:
        raise ValueError(f"{phase.series_uid}/{prefix}: phase_uid mismatch")
    if phase.frame_paths != expected_paths:
        raise ValueError(f"{phase.phase_uid}: temporal frame paths differ from pair manifest")
    if phase.frame_list_hash != str(pair_row[f"{prefix}_frame_list_hash"]):
        raise ValueError(f"{phase.phase_uid}: frame list hash differs from pair manifest")
    if len(phase.frame_paths) != int(float(pair_row[f"{prefix}_n_frames"])):
        raise ValueError(f"{phase.phase_uid}: pair n_frames differs")
    if phase.png_key != str(pair_row[f"{prefix}_png_key"]):
        raise ValueError(f"{phase.phase_uid}: png_key differs from pair manifest")


def load_local_reference_pairs(cfg: dict[str, Any], *, split: str | None = None) -> list[PairRecord]:
    paths = cfg["paths"]
    locks = cfg["locks"]
    require_sha256(paths["temporal_pairs"], locks["temporal_pairs_sha256"], "temporal_pairs")
    require_sha256(paths["roi_phase_manifest"], locks["roi_phase_manifest_sha256"], "roi_phase_manifest")

    pairs = pd.read_csv(paths["temporal_pairs"], dtype=str, keep_default_na=False)
    roi = pd.read_csv(paths["roi_phase_manifest"], dtype=str, keep_default_na=False)
    _require_columns(pairs, PAIR_REQUIRED, "temporal_pairs")
    _require_columns(roi, ROI_REQUIRED, "roi_phase_manifest")
    if pairs["series_uid"].duplicated().any():
        raise AssertionError("temporal pair manifest has duplicate series_uid")
    if roi["phase_uid"].duplicated().any():
        raise AssertionError("ROI manifest has duplicate phase_uid")

    expected = cfg["expected"]
    if len(pairs) != int(expected["paired_series"]):
        raise AssertionError(f"paired series={len(pairs)}, expected={expected['paired_series']}")
    if len(roi) != int(expected["eligible_phases"]):
        raise AssertionError(f"eligible ROI phases={len(roi)}, expected={expected['eligible_phases']}")
    split_counts = pairs["split"].value_counts().to_dict()
    if int(split_counts.get("Train", 0)) != int(expected["paired_series_train"]):
        raise AssertionError("Train pair count changed")
    if int(split_counts.get("Valid", 0)) != int(expected["paired_series_valid"]):
        raise AssertionError("Valid pair count changed")

    roi_by_uid = {str(row["phase_uid"]): _phase_from_row(row) for row in roi.to_dict("records")}
    records: list[PairRecord] = []
    for row in pairs.to_dict("records"):
        row_split = str(row["split"])
        if split is not None and row_split != split:
            continue
        pre = roi_by_uid.get(str(row["pre_phase_uid"]))
        post = roi_by_uid.get(str(row["post_phase_uid"]))
        if pre is None or post is None:
            raise KeyError(f"{row['series_uid']}: phase absent from old eligible ROI manifest")
        _check_pair_row(row, pre, "pre")
        _check_pair_row(row, post, "post")
        if pre.phase != "pre" or post.phase != "post":
            raise AssertionError(f"{row['series_uid']}: incorrect phase labels")
        for phase in (pre, post):
            if (phase.split, phase.patient_id, phase.series_uid) != (
                row_split, str(row["patient_id"]), str(row["series_uid"])
            ):
                raise AssertionError(f"{phase.phase_uid}: pair identity mismatch")
        records.append(PairRecord(
            split=row_split, patient_id=str(row["patient_id"]), series_uid=str(row["series_uid"]),
            series_id=str(row["series_id"]), pre=pre, post=post,
        ))
    return records


def audit_input_paths(records: list[PairRecord]) -> list[dict[str, object]]:
    """Stat all input assets while preserving both phases and every temporal frame."""
    rows: list[dict[str, object]] = []
    for pair in records:
        for phase in (pair.pre, pair.post):
            missing_frames = sum(not Path(path).is_file() for path in phase.frame_paths)
            rows.append({
                "series_uid": pair.series_uid, "phase_uid": phase.phase_uid, "split": pair.split,
                "phase": phase.phase, "reference_exists": phase.reference_image_path.is_file(),
                "mask_exists": phase.mask_path.is_file(), "frame_count": len(phase.frame_paths),
                "missing_frame_count": missing_frames,
            })
    return rows
