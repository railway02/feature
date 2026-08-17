#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/aneurysm")
sys.path.insert(0, str(ROOT / "code/api_adverse_lesion_cave_v1"))
from assets import (  # noqa: E402
    apply_orientation,
    largest_component,
    lesion_and_context_masks,
    load_nifti_image,
    load_nifti_mask,
    make_resize_transform,
    read_frames,
    resize_mask_to_model,
    resize_stack_to_model,
)


LESION_LABELS = [2, 3, 4, 5, 6]
CONTEXT_LABELS = [1]
IMAGE_SIZE = 1024
VIEW_ID = "reference_upper_exact_frame_median_enhancement_phase_v1"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_pipe(value: str) -> list[str]:
    return [item for item in str(value).split("|") if item]


def normalize_with_bounds(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(value, dtype=np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)


def normalize_channel(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return np.zeros_like(array, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    scaled = normalize_with_bounds(array, float(low), float(high))
    return np.rint(scaled * 255.0).astype(np.uint8)


def resize_like(array: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    if array.shape == shape:
        return array
    source_ratio = array.shape[1] / max(array.shape[0], 1)
    target_ratio = shape[1] / max(shape[0], 1)
    if abs(source_ratio - target_ratio) > 0.02:
        raise AssertionError(f"Aspect mismatch {array.shape} -> {shape}")
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


def reference_stack(frames: np.ndarray, reference: np.ndarray, polarity: float) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise AssertionError("No finite frame pixels")
    low, high = [float(item) for item in np.percentile(finite, [1.0, 99.0])]
    normalized = normalize_with_bounds(values, low, high)
    reference_normalized = normalize_with_bounds(reference, low, high)
    median = np.median(normalized, axis=0)
    early_count = max(1, min(3, len(normalized) // 4 if len(normalized) >= 4 else 1))
    baseline = np.median(normalized[:early_count], axis=0)
    if polarity < 0:
        reference_contrast = 1.0 - reference_normalized
        median_contrast = 1.0 - median
        enhancement = np.maximum(baseline - reference_normalized, 0.0)
    else:
        reference_contrast = reference_normalized
        median_contrast = median
        enhancement = np.maximum(reference_normalized - baseline, 0.0)
    return np.stack(
        [
            normalize_channel(reference_contrast),
            normalize_channel(median_contrast),
            normalize_channel(enhancement),
        ]
    )


def sample_uid(row: dict) -> str:
    raw = "|".join(
        [
            str(row["record_uid"]),
            str(row["phase"]),
            str(row["frame_list_hash"]),
            str(row["reference_image_path"]),
            VIEW_ID,
            str(IMAGE_SIZE),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_one(row: dict, output_root: Path) -> dict:
    uid = sample_uid(row)
    output_path = output_root / "samples" / str(row["phase"]) / f"{uid}.npz"
    if output_path.is_file():
        raw = np.load(output_path, allow_pickle=False)
        metadata = json.loads(str(raw["metadata"].item()))
        return {**metadata, "sample_path": str(output_path)}

    frame_paths = parse_pipe(row["frame_paths"])
    frames = read_frames(frame_paths)
    reference, reference_info = load_nifti_image(Path(row["reference_image_path"]))
    reference = apply_orientation(reference, row["orientation_transform"])
    reference = resize_like(reference, tuple(frames.shape[1:]), cv2.INTER_LINEAR)

    mask, mask_info = load_nifti_mask(Path(row["segmentation_path"]))
    mask = apply_orientation(mask, row["orientation_transform"])
    mask = resize_like(mask, tuple(frames.shape[1:]), cv2.INTER_NEAREST)
    lesion, _context, _all_nonzero = lesion_and_context_masks(
        mask,
        LESION_LABELS,
        CONTEXT_LABELS,
    )
    if int(lesion.sum()) < 8:
        alternative = ((mask != 0) & (~np.isin(mask, CONTEXT_LABELS))).astype(np.uint8)
        lesion = largest_component(alternative)
    if int(lesion.sum()) < 8:
        raise AssertionError("No lesion target")

    stack = reference_stack(frames, reference, float(row["polarity"]))
    transform = make_resize_transform(frames.shape[1], frames.shape[2], IMAGE_SIZE)
    image = resize_stack_to_model(stack, transform).astype(np.uint8)
    target = (resize_mask_to_model(lesion, transform) > 0).astype(np.uint8)
    if int(target.sum()) < 4:
        raise AssertionError("Lesion vanished after resize")
    yy, xx = np.where(target > 0)

    metadata = {
        "sample_uid": uid,
        "record_uid": row["record_uid"],
        "patient_id": row["patient_id"],
        "series_uid": row["series_uid"],
        "phase_uid": row["phase_uid"],
        "phase": row["phase"],
        "fold": int(row["fold"]),
        "annotation_grade": row["annotation_grade"],
        "mapping_tier": row["mapping_tier"],
        "frame_list_hash": row["frame_list_hash"],
        "segmentation_path": row["segmentation_path"],
        "reference_image_path": row["reference_image_path"],
        "matched_reference_frame_path": row["matched_reference_frame_path"],
        "matched_reference_frame_position": int(float(row["matched_reference_frame_position"])),
        "reference_ncc": float(row["reference_ncc"]),
        "reference_ssim": float(row["reference_ssim"]),
        "reference_mae": float(row["reference_mae"]),
        "exact_reference_fallback": str(row["exact_reference_fallback"]).casefold() == "true",
        "orientation_transform": row["orientation_transform"],
        "polarity": float(row["polarity"]),
        "polarity_label": row["polarity_label"],
        "input_channels": ["exact_annotation_reference", "temporal_median", "reference_enhancement"],
        "phase_channel_added_at_load": True,
        "image_size": IMAGE_SIZE,
        "n_frames": int(len(frames)),
        "original_h": int(frames.shape[1]),
        "original_w": int(frames.shape[2]),
        "lesion_pixels_model": int(target.sum()),
        "lesion_area_ratio_model": float(target.mean()),
        "lesion_center_x_model": float(xx.mean()),
        "lesion_center_y_model": float(yy.mean()),
        "mask_labels": mask_info["labels"],
        "reference_raw_shape": reference_info["raw_shape"],
        "resize_transform": transform.to_json(),
        "view_id": VIEW_ID,
        "label_semantics_status": "engineering_assumption_labels_2_to_6_pending_hospital_confirmation",
    }
    atomic_npz(
        output_path,
        image=image,
        mask=target,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    return {**metadata, "sample_path": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    manifest_root = ROOT / "manifests/api_adverse_lesion_record_v2"
    suffix = "_smoke" if args.max_samples is not None else ""
    output_root = ROOT / f"outputs/api_adverse_lesion_record_v2/reference_frame_upper_pilot{suffix}"
    report_root = ROOT / "reports/api_adverse_lesion_record_v2"
    phase = pd.read_csv(
        manifest_root / "segmentation_phase_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    temporal = pd.read_csv(
        manifest_root / "frozen_temporal_view_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    phase = phase[(phase.split == "Train") & (phase.annotation_grade == "A")].copy()
    temporal = temporal[temporal.split == "Train"][
        ["record_uid", "phase", "frame_list_hash", "polarity", "polarity_label"]
    ].copy()
    merged = phase.merge(
        temporal,
        on=["record_uid", "phase", "frame_list_hash"],
        how="left",
        validate="one_to_one",
    )
    required = [
        "reference_image_path",
        "matched_reference_frame_path",
        "matched_reference_frame_position",
        "polarity",
    ]
    for column in required:
        if merged[column].eq("").any() or merged[column].isna().any():
            raise AssertionError(f"A-grade sample missing {column}")
    if merged.phase_uid.duplicated().any():
        raise AssertionError("Duplicate phase_uid in reference-frame Pilot")
    merged["fold"] = pd.to_numeric(merged["fold"], errors="raise").astype(int)
    merged = merged.sort_values(["fold", "patient_id", "series_uid", "phase"]).reset_index(drop=True)
    if args.max_samples:
        merged = merged.head(int(args.max_samples)).copy()

    built, excluded = [], []
    records = merged.to_dict("records")
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(build_one, row, output_root): row for row in records}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                built.append(future.result())
            except Exception as exc:
                excluded.append(
                    {
                        "record_uid": row["record_uid"],
                        "series_uid": row["series_uid"],
                        "phase": row["phase"],
                        "reason": repr(exc),
                    }
                )
            if index % 25 == 0 or index == len(records):
                print(f"[{index}/{len(records)}] built={len(built)} excluded={len(excluded)}", flush=True)

    index_frame = pd.DataFrame(built)
    if index_frame.empty:
        raise RuntimeError("No reference-frame Pilot samples built")
    index_frame = index_frame.sort_values(["fold", "patient_id", "series_uid", "phase"]).reset_index(drop=True)
    atomic_csv(index_frame, manifest_root / f"reference_frame_upper_pilot{suffix}_index.csv")
    atomic_csv(pd.DataFrame(excluded), report_root / f"reference_frame_upper_pilot{suffix}_excluded.csv")
    audit = {
        "status": "complete" if len(index_frame) == len(merged) and not excluded else "incomplete",
        "smoke": args.max_samples is not None,
        "view_id": VIEW_ID,
        "image_size": IMAGE_SIZE,
        "samples_expected": int(len(merged)),
        "samples_built": int(len(index_frame)),
        "excluded": int(len(excluded)),
        "patients": int(index_frame.patient_id.nunique()),
        "by_fold": {str(key): int(value) for key, value in index_frame.groupby("fold").size().to_dict().items()},
        "by_phase": {str(key): int(value) for key, value in index_frame.phase.value_counts().to_dict().items()},
        "median_lesion_area_ratio": float(index_frame.lesion_area_ratio_model.median()),
        "reference_ssim_median": float(index_frame.reference_ssim.median()),
        "reference_relative_position_median": float(
            (
                index_frame.matched_reference_frame_position.astype(float)
                / np.maximum(index_frame.n_frames.astype(float) - 1.0, 1.0)
            ).median()
        ),
        "input_channels": ["exact_annotation_reference", "temporal_median", "reference_enhancement", "phase_channel"],
        "lesion_labels_engineering_assumption": LESION_LABELS,
        "label_semantics_confirmation": "pending_hospital_confirmation",
    }
    atomic_json(audit, report_root / f"reference_frame_upper_pilot{suffix}_dataset_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
