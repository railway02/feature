#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/aneurysm")
sys.path.insert(0, str(ROOT / "code/api_adverse_lesion_cave_v1"))
from assets import (  # noqa: E402
    apply_orientation,
    largest_component,
    lesion_and_context_masks,
    load_nifti_mask,
    make_resize_transform,
    read_frames,
    resize_mask_to_model,
    resize_stack_to_model,
)


LESION_LABELS = [2, 3, 4, 5, 6]
CONTEXT_LABELS = [1]
IMAGE_SIZE = 1024
VIEW_ID = "p2_polarity_minip_median_q95q05_phase"


def atomic_npz(path: Path, **arrays) -> None:
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


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise AssertionError("No finite frame pixels")
    low, high = np.percentile(finite, [1.0, 99.0])
    return np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def normalize_channel(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return np.zeros_like(array, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    scaled = np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def representative_stack(frames: np.ndarray, polarity: float) -> np.ndarray:
    normalized = normalize_frames(frames)
    contrast = 1.0 - normalized if polarity < 0 else normalized
    polarity_minip = np.max(contrast, axis=0)
    median = np.median(contrast, axis=0)
    q95, q05 = np.quantile(normalized, [0.95, 0.05], axis=0)
    robust_range = q95 - q05
    return np.stack([
        normalize_channel(polarity_minip),
        normalize_channel(median),
        normalize_channel(robust_range),
    ])


def sample_uid(record_uid: str, phase: str, frame_hash: str) -> str:
    raw = f"{record_uid}|{phase}|{frame_hash}|{VIEW_ID}|{IMAGE_SIZE}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_one(row: dict, output_root: Path) -> dict:
    uid = sample_uid(row["record_uid"], row["phase"], row["frame_list_hash"])
    path = output_root / "samples" / row["phase"] / f"{uid}.npz"
    if path.is_file():
        raw = np.load(path, allow_pickle=False)
        metadata = json.loads(str(raw["metadata"].item()))
        return {**metadata, "sample_path": str(path)}

    frames = read_frames(parse_pipe(row["frame_paths"]))
    mask, mask_info = load_nifti_mask(Path(row["segmentation_path"]))
    oriented = apply_orientation(mask, row["orientation_transform"])
    if oriented.shape != frames.shape[1:]:
        import cv2
        source_ratio = oriented.shape[1] / max(oriented.shape[0], 1)
        frame_ratio = frames.shape[2] / max(frames.shape[1], 1)
        if abs(source_ratio - frame_ratio) > 0.02:
            raise AssertionError(f"Mask/frame aspect mismatch {oriented.shape} {frames.shape[1:]}")
        oriented = cv2.resize(
            oriented,
            (frames.shape[2], frames.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )

    lesion, _context, _all_nonzero = lesion_and_context_masks(
        oriented,
        LESION_LABELS,
        CONTEXT_LABELS,
    )
    if int(lesion.sum()) < 8:
        alternative = ((oriented != 0) & (~np.isin(oriented, CONTEXT_LABELS))).astype(np.uint8)
        lesion = largest_component(alternative)
    if int(lesion.sum()) < 8:
        raise AssertionError("No lesion target")

    stack = representative_stack(frames, float(row["polarity"]))
    transform = make_resize_transform(frames.shape[1], frames.shape[2], IMAGE_SIZE)
    image = resize_stack_to_model(stack, transform).astype(np.uint8)
    target = (resize_mask_to_model(lesion, transform) > 0).astype(np.uint8)
    if int(target.sum()) < 4:
        raise AssertionError("Lesion vanished after resize")

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
        "orientation_transform": row["orientation_transform"],
        "polarity": float(row["polarity"]),
        "polarity_label": row["polarity_label"],
        "input_channels": ["polarity_minip", "temporal_median", "q95_minus_q05"],
        "phase_channel_added_at_load": True,
        "image_size": IMAGE_SIZE,
        "original_h": int(frames.shape[1]),
        "original_w": int(frames.shape[2]),
        "lesion_pixels_model": int(target.sum()),
        "lesion_area_ratio_model": float(target.mean()),
        "mask_labels": mask_info["labels"],
        "resize_transform": transform.to_json(),
        "view_id": VIEW_ID,
    }
    atomic_npz(
        path,
        image=image,
        mask=target,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    return {**metadata, "sample_path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest_root = ROOT / "manifests/api_adverse_lesion_record_v2"
    output_root = ROOT / "outputs/api_adverse_lesion_record_v2/segmentation_pilot_p2"
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
    phase = phase[phase.split == "Train"].copy()
    temporal = temporal[temporal.split == "Train"][
        ["record_uid", "phase", "frame_list_hash", "polarity", "polarity_label"]
    ].copy()
    if temporal.duplicated(["record_uid", "phase"]).any():
        raise AssertionError("Duplicate frozen temporal row")
    merged = phase.merge(
        temporal,
        on=["record_uid", "phase", "frame_list_hash"],
        how="left",
        validate="one_to_one",
    )
    if merged["polarity"].eq("").any() or merged["polarity"].isna().any():
        raise AssertionError("Missing frozen polarity")
    merged["fold"] = pd.to_numeric(merged["fold"], errors="raise").astype(int)
    merged = merged.sort_values(["fold", "patient_id", "series_uid", "phase"]).reset_index(drop=True)
    if args.max_samples:
        merged = merged.head(args.max_samples).copy()

    rows, excluded = [], []
    records = merged.to_dict("records")
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(build_one, row, output_root): row for row in records}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                excluded.append({
                    "record_uid": row["record_uid"],
                    "series_uid": row["series_uid"],
                    "phase": row["phase"],
                    "reason": repr(exc),
                })
            if index % 25 == 0 or index == len(records):
                print(f"[{index}/{len(records)}] built={len(rows)} excluded={len(excluded)}", flush=True)

    index_frame = pd.DataFrame(rows)
    index_frame = index_frame.sort_values(["fold", "patient_id", "series_uid", "phase"]).reset_index(drop=True)
    if index_frame.empty:
        raise RuntimeError("No pilot samples built")
    if index_frame["sample_uid"].duplicated().any():
        raise AssertionError("Duplicate pilot sample_uid")
    atomic_csv(index_frame, manifest_root / "segmentation_pilot_p2_index.csv")
    atomic_csv(pd.DataFrame(excluded), report_root / "segmentation_pilot_p2_excluded.csv")
    expected = len(merged)
    audit = {
        "smoke": args.max_samples is not None,
        "view_id": VIEW_ID,
        "image_size": IMAGE_SIZE,
        "input_channels": ["polarity_minip", "temporal_median", "q95_minus_q05", "phase_channel"],
        "lesion_labels_engineering_assumption": LESION_LABELS,
        "context_labels_excluded": CONTEXT_LABELS,
        "samples_expected": expected,
        "samples_built": len(index_frame),
        "excluded": len(excluded),
        "patients": int(index_frame.patient_id.nunique()),
        "by_fold": {str(key): int(value) for key, value in index_frame.groupby("fold").size().to_dict().items()},
        "by_phase": index_frame.phase.value_counts().to_dict(),
        "by_grade": index_frame.annotation_grade.value_counts().to_dict(),
        "median_lesion_area_ratio": float(index_frame.lesion_area_ratio_model.median()),
        "complete": len(index_frame) == expected and not excluded,
    }
    atomic_json(audit, report_root / "segmentation_pilot_p2_dataset_audit.json")
    marker = report_root / ".SEG_PILOT_DATA_PASS"
    if audit["complete"] and args.max_samples is None:
        marker.write_text("pass\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
