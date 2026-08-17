#!/usr/bin/env python3
"""ROI QA overlays for eligible phases.

Frame picks come from the frozen Whole-CAVE contrast_core20 indices (contrast
peak region), never just the first frame — early frames may carry no
contrast. Sampling: every non-upstream/manual mapping (unverified
orientation), all anomalies, ROI side extremes, one row per resolution, plus
a seeded random upstream sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, parse_pipe
from nifti_io import apply_orientation, load_label_mask, resize_labels
from roi import bbox_from_text, crop_frame


def normalize_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def overlay(image: np.ndarray, labels: np.ndarray, tight: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    canvas = cv2.cvtColor(normalize_u8(image), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((labels != 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    cv2.rectangle(canvas, (tight[0], tight[1]), (tight[2] - 1, tight[3] - 1), (0, 0, 255), 2)
    cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (0, 255, 0), 2)
    return canvas


def representative_indices(blocks_json: str, n_frames: int) -> list[int]:
    """Contrast-core representative frame indices (25/50/75 percentiles)."""
    fallback = sorted({0, n_frames // 2, n_frames - 1})
    try:
        blocks = json.loads(blocks_json)
        core: list[int] = []
        for block in blocks:
            values = block.get("view_indices", {}).get("contrast_core20")
            if values:
                core = [int(v) for v in values]
                break
        if not core:
            return fallback
        picks = {core[0], core[len(core) // 4], core[len(core) // 2], core[(3 * len(core)) // 4], core[-1]}
        picks = {min(max(int(v), 0), n_frames - 1) for v in picks}
        return sorted(picks)[:3] if len(picks) >= 3 else sorted(set(list(picks) + fallback))[:3]
    except Exception:
        return fallback


def select_samples(group: pd.DataFrame, upstream_sample: int, seed: int) -> pd.DataFrame:
    ratio = pd.to_numeric(group["roi_area_ratio"], errors="coerce")
    anomaly = group[(ratio > 0.50) | (ratio < 0.005) | (group["mask_resized_to_frame"].astype(str).str.casefold() == "true")]
    review = group[~group["mapping_method"].isin(["upstream", "manual"])]
    side = pd.to_numeric(group["roi_side"], errors="coerce")
    extremes = group[(side == side.max()) | (side == side.min())]
    resolutions = group.drop_duplicates(subset=["frame_height", "frame_width"])
    base = pd.concat([anomaly, review, extremes, resolutions]).drop_duplicates("phase_uid")
    rest = group.drop(index=base.index, errors="ignore")
    upstream_rest = rest[rest["mapping_method"].isin(["upstream", "manual"])]
    sample = upstream_rest.sample(n=min(upstream_sample, len(upstream_rest)), random_state=seed) if len(upstream_rest) else upstream_rest
    return pd.concat([base, sample]).drop_duplicates("phase_uid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    qa_root = reports / "roi_qa"
    roi = pd.read_csv(manifests / "roi_phase_manifest_eligible.csv", dtype=str, keep_default_na=False)
    views = pd.read_csv(manifests / "whole_temporal_views_eligible.csv", dtype=str, keep_default_na=False)
    blocks_by_hash = {str(r.frame_list_hash): str(r.blocks_json) for r in views.itertuples(index=False)}

    samples: list[pd.DataFrame] = []
    for split in ("Train", "Valid"):
        group = roi[roi["split"] == split].copy()
        samples.append(select_samples(group, args.upstream_sample, args.seed))
    sample = pd.concat(samples, ignore_index=True)

    index_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in sample.to_dict("records"):
        try:
            paths = parse_pipe(row["frame_paths"])
            if not paths:
                raise ValueError("empty frame paths")
            selected = representative_indices(blocks_by_hash.get(str(row["frame_list_hash"]), ""), len(paths))
            frames, valid_indices = [], []
            for index in selected:
                image = cv2.imread(paths[index], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    frames.append(image)
                    valid_indices.append(index)
            if not frames:
                raise FileNotFoundError("QA frames unreadable")
            labels, _ = load_label_mask(row["mask_path"])
            labels = apply_orientation(labels, row["orientation_transform"])
            if labels.shape != frames[0].shape:
                labels = resize_labels(labels, frames[0].shape)
            tight = bbox_from_text(row["original_bbox"])
            box = bbox_from_text(row["expanded_bbox"])
            panels = []
            for frame, frame_index in zip(frames, valid_indices):
                full = overlay(frame, labels, tight, box)
                local = cv2.cvtColor(normalize_u8(crop_frame(frame, box)), cv2.COLOR_GRAY2BGR)
                local = cv2.resize(local, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_AREA)
                cv2.putText(full, f"frame={frame_index} full", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(local, f"frame={frame_index} local", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(np.concatenate([full, local], axis=1))
            contact = np.concatenate(panels, axis=0)
            safe_uid = str(row["series_uid"]).replace("/", "_").replace("\\", "_")
            out = qa_root / row["split"].casefold() / str(row["patient_id"]) / f"{safe_uid}__{row['phase']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), contact, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(f"cannot write {out}")
            index_rows.append({
                "phase_uid": row["phase_uid"], "split": row["split"], "patient_id": row["patient_id"],
                "series_uid": row["series_uid"], "phase": row["phase"], "mapping_method": row["mapping_method"],
                "orientation_transform": row["orientation_transform"], "orientation_status": row.get("orientation_status", ""),
                "qa_frame_indices": "|".join(map(str, valid_indices)),
                "mask_path": row["mask_path"], "qa_path": str(out),
            })
        except Exception as exc:
            failures.append({**row, "qa_error": f"{type(exc).__name__}:{exc}"})

    atomic_csv(pd.DataFrame(index_rows), reports / "04_roi_qa_index.csv")
    atomic_csv(pd.DataFrame(failures), reports / "04_roi_qa_failures.csv")
    method_counts = pd.Series([r["mapping_method"] for r in index_rows]).value_counts().to_dict()
    summary = {
        "qa_images": len(index_rows),
        "qa_failures": len(failures),
        "qa_root": str(qa_root),
        "qa_by_mapping_method": method_counts,
        "frame_policy": "contrast_core20_representative_indices",
    }
    atomic_json(summary, reports / "04_roi_qa_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
