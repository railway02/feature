#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from assets import (
    apply_orientation, largest_component, lesion_and_context_masks, load_nifti_mask, make_resize_transform,
    read_frames, resize_mask_to_model, resize_stack_to_model, summary_stack_u8,
)
from common import atomic_csv, atomic_json, configure_runtime, load_config, parse_pipe, safe_uid, sha256_file, stage_logger, write_marker


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--max-rows", type=int); args = parser.parse_args()
    config = load_config(args.config); configure_runtime(config)
    finish = stage_logger("04_build_segmentation_dataset")
    manifests = Path(config["paths"]["manifests"]); outputs = Path(config["paths"]["outputs"]); reports = Path(config["paths"]["reports"])
    source_path = manifests / "authoritative_roi_manifest_primary.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    if args.max_rows: source = source.head(args.max_rows).copy()
    mask_cfg = config["mask"]; seg_cfg = config["segmentation"]
    image_size = int(os.environ.get("SEG_IMAGE_SIZE", seg_cfg["image_size"]))
    channels = list(seg_cfg["input_channels"])
    dataset_root = outputs / "segmentation_dataset"
    rows, excluded = [], []
    for record in source.to_dict("records"):
        try:
            frame_paths = parse_pipe(record["frame_paths"]); frames = read_frames(frame_paths)
            raw_mask, mask_info = load_nifti_mask(Path(record["segmentation_path"]))
            oriented = apply_orientation(raw_mask, record["orientation_transform"])
            if oriented.shape != frames.shape[1:]:
                source_ratio = oriented.shape[1] / max(oriented.shape[0], 1)
                frame_ratio = frames.shape[2] / max(frames.shape[1], 1)
                if abs(source_ratio - frame_ratio) > 0.02:
                    raise AssertionError(f"Mask/frame aspect mismatch mask={oriented.shape} frames={frames.shape[1:]}")
                oriented = cv2.resize(oriented, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_NEAREST)
                mask_resized_to_frames = True
            else:
                mask_resized_to_frames = False
            lesion, context, all_nonzero = lesion_and_context_masks(oriented, mask_cfg["lesion_labels"], mask_cfg["context_labels"])
            lesion_pixels = int(lesion.sum())
            if lesion_pixels < int(mask_cfg["minimum_pixels"]):
                alternative = ((oriented != 0) & (~np.isin(oriented, mask_cfg["context_labels"]))).astype(np.uint8)
                if int(alternative.sum()) >= int(mask_cfg["minimum_pixels"]):
                    lesion = largest_component(alternative); lesion_pixels = int(lesion.sum()); lesion_fallback = "largest_nonzero_noncontext_component"
                    if lesion_pixels < int(mask_cfg["minimum_pixels"]): raise AssertionError("no_lesion_target_after_component_filter")
                else:
                    raise AssertionError("no_lesion_target")
            else:
                lesion_fallback = "none"
            area_ratio = lesion_pixels / float(lesion.size)
            if area_ratio > float(mask_cfg["maximum_area_ratio"]):
                raise AssertionError(f"lesion_area_ratio_too_large:{area_ratio}")
            stack = summary_stack_u8(frames, channels)
            transform = make_resize_transform(frames.shape[1], frames.shape[2], image_size)
            model_input = resize_stack_to_model(stack, transform)
            model_mask = resize_mask_to_model(lesion, transform)
            model_context = resize_mask_to_model(context, transform)
            model_all_nonzero = resize_mask_to_model(all_nonzero, transform)
            sample_uid = safe_uid(record["phase_uid"], record["segmentation_path"], image_size, channels)
            sample_path = dataset_root / record["split"].casefold() / record["phase"] / f"{sample_uid}.npz"
            metadata = {
                "sample_uid": sample_uid, "phase_uid": record["phase_uid"], "patient_id": record["patient_id"],
                "split": record["split"], "phase": record["phase"], "series_uid": record["series_uid"],
                "series_id": record["series_id"], "source_type": record["source_type"],
                "annotation_grade": record["annotation_grade"], "annotation_layout": record["annotation_layout"],
                "segmentation_path": record["segmentation_path"], "orientation_transform": record["orientation_transform"],
                "frame_list_hash": record["frame_list_hash"], "frame_paths": frame_paths,
                "input_channels": channels, "resize_transform": transform.to_json(),
                "original_shape": [int(frames.shape[1]), int(frames.shape[2])],
                "mask_labels": mask_info["labels"], "lesion_pixels": lesion_pixels,
                "lesion_area_ratio": area_ratio, "lesion_target_fallback": lesion_fallback,
                "mask_resized_to_frames": mask_resized_to_frames,
            }
            atomic_npz(
                sample_path, image=model_input.astype(np.uint8), mask=(model_mask > 0).astype(np.uint8),
                context=(model_context > 0).astype(np.uint8), all_nonzero=(model_all_nonzero > 0).astype(np.uint8),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            )
            rows.append({
                **{key: metadata[key] for key in ["sample_uid","phase_uid","patient_id","split","phase","series_uid","series_id","source_type","annotation_grade","annotation_layout","segmentation_path","orientation_transform","frame_list_hash","lesion_pixels","lesion_area_ratio","lesion_target_fallback","mask_resized_to_frames"]},
                "sample_path": str(sample_path), "original_h": frames.shape[1], "original_w": frames.shape[2],
                "image_size": image_size, "input_channels": "|".join(channels),
            })
        except Exception as exc:
            excluded.append({**record, "dataset_exclusion_reason": repr(exc)})
    index = pd.DataFrame(rows).sort_values(["split", "phase", "patient_id", "series_uid"])
    if index.empty: raise RuntimeError("No segmentation samples built")
    if index["sample_uid"].duplicated().any(): raise AssertionError("Duplicate sample_uid")
    atomic_csv(index, manifests / "segmentation_dataset_index.csv")
    atomic_csv(pd.DataFrame(excluded), manifests / "segmentation_dataset_excluded.csv")
    summary = {
        "samples": len(index), "patients": int(index["patient_id"].nunique()),
        "by_split_phase": {"|".join(key): int(value) for key, value in index.groupby(["split","phase"]).size().to_dict().items()},
        "by_grade": index["annotation_grade"].value_counts().to_dict(),
        "excluded": len(excluded), "image_size": image_size, "input_channels": channels,
        "source_manifest_sha256": sha256_file(source_path),
    }
    atomic_json(summary, reports / "segmentation_dataset_audit.json")
    write_marker(reports / ".SEG_DATA_SUCCESS", "04_build_segmentation_dataset", config, {"source_manifest_sha256": sha256_file(source_path)}, summary)
    finish(summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
