#!/usr/bin/env python3
"""Audit and freeze paired 2-D mean-image/GT-mask PNG mapping inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"unreadable 2-D grayscale PNG: {path}")
    return image


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def resize_intensity(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape == shape:
        return image.astype(np.float32, copy=False)
    interpolation = (
        cv2.INTER_AREA
        if image.shape[0] > shape[0] or image.shape[1] > shape[1]
        else cv2.INTER_CUBIC
    )
    return cv2.resize(
        image, (shape[1], shape[0]), interpolation=interpolation
    ).astype(np.float32)


def source_mean(frame_paths: str) -> np.ndarray:
    paths = [Path(item) for item in str(frame_paths).split("|") if item]
    if not paths:
        raise ValueError("empty frame_paths")
    total: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    for path in paths:
        frame = read_gray(path)
        if shape is None:
            shape = frame.shape
            total = np.zeros(shape, dtype=np.float64)
        elif frame.shape != shape:
            raise ValueError(f"mixed frame shape: {path}={frame.shape}, expected={shape}")
        assert total is not None
        total += frame
    return (total / len(paths)).astype(np.float32)


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda image: image,
    "flip_x": np.fliplr,
    "flip_y": np.flipud,
    "flip_xy": lambda image: np.flipud(np.fliplr(image)),
    "transpose": lambda image: image.T,
    "transpose_flip_x": lambda image: np.fliplr(image.T),
    "transpose_flip_y": lambda image: np.flipud(image.T),
    "transpose_flip_xy": lambda image: np.flipud(np.fliplr(image.T)),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    annotation = cfg["annotation"]
    mapping_cfg = cfg["mapping"]
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    image_dir = Path(annotation["image_dir"]).resolve()
    mask_dir = Path(annotation["mask_dir"]).resolve()
    accepted_path = Path(mapping_cfg["accepted_csv"]).resolve()
    unresolved_path = Path(mapping_cfg["unresolved_csv"]).resolve()
    conflict_path = Path(mapping_cfg["conflict_csv"]).resolve()
    summary_path = Path(mapping_cfg["summary_json"]).resolve()

    images = {path.stem: path.resolve() for path in sorted(image_dir.glob("*.png"))}
    masks = {path.stem: path.resolve() for path in sorted(mask_dir.glob("*.png"))}
    failures: list[str] = []
    if set(images) != set(masks):
        failures.append(
            f"image/mask key mismatch: images_only={len(set(images)-set(masks))}, "
            f"masks_only={len(set(masks)-set(images))}"
        )
    expected_pairs = int(annotation["expected_paired_png_keys"])
    if len(images) != expected_pairs or len(masks) != expected_pairs:
        failures.append(
            f"paired PNG count mismatch: images={len(images)}, masks={len(masks)}, "
            f"expected={expected_pairs}"
        )

    accepted = pd.read_csv(accepted_path, dtype=str, keep_default_na=False)
    unresolved = pd.read_csv(unresolved_path, dtype=str, keep_default_na=False)
    conflict = pd.read_csv(conflict_path, dtype=str, keep_default_na=False)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "png_key", "patient_id", "phase", "image_path", "mask_path",
        "image_sha256", "mask_sha256", "mask_shape", "mask_label_values",
        "candidate_count", "identity_pearson_correlation", "orientation_transform",
        "split", "series_uid", "phase_uid", "frame_paths", "frame_list_hash",
        "reference_image_shape", "series_mean_shape", "mapping_status",
        "mapping_method", "manual_confirmation_note",
    }
    missing = sorted(required - set(accepted.columns))
    if missing:
        failures.append(f"accepted mapping missing columns: {missing}")
    if len(accepted) != int(mapping_cfg["expected_accepted"]):
        failures.append(f"accepted count={len(accepted)}")
    if len(unresolved) != int(mapping_cfg["expected_unresolved"]):
        failures.append(f"unresolved count={len(unresolved)}")
    if len(conflict) != int(mapping_cfg["expected_conflict_rows"]):
        failures.append(f"conflict rows={len(conflict)}")
    for key in ("png_key", "phase_uid", "frame_list_hash", "image_sha256", "mask_sha256"):
        if key in accepted and accepted[key].duplicated().any():
            failures.append(f"accepted duplicate {key}")
    if "mapping_status" in accepted and not accepted["mapping_status"].eq("accepted").all():
        failures.append("accepted mapping contains non-accepted status")
    if "orientation_transform" in accepted and not accepted["orientation_transform"].eq("identity").all():
        failures.append("accepted mapping contains non-identity orientation")
    methods = accepted["mapping_method"].astype(str)
    expected_methods = {
        str(key): int(value)
        for key, value in mapping_cfg["expected_mapping_method_counts"].items()
    }
    if methods.value_counts().to_dict() != expected_methods:
        failures.append(
            f"mapping method counts={methods.value_counts().to_dict()}"
        )
    manual = methods.eq("manual_visual_identity_confirmed")
    if accepted.loc[manual, "manual_confirmation_note"].astype(str).str.strip().eq("").any():
        failures.append("manual mapping lacks confirmation note")
    correlations = pd.to_numeric(
        accepted.get("identity_pearson_correlation"), errors="coerce"
    )
    automated = ~manual
    if correlations.isna().any() or (
        correlations.loc[automated]
        < float(mapping_cfg["min_accepted_identity_correlation"])
    ).any():
        failures.append("automated mapping has invalid/below-threshold correlation")
    if not bool(source_summary.get("identity_only")):
        failures.append("mapping summary is not identity_only")

    allowed = {int(value) for value in annotation["allowed_mask_labels"]}
    inventory_rows: list[dict[str, Any]] = []
    for key in sorted(set(images) & set(masks)):
        image_path = images[key]
        mask_path = masks[key]
        try:
            image = read_gray(image_path)
            mask = read_gray(mask_path)
            labels, counts = np.unique(mask, return_counts=True)
            label_set = {int(value) for value in labels}
            if image.shape != mask.shape:
                raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")
            if not label_set.issubset(allowed):
                raise ValueError(f"unexpected labels: {sorted(label_set)}")
            if not np.count_nonzero(mask):
                raise ValueError("empty GT foreground")
            inventory_rows.append({
                "png_key": key,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "image_shape": f"{image.shape[0]}x{image.shape[1]}",
                "mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
                "mask_label_values": json.dumps(sorted(label_set), separators=(",", ":")),
                "mask_label_pixel_counts": json.dumps(
                    {str(int(v)): int(c) for v, c in zip(labels, counts)},
                    separators=(",", ":"),
                ),
                "mask_nonzero_pixels": int(np.count_nonzero(mask)),
            })
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")

    inventory = pd.DataFrame(inventory_rows)
    if len(inventory) == expected_pairs:
        accepted_checks = accepted.merge(
            inventory,
            on="png_key",
            how="left",
            validate="one_to_one",
            suffixes=("_mapping", "_audit"),
        )
        for key in ("image_sha256", "mask_sha256", "mask_shape", "mask_label_values"):
            left = accepted_checks[f"{key}_mapping"].astype(str)
            right = accepted_checks[f"{key}_audit"].astype(str)
            if not left.eq(right).all():
                failures.append(f"mapping/inventory mismatch: {key}")
    atomic_csv(inventory, manifests / "png_annotation_inventory.csv")

    threshold = float(mapping_cfg["weak_mapping_qa_threshold"])
    weak = accepted[correlations < threshold].copy()
    orientation_rows: list[dict[str, Any]] = []
    for row in weak.to_dict("records"):
        reference = read_gray(Path(row["image_path"]))
        mean = source_mean(row["frame_paths"])
        scores = {
            name: pearson(resize_intensity(transform(reference), mean.shape), mean)
            for name, transform in TRANSFORMS.items()
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        orientation_rows.append({
            "png_key": row["png_key"],
            "phase_uid": row["phase_uid"],
            "mapping_method": row["mapping_method"],
            "verification_class": (
                "manual_visual"
                if row["mapping_method"] == "manual_visual_identity_confirmed"
                else "automated_identity"
            ),
            "manual_confirmation_note": row.get("manual_confirmation_note", ""),
            "identity_score": scores["identity"],
            "best_transform": ranked[0][0],
            "best_score": ranked[0][1],
            "runner_up_transform": ranked[1][0],
            "runner_up_score": ranked[1][1],
            "best_minus_identity": ranked[0][1] - scores["identity"],
            "identity_minus_runner_up": scores["identity"] - ranked[1][1],
            "all_scores_json": json.dumps(scores, sort_keys=True, separators=(",", ":")),
        })
    orientation_audit = pd.DataFrame(orientation_rows)
    if len(orientation_audit):
        automated_weak = orientation_audit[
            orientation_audit["verification_class"] == "automated_identity"
        ]
        if not automated_weak["best_transform"].eq("identity").all():
            failures.append("weak automated mapping has non-identity best transform")
        if (pd.to_numeric(automated_weak["identity_score"]) < float(
            mapping_cfg["min_accepted_identity_correlation"]
        )).any():
            failures.append("weak automated identity score below threshold")
    atomic_csv(orientation_audit, reports / "01_weak_identity_orientation_audit.csv")

    input_hashes = {
        str(path): sha256_file(path)
        for path in (accepted_path, unresolved_path, conflict_path, summary_path)
    }
    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "annotation_source": "paired_png_2d_mean_and_gt_mask",
        "paired_png_keys": len(inventory),
        "accepted_mapping_rows": len(accepted),
        "unresolved_rows": len(unresolved),
        "conflict_rows": len(conflict),
        "allowed_mask_labels": sorted(allowed),
        "mask_label_set_counts": inventory["mask_label_values"].value_counts().to_dict(),
        "weak_mapping_threshold": threshold,
        "weak_mapping_rows": len(orientation_audit),
        "weak_identity_best_rows": int(
            orientation_audit["best_transform"].eq("identity").sum()
        ) if len(orientation_audit) else 0,
        "accepted_identity_correlation": {
            "min": float(correlations.min()),
            "median": float(correlations.median()),
            "max": float(correlations.max()),
        },
        "input_hashes": input_hashes,
        "segmentation_model_used": False,
    }
    atomic_json(summary, reports / "01_png2d_input_audit.json")
    atomic_json(summary, manifests / "png2d_input_lock.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

