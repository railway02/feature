from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .local_geometry import BBox, crop_with_border_median_padding, resize_whole_canvas, scale_bbox
from .preprocessing_adapter import PairRecord, PhaseRecord


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"Unreadable grayscale image: {path}")
    return image


def _normalise(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, [1, 99])
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _native_mask(phase: PhaseRecord) -> np.ndarray:
    mask = read_gray(phase.mask_path)
    return resize_whole_canvas(mask, phase.canvas_shape_yx, is_mask=True) > 0


def _overlay(image: np.ndarray, mask: np.ndarray, box: BBox) -> np.ndarray:
    base = _normalise(image)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    rgb[edge] = np.asarray([1.0, 0.0, 0.0])
    cv2.rectangle(rgb, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), (0.0, 1.0, 0.0), 2)
    return rgb


def _phase_raw_frame(phase: PhaseRecord, frame_position: int) -> np.ndarray:
    if frame_position < 0 or frame_position >= len(phase.frame_paths):
        raise IndexError(f"{phase.phase_uid}: frame position {frame_position} out of range")
    image = read_gray(phase.frame_paths[frame_position])
    if image.shape != phase.canvas_shape_yx:
        raise AssertionError(f"{phase.phase_uid}: raw frame {image.shape} != manifest {phase.canvas_shape_yx}")
    return image


def select_geometry_cases(records: Iterable[PairRecord], n_cases: int = 10) -> list[PairRecord]:
    """Deterministic Train-only selection using geometry, never outcomes or registration QC."""
    candidates = sorted(records, key=lambda record: record.series_uid)
    if len(candidates) < n_cases:
        raise ValueError(f"Need {n_cases} cases, have {len(candidates)}")
    by_patient: dict[str, int] = {}
    for record in candidates:
        by_patient[record.patient_id] = by_patient.get(record.patient_id, 0) + 1

    def max_side(record: PairRecord) -> int:
        return max(record.pre.roi_side, record.post.roi_side)

    def first(predicate):
        return next((record for record in candidates if predicate(record)), None)

    desired = [
        first(lambda r: r.pre.canvas_shape_yx != r.post.canvas_shape_yx),
        first(lambda r: r.pre.canvas_shape_yx == r.post.canvas_shape_yx),
        first(lambda r: r.pre.mask_resized_to_frame or r.post.mask_resized_to_frame),
        first(lambda r: "manual_visual" in r.pre.mapping_method or "manual_visual" in r.post.mapping_method),
        first(lambda r: any((r.pre.padding_left, r.pre.padding_top, r.pre.padding_right, r.pre.padding_bottom,
                             r.post.padding_left, r.post.padding_top, r.post.padding_right, r.post.padding_bottom))),
        first(lambda r: by_patient[r.patient_id] > 1),
    ]
    sides = np.asarray([max_side(record) for record in candidates], dtype=float)
    for quantile in (0.0, 0.5, 0.9, 1.0):
        target = float(np.quantile(sides, quantile))
        desired.append(min(candidates, key=lambda r: (abs(max_side(r) - target), r.series_uid)))

    selected: list[PairRecord] = []
    for record in desired:
        if record is not None and record.series_uid not in {item.series_uid for item in selected}:
            selected.append(record)
    if len(selected) < n_cases:
        positions = np.linspace(0, len(candidates) - 1, num=n_cases, dtype=int)
        for position in positions:
            record = candidates[int(position)]
            if record.series_uid not in {item.series_uid for item in selected}:
                selected.append(record)
            if len(selected) == n_cases:
                break
    if len(selected) != n_cases:
        raise AssertionError(f"Could not select {n_cases} unique geometry cases")
    return selected


def geometry_selection_table(records: Iterable[PairRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        pre, post = record.pre, record.post
        rows.append({
            "series_uid": record.series_uid, "patient_id": record.patient_id, "split": record.split,
            "pre_canvas": f"{pre.frame_height}x{pre.frame_width}",
            "post_canvas": f"{post.frame_height}x{post.frame_width}",
            "same_raw_canvas": pre.canvas_shape_yx == post.canvas_shape_yx,
            "pre_expanded_bbox": pre.expanded_bbox.as_text(), "post_expanded_bbox": post.expanded_bbox.as_text(),
            "pre_local_size": f"{pre.expanded_bbox.height}x{pre.expanded_bbox.width}",
            "post_local_size": f"{post.expanded_bbox.height}x{post.expanded_bbox.width}",
            "same_local_matrix_size": (pre.expanded_bbox.height, pre.expanded_bbox.width) == (post.expanded_bbox.height, post.expanded_bbox.width),
            "pre_mask_resized_to_frame": pre.mask_resized_to_frame,
            "post_mask_resized_to_frame": post.mask_resized_to_frame,
            "pre_mapping_method": pre.mapping_method, "post_mapping_method": post.mapping_method,
            "pre_source_spacing_xy": "|".join(f"{value:.8g}" for value in pre.source_spacing_xy),
            "post_source_spacing_xy": "|".join(f"{value:.8g}" for value in post.source_spacing_xy),
        })
    return pd.DataFrame(rows)


def make_independent_local_crop_sheet(
    record: PairRecord, output_path: str | Path, *, frame_position: int, g1_canvas_yx: tuple[int, int], dpi: int = 160
) -> dict[str, object]:
    """Create Stage B evidence: G0 independent local crops plus G1 audit-only preview."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phases = (record.pre, record.post)
    raw = [_phase_raw_frame(phase, frame_position) for phase in phases]
    masks = [_native_mask(phase) for phase in phases]
    g0 = [crop_with_border_median_padding(image, phase.expanded_bbox) for image, phase in zip(raw, phases)]
    g1_canvas = [resize_whole_canvas(image, g1_canvas_yx) for image in raw]
    g1_mask = [resize_whole_canvas(mask.astype(np.uint8), g1_canvas_yx, is_mask=True) > 0 for mask in masks]
    g1_boxes = [scale_bbox(phase.expanded_bbox, phase.canvas_shape_yx, g1_canvas_yx) for phase in phases]
    g1 = [crop_with_border_median_padding(image, box) for image, box in zip(g1_canvas, g1_boxes)]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for row, (name, phase, image, mask, g0_crop, g1_image, g1_binary, g1_box, g1_crop) in enumerate(
        zip(("Pre moving", "Post fixed"), phases, raw, masks, g0, g1_canvas, g1_mask, g1_boxes, g1)
    ):
        axes[row, 0].imshow(_overlay(image, mask, phase.expanded_bbox))
        axes[row, 0].set_title(f"{name}: native canvas + old bbox\n{image.shape[1]}×{image.shape[0]}")
        axes[row, 1].imshow(_normalise(g0_crop.image), cmap="gray")
        axes[row, 1].set_title(f"G0 primary independent crop\n{g0_crop.image.shape[1]}×{g0_crop.image.shape[0]}")
        axes[row, 2].imshow(_overlay(g1_image, g1_binary, g1_box))
        axes[row, 2].set_title(f"G1 audit-only whole-canvas preview\n{g1_image.shape[1]}×{g1_image.shape[0]}")
        axes[row, 3].imshow(_normalise(g1_crop.image), cmap="gray")
        axes[row, 3].set_title(f"G1 independent crop (audit only)\n{g1_crop.image.shape[1]}×{g1_crop.image.shape[0]}")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle(
        f"{record.series_uid} | fixed=Post, moving=Pre | independent phase-specific old expanded bboxes",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "series_uid": record.series_uid,
        "contact_sheet": str(output_path),
        "pre_g0_shape": list(g0[0].image.shape), "post_g0_shape": list(g0[1].image.shape),
        "pre_g1_shape": list(g1[0].image.shape), "post_g1_shape": list(g1[1].image.shape),
        "pre_g1_bbox": g1_boxes[0].as_text(), "post_g1_bbox": g1_boxes[1].as_text(),
        "pre_padding": [g0[0].padding_left, g0[0].padding_top, g0[0].padding_right, g0[0].padding_bottom],
        "post_padding": [g0[1].padding_left, g0[1].padding_top, g0[1].padding_right, g0[1].padding_bottom],
    }
