#!/usr/bin/env python3
"""Prepare full-resolution, label-blind ROI Pilot visual-model inputs.

For the 30 Train Pilot lesions only, this script organizes every candidate
series and every frozen full-resolution frame, then creates 10-frame contact
sheets, GIF/MP4 playback, phase p1/p99 normalized caches, raw summaries, FOV,
global activity/TDC, provisional TOA/TTP maps, vesselness, and registration QC.
It does not run SEA-RAFT, infer lesion identity, create ROI annotations, or
calculate final local features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT = Path("/root/autodl-tmp/aneurysm")
EXPECTED_LESIONS = 30
EXPECTED_CANDIDATE_OCCURRENCES = 42
EXPECTED_FRAME_PATHS = 1788
FRAME_RE = re.compile(r"IMG-(\d+)-(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to outputs/api_fullseq_v3_roi_pilot_ai_cache.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--contact-frames", type=int, default=10)
    parser.add_argument("--media-max-side", type=int, default=768)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def frame_index(path: str) -> int:
    match = FRAME_RE.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"Frozen strict frame path has nonstandard filename: {path}")
    return int(match.group(2))


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(
        temp,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    os.replace(temp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    if not cv2.imwrite(str(temp), image):
        raise IOError(f"Failed to write image: {path}")
    os.replace(temp, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def sample_indices(count: int, target: int) -> list[int]:
    if count <= target:
        return list(range(count))
    return sorted(
        {round(position * (count - 1) / (target - 1)) for position in range(target)}
    )


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def normalize_stack(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    p1, p99 = np.percentile(raw, [1.0, 99.0])
    if not math.isfinite(float(p1)) or not math.isfinite(float(p99)) or p99 <= p1:
        p1 = float(raw.min())
        p99 = float(raw.max())
    if p99 <= p1:
        return np.zeros_like(raw, dtype=np.uint8), float(p1), float(p99)
    normalized = np.clip((raw.astype(np.float32) - p1) * 255.0 / (p99 - p1), 0, 255)
    return normalized.astype(np.uint8), float(p1), float(p99)


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    result = np.zeros_like(binary)
    result[labels == largest] = 255
    return result


def build_fov(normalized: np.ndarray) -> np.ndarray:
    median_image = np.median(normalized, axis=0).astype(np.uint8)
    border = np.concatenate(
        [
            median_image[0, :],
            median_image[-1, :],
            median_image[:, 0],
            median_image[:, -1],
        ]
    )
    threshold = max(2, int(np.percentile(border, 90)) + 1)
    mask = (median_image > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = largest_component(mask)
    if (mask > 0).mean() < 0.2:
        mask[:] = 255
    return mask


def provisional_vesselness(activity_map: np.ndarray, fov: np.ndarray) -> np.ndarray:
    source = activity_map.astype(np.float32) / 255.0
    best = np.zeros_like(source, dtype=np.float32)
    for sigma in (1.0, 2.0, 4.0):
        blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * sigma * sigma
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * sigma * sigma
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * sigma * sigma
        root = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy**2)
        lambda1 = 0.5 * (dxx + dyy + root)
        lambda2 = 0.5 * (dxx + dyy - root)
        swap = np.abs(lambda1) > np.abs(lambda2)
        small = np.where(swap, lambda2, lambda1)
        large = np.where(swap, lambda1, lambda2)
        rb = np.abs(small) / (np.abs(large) + 1e-6)
        scale = np.sqrt(small**2 + large**2)
        c_value = max(float(np.percentile(scale, 95)), 1e-4)
        response = np.exp(-(rb**2) / (2 * 0.5**2)) * (
            1.0 - np.exp(-(scale**2) / (2 * c_value**2))
        )
        response[large >= 0] = 0
        best = np.maximum(best, response)
    best[fov == 0] = 0
    high = float(np.percentile(best[fov > 0], 99)) if np.any(fov > 0) else 0.0
    if high <= 0:
        return np.zeros_like(activity_map, dtype=np.uint8)
    return np.clip(best * 255.0 / high, 0, 255).astype(np.uint8)


def build_contact_sheet(
    normalized: np.ndarray,
    indices: list[int],
    selected_positions: list[int],
) -> np.ndarray:
    thumbs: list[np.ndarray] = []
    for position in selected_positions:
        image = normalized[position]
        thumb = resize_max_side(image, 320)
        canvas = np.zeros((360, 340), dtype=np.uint8)
        top = 25 + (320 - thumb.shape[0]) // 2
        left = 10 + (320 - thumb.shape[1]) // 2
        canvas[top : top + thumb.shape[0], left : left + thumb.shape[1]] = thumb
        cv2.putText(
            canvas,
            f"frame={indices[position]}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            255,
            1,
            cv2.LINE_AA,
        )
        thumbs.append(canvas)
    columns = 5
    rows = math.ceil(len(thumbs) / columns)
    sheet = np.zeros((rows * 360, columns * 340), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet[row * 360 : (row + 1) * 360, column * 340 : (column + 1) * 340] = thumb
    return sheet


def write_gif(normalized: np.ndarray, path: Path, max_side: int) -> None:
    frames = [
        Image.fromarray(resize_max_side(frame, max_side), mode="L") for frame in normalized
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.gif")
    frames[0].save(
        temp,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    os.replace(temp, path)


def write_mp4(normalized: np.ndarray, path: Path, max_side: int) -> bool:
    first = resize_max_side(normalized[0], max_side)
    height, width = first.shape
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.mp4")
    writer = cv2.VideoWriter(
        str(temp), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height), False
    )
    if not writer.isOpened():
        return False
    try:
        for frame in normalized:
            resized = resize_max_side(frame, max_side)
            canvas = np.zeros((height, width), dtype=np.uint8)
            canvas[: resized.shape[0], : resized.shape[1]] = resized
            writer.write(canvas)
    finally:
        writer.release()
    if not temp.is_file() or temp.stat().st_size == 0:
        return False
    os.replace(temp, path)
    return True


def registration_qc(normalized: np.ndarray, indices: list[int]) -> pd.DataFrame:
    reference = normalized[0].astype(np.float32)
    window = cv2.createHanningWindow(
        (reference.shape[1], reference.shape[0]), cv2.CV_32F
    )
    rows: list[dict[str, Any]] = []
    for position, frame in enumerate(normalized):
        if position == 0:
            shift = (0.0, 0.0)
            response = 1.0
        else:
            shift, response = cv2.phaseCorrelate(
                reference * window, frame.astype(np.float32) * window
            )
        rows.append(
            {
                "frame_position": position,
                "frame_index": indices[position],
                "translation_dx_pixels": float(shift[0]),
                "translation_dy_pixels": float(shift[1]),
                "phase_correlation_response": float(response),
                "translation_magnitude_pixels": float(math.hypot(*shift)),
                "registration_qc_only": True,
            }
        )
    return pd.DataFrame(rows)


def raw_summary(raw: np.ndarray, paths: list[str], indices: list[int]) -> pd.DataFrame:
    rows = []
    for position, (image, path, index) in enumerate(zip(raw, paths, indices)):
        rows.append(
            {
                "frame_position": position,
                "frame_index": index,
                "original_path": path,
                "height": image.shape[0],
                "width": image.shape[1],
                "raw_min": int(image.min()),
                "raw_p1": float(np.percentile(image, 1)),
                "raw_median": float(np.median(image)),
                "raw_mean": float(image.mean()),
                "raw_std": float(image.std()),
                "raw_p99": float(np.percentile(image, 99)),
                "raw_max": int(image.max()),
            }
        )
    return pd.DataFrame(rows)


def load_sequence(paths: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    ordered = sorted(paths, key=lambda path: (frame_index(path), natural_key(path)))
    images: list[np.ndarray] = []
    indices: list[int] = []
    shape: tuple[int, int] | None = None
    for path in ordered:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Unreadable frozen frame: {path}")
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f"Mixed dimensions within internal series: {shape} vs {image.shape}")
        images.append(image)
        indices.append(frame_index(path))
    return np.stack(images, axis=0), indices, ordered


def process_internal_sequence(
    task: dict[str, Any], contact_frames: int, media_max_side: int, resume: bool
) -> dict[str, Any]:
    output_dir = Path(task["output_dir"])
    complete_path = output_dir / "cache_complete.json"
    path_hash = sha256_text("\n".join(task["paths"]))
    if resume and complete_path.is_file():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("frozen_path_list_sha256") == path_hash:
            return existing

    raw, indices, paths = load_sequence(task["paths"])
    normalized, p1, p99 = normalize_stack(raw)
    fov = build_fov(normalized)
    baseline_count = min(max(2, len(normalized) // 5), 3, len(normalized))
    baseline = np.median(normalized[:baseline_count], axis=0).astype(np.float32)
    activity = np.abs(normalized.astype(np.float32) - baseline[None, ...])
    activity[:, fov == 0] = 0
    max_activity = np.max(activity, axis=0)
    mean_activity = np.mean(activity, axis=0)
    max_scale = max(float(np.percentile(max_activity[fov > 0], 99)), 1.0)
    max_enhancement = np.clip(max_activity * 255.0 / max_scale, 0, 255).astype(np.uint8)
    mean_scale = max(float(np.percentile(mean_activity[fov > 0], 99)), 1.0)
    global_activity = np.clip(mean_activity * 255.0 / mean_scale, 0, 255).astype(np.uint8)

    ttp_positions = np.argmax(activity, axis=0).astype(np.uint16)
    threshold = np.maximum(8.0, max_activity * 0.25)
    crossings = activity >= threshold[None, ...]
    toa_positions = np.argmax(crossings, axis=0).astype(np.uint16)
    valid_activity = (max_activity >= 8.0) & (fov > 0)
    toa_positions[~valid_activity] = np.iinfo(np.uint16).max
    ttp_positions[~valid_activity] = np.iinfo(np.uint16).max

    denominator = max(len(normalized) - 1, 1)
    ttp_display = np.clip(ttp_positions.astype(np.float32) * 255 / denominator, 0, 255).astype(
        np.uint8
    )
    toa_display = np.clip(toa_positions.astype(np.float32) * 255 / denominator, 0, 255).astype(
        np.uint8
    )
    ttp_color = cv2.applyColorMap(ttp_display, cv2.COLORMAP_TURBO)
    toa_color = cv2.applyColorMap(toa_display, cv2.COLORMAP_TURBO)
    ttp_color[~valid_activity] = 0
    toa_color[~valid_activity] = 0
    vesselness = provisional_vesselness(max_enhancement, fov)
    vesselness_color = cv2.applyColorMap(vesselness, cv2.COLORMAP_INFERNO)
    vesselness_color[fov == 0] = 0

    fov_boolean = fov > 0
    tdc = activity[:, fov_boolean].mean(axis=1) if np.any(fov_boolean) else activity.mean(axis=(1, 2))
    tdc_frame = pd.DataFrame(
        {
            "frame_position": list(range(len(indices))),
            "frame_index": indices,
            "global_activity_mean_provisional": tdc.astype(float),
        }
    )
    summary = raw_summary(raw, paths, indices)
    summary["phase_p1"] = p1
    summary["phase_p99"] = p99
    summary["global_activity_mean_provisional"] = tdc.astype(float)
    registration = registration_qc(normalized, indices)
    selected_positions = sample_indices(len(indices), contact_frames)
    contact_sheet = build_contact_sheet(normalized, indices, selected_positions)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(summary, output_dir / "original_frames.csv")
    atomic_write_csv(tdc_frame, output_dir / "global_tdc_provisional.csv")
    atomic_write_csv(registration, output_dir / "registration_qc.csv")
    atomic_write_npz(
        output_dir / "phase_normalized_cache.npz",
        frames=normalized,
        frame_indices=np.asarray(indices, dtype=np.int32),
        original_paths=np.asarray(paths, dtype=np.str_),
        phase_p1=np.asarray([p1], dtype=np.float32),
        phase_p99=np.asarray([p99], dtype=np.float32),
    )
    atomic_write_npz(
        output_dir / "provisional_kinetic_maps.npz",
        toa_frame_position=toa_positions,
        ttp_frame_position=ttp_positions,
        max_activity=max_activity.astype(np.float32),
        mean_activity=mean_activity.astype(np.float32),
    )
    atomic_write_image(output_dir / "fov_mask.png", fov)
    atomic_write_image(output_dir / "contact_sheet_10frames.png", contact_sheet)
    atomic_write_image(output_dir / "max_enhancement_provisional.png", max_enhancement)
    atomic_write_image(output_dir / "global_activity_map_provisional.png", global_activity)
    atomic_write_image(output_dir / "toa_map_provisional.png", toa_color)
    atomic_write_image(output_dir / "ttp_map_provisional.png", ttp_color)
    atomic_write_image(output_dir / "vesselness_provisional.png", vesselness_color)
    write_gif(normalized, output_dir / "sequence_preview.gif", media_max_side)
    mp4_written = write_mp4(normalized, output_dir / "sequence_preview.mp4", media_max_side)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(indices, tdc, marker="o", markersize=2, linewidth=1)
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Global activity (provisional)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    temp_plot = output_dir / f".global_tdc.{uuid.uuid4().hex}.png"
    figure.savefig(temp_plot, dpi=140)
    plt.close(figure)
    os.replace(temp_plot, output_dir / "global_tdc_provisional.png")

    registration_magnitude = registration["translation_magnitude_pixels"].to_numpy(float)
    registration_response = registration["phase_correlation_response"].to_numpy(float)
    result = {
        "lesion_uid": task["lesion_uid"],
        "candidate_uid": task["candidate_uid"],
        "candidate_index": task["candidate_index"],
        "candidate_series_id": task["candidate_series_id"],
        "candidate_series_path": task["candidate_series_path"],
        "phase": task["phase"],
        "internal_series": task["internal_series"],
        "n_frames": len(paths),
        "n_contiguous_pairs": sum(
            right - left == 1 for left, right in zip(indices, indices[1:])
        ),
        "frame_indices": indices,
        "frozen_path_list_sha256": path_hash,
        "height": raw.shape[1],
        "width": raw.shape[2],
        "phase_p1": p1,
        "phase_p99": p99,
        "contact_frame_positions": selected_positions,
        "contact_frame_indices": [indices[position] for position in selected_positions],
        "mp4_written": mp4_written,
        "media_playback_fps_nonphysical": 10.0,
        "registration_median_translation_pixels": float(np.median(registration_magnitude)),
        "registration_max_translation_pixels": float(np.max(registration_magnitude)),
        "registration_low_response_frames": int((registration_response < 0.05).sum()),
        "timing_seconds_available": False,
        "formal_local_features_generated": False,
        "sea_raft_run": False,
        "output_dir": str(output_dir),
    }
    atomic_write_text(
        json.dumps(result, ensure_ascii=False, indent=2), complete_path
    )
    return result


def candidate_uid(lesion_uid: str, candidate: dict[str, Any]) -> str:
    source = candidate["candidate_source"]
    material = "|".join(
        [
            lesion_uid,
            str(source.get("discovery_rank", "")),
            str(source.get("series_id", "")),
            str(source.get("series_path", "")),
        ]
    )
    return f"candidate_{sha256_text(material)[:20]}"


def build_tasks(root: Path, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot = pd.read_csv(
        root / "manifests/api_fullseq_v3_roi_pilot_train.csv",
        dtype=str,
        keep_default_na=False,
    )
    registry = pd.read_csv(
        root / "metadata/api_fullseq_v3/lesion_registry_train_blinded.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index("lesion_uid")
    if len(pilot) != EXPECTED_LESIONS or not pilot["lesion_uid"].is_unique:
        raise AssertionError("Expected 30 unique Train ROI Pilot lesion UIDs")
    tasks: list[dict[str, Any]] = []
    lesion_packages: list[dict[str, Any]] = []
    all_paths: list[str] = []
    candidate_occurrences = 0

    for pilot_row in pilot.itertuples(index=False):
        lesion_uid = pilot_row.lesion_uid
        if lesion_uid not in registry.index:
            raise AssertionError(f"Pilot lesion not found in Train blinded registry: {lesion_uid}")
        registry_row = registry.loc[lesion_uid]
        payload = json.loads(registry_row["candidate_series_registry_json"])
        lesion_entry = {
            "lesion_uid": lesion_uid,
            "patient_id": pilot_row.patient_id,
            "side_raw": pilot_row.side_raw,
            "side_normalized": pilot_row.side_normalized,
            "location_raw": pilot_row.location_raw,
            "location_normalized": pilot_row.location_normalized,
            "lesion_index_normalized": pilot_row.lesion_index_normalized,
            "candidate_count": len(payload["candidates"]),
            "candidates": [],
        }
        for candidate_index, candidate in enumerate(payload["candidates"], start=1):
            candidate_occurrences += 1
            uid = candidate_uid(lesion_uid, candidate)
            source = candidate["candidate_source"]
            audit = candidate["candidate_audit"]
            candidate_entry = {
                "candidate_uid": uid,
                "candidate_index": candidate_index,
                "candidate_source": source,
                "candidate_audit": audit,
                "phases": {},
                "v2_pairdata_reference": {},
            }
            if audit.get("selected_candidate_in_v2"):
                for phase in ("pre", "post"):
                    reference = root / "outputs/api_fullseq_v2_pairdata/full/train" / str(
                        pilot_row.patient_id
                    ) / phase
                    candidate_entry["v2_pairdata_reference"][phase] = (
                        str(reference) if reference.is_dir() else ""
                    )
            for phase in ("pre", "post"):
                phase_entries: list[dict[str, Any]] = []
                paths_by_internal = candidate[phase].get(
                    "strict_frame_paths_by_internal_series", {}
                )
                for internal_series, paths in sorted(
                    paths_by_internal.items(), key=lambda item: natural_key(str(item[0]))
                ):
                    if not paths:
                        continue
                    all_paths.extend(paths)
                    phase_output = (
                        output_root
                        / lesion_uid
                        / uid
                        / phase
                        / f"internal_{internal_series}"
                    )
                    task = {
                        "lesion_uid": lesion_uid,
                        "candidate_uid": uid,
                        "candidate_index": candidate_index,
                        "candidate_series_id": source.get("series_id", ""),
                        "candidate_series_path": source.get("series_path", ""),
                        "phase": phase,
                        "internal_series": str(internal_series),
                        "paths": list(paths),
                        "output_dir": str(phase_output),
                    }
                    tasks.append(task)
                    phase_entries.append(
                        {
                            "internal_series": str(internal_series),
                            "n_frames": len(paths),
                            "n_contiguous_pairs": sum(
                                right - left == 1
                                for left, right in zip(
                                    sorted(frame_index(path) for path in paths),
                                    sorted(frame_index(path) for path in paths)[1:],
                                )
                            ),
                            "selected_internal_series_in_v2": str(
                                candidate[phase].get("selected_internal_series_in_v2", "")
                            )
                            == str(internal_series),
                            "cache_dir": str(phase_output),
                        }
                    )
                candidate_entry["phases"][phase] = phase_entries
            lesion_entry["candidates"].append(candidate_entry)
        lesion_packages.append(lesion_entry)

    if candidate_occurrences != EXPECTED_CANDIDATE_OCCURRENCES:
        raise AssertionError(
            f"Expected {EXPECTED_CANDIDATE_OCCURRENCES} candidate occurrences, got {candidate_occurrences}"
        )
    if len(all_paths) != EXPECTED_FRAME_PATHS or len(set(all_paths)) != EXPECTED_FRAME_PATHS:
        raise AssertionError(
            f"Expected {EXPECTED_FRAME_PATHS} unique Pilot frozen frame paths, got {len(set(all_paths))}"
        )
    return tasks, lesion_packages


def main() -> int:
    args = parse_args()
    if args.workers < 1 or not 8 <= args.contact_frames <= 12:
        raise ValueError("--workers must be positive and --contact-frames must be 8..12")
    root = args.project_root.resolve()
    output_root = (
        args.output_root or root / "outputs/api_fullseq_v3_roi_pilot_ai_cache"
    ).resolve()
    tasks, lesion_packages = build_tasks(root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        json.dumps(
            {
                "pilot_lesions": len(lesion_packages),
                "internal_sequence_tasks": len(tasks),
                "lesions": lesion_packages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        output_root / "pilot_visual_input_manifest.json",
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_internal_sequence,
                task,
                args.contact_frames,
                args.media_max_side,
                args.resume,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "precache_completed": completed,
                        "precache_total": len(tasks),
                        "lesion_uid": result["lesion_uid"],
                        "candidate_uid": result["candidate_uid"],
                        "phase": result["phase"],
                        "internal_series": result["internal_series"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results.sort(
        key=lambda row: (
            natural_key(row["lesion_uid"]),
            row["candidate_index"],
            row["phase"],
            natural_key(row["internal_series"]),
        )
    )
    manifest = pd.DataFrame(results)
    atomic_write_csv(manifest, output_root / "cache_manifest.csv")
    audit_lines = [
        "# api_fullseq_v3 ROI Pilot AI Precache Audit",
        "",
        f"- Pilot lesions: **{len(lesion_packages)}**",
        f"- Candidate occurrences: **{EXPECTED_CANDIDATE_OCCURRENCES}**",
        f"- Frozen full-resolution frame paths: **{int(manifest['n_frames'].sum())}**",
        f"- Internal phase sequences: **{len(manifest)}**",
        f"- GIF files: **{sum((Path(path) / 'sequence_preview.gif').is_file() for path in manifest['output_dir'])}**",
        f"- MP4 files: **{sum((Path(path) / 'sequence_preview.mp4').is_file() for path in manifest['output_dir'])}**",
        "- Contact sheets use 8-12 uniformly sampled frames; all original paths and full-resolution normalized stacks are retained separately.",
        "- TOA/TTP, vesselness, global activity/TDC, and registration outputs are provisional QC/cache artifacts only.",
        "- No private labels, SEA-RAFT execution, ROI annotations, or formal local features were used/generated.",
        "",
    ]
    report_path = root / "reports/api_fullseq_v3/roi_pilot_ai_precache_audit.md"
    atomic_write_text("\n".join(audit_lines), report_path)
    print(
        json.dumps(
            {
                "pilot_lesions": len(lesion_packages),
                "candidate_occurrences": EXPECTED_CANDIDATE_OCCURRENCES,
                "full_resolution_frame_paths": int(manifest["n_frames"].sum()),
                "internal_sequences": len(manifest),
                "output_root": str(output_root),
                "audit": str(report_path),
                "private_labels_read": False,
                "sea_raft_run": False,
                "formal_local_features_generated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
