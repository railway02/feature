#!/usr/bin/env python3
"""Conservatively map paired 2-D PNG annotations onto frozen DSA phases.

The PNG stem is the only identifier used to build candidates:
``<patient_id>_<Pre|Post>``.  Candidate selection starts in identity
orientation and compares the annotation-side image with the temporal mean of
each candidate phase.  Rotation/flip search is intentionally not automatic;
ambiguous or weak cases are written for manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


PHASE_BY_SUFFIX = {"pre": "pre", "post": "post"}


@dataclass(frozen=True)
class Candidate:
    patient_id: str
    split: str
    series_uid: str
    series_path: str
    phase: str
    api_dir: str
    frame_paths: tuple[str, ...]
    frame_list_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        frame.to_csv(temp, index=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp = Path(name)
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def parse_png_key(path: Path) -> tuple[str, str] | None:
    if path.suffix.casefold() != ".png" or "_" not in path.stem:
        return None
    patient_id, suffix = path.stem.rsplit("_", 1)
    phase = PHASE_BY_SUFFIX.get(suffix.casefold())
    if not patient_id or phase is None:
        return None
    return patient_id, phase


def series_path_alias(series_path: str) -> str:
    """Flatten the directory below tiantanDSA, e.g. 481684/L -> 481684L."""
    parts = Path(series_path).parts
    try:
        start = parts.index("tiantanDSA") + 1
    except ValueError:
        return ""
    return "".join(parts[start:])


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 2:
        raise ValueError(f"expected 2-D grayscale image, got shape={image.shape}")
    return image


def read_candidate_mean(paths: tuple[str, ...]) -> np.ndarray:
    accumulator: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    for path_text in paths:
        image = read_gray(Path(path_text))
        if shape is None:
            shape = image.shape
            accumulator = np.zeros(shape, dtype=np.float64)
        elif image.shape != shape:
            raise ValueError(f"mixed frame dimensions: expected={shape}, got={image.shape}, frame={path_text}")
        assert accumulator is not None
        accumulator += image
    if accumulator is None or not paths:
        raise ValueError("empty candidate frame list")
    return (accumulator / len(paths)).astype(np.float32)


def resize_for_score(image: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    if image.shape == target_shape:
        return image.astype(np.float32, copy=False), False
    interpolation = cv2.INTER_AREA if image.shape[0] > target_shape[0] or image.shape[1] > target_shape[1] else cv2.INTER_CUBIC
    resized = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=interpolation)
    return resized.astype(np.float32, copy=False), True


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def gradient_correlation(left: np.ndarray, right: np.ndarray) -> float:
    lx = cv2.Sobel(left, cv2.CV_32F, 1, 0)
    ly = cv2.Sobel(left, cv2.CV_32F, 0, 1)
    rx = cv2.Sobel(right, cv2.CV_32F, 1, 0)
    ry = cv2.Sobel(right, cv2.CV_32F, 0, 1)
    return pearson_correlation(cv2.magnitude(lx, ly), cv2.magnitude(rx, ry))


def build_candidates(manifest_paths: dict[str, Path]) -> dict[tuple[str, str], list[Candidate]]:
    required = {
        "patient_id", "split", "series_uid", "series_path",
        "pre_api_dir", "post_api_dir", "pre_frame_paths", "post_frame_paths",
        "pre_frame_list_hash", "post_frame_list_hash", "can_run_pre", "can_run_post",
    }
    by_key: dict[tuple[str, str], list[Candidate]] = {}
    for declared_split, path in manifest_paths.items():
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = sorted(required - set(source.columns))
        if missing:
            raise KeyError(f"{path} missing required columns: {missing}")
        if not source["split"].eq(declared_split).all():
            values = sorted(source.loc[~source["split"].eq(declared_split), "split"].unique())
            raise ValueError(f"{path} contains split values inconsistent with {declared_split}: {values}")
        for row in source.to_dict("records"):
            for phase in ("pre", "post"):
                if str(row[f"can_run_{phase}"]).strip().casefold() != "true":
                    continue
                paths = tuple(part for part in str(row[f"{phase}_frame_paths"]).split("|") if part)
                if not paths:
                    raise ValueError(f"empty {phase} frame list: {path}/{row['series_uid']}")
                candidate = Candidate(
                    patient_id=str(row["patient_id"]),
                    split=declared_split,
                    series_uid=str(row["series_uid"]),
                    series_path=str(row["series_path"]),
                    phase=phase,
                    api_dir=str(row[f"{phase}_api_dir"]),
                    frame_paths=paths,
                    frame_list_hash=str(row[f"{phase}_frame_list_hash"]),
                )
                by_key.setdefault((candidate.patient_id, phase), []).append(candidate)
                alias = series_path_alias(candidate.series_path)
                if alias and alias != candidate.patient_id:
                    # The 2-D export flattens nested DSA folders into its filename.
                    by_key.setdefault((alias, phase), []).append(candidate)
    for candidates in by_key.values():
        candidates.sort(key=lambda item: (item.split, item.series_uid))
    return by_key


def candidate_row(candidate: Candidate, *, rank: int, identity_corr: float, gradient_corr: float,
                  reference_shape: tuple[int, int], mean_shape: tuple[int, int], resized: bool) -> dict[str, Any]:
    return {
        "candidate_rank": rank,
        "phase_uid": f"{candidate.series_uid}::{candidate.phase}",
        "patient_id": candidate.patient_id,
        "split": candidate.split,
        "series_uid": candidate.series_uid,
        "series_path": candidate.series_path,
        "api_dir": candidate.api_dir,
        "phase": candidate.phase,
        "n_frames": len(candidate.frame_paths),
        "frame_paths": "|".join(candidate.frame_paths),
        "frame_list_hash": candidate.frame_list_hash,
        "reference_image_shape": f"{reference_shape[0]}x{reference_shape[1]}",
        "series_mean_shape": f"{mean_shape[0]}x{mean_shape[1]}",
        "reference_resized_for_score": int(resized),
        "identity_pearson_correlation": identity_corr,
        "identity_gradient_correlation": gradient_corr,
        "orientation_transform": "identity",
        "orientation_check": "identity_only",
    }


def annotation_fields(key: str, patient_id: str, phase: str, image_path: Path, mask_path: Path,
                      mask: np.ndarray) -> dict[str, Any]:
    labels, counts = np.unique(mask, return_counts=True)
    return {
        "png_patient_id": patient_id,
        "png_key": key,
        "patient_id": patient_id,
        "phase": phase,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "mask_sha256": sha256_file(mask_path),
        "mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
        "mask_label_values": json.dumps([int(value) for value in labels], separators=(",", ":")),
        "mask_label_pixel_counts": json.dumps(
            {str(int(value)): int(count) for value, count in zip(labels, counts)},
            separators=(",", ":"),
        ),
        "mask_nonzero_pixels": int(np.count_nonzero(mask)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-identity-correlation", type=float, default=0.70)
    parser.add_argument("--min-multi-candidate-margin", type=float, default=0.05)
    args = parser.parse_args()
    if not -1.0 <= args.min_identity_correlation <= 1.0:
        raise ValueError("--min-identity-correlation must be in [-1, 1]")
    if args.min_multi_candidate_margin < 0.0:
        raise ValueError("--min-multi-candidate-margin must be non-negative")

    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("--image-dir and --mask-dir must both exist")
    candidates_by_key = build_candidates({"Train": args.source_train, "Valid": args.source_valid})

    image_by_key: dict[str, Path] = {}
    invalid_images: list[dict[str, Any]] = []
    for path in sorted(image_dir.glob("*.png")):
        parsed = parse_png_key(path)
        if parsed is None:
            invalid_images.append({"png_key": path.stem, "image_path": str(path.resolve()), "reason": "invalid_png_filename"})
            continue
        if path.stem in image_by_key:
            invalid_images.append({"png_key": path.stem, "image_path": str(path.resolve()), "reason": "duplicate_image_png_key"})
            continue
        image_by_key[path.stem] = path

    mask_by_key: dict[str, Path] = {}
    invalid_masks: list[dict[str, Any]] = []
    for path in sorted(mask_dir.glob("*.png")):
        parsed = parse_png_key(path)
        if parsed is None:
            invalid_masks.append({"png_key": path.stem, "mask_path": str(path.resolve()), "reason": "invalid_png_filename"})
            continue
        if path.stem in mask_by_key:
            invalid_masks.append({"png_key": path.stem, "mask_path": str(path.resolve()), "reason": "duplicate_mask_png_key"})
            continue
        mask_by_key[path.stem] = path

    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = [*invalid_images, *invalid_masks]
    conflicts: list[dict[str, Any]] = []
    mean_cache: dict[str, np.ndarray] = {}
    active_patient_id = ""

    for key in sorted(set(image_by_key) | set(mask_by_key)):
        image_path = image_by_key.get(key)
        mask_path = mask_by_key.get(key)
        parsed = parse_png_key((image_path or mask_path))
        assert parsed is not None
        patient_id, phase = parsed
        if patient_id != active_patient_id:
            # Candidates never cross patients; retaining all means would use gigabytes.
            mean_cache.clear()
            active_patient_id = patient_id
        base = {
            "png_key": key,
            "patient_id": patient_id,
            "phase": phase,
            "image_path": str(image_path.resolve()) if image_path else "",
            "mask_path": str(mask_path.resolve()) if mask_path else "",
        }
        if image_path is None or mask_path is None:
            unresolved.append({
                **base,
                "reason": "missing_paired_image_or_mask",
                "missing_image": int(image_path is None),
                "missing_mask": int(mask_path is None),
            })
            continue
        try:
            reference = read_gray(image_path)
            mask = read_gray(mask_path)
        except Exception as exc:
            unresolved.append({**base, "reason": f"png_unreadable:{type(exc).__name__}:{exc}"})
            continue
        if reference.shape != mask.shape:
            unresolved.append({
                **base,
                "reason": "image_mask_shape_mismatch",
                "image_shape": f"{reference.shape[0]}x{reference.shape[1]}",
                "mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
            })
            continue
        candidates = candidates_by_key.get((patient_id, phase), [])
        if not candidates:
            unresolved.append({**base, "reason": "no_candidate_in_frozen_source_manifest"})
            continue

        scored: list[tuple[Candidate, dict[str, Any]]] = []
        candidate_error: str | None = None
        for candidate in candidates:
            try:
                # The same series_uid has distinct Pre/Post frame lists.
                cache_key = candidate.frame_list_hash
                if cache_key not in mean_cache:
                    mean_cache[cache_key] = read_candidate_mean(candidate.frame_paths)
                mean = mean_cache[cache_key]
                comparable, resized = resize_for_score(reference, mean.shape)
                identity_corr = pearson_correlation(comparable, mean)
                gradient_corr = gradient_correlation(comparable, mean)
                scored.append((candidate, candidate_row(
                    candidate,
                    rank=0,
                    identity_corr=identity_corr,
                    gradient_corr=gradient_corr,
                    reference_shape=reference.shape,
                    mean_shape=mean.shape,
                    resized=resized,
                )))
            except Exception as exc:
                candidate_error = f"candidate_mean_error:{candidate.series_uid}:{type(exc).__name__}:{exc}"
                break
        if candidate_error:
            unresolved.append({**base, "reason": candidate_error, "candidate_count": len(candidates)})
            continue

        scored.sort(key=lambda item: (-float(item[1]["identity_pearson_correlation"]), item[0].split, item[0].series_uid))
        ranked: list[dict[str, Any]] = []
        for rank, (_, row) in enumerate(scored, start=1):
            ranked.append({**row, "candidate_rank": rank})
        best = ranked[0]
        runner_score = float(ranked[1]["identity_pearson_correlation"]) if len(ranked) > 1 else ""
        margin = float(best["identity_pearson_correlation"]) - float(runner_score) if runner_score != "" else ""
        annotation = annotation_fields(key, patient_id, phase, image_path, mask_path, mask)
        common = {
            "mapping_key_type": "patient_id_exact" if str(best["patient_id"]) == patient_id else "series_path_alias",
            **annotation,
            "candidate_count": len(ranked),
            "best_identity_pearson_correlation": best["identity_pearson_correlation"],
            "runner_up_identity_pearson_correlation": runner_score,
            "identity_score_margin": margin,
            "orientation_transform": "identity",
            "orientation_check": "identity_only",
        }

        if len(ranked) == 1:
            if float(best["identity_pearson_correlation"]) < args.min_identity_correlation:
                unresolved.append({
                    **common,
                    **best,
                    "reason": "single_candidate_identity_correlation_below_threshold",
                    "min_identity_correlation": args.min_identity_correlation,
                })
                continue
            accepted.append({
                **common,
                **best,
                "mapping_method": "unique_patient_phase_identity_mean_verified",
                "mapping_status": "accepted",
            })
            continue

        clear_winner = (
            float(best["identity_pearson_correlation"]) >= args.min_identity_correlation
            and float(margin) >= args.min_multi_candidate_margin
        )
        if clear_winner:
            accepted.append({
                **common,
                **best,
                "mapping_method": "multi_candidate_identity_mean_correlation",
                "mapping_status": "accepted",
            })
            continue
        for row in ranked:
            conflicts.append({
                **common,
                **row,
                "reason": "multi_candidate_not_uniquely_resolved_in_identity_orientation",
                "min_identity_correlation": args.min_identity_correlation,
                "min_multi_candidate_margin": args.min_multi_candidate_margin,
                "manual_action": "confirm_series_or_test_orientation_for_this_png_only",
            })

    output_dir = args.output_dir.resolve()
    mapping_frame = pd.DataFrame(accepted)
    unresolved_frame = pd.DataFrame(unresolved)
    conflict_frame = pd.DataFrame(conflicts)
    if conflict_frame.empty:
        conflict_frame = pd.DataFrame(columns=[
            *mapping_frame.columns, "reason", "min_identity_correlation",
            "min_multi_candidate_margin", "manual_action",
        ])
    atomic_csv(mapping_frame, output_dir / "png_to_phase_mapping.csv")
    atomic_csv(unresolved_frame, output_dir / "unresolved_mapping.csv")
    atomic_csv(conflict_frame, output_dir / "conflict_mapping.csv")
    summary = {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "source_train": str(args.source_train.resolve()),
        "source_valid": str(args.source_valid.resolve()),
        "identity_only": True,
        "min_identity_correlation": args.min_identity_correlation,
        "min_multi_candidate_margin": args.min_multi_candidate_margin,
        "accepted": len(accepted),
        "unresolved": len(unresolved),
        "conflict_rows": len(conflicts),
        "conflict_png_keys": len({row["png_key"] for row in conflicts}),
        "paired_png_keys": len(set(image_by_key) & set(mask_by_key)),
    }
    atomic_json(summary, output_dir / "png_mapping_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
