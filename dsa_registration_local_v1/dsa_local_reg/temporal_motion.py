"""Native-local, frozen-peak frame-to-peak Rigid correction.

Transforms are estimated on structure images and applied, unchanged, to raw DSA
and real-source support.  This separation is intentional: no vesselness value is
ever used as a TDC intensity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .local_geometry import crop_with_border_median_padding, resize_whole_canvas
from .temporal_contract import FrozenPhaseContract
from .v5_adapter import load_v5_module


@dataclass
class LocalPhaseSequence:
    raw: np.ndarray
    crop_valid: np.ndarray
    lesion_mask: np.ndarray
    source_positions: np.ndarray
    source_frame_indices: np.ndarray
    frozen_peak_index: int


@dataclass
class CorrectedPhaseSequence:
    corrected_signal: np.ndarray
    corrected_valid: np.ndarray
    stable_valid: np.ndarray
    source_frame_indices: np.ndarray
    applied_transforms: list[Any]
    frame_metadata: list[dict[str, Any]]
    local: LocalPhaseSequence


def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read frozen DSA frame: {path}")
    return image


def _source_index(path: str) -> int:
    name = Path(path).stem
    digits = "".join(reversed("".join(reversed(name)).split("-")[0]))
    try:
        return int(digits)
    except ValueError as exc:
        raise ValueError(f"Cannot parse source frame index: {path}") from exc


def reconstruct_local_phase(contract: FrozenPhaseContract) -> LocalPhaseSequence:
    """Rebuild every native frame using exactly the G0 phase-specific bbox."""
    rec = contract.record
    crops = [crop_with_border_median_padding(_read_gray(path), rec.expanded_bbox) for path in rec.frame_paths]
    raw = np.stack([item.image for item in crops], axis=0).astype(np.uint8, copy=False)
    crop_valid = crops[0].valid_support.astype(bool, copy=False)
    if any(not np.array_equal(crop_valid, item.valid_support) for item in crops[1:]):
        raise AssertionError(f"{rec.phase_uid}: crop support changed across same-size native frames")
    mask_source = _read_gray(rec.mask_path)
    mask_native = resize_whole_canvas(mask_source, rec.canvas_shape_yx, is_mask=True) > 0
    lesion = crop_with_border_median_padding(mask_native.astype(np.uint8), rec.expanded_bbox).image > 0
    if lesion.shape != crop_valid.shape:
        raise AssertionError(f"{rec.phase_uid}: mask crop shape mismatch")
    positions = np.asarray(contract.selected_block_positions, dtype=int)
    if positions.size == 0 or contract.frozen_peak_index not in positions:
        raise AssertionError(f"{rec.phase_uid}: frozen peak not in acquisition block")
    return LocalPhaseSequence(
        raw=raw, crop_valid=crop_valid, lesion_mask=lesion, source_positions=positions,
        source_frame_indices=np.asarray(contract.frozen_source_indices, dtype=int),
        frozen_peak_index=int(contract.frozen_peak_index),
    )


def _ncc(fixed: np.ndarray, moving: np.ndarray, support: np.ndarray) -> float:
    a = np.asarray(fixed, dtype=np.float64)[support]
    b = np.asarray(moving, dtype=np.float64)[support]
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 16 or np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_structure_sequence(raw: np.ndarray, valid: np.ndarray, cfg: dict[str, Any], frame_workers: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pre = load_v5_module(cfg, "preprocessing.py")
    normal = pre.common_percentile_normalize(np.asarray(raw, dtype=np.float32))
    vessels = pre.sequence_vesselness(normal, workers=max(1, int(frame_workers)))
    structures = []
    scores = []
    for vessel in vessels:
        structure, _, _ = pre.build_structure_map(vessel, valid_mask=valid)
        structures.append(structure)
        values = vessel[valid & np.isfinite(vessel)]
        scores.append(float(np.mean(values)) if values.size else 0.0)
    diagnostic_best, diagnostic_combined, *_ = pre.choose_contrast_peak_index(
        normal, vessels, spatial_mask=valid
    )
    return np.stack(structures, axis=0).astype(np.float32), np.asarray(scores, dtype=np.float32), {
        "diagnostic_best_index_within_selected_block": int(diagnostic_best),
        "diagnostic_combined_scores": diagnostic_combined.astype(float).tolist(),
        "frozen_peak_overrode_diagnostic": True,
    }


def _identity_rigid(shape: tuple[int, int]):
    import SimpleITK as sitk
    tx = sitk.Euler2DTransform()
    tx.SetCenter(((shape[1] - 1.0) / 2.0, (shape[0] - 1.0) / 2.0))
    return tx


def estimate_reliable_rigids(structures: np.ndarray, scores: np.ndarray, peak_local_index: int,
                             valid: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    """Estimate only reliable frame→frozen-peak Rigid transforms."""
    sitk = load_v5_module(cfg, "registration_sitk.py")
    pars = cfg["jacobian_hemo"]["linear"]
    peak = structures[peak_local_index]
    maximum = max(float(np.nanmax(scores)), 1e-12)
    transforms: dict[int, Any] = {peak_local_index: _identity_rigid(peak.shape)}
    meta: dict[int, dict[str, Any]] = {
        peak_local_index: {"provenance": "peak_identity", "reliable": True, "ncc_before": 1.0, "ncc_after": 1.0,
                           "vessel_score": float(scores[peak_local_index]), "registration": {}}
    }
    for i in range(len(structures)):
        if i == peak_local_index:
            continue
        row: dict[str, Any] = {"vessel_score": float(scores[i]), "provenance": "unresolved", "reliable": False}
        if not np.isfinite(scores[i]) or float(scores[i]) / maximum < float(pars["min_vessel_score_ratio"]):
            row["reason"] = "low_vessel_score"
            meta[i] = row
            continue
        try:
            before = _ncc(peak, structures[i], valid)
            tx, registration = sitk.register_pair(
                peak, structures[i], kind="rigid", fixed_mask=valid, moving_mask=valid,
                metric=pars["metric"], shrink_factors=pars["shrink_factors"], smoothing_sigmas=pars["smoothing_sigmas"],
                learning_rate=pars["learning_rate"], min_step=pars["min_step"], iterations=pars["iterations"],
                gradient_tolerance=pars["gradient_tolerance"],
            )
            warped = sitk.resample(structures[i], peak, tx, default=0.0)
            after = _ncc(peak, warped, valid)
            row.update({"ncc_before": before, "ncc_after": after, "registration": registration})
            if not np.isfinite(after) or after < float(pars["min_frame_ncc_after"]):
                row["reason"] = "ncc_after_below_min"
            elif np.isfinite(before) and after < before - float(pars["max_frame_ncc_degradation"]):
                row["reason"] = "ncc_degraded"
            else:
                transforms[i] = tx
                row.update({"provenance": "estimated", "reliable": True})
        except Exception as exc:  # A data-local registration failure is explicitly retained for borrowing.
            row["reason"] = f"rigid_estimation:{type(exc).__name__}:{exc}"
        meta[i] = row
    return transforms, meta


def reject_trajectory_outliers(transforms: dict[int, Any], metadata: dict[int, dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    """Reject obvious estimated-trajectory spikes before interpolation/borrowing."""
    sitk = load_v5_module(cfg, "registration_sitk.py")
    pars = cfg["jacobian_hemo"]["linear"]
    estimated = sorted(i for i, item in metadata.items() if item.get("provenance") == "estimated" and i in transforms)
    if len(estimated) < 3:
        return transforms, metadata
    x = np.asarray(estimated, dtype=float)
    params = np.asarray([[sitk.canonical_parameters(transforms[i], "rigid")[key] for key in ("tx", "ty", "rotation_deg")] for i in estimated], dtype=float)
    residual = np.empty_like(params)
    for column in range(3):
        fit = np.polyfit(x, params[:, column], deg=1)
        residual[:, column] = params[:, column] - np.polyval(fit, x)
    for i, (rx, ry, rr) in zip(estimated, residual):
        if math.hypot(float(rx), float(ry)) > float(pars["max_trajectory_translation_residual"]) or abs(float(rr)) > float(pars["max_trajectory_rotation_residual_deg"]):
            transforms.pop(i, None)
            metadata[i].update({"reliable": False, "provenance": "rejected_trajectory", "reason": "trajectory_spike"})
    return transforms, metadata


def fill_missing_transforms(transforms: dict[int, Any], metadata: dict[int, dict[str, Any]], n_frames: int,
                            cfg: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    sitk = load_v5_module(cfg, "registration_sitk.py")
    if not transforms:
        raise RuntimeError("No reliable transforms available for interpolation/borrowing")
    available = sorted(transforms)
    full: list[Any] = []
    rows: list[dict[str, Any]] = []
    for i in range(n_frames):
        if i in transforms:
            full.append(transforms[i]); rows.append(metadata[i]); continue
        left = max((j for j in available if j < i), default=None)
        right = min((j for j in available if j > i), default=None)
        row = dict(metadata.get(i, {}))
        if left is not None and right is not None:
            alpha = (i - left) / float(right - left)
            tx = sitk.interpolate_rigid_transforms(transforms[left], transforms[right], alpha)
            row.update({"provenance": "interpolated", "left_reliable_frame": int(left), "right_reliable_frame": int(right), "alpha": float(alpha)})
        else:
            nearest = left if right is None else right if left is None else min((left, right), key=lambda j: abs(j - i))
            tx = transforms[nearest]
            row.update({"provenance": "borrowed", "borrowed_reliable_frame": int(nearest)})
        full.append(tx); rows.append(row)
    return full, rows


def _border_median(frame: np.ndarray) -> float:
    return float(np.median(np.concatenate((frame[0], frame[-1], frame[:, 0], frame[:, -1]))))


def apply_signal_and_support(raw: np.ndarray, crop_valid: np.ndarray, transforms: list[Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sitk = load_v5_module(cfg, "registration_sitk.py")
    if len(raw) != len(transforms):
        raise ValueError("signal/transform count mismatch")
    signal, valid = [], []
    fixed_shape = raw[0]
    for frame, tx in zip(raw, transforms):
        signal.append(sitk.resample(frame.astype(np.float32), fixed_shape, tx, is_mask=False, default=_border_median(frame)))
        valid.append(sitk.resample(crop_valid.astype(np.uint8), fixed_shape, tx, is_mask=True, default=0.0))
    return np.stack(signal, axis=0).astype(np.float32), np.stack(valid, axis=0).astype(bool)


def correct_phase_sequence(contract: FrozenPhaseContract, cfg: dict[str, Any], output_dir: str | Path,
                           *, frame_workers: int = 2) -> CorrectedPhaseSequence:
    local = reconstruct_local_phase(contract)
    positions = local.source_positions
    raw = local.raw[positions]
    peak_local = int(np.where(positions == local.frozen_peak_index)[0][0])
    structures, scores, diag = compute_structure_sequence(raw, local.crop_valid, cfg, frame_workers)
    estimated, metadata = estimate_reliable_rigids(structures, scores, peak_local, local.crop_valid, cfg)
    estimated, metadata = reject_trajectory_outliers(estimated, metadata, cfg)
    transforms, rows = fill_missing_transforms(estimated, metadata, len(raw), cfg)
    corrected, corrected_valid = apply_signal_and_support(raw, local.crop_valid, transforms, cfg)
    stable = np.all(corrected_valid, axis=0)
    if not np.any(stable):
        raise RuntimeError(f"{contract.phase_uid}: no stable real-source support after frame-to-peak correction")
    sitk = load_v5_module(cfg, "registration_sitk.py")
    phase_root = Path(output_dir)
    transforms_dir = phase_root / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    for pos, source, tx, row in zip(positions, local.source_frame_indices, transforms, rows):
        name = f"frame_{int(pos):03d}_source_{int(source):04d}_to_peak.tfm"
        sitk.save_transform(tx, transforms_dir / name)
        row.update({"manifest_frame_position": int(pos), "source_frame_index": int(source), "transform_path": str((transforms_dir / name).name),
                    "canonical_parameters": sitk.canonical_parameters(tx, "rigid")})
    import json
    payload = {
        "phase_uid": contract.phase_uid, "phase": contract.phase, "frozen_peak_index": contract.frozen_peak_index,
        "frozen_peak_score": contract.frozen_peak_score, "selected_manifest_positions": positions.astype(int).tolist(),
        "selected_source_frame_indices": local.source_frame_indices.astype(int).tolist(), "diagnostic": diag,
        "stable_valid_fraction": float(np.mean(stable)), "frames": rows,
    }
    (phase_root / "intra_registration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (phase_root / "local_sequence_contract.json").write_text(json.dumps({
        "phase_uid": contract.phase_uid, "frame_paths": list(contract.frame_paths), "frame_list_hash": contract.record.frame_list_hash,
        "n_frames": len(contract.frame_paths), "source_shape": list(contract.canvas_shape_yx),
        "expanded_bbox": contract.expanded_bbox.as_text(), "output_shape": list(local.crop_valid.shape),
        "padding": [contract.record.padding_left, contract.record.padding_top, contract.record.padding_right, contract.record.padding_bottom],
        "valid_fraction": float(np.mean(local.crop_valid)), "frozen_peak_index": contract.frozen_peak_index,
        "frozen_peak_score": contract.frozen_peak_score, "mask_path": str(contract.mask_path),
        "effective_mask_hash": contract.record.effective_mask_array_sha256,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CorrectedPhaseSequence(corrected, corrected_valid, stable, local.source_frame_indices, transforms, rows, local)
