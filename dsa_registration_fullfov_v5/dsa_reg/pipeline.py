from __future__ import annotations
import json
import gc
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation

from .manifest import apply_path_remap, parse_frame_paths, frame_number, frozen_candidate_numbers
from .io import (
    read_mask, read_gray, crop_fixed_2d, crop_valid_mask, resize_image_to_canvas,
    read_sequence_canvas, joint_mask_center, crop_origin_yx, mask_center,
)
from .preprocessing import (
    common_percentile_normalize, sequence_vesselness,
    choose_contrast_peak_index, build_structure_map, vesselness, make_peak_search_mask,
    central_fov_support, temporal_vascular_aggregate,
    centerline_similarity_map, suppress_linear_border_artifacts,
)
from .regions import build_anchor, measurement_regions
from .registration_sitk import (
    register_pair, resample, save_transform, canonical_parameters, interpolate_rigid_transforms,
)
from .global_correspondence import (
    coarse_similarity_candidates, compatibility_metrics, broad_fixed_metric_support,
    global_compatibility_status,
)
from .registration_ants import run_syn_residual, warp_mask_to_fixed
from .hemodynamics import time_density_curve, curve_features, normalized_phase_features, delta_features
from .features import morphology_features, deformation_features
from .qc import (
    masked_ncc, vascular_ncc, dice, symmetric_chamfer, initial_qreg, registration_validity,
    save_overlay, save_heatmap, save_peak_selection_plot, save_displacement_map,
)
from .utils import ensure_dir, json_dump, sanitize_key


def _candidate_positions(row, phase: str, frame_paths, cfg):
    if not cfg["sequence"].get("use_frozen_contrast_candidates", True):
        return list(range(len(frame_paths)))
    col = f"{phase}_frozen_temporal_blocks_json"
    if col not in row.index:
        return list(range(len(frame_paths)))
    nums = set(frozen_candidate_numbers(row[col], cfg["sequence"].get("candidate_view_name", "contrast_core20")))
    positions = [i for i, p in enumerate(frame_paths) if frame_number(p) in nums]
    return positions if positions else list(range(len(frame_paths)))


def resolve_expansion_tau(cfg: dict) -> tuple[float, str]:
    fcfg = cfg["features"]
    artifact = fcfg.get("expansion_tau_artifact")
    if artifact:
        p = Path(str(artifact))
        obj = json.loads(p.read_text())
        if str(obj.get("split")) != "Train":
            raise ValueError(f"tau artifact must be calibrated on Train, got {obj.get('split')!r}")
        tau = float(obj["tau"])
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError(f"Invalid tau in {p}: {tau}")
        return tau, str(p)
    tau = float(fcfg["expansion_tau_fallback"])
    return tau, "exploratory_fallback"


def _structure_ncc(a, b, valid_mask):
    support = np.asarray(valid_mask, dtype=bool)
    return masked_ncc(a, b, support)


def identity_mapping_is_independently_verified(mapping_score, mapping_method, cfg) -> bool:
    """Whether identity reference→temporal geometry has independent manifest evidence.

    The reference image/mask may be used without a newly accepted geometric transform
    only when the versioned manifest explicitly verifies an identity mapping.  Scores
    alone are not sufficient: the method must identify an identity or manual visual
    verification.  This prevents a failed reference→peak registration from silently
    becoming an unconditional identity fallback.
    """
    method = str(mapping_method).lower()
    geometry = cfg.get("geometry", {})
    tokens = [str(x).lower() for x in geometry.get(
        "identity_mapping_method_tokens", ["identity", "manual_visual"]
    )]
    method_ok = any(token and token in method for token in tokens)
    if not method_ok:
        return False
    if "manual" in method:
        return bool(geometry.get("allow_manual_identity_mapping", True))
    try:
        return bool(np.isfinite(float(mapping_score)) and float(mapping_score) >= float(
            geometry.get("min_mapping_score", 0.75)
        ))
    except (TypeError, ValueError):
        return False


def _direct_reference_to_peak(reference_crop, mask_native, fixed_struct, valid_mask, cfg, phase_dir: Path,
                              mapping_score=np.nan, mapping_method=""):
    """Register the 2-D reference image directly to the selected peak frame.

    The supplied manifest verifies the reference image belongs to the same patient/phase,
    but it does not prove that the reference PNG is one particular temporal frame. The
    previous implementation guessed a temporal frame by NCC. That assumption is removed.
    """
    pre = cfg["preprocess"]
    icfg = cfg["intra_registration"]
    ref01 = common_percentile_normalize(reference_crop[None, ...], pre["percentile_low"], pre["percentile_high"])[0]
    rv = vesselness(ref01, tuple(pre["frangi_sigmas"]))
    ref_struct, _, _ = build_structure_map(
        rv, pre["vessel_threshold_percentile"], pre["vessel_min_size"], pre["centerline_sigma_px"],
        valid_mask=valid_mask,
    )
    before = _structure_ncc(ref_struct, fixed_struct, valid_mask)
    accepted = False
    tx = None
    after = before
    error = None
    reference_kind = str(icfg.get("reference_transform", "similarity"))
    try:
        tx_try, meta = register_pair(
            fixed_struct, ref_struct, kind=reference_kind, metric=icfg["metric"],
            shrink_factors=icfg["shrink_factors"], smoothing_sigmas=icfg["smoothing_sigmas"],
            learning_rate=icfg["learning_rate"], min_step=icfg["min_step"],
            iterations=icfg["iterations"], gradient_tolerance=icfg["gradient_tolerance"],
        )
        warped = resample(ref_struct, fixed_struct, tx_try, is_mask=False)
        after = _structure_ncc(warped, fixed_struct, valid_mask)
        min_final = float(icfg.get("reference_min_ncc", 0.15))
        min_gain = float(icfg.get("reference_min_ncc_gain", 0.01))
        # If identity is already excellent, do not move the gold-standard mask unnecessarily.
        if np.isfinite(after) and after >= min_final and (
            not np.isfinite(before) or after >= before + min_gain
        ):
            accepted = True
            tx = tx_try
            save_transform(tx, phase_dir / "transforms" / "reference_to_peak.tfm")
        else:
            meta = {**meta, "rejected": True}
    except Exception as e:
        meta = {}
        error = repr(e)

    identity_evidence = identity_mapping_is_independently_verified(mapping_score, mapping_method, cfg)
    if accepted:
        mask_peak = resample(mask_native.astype(np.uint8), fixed_struct, tx, is_mask=True)
        params = canonical_parameters(tx, reference_kind)
        phase_geometry_valid = True
        geometry_source = "reference_to_peak_registration"
    elif identity_evidence:
        # This is an explicitly evidenced identity mapping, not a fallback triggered by
        # registration failure.  Preserve it as an auditable separate path.
        mask_peak = mask_native.astype(bool)
        params = {"rotation_rad": 0.0, "rotation_deg": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0}
        phase_geometry_valid = True
        geometry_source = "manifest_identity_evidence"
    else:
        # Do not return a lesion mask that could leak into TDC, morphology, SyN's lesion
        # metric, or Jacobian regions.  The caller must fail this series closed.
        mask_peak = None
        params = {"rotation_rad": 0.0, "rotation_deg": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0}
        phase_geometry_valid = False
        geometry_source = "invalid_no_independent_identity_evidence"

    return mask_peak, {
        "accepted_transform": bool(accepted),
        "transform_kind": reference_kind,
        "ncc_before": float(before) if np.isfinite(before) else np.nan,
        "ncc_after": float(after) if np.isfinite(after) else np.nan,
        "error": error,
        "identity_evidence": bool(identity_evidence),
        "phase_geometry_valid": bool(phase_geometry_valid),
        "geometry_source": geometry_source,
        **params,
    }


def _intra_correct(seq_raw, mask_native, reference_crop, valid_mask, candidate_positions, cfg, phase_dir: Path,
                   mapping_score=np.nan, mapping_method=""):
    pre = cfg["preprocess"]
    seq01 = common_percentile_normalize(seq_raw, pre["percentile_low"], pre["percentile_high"])
    vseq = sequence_vesselness(
        seq01, tuple(pre["frangi_sigmas"]), workers=pre.get("vesselness_workers", 1)
    )
    # Peak selection must not depend on an as-yet unverified reference-mask mapping.
    # Use the central whole-FOV acquisition support instead of a lesion-centred window.
    peak_spatial_mask = central_fov_support(
        valid_mask, int(cfg["sequence"].get("full_fov_border_px", 24))
    )
    (peak_i, peak_scores, contrast_peak_scores, vessel_peak_scores,
     contrast_peak_scores_norm, vessel_peak_scores_norm) = choose_contrast_peak_index(
        seq01, vseq, candidate_positions, cfg["sequence"]["peak_score_top_fraction"],
        spatial_mask=peak_spatial_mask,
        baseline_n_frames=cfg["hemodynamics"]["baseline_n_frames"],
        vessel_weight=cfg["sequence"].get("peak_vessel_weight", 0.25),
    )

    structures, vessels, skeletons = [], [], []
    for v in vseq:
        s, m, sk = build_structure_map(
            v, pre["vessel_threshold_percentile"], pre["vessel_min_size"], pre["centerline_sigma_px"],
            valid_mask=valid_mask,
        )
        structures.append(s); vessels.append(m); skeletons.append(sk)
    structures = np.stack(structures)
    vessels = np.stack(vessels)
    skeletons = np.stack(skeletons)

    fixed_struct = structures[peak_i]
    icfg = cfg["intra_registration"]
    min_ratio = float(icfg.get("min_vessel_score_ratio", 0.15))
    peak_score_value = float(peak_scores[peak_i])
    score_good = np.isfinite(peak_scores) & (peak_scores >= max(1e-8, min_ratio * peak_score_value))
    score_good[peak_i] = True

    # First estimate transforms only where enough vascular signal exists. Very early/late
    # low-contrast frames otherwise produce arbitrary transforms. Their motion estimate is
    # borrowed from the nearest reliable temporal neighbor, consistent with slowly varying
    # head motion and preferable to fitting contrast noise.
    tx_by_frame = {peak_i: None}
    meta_by_frame = {
        peak_i: {"frame_pos": peak_i, "identity": True, "estimated": True,
                 "vessel_score": peak_score_value, "ncc_before": 1.0, "ncc_after": 1.0}
    }
    min_after_ncc = float(icfg.get("min_frame_ncc_after", 0.10))
    max_degradation = float(icfg.get("max_frame_ncc_degradation", 0.02))

    if icfg.get("enabled", True):
        for i in range(len(seq01)):
            if i == peak_i or not score_good[i]:
                continue
            before = _structure_ncc(structures[i], fixed_struct, valid_mask)
            try:
                tx, meta = register_pair(
                    fixed_struct, structures[i], kind="rigid", metric=icfg["metric"],
                    shrink_factors=icfg["shrink_factors"], smoothing_sigmas=icfg["smoothing_sigmas"],
                    learning_rate=icfg["learning_rate"], min_step=icfg["min_step"],
                    iterations=icfg["iterations"], gradient_tolerance=icfg["gradient_tolerance"],
                )
                warped = resample(structures[i], fixed_struct, tx, is_mask=False)
                after = _structure_ncc(warped, fixed_struct, valid_mask)
                accepted = np.isfinite(after) and after >= min_after_ncc and (
                    not np.isfinite(before) or after + max_degradation >= before
                )
                if accepted:
                    tx_by_frame[i] = tx
                    save_transform(tx, phase_dir / "transforms" / f"frame_{i:03d}_to_peak.tfm")
                meta_by_frame[i] = {
                    "frame_pos": i, "identity": False, "estimated": True, "accepted": bool(accepted),
                    "vessel_score": float(peak_scores[i]),
                    "ncc_before": float(before) if np.isfinite(before) else np.nan,
                    "ncc_after": float(after) if np.isfinite(after) else np.nan,
                    **meta,
                }
            except Exception as e:
                meta_by_frame[i] = {
                    "frame_pos": i, "identity": False, "estimated": True, "accepted": False,
                    "vessel_score": float(peak_scores[i]), "error": repr(e),
                }

    # Reject isolated transform spikes before filling low-contrast frames.  True DSA head
    # motion is expected to vary smoothly over adjacent frames; an estimate that strongly
    # disagrees with interpolation between two reliable neighbours is more likely a
    # contrast-driven registration error.
    jump_rejections = []
    max_jump_px = float(icfg.get("max_interpolated_translation_residual_px", 16.0))
    max_jump_deg = float(icfg.get("max_interpolated_rotation_residual_deg", 3.0))

    def rigid_vec(tx):
        if tx is None:
            return np.asarray([0.0, 0.0, 0.0], dtype=float)
        p = canonical_parameters(tx, "rigid")
        return np.asarray([p["tx"], p["ty"], p["rotation_deg"]], dtype=float)

    initial_reliable = sorted(tx_by_frame)
    for pos, i in enumerate(initial_reliable[1:-1], start=1):
        left_i, right_i = initial_reliable[pos - 1], initial_reliable[pos + 1]
        alpha = (i - left_i) / float(right_i - left_i)
        left_v, right_v, current_v = rigid_vec(tx_by_frame[left_i]), rigid_vec(tx_by_frame[right_i]), rigid_vec(tx_by_frame[i])
        predicted = left_v + alpha * (right_v - left_v)
        translation_residual = float(np.linalg.norm(current_v[:2] - predicted[:2]))
        rotation_residual = float(abs(current_v[2] - predicted[2]))
        if translation_residual > max_jump_px or rotation_residual > max_jump_deg:
            tx_by_frame.pop(i, None)
            jump_rejections.append({
                "frame_pos": int(i), "translation_residual_px": translation_residual,
                "rotation_residual_deg": rotation_residual,
            })
            meta_by_frame[i] = {
                **meta_by_frame.get(i, {"frame_pos": i}),
                "accepted": False, "trajectory_outlier": True,
                "translation_residual_px": translation_residual,
                "rotation_residual_deg": rotation_residual,
            }

    reliable = sorted(tx_by_frame.keys())
    corrected_signal = np.empty_like(seq_raw, dtype=np.float32)
    corrected_struct = np.empty_like(structures, dtype=np.float32)
    used_tx = []
    borrowed_count = 0
    for i in range(len(seq_raw)):
        if i in tx_by_frame:
            tx = tx_by_frame[i]
            source_i = i
        else:
            lower = [j for j in reliable if j < i]
            upper = [j for j in reliable if j > i]
            if (cfg["intra_registration"].get("low_contrast_transform_mode", "interpolate") == "interpolate"
                    and lower and upper):
                left_i, right_i = max(lower), min(upper)
                alpha = (i - left_i) / float(right_i - left_i)
                tx = interpolate_rigid_transforms(tx_by_frame[left_i], tx_by_frame[right_i], alpha)
                source_i = None
                source_meta = {"interpolated": True, "interpolated_from": [int(left_i), int(right_i)]}
            else:
                source_i = min(reliable, key=lambda j: abs(j - i))
                tx = tx_by_frame[source_i]
                source_meta = {"interpolated": False, "borrowed_from": int(source_i)}
            borrowed_count += 1
            old = meta_by_frame.get(i, {"frame_pos": i, "vessel_score": float(peak_scores[i])})
            meta_by_frame[i] = {**old, "estimated": False, **source_meta}
        used_tx.append(tx)
        if tx is None:
            corrected_signal[i] = seq_raw[i].astype(np.float32)
            corrected_struct[i] = structures[i]
        else:
            corrected_signal[i] = resample(seq_raw[i], fixed_struct, tx, is_mask=False)
            corrected_struct[i] = resample(structures[i], fixed_struct, tx, is_mask=False)

    # Map the gold-standard mask from its reference PNG directly to peak space; do not
    # guess that the reference image is a particular temporal frame.
    mask_peak, ref_meta = _direct_reference_to_peak(
        reference_crop, mask_native, fixed_struct, valid_mask, cfg, phase_dir,
        mapping_score=mapping_score, mapping_method=mapping_method,
    )

    radius = int(cfg["sequence"].get("peak_window_radius", 1))
    idxs = [i for i in range(max(0, peak_i - radius), min(len(seq_raw), peak_i + radius + 1))]
    template = np.mean(corrected_struct[idxs], axis=0).astype(np.float32)
    _, template_vessel, template_skel = build_structure_map(
        template, pre["vessel_threshold_percentile"], pre["vessel_min_size"], pre["centerline_sigma_px"],
        valid_mask=valid_mask,
    )

    vascular_aggregate, aggregate_meta = temporal_vascular_aggregate(
        corrected_signal, valid_mask,
        baseline_n_frames=cfg["hemodynamics"]["baseline_n_frames"],
        frame_strength_fraction=cfg["sequence"].get("vascular_aggregate_frame_strength_fraction", 0.35),
        aggregate_percentile=cfg["sequence"].get("vascular_aggregate_percentile", 85.0),
        border_px=cfg["sequence"].get("full_fov_border_px", 24),
    )
    global_vmap = vesselness(vascular_aggregate, tuple(pre["frangi_sigmas"]))
    _, global_vessel, _ = build_structure_map(
        global_vmap,
        pre.get("global_vessel_threshold_percentile", pre["vessel_threshold_percentile"]),
        pre.get("global_vessel_min_size", pre["vessel_min_size"]),
        pre["centerline_sigma_px"], valid_mask=valid_mask,
    )
    global_vessel, border_artifacts = suppress_linear_border_artifacts(
        global_vessel,
        border_fraction=pre.get("border_artifact_fraction", 0.15),
        min_span_fraction=pre.get("border_artifact_min_span_fraction", 0.25),
        max_thickness_fraction=pre.get("border_artifact_max_thickness_fraction", 0.03),
    )
    global_similarity, global_skeleton = centerline_similarity_map(
        global_vessel, pre["centerline_sigma_px"]
    )
    structure_support = binary_dilation(
        global_vessel,
        iterations=int(pre.get("global_structure_support_dilation_px", 16)),
    ) & np.asarray(valid_mask, bool)
    global_structure = (
        (0.75 * global_similarity + 0.25 * global_vmap) * structure_support
    ).astype(np.float32)
    np.savez_compressed(
        phase_dir / "temporal_vascular_aggregate.npz",
        contrast_aggregate=vascular_aggregate,
        vesselness=global_vmap,
        structure=global_structure,
        vessel=global_vessel.astype(np.uint8),
        skeleton=global_skeleton.astype(np.uint8),
    )

    if cfg["sequence"].get("save_corrected_sequence_npz", False):
        np.savez_compressed(
            phase_dir / "corrected_sequence.npz",
            corrected_signal=corrected_signal,
            corrected_structure=corrected_struct,
            template=template,
            peak_index=peak_i,
            peak_scores=peak_scores,
        )

    # Visual QC is mandatory for the real pilot.  It also makes the two score units and
    # their normalisation transparent rather than hiding them in JSON only.
    save_peak_selection_plot(
        contrast_peak_scores, contrast_peak_scores_norm, vessel_peak_scores,
        vessel_peak_scores_norm, peak_scores, peak_i, phase_dir.parent / "qc" /
        f"{phase_dir.name}_peak_selection.png", f"{phase_dir.name}: selected peak={peak_i}",
    )
    save_overlay(np.median(corrected_struct, axis=0), fixed_struct,
                 phase_dir.parent / "qc" / f"{phase_dir.name}_motion_overlay.png",
                 f"{phase_dir.name}: median motion-corrected structure vs peak")

    meta_list = [meta_by_frame.get(i, {"frame_pos": i}) for i in range(len(seq_raw))]
    intra_valid_fraction = float(sum(1 for i in tx_by_frame if i != peak_i) / max(1, len(seq_raw) - 1))
    json_dump({
        "peak_index_position": int(peak_i),
        "peak_scores": peak_scores.tolist(),
        "contrast_peak_scores": contrast_peak_scores.tolist(),
        "normalized_contrast_peak_scores": contrast_peak_scores_norm.tolist(),
        "vessel_peak_scores": vessel_peak_scores.tolist(),
        "normalized_vessel_peak_scores": vessel_peak_scores_norm.tolist(),
        "reliable_estimated_frames": reliable,
        "borrowed_frame_count": int(borrowed_count),
        "estimated_transform_fraction": intra_valid_fraction,
        "trajectory_outlier_rejections": jump_rejections,
        "reference_to_peak": ref_meta,
        "temporal_vascular_aggregate": aggregate_meta,
        "removed_linear_border_artifacts": border_artifacts,
        "frames": meta_list,
    }, phase_dir / "intra_registration.json")

    return {
        "corrected": corrected_signal,
        "template": template,
        "vessel": template_vessel,
        "skeleton": template_skel,
        "global_structure": global_structure,
        "global_vessel": global_vessel,
        "global_skeleton": global_skeleton,
        "peak_i": int(peak_i),
        "peak_scores": peak_scores,
        "mask_peak": mask_peak,
        "reference_to_peak": ref_meta,
        "intra_estimated_fraction": intra_valid_fraction,
        "intra_borrowed_fraction": float(borrowed_count / max(1, len(seq_raw))),
        "trajectory_outlier_count": int(len(jump_rejections)),
    }


def _hemo_for_phase(corrected, mask, valid_mask, cfg):
    r = cfg["regions"]
    lesion = np.asarray(mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    peri = binary_dilation(lesion, iterations=int(r["peri_outer_px"])) & ~binary_dilation(
        lesion, iterations=int(r["peri_inner_px"])
    )
    peri &= np.asarray(valid_mask, dtype=bool)
    regions = {"lesion": lesion, "peri": peri}
    out, curves, qc = {}, {}, {}
    for name in cfg["hemodynamics"]["regions"]:
        m = regions.get(name)
        if m is None or not np.any(m):
            continue
        curve, polarity = time_density_curve(corrected, m, cfg["hemodynamics"]["baseline_n_frames"])
        feats = curve_features(
            curve, cfg["sequence"].get("frame_interval_seconds"), cfg["hemodynamics"]["arrival_fraction"]
        )
        normalised, phase_curve, phase_meta = normalized_phase_features(
            curve,
            n_samples=cfg["hemodynamics"].get("normalized_phase_samples", 32),
            arrival_fraction=cfg["hemodynamics"]["arrival_fraction"],
            washout_fraction=cfg["hemodynamics"].get("normalized_phase_washout_fraction", 0.10),
        )
        feats.update(normalised)
        out[name] = feats
        curves[name] = curve
        curves[f"{name}_normalised_phase"] = phase_curve
        qc[name] = {
            "polarity": float(polarity), "n_pixels": int(np.sum(m)),
            "time_unit": "seconds" if cfg["sequence"].get("frame_interval_seconds") is not None else "frames",
            "normalised_phase": phase_meta,
        }
    return out, curves, qc


def _stable_metric_masks(warped_vessel, fixed_vessel, warped_mask, fixed_mask, valid, exclusion_px):
    lesion_union = np.asarray(warped_mask, dtype=bool) | np.asarray(fixed_mask, dtype=bool)
    stable_support = (np.asarray(warped_vessel, dtype=bool) | np.asarray(fixed_vessel, dtype=bool))
    stable_support &= ~binary_dilation(lesion_union, iterations=int(exclusion_px))
    stable_support &= np.asarray(valid, dtype=bool)
    return stable_support


def _invalid_feature_record(row, reason: str, **extra):
    """Return a retained, explicitly invalid series row without deformation values.

    Invalid geometry/registration is represented by ``registration_valid=0`` and
    ``q_reg=0``.  Deformation quantities are deliberately absent/NaN, never median-filled
    here; downstream preprocessing may use their missing indicators while the 3171 gate
    exactly preserves the PredROI identity path.
    """
    out = {
        "patient_id": int(row.patient_id), "split": str(row.split),
        "series_uid": str(row.series_uid), "series_id": str(row.series_id),
        "registration_valid": 0, "q_reg": 0.0,
        "registration_invalid_reasons": str(reason), "neck_feature_available": False,
    }
    out.update(extra)
    return out


def _peak_tdc_qc(curves: dict, selected_index: int, max_delta: int) -> dict:
    """Compare a structural/contrast-selected peak with the corrected lesion TDC maximum."""
    curve = np.asarray(curves.get("lesion", []), dtype=float)
    if curve.size == 0 or not np.any(np.isfinite(curve)):
        return {"available": False, "warning": True, "reason": "missing_lesion_tdc"}
    tdc_peak = int(np.nanargmax(np.maximum(curve, 0.0)))
    delta = abs(int(selected_index) - tdc_peak)
    return {
        "available": True, "tdc_peak_position": tdc_peak,
        "selected_to_tdc_peak_delta_frames": int(delta),
        "warning": bool(delta > int(max_delta)),
        "max_allowed_delta_frames": int(max_delta),
    }


def _adaptive_post_global_crop(mask_a: np.ndarray, mask_b: np.ndarray, cfg: dict):
    """Choose one shared local ROI after both masks occupy the Post canvas.

    The crop is never used to estimate the global transform.  Its origin is defined in
    the full Post reference canvas and is shared by Pre-global and Post.  The smallest
    configured square covering the lesion union plus context is selected.
    """
    union = np.asarray(mask_a, bool) | np.asarray(mask_b, bool)
    yy, xx = np.nonzero(union)
    if not len(xx):
        raise ValueError("Cannot define post-global ROI from empty lesion masks")
    margin = int(cfg["roi"].get("post_global_margin_px", 96))
    need = max(int(yy.max() - yy.min() + 1 + 2 * margin),
               int(xx.max() - xx.min() + 1 + 2 * margin))
    allowed = sorted(int(x) for x in cfg["roi"].get("adaptive_sizes", [512, 640, 768, 1024]))
    size = next((x for x in allowed if x >= need), allowed[-1])
    size = min(size, max(union.shape))
    center = ((float(yy.min()) + float(yy.max())) / 2.0,
              (float(xx.min()) + float(xx.max())) / 2.0)
    origin = crop_origin_yx(center, (size, size))
    return center, (size, size), origin


def process_series(row, cfg):
    remap = cfg["paths"].get("remap", {})
    series_key = sanitize_key(str(row.series_uid))
    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / str(row.split) / str(row.patient_id) / series_key)
    pre_dir = ensure_dir(out_dir / "intra_pre")
    post_dir = ensure_dir(out_dir / "intra_post")
    global_dir = ensure_dir(out_dir / "global")
    syn_dir = ensure_dir(out_dir / "nonrigid")
    qc_dir = ensure_dir(out_dir / "qc")

    pre_paths = parse_frame_paths(row.pre_frame_paths, remap)
    post_paths = parse_frame_paths(row.post_frame_paths, remap)
    if len(pre_paths) != int(row.pre_n_frames) or len(post_paths) != int(row.post_n_frames):
        raise ValueError("Manifest frame count != parsed frame paths")

    pre_mask_full = read_mask(apply_path_remap(row.pre_mask_path, remap))
    post_mask_full = read_mask(apply_path_remap(row.post_mask_path, remap))
    pre_ref_full = read_gray(apply_path_remap(row.pre_reference_image_path, remap))
    post_ref_full = read_gray(apply_path_remap(row.post_reference_image_path, remap))

    geom_cfg = cfg.get("geometry", {})
    canvas_hw = tuple(int(x) for x in geom_cfg.get("canvas_size", pre_ref_full.shape))
    if pre_ref_full.shape != pre_mask_full.shape or post_ref_full.shape != post_mask_full.shape:
        raise ValueError(
            f"Reference/mask shape mismatch: pre={pre_ref_full.shape}/{pre_mask_full.shape}, "
            f"post={post_ref_full.shape}/{post_mask_full.shape}"
        )
    pre_ref_canvas = resize_image_to_canvas(pre_ref_full, canvas_hw)
    post_ref_canvas = resize_image_to_canvas(post_ref_full, canvas_hw)
    pre_mask_canvas = resize_image_to_canvas(pre_mask_full, canvas_hw, is_mask=True)
    post_mask_canvas = resize_image_to_canvas(post_mask_full, canvas_hw, is_mask=True)

    aspect_tol = float(geom_cfg.get("uniform_scale_tolerance", 0.01))
    # V5 invariant: global correspondence is established on the whole canvas.  No
    # lesion-defined crop or origin exists before global registration.
    pre_seq, pre_geom = read_sequence_canvas(pre_paths, canvas_hw, aspect_tol)
    post_seq, post_geom = read_sequence_canvas(post_paths, canvas_hw, aspect_tol)
    pre_mask, post_mask = pre_mask_canvas, post_mask_canvas
    pre_valid = np.ones(canvas_hw, dtype=bool)
    post_valid = np.ones(canvas_hw, dtype=bool)
    pre_ref, post_ref = pre_ref_canvas, post_ref_canvas
    pre_center, post_center = mask_center(pre_mask), mask_center(post_mask)
    pre_coverage = post_coverage = 1.0

    pre_cand = _candidate_positions(row, "pre", pre_paths, cfg)
    post_cand = _candidate_positions(row, "post", post_paths, cfg)
    pre = _intra_correct(
        pre_seq, pre_mask, pre_ref, pre_valid, pre_cand, cfg, pre_dir,
        mapping_score=getattr(row, "pre_mapping_score", np.nan),
        mapping_method=getattr(row, "pre_mapping_method", ""),
    )
    post = _intra_correct(
        post_seq, post_mask, post_ref, post_valid, post_cand, cfg, post_dir,
        mapping_score=getattr(row, "post_mapping_score", np.nan),
        mapping_method=getattr(row, "post_mapping_method", ""),
    )
    del pre_seq, post_seq, pre_ref, post_ref, pre_ref_canvas, post_ref_canvas
    phase_valid = bool(pre["reference_to_peak"]["phase_geometry_valid"] and
                       post["reference_to_peak"]["phase_geometry_valid"])
    if not phase_valid:
        reasons = []
        for phase, obj in (("pre", pre), ("post", post)):
            if not obj["reference_to_peak"]["phase_geometry_valid"]:
                reasons.append(f"{phase}_reference_to_peak_geometry_invalid")
        feats = _invalid_feature_record(
            row, "|".join(reasons), phase_geometry_valid=0,
            pre_reference_to_peak_geometry_source=pre["reference_to_peak"]["geometry_source"],
            post_reference_to_peak_geometry_source=post["reference_to_peak"]["geometry_source"],
            pre_reference_to_peak_ncc_before=pre["reference_to_peak"]["ncc_before"],
            pre_reference_to_peak_ncc_after=pre["reference_to_peak"]["ncc_after"],
            post_reference_to_peak_ncc_before=post["reference_to_peak"]["ncc_before"],
            post_reference_to_peak_ncc_after=post["reference_to_peak"]["ncc_after"],
        )
        json_dump(feats, out_dir / "features.json")
        return feats
    pre_mask_peak, post_mask_peak = pre["mask_peak"], post["mask_peak"]

    save_overlay(pre_mask_peak.astype(np.float32), pre["template"], qc_dir / "pre_reference_to_peak_mask_overlay.png",
                 "Pre reference mask mapped to temporal peak")
    save_overlay(post_mask_peak.astype(np.float32), post["template"], qc_dir / "post_reference_to_peak_mask_overlay.png",
                 "Post reference mask mapped to temporal peak")

    pre_hemo, pre_curves, pre_hemo_qc = _hemo_for_phase(pre["corrected"], pre_mask_peak, pre_valid, cfg)
    post_hemo, post_curves, post_hemo_qc = _hemo_for_phase(post["corrected"], post_mask_peak, post_valid, cfg)
    np.savez_compressed(
        out_dir / "hemodynamic_curves.npz",
        **{f"pre_{k}": v for k, v in pre_curves.items()},
        **{f"post_{k}": v for k, v in post_curves.items()},
    )
    json_dump({"pre": pre_hemo_qc, "post": post_hemo_qc}, out_dir / "hemodynamic_qc.json")
    pre_peak_tdc_qc = _peak_tdc_qc(
        pre_curves, pre["peak_i"], cfg["sequence"].get("peak_tdc_max_index_delta", 4)
    )
    post_peak_tdc_qc = _peak_tdc_qc(
        post_curves, post["peak_i"], cfg["sequence"].get("peak_tdc_max_index_delta", 4)
    )
    json_dump({"pre": pre_peak_tdc_qc, "post": post_peak_tdc_qc}, out_dir / "peak_tdc_qc.json")
    # Corrected temporal stacks are no longer needed after TDC extraction.
    del pre["corrected"], post["corrected"]

    regcfg = cfg["global_registration"]
    rg = cfg["regions"]
    anchor_radius = regcfg.get("anchor_max_distance_px")
    pre_anchor = build_anchor(
        pre["global_vessel"], pre_mask_peak, rg["anchor_exclusion_px"], pre_valid, anchor_radius
    )
    post_anchor = build_anchor(
        post["global_vessel"], post_mask_peak, rg["anchor_exclusion_px"], post_valid, anchor_radius
    )
    fixed_metric_support = broad_fixed_metric_support(
        post_anchor, post_valid, regcfg.get("fixed_support_dilation_px", 32)
    )
    pre_global_exclusion = binary_dilation(
        pre_mask_peak, iterations=int(rg["anchor_exclusion_px"])
    )
    post_global_exclusion = binary_dilation(
        post_mask_peak, iterations=int(rg["anchor_exclusion_px"])
    )
    pre_reg_structure = pre["global_structure"] * (~pre_global_exclusion)
    post_reg_structure = post["global_structure"] * (~post_global_exclusion)
    pre_reg_skeleton = pre["global_skeleton"] & ~pre_global_exclusion
    post_reg_skeleton = post["global_skeleton"] & ~post_global_exclusion

    globals_out = {}
    global_errors = {}
    for method in regcfg["run_methods"]:
        try:
            coarse = coarse_similarity_candidates(
                post_reg_structure, pre_reg_structure,
                post_reg_skeleton, pre_reg_skeleton,
                post_valid, pre_valid, regcfg, method,
            )
            attempts = []
            refine_cfg = regcfg.get("refine_top_candidates", 1)
            refine_count = int(refine_cfg.get(method, 1) if isinstance(refine_cfg, dict) else refine_cfg)
            for rank, candidate in enumerate(coarse):
                candidate_metrics = compatibility_metrics(
                    pre_reg_skeleton, post_reg_skeleton,
                    pre_valid, post_valid, candidate["transform"],
                )
                candidate_meta = {
                    k: v for k, v in candidate.items()
                    if k not in {"transform", "canonical_matrix"}
                }
                attempts.append({
                    "tx": candidate["transform"], "stage": "coarse", "rank": rank,
                    "meta": candidate_meta, "metrics": candidate_metrics,
                })
                if rank >= max(0, refine_count):
                    continue
                try:
                    refined_tx, refined_meta = register_pair(
                        fixed=post_reg_structure, moving=pre_reg_structure, kind=method,
                        fixed_mask=fixed_metric_support, moving_mask=None,
                        metric=regcfg["metric"], shrink_factors=regcfg["shrink_factors"],
                        smoothing_sigmas=regcfg["smoothing_sigmas"], learning_rate=regcfg["learning_rate"],
                        min_step=regcfg["min_step"], iterations=regcfg["iterations"],
                        gradient_tolerance=regcfg["gradient_tolerance"], use_moving_mask=False,
                        initial_transform=candidate["transform"],
                    )
                    refined_metrics = compatibility_metrics(
                        pre_reg_skeleton, post_reg_skeleton,
                        pre_valid, post_valid, refined_tx,
                    )
                    attempts.append({
                        "tx": refined_tx, "stage": "refined", "rank": rank,
                        "meta": {"coarse": candidate_meta, "optimizer": refined_meta},
                        "metrics": refined_metrics,
                    })
                except Exception as refine_error:
                    attempts.append({
                        "tx": candidate["transform"], "stage": "refinement_failed", "rank": rank,
                        "meta": {"coarse": candidate_meta, "error": repr(refine_error)},
                        "metrics": candidate_metrics,
                    })
            selected = max(attempts, key=lambda x: x["metrics"]["score"])
            tx, meta = selected["tx"], selected["meta"]
            warped_template = resample(pre["global_structure"], post["global_structure"], tx)
            warped_vessel = resample(pre["global_vessel"].astype(np.uint8), post["global_structure"], tx, is_mask=True)
            warped_skel = resample(pre["global_skeleton"].astype(np.uint8), post["global_structure"], tx, is_mask=True)
            warped_peak_template = resample(pre["template"], post["template"], tx)
            warped_mask = resample(pre_mask_peak.astype(np.uint8), post["global_structure"], tx, is_mask=True)
            warped_anchor = resample(pre_anchor.astype(np.uint8), post["global_structure"], tx, is_mask=True)
            warped_valid = resample(pre_valid.astype(np.uint8), post["global_structure"], tx, is_mask=True)
            valid_pair = post_valid & warped_valid
            stable_support = _stable_metric_masks(
                warped_vessel, post["global_vessel"], warped_mask, post_mask_peak,
                valid_pair, rg["anchor_exclusion_px"]
            )
            params = canonical_parameters(tx, method)
            compatibility = compatibility_metrics(
                pre_reg_skeleton, post_reg_skeleton, pre_valid, post_valid, tx
            )
            global_status, global_reasons = global_compatibility_status(
                compatibility, params, regcfg
            )
            save_transform(tx, global_dir / f"{method}.tfm")
            save_overlay(warped_template, post["global_structure"], qc_dir / f"global_{method}.png",
                         f"full-FOV {method}: green Pre, red Post")
            globals_out[method] = {
                "tx": tx, "meta": meta, "template": warped_template, "vessel": warped_vessel,
                "skeleton": warped_skel, "mask": warped_mask, "anchor": warped_anchor, "valid": valid_pair,
                "peak_template": warped_peak_template,
                "params": params, "compatibility": compatibility,
                "global_status": global_status, "global_reasons": global_reasons,
                "selected_stage": selected["stage"], "selected_candidate_rank": selected["rank"],
                "ncc": vascular_ncc(warped_vessel, post["global_vessel"], stable_support),
                "structure_ncc": masked_ncc(warped_template, post["global_structure"], stable_support),
                "dice_vessel": dice(warped_vessel & stable_support, post["global_vessel"] & stable_support),
                "chamfer": symmetric_chamfer(warped_skel & stable_support, post["global_skeleton"] & stable_support),
                "stable_pixels": int(np.sum(stable_support)),
            }
        except Exception as e:
            global_errors[method] = repr(e)

    primary = regcfg["primary_method"]
    if primary not in globals_out:
        raise RuntimeError(f"Primary global registration '{primary}' failed: {global_errors.get(primary)}")
    g = globals_out[primary]
    save_overlay(pre["global_structure"], post["global_structure"], qc_dir / "global_before.png",
                 "Whole-FOV temporal vascular aggregate before global registration")
    json_dump({
        "primary_method": primary,
        "errors": global_errors,
        "methods": {m: {"meta": o["meta"], "params": o["params"], "ncc": o["ncc"],
                        "structure_ncc": o["structure_ncc"], "vessel_dice": o["dice_vessel"], "centerline_chamfer": o["chamfer"],
                        "stable_pixels": o["stable_pixels"], "compatibility": o["compatibility"],
                        "global_status": o["global_status"], "global_reasons": o["global_reasons"],
                        "selected_stage": o["selected_stage"],
                        "selected_candidate_rank": o["selected_candidate_rank"]}
                    for m, o in globals_out.items()},
    }, global_dir / "global_registration.json")

    feats = {
        "patient_id": int(row.patient_id), "split": str(row.split),
        "series_uid": str(row.series_uid), "series_id": str(row.series_id),
        "pre_peak_position": int(pre["peak_i"]), "post_peak_position": int(post["peak_i"]),
        "pre_intra_estimated_fraction": float(pre["intra_estimated_fraction"]),
        "post_intra_estimated_fraction": float(post["intra_estimated_fraction"]),
        "pre_intra_borrowed_fraction": float(pre["intra_borrowed_fraction"]),
        "post_intra_borrowed_fraction": float(post["intra_borrowed_fraction"]),
        "pre_intra_trajectory_outlier_count": int(pre["trajectory_outlier_count"]),
        "post_intra_trajectory_outlier_count": int(post["trajectory_outlier_count"]),
        "pre_reference_to_peak_ncc_before": pre["reference_to_peak"]["ncc_before"],
        "pre_reference_to_peak_ncc_after": pre["reference_to_peak"]["ncc_after"],
        "post_reference_to_peak_ncc_before": post["reference_to_peak"]["ncc_before"],
        "post_reference_to_peak_ncc_after": post["reference_to_peak"]["ncc_after"],
        "pre_reference_to_peak_transform_accepted": int(pre["reference_to_peak"]["accepted_transform"]),
        "post_reference_to_peak_transform_accepted": int(post["reference_to_peak"]["accepted_transform"]),
        "phase_geometry_valid": 1,
        "pre_reference_to_peak_geometry_source": pre["reference_to_peak"]["geometry_source"],
        "post_reference_to_peak_geometry_source": post["reference_to_peak"]["geometry_source"],
        "pre_peak_tdc_warning": int(pre_peak_tdc_qc["warning"]),
        "post_peak_tdc_warning": int(post_peak_tdc_qc["warning"]),
        "pre_peak_tdc_delta_frames": pre_peak_tdc_qc.get("selected_to_tdc_peak_delta_frames", np.nan),
        "post_peak_tdc_delta_frames": post_peak_tdc_qc.get("selected_to_tdc_peak_delta_frames", np.nan),
        "neck_feature_available": False,
        "pre_crop_center_y": float(pre_center[0]), "pre_crop_center_x": float(pre_center[1]),
        "post_crop_center_y": float(post_center[0]), "post_crop_center_x": float(post_center[1]),
        "pre_global_crop_applied": 0,
        "shared_crop_origin_y": np.nan, "shared_crop_origin_x": np.nan,
        "pre_mask_crop_coverage": pre_coverage, "post_mask_crop_coverage": post_coverage,
        **{f"pre_geometry_{k}": v for k, v in pre_geom.items()},
        **{f"post_geometry_{k}": v for k, v in post_geom.items()},
        "pre_geometry_reference_source_height": int(pre_ref_full.shape[0]),
        "pre_geometry_reference_source_width": int(pre_ref_full.shape[1]),
        "post_geometry_reference_source_height": int(post_ref_full.shape[0]),
        "post_geometry_reference_source_width": int(post_ref_full.shape[1]),
        "global_primary_method": primary,
        "global_compatibility_status": g["global_status"],
        "global_compatibility_reasons": "|".join(g["global_reasons"]),
        "vascular_anchor_type": "pseudo_automatic_motion_corrected_temporal_full_fov",
        "vascular_anchor_max_distance_px": anchor_radius if anchor_radius is not None else np.nan,
        "pre_mapping_method": str(getattr(row, "pre_mapping_method", "")),
        "post_mapping_method": str(getattr(row, "post_mapping_method", "")),
        "pre_mapping_score": float(getattr(row, "pre_mapping_score", np.nan)),
        "post_mapping_score": float(getattr(row, "post_mapping_score", np.nan)),
    }
    for method, obj in globals_out.items():
        for k, v in obj["params"].items():
            feats[f"global_{method}_{k}"] = v
        feats[f"global_{method}_ncc"] = obj["ncc"]
        feats[f"global_{method}_structure_ncc"] = obj["structure_ncc"]
        feats[f"global_{method}_vessel_dice"] = obj["dice_vessel"]
        feats[f"global_{method}_centerline_chamfer"] = obj["chamfer"]
        feats[f"global_{method}_stable_pixels"] = obj["stable_pixels"]
        feats[f"global_{method}_status"] = obj["global_status"]
        feats[f"global_{method}_selected_stage"] = obj["selected_stage"]
        for k, v in obj["compatibility"].items():
            feats[f"global_{method}_{k}"] = v
    # Keep only the selected global result before entering memory-intensive ANTs.
    for method in list(globals_out):
        if method != primary:
            del globals_out[method]
    gc.collect()

    before_valid = pre_valid & post_valid
    before_exclusion = binary_dilation(pre_mask_peak | post_mask_peak, iterations=int(rg["anchor_exclusion_px"]))
    before_stable = (pre["global_vessel"] | post["global_vessel"]) & ~before_exclusion & before_valid
    feats["global_ncc_before"] = vascular_ncc(pre["global_vessel"], post["global_vessel"], before_stable)
    feats["global_structure_ncc_before"] = masked_ncc(
        pre["global_structure"], post["global_structure"], before_stable
    )
    feats["global_centerline_chamfer_before"] = symmetric_chamfer(
        pre["global_skeleton"] & before_stable, post["global_skeleton"] & before_stable
    )
    feats.update(morphology_features(g["mask"], post_mask_peak))

    for region in sorted(set(pre_hemo) | set(post_hemo)):
        if region in pre_hemo and region in post_hemo:
            feats.update(delta_features(pre_hemo[region], post_hemo[region], prefix=f"hemo_{region}"))
            feats[f"hemo_{region}_pre_polarity"] = pre_hemo_qc[region]["polarity"]
            feats[f"hemo_{region}_post_polarity"] = post_hemo_qc[region]["polarity"]

    # SyN is residual/local by definition and must never be asked to repair a failed
    # acquisition-level correspondence.  Retain the series with neutral registration
    # representation and an explicit failure reason.
    if g["global_status"] == "GLOBAL_FAIL":
        feats["registration_valid"] = 0
        feats["q_reg"] = 0.0
        feats["registration_invalid_reasons"] = "global_correspondence_failed:" + "|".join(g["global_reasons"])
        json_dump(feats, out_dir / "features.json")
        return feats

    shared_center, out_hw, shared_origin = _adaptive_post_global_crop(
        g["mask"], post_mask_peak, cfg
    )
    pad = float(cfg["roi"].get("pad_value", 0.0))

    def crop_img(x, fill=0.0):
        return crop_fixed_2d(np.asarray(x), shared_center, out_hw, fill)

    g_mask_full = g["mask"]
    post_mask_full_global = post_mask_peak
    g = {
        **g,
        "template": crop_img(g["template"], pad).astype(np.float32),
        "peak_template": crop_img(g["peak_template"], pad).astype(np.float32),
        "vessel": crop_img(g["vessel"].astype(np.uint8), 0).astype(bool),
        "skeleton": crop_img(g["skeleton"].astype(np.uint8), 0).astype(bool),
        "mask": crop_img(g["mask"].astype(np.uint8), 0).astype(bool),
        "anchor": crop_img(g["anchor"].astype(np.uint8), 0).astype(bool),
        "valid": crop_img(g["valid"].astype(np.uint8), 0).astype(bool),
    }
    post = {
        **post,
        "template": crop_img(post["global_structure"], pad).astype(np.float32),
        "vessel": crop_img(post["global_vessel"].astype(np.uint8), 0).astype(bool),
        "skeleton": crop_img(post["global_skeleton"].astype(np.uint8), 0).astype(bool),
    }
    post_mask_peak = crop_img(post_mask_peak.astype(np.uint8), 0).astype(bool)
    post_anchor = crop_img(post_anchor.astype(np.uint8), 0).astype(bool)
    post_valid = crop_img(post_valid.astype(np.uint8), 0).astype(bool)
    pre_local_coverage = float(g["mask"].sum() / max(1, g_mask_full.sum()))
    post_local_coverage = float(post_mask_peak.sum() / max(1, post_mask_full_global.sum()))
    if min(pre_local_coverage, post_local_coverage) < float(cfg["roi"].get("min_mask_coverage", 0.995)):
        feats["registration_valid"] = 0
        feats["q_reg"] = 0.0
        feats["registration_invalid_reasons"] = "post_global_adaptive_crop_mask_clipping"
        json_dump(feats, out_dir / "features.json")
        return feats
    feats.update({
        "post_global_crop_center_y": float(shared_center[0]),
        "post_global_crop_center_x": float(shared_center[1]),
        "post_global_crop_origin_y": int(shared_origin[0]),
        "post_global_crop_origin_x": int(shared_origin[1]),
        "post_global_crop_height": int(out_hw[0]),
        "post_global_crop_width": int(out_hw[1]),
        "post_global_pre_mask_coverage": pre_local_coverage,
        "post_global_post_mask_coverage": post_local_coverage,
    })
    global_local_stable = _stable_metric_masks(
        g["vessel"], post["vessel"], g["mask"], post_mask_peak,
        g["valid"] & post_valid, rg["anchor_exclusion_px"],
    )
    g["local_ncc"] = vascular_ncc(g["vessel"], post["vessel"], global_local_stable)
    g["local_structure_ncc"] = masked_ncc(g["template"], post["template"], global_local_stable)
    g["local_dice_vessel"] = dice(
        g["vessel"] & global_local_stable, post["vessel"] & global_local_stable
    )
    g["local_chamfer"] = symmetric_chamfer(
        g["skeleton"] & global_local_stable, post["skeleton"] & global_local_stable
    )
    feats.update({
        "global_primary_local_ncc": g["local_ncc"],
        "global_primary_local_structure_ncc": g["local_structure_ncc"],
        "global_primary_local_vessel_dice": g["local_dice_vessel"],
        "global_primary_local_centerline_chamfer": g["local_chamfer"],
        "global_primary_local_stable_pixels": int(np.sum(global_local_stable)),
    })
    save_overlay(g["template"], post["template"], qc_dir / "global_primary_local_roi.png",
                 f"Post-global adaptive ROI ({out_hw[0]} px)")

    if cfg["nonrigid"].get("enabled", True):
        syn_margin = int(cfg["nonrigid"].get("metric_roi_margin_px", 24))
        fixed_syn_metric_mask = binary_dilation(
            post_anchor | post_mask_peak, iterations=max(0, syn_margin)
        ) & post_valid
        moving_syn_metric_mask = binary_dilation(
            g["anchor"] | g["mask"], iterations=max(0, syn_margin)
        ) & g["valid"]
        syn = run_syn_residual(
            fixed=post["template"], moving_global=g["template"],
            fixed_anchor=fixed_syn_metric_mask, moving_anchor=moving_syn_metric_mask,
            outprefix=str(syn_dir / "syn_"), cfg=cfg["nonrigid"],
            fixed_lesion=post_mask_peak, moving_lesion=g["mask"],
        )

        nr_vessel = warp_mask_to_fixed(g["vessel"], post["template"], g["template"], syn["fwdtransforms"])
        nr_skel = warp_mask_to_fixed(g["skeleton"], post["template"], g["template"], syn["fwdtransforms"])
        nr_anchor = warp_mask_to_fixed(g["anchor"], post["template"], g["template"], syn["fwdtransforms"])
        nr_mask = warp_mask_to_fixed(g["mask"], post["template"], g["template"], syn["fwdtransforms"])
        nr_stable = _stable_metric_masks(
            nr_vessel, post["vessel"], nr_mask, post_mask_peak, g["valid"], rg["anchor_exclusion_px"]
        )
        nr_ncc = vascular_ncc(nr_vessel, post["vessel"], nr_stable)
        nr_structure_ncc = masked_ncc(syn["warped_moving"], post["template"], nr_stable)
        nr_dice = dice(nr_vessel & nr_stable, post["vessel"] & nr_stable)
        nr_chamfer = symmetric_chamfer(nr_skel & nr_stable, post["skeleton"] & nr_stable)

        regions = measurement_regions(
            g["mask"], post_mask_peak, pre_global_vessel=g["vessel"], post_vessel=post["vessel"],
            valid_mask=g["valid"], boundary_inner=rg["boundary_inner_px"],
            boundary_outer=rg["boundary_outer_px"], peri_inner=rg["peri_inner_px"],
            peri_outer=rg["peri_outer_px"], roi_margin=rg.get("measurement_roi_margin_px", 40),
            vessel_roi_dilate=rg.get("vessel_roi_dilate_px", 5),
        )
        regions["stable"] = nr_stable & syn["canonical_valid"]

        tau, tau_source = resolve_expansion_tau(cfg)
        feats.update(deformation_features(
            syn["displacement"], syn["canonical_logjac"], syn["canonical_jac"],
            syn["canonical_folding"], syn["canonical_valid"], regions, tau,
            tuple(cfg["features"]["quantiles"]),
        ))
        feats["expansion_tau"] = tau
        feats["expansion_tau_source"] = tau_source
        feats["nonrigid_anchor_ncc"] = nr_ncc
        feats["nonrigid_structure_ncc"] = nr_structure_ncc
        feats["nonrigid_vessel_dice"] = nr_dice
        feats["nonrigid_centerline_chamfer"] = nr_chamfer
        feats["nonrigid_stable_pixels"] = int(np.sum(nr_stable))
        feats["inverse_consistency_logjac_mae"] = syn["inverse_consistency_logjac_mae"]
        feats["nonrigid_lesion_metric_enabled"] = int(syn["lesion_metric_enabled"])
        feats["nonrigid_lesion_metric_weight"] = float(syn["lesion_metric_weight"])

        feats["q_reg"] = initial_qreg(
            g["local_ncc"], nr_ncc, g["local_chamfer"], nr_chamfer,
            feats["folding_rate"], feats["abs_logjac_p99"], syn["inverse_consistency_logjac_mae"], cfg["qc"],
            dice_global=g["local_dice_vessel"], dice_nonrigid=nr_dice,
        )
        if g["global_status"] == "GLOBAL_PASS_WITH_CAUTION":
            feats["q_reg"] *= float(cfg["qc"].get("global_caution_q_multiplier", 0.70))
        valid, reasons = registration_validity(
            g["local_ncc"], nr_ncc, g["local_chamfer"], nr_chamfer,
            feats["folding_rate"], feats["abs_logjac_p99"], syn["inverse_consistency_logjac_mae"],
            int(np.sum(nr_stable)), cfg["qc"],
            dice_global=g["local_dice_vessel"], dice_nonrigid=nr_dice,
        )
        feats["registration_valid"] = int(valid)
        feats["registration_invalid_reasons"] = "|".join(reasons)
        if not valid:
            # Invalid registration must not inject deformation into the downstream residual branch.
            feats["q_reg"] = 0.0

        np.savez_compressed(
            out_dir / "change_maps.npz",
            pre_global=g["template"], post=post["template"],
            global_absdiff=np.abs(post["template"] - g["template"]),
            displacement=syn["displacement"], canonical_logjac=syn["canonical_logjac"],
            canonical_jac=syn["canonical_jac"], canonical_valid=syn["canonical_valid"].astype(np.uint8),
            canonical_folding=syn["canonical_folding"].astype(np.uint8),
            forward_pull_jac=syn["forward_pull_jac"],
            lesion=regions["lesion"].astype(np.uint8), boundary=regions["boundary"].astype(np.uint8),
            peri=regions["peri"].astype(np.uint8), stable=regions["stable"].astype(np.uint8),
            roi=regions["roi"].astype(np.uint8),
        )
        save_overlay(syn["warped_moving"], post["template"], qc_dir / "nonrigid_overlay.png",
                     "SyN residual: green Pre, red Post")
        save_heatmap(post["template"], syn["canonical_logjac"], qc_dir / "canonical_logjac.png",
                     "Pre→Post local expansion (+)", regions["roi"] & syn["canonical_valid"])
        save_displacement_map(post["template"], syn["displacement"], qc_dir / "displacement.png",
                              regions["roi"] & syn["canonical_valid"])
    else:
        feats["q_reg"] = np.nan
        feats["registration_valid"] = 0
        feats["registration_invalid_reasons"] = "nonrigid_disabled"

    json_dump(feats, out_dir / "features.json")
    return feats
