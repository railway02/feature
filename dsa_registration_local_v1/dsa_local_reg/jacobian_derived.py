"""Read-only canonical residual Jacobian derivation for frozen G0 assets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label

from .local_geometry import crop_with_border_median_padding, resize_whole_canvas
from .temporal_contract import FrozenSeriesContract
from .v5_adapter import load_v5_module


REGION_NAMES = ("lesion", "peri_lesion", "whole_valid_local_roi")
EXISTING42_METRICS = (
    "logJ_mean", "logJ_median", "logJ_std", "logJ_P10", "logJ_P25", "logJ_P75", "logJ_P90", "logJ_P95",
    "abs_logJ_median", "abs_logJ_P90", "abs_logJ_P95", "disp_median", "disp_P90", "disp_P95",
)


def existing42_columns() -> list[str]:
    return [f"{region}_{metric}" for region in REGION_NAMES for metric in EXISTING42_METRICS]


def extended28_columns() -> list[str]:
    names: list[str] = []
    for region in REGION_NAMES:
        names.extend([f"{region}_positive_logJ_burden", f"{region}_negative_logJ_burden"])
        for tau in ("0p025", "0p050", "0p100"):
            names.append(f"{region}_expansion_fraction_tau_{tau}")
        for tau in ("0p025", "0p050", "0p100"):
            names.append(f"{region}_contraction_fraction_tau_{tau}")
    return names + [
        "whole_valid_local_roi_largest_expansion_component_ratio_tau_0p050",
        "whole_valid_local_roi_largest_contraction_component_ratio_tau_0p050",
        "whole_valid_local_roi_expansion_component_count_tau_0p050",
        "whole_valid_local_roi_contraction_component_count_tau_0p050",
    ]


def _read_gray(path: Path) -> np.ndarray:
    item = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if item is None:
        raise FileNotFoundError(path)
    return item


def assert_existing_residual_identity(case_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    ants = load_v5_module(cfg, "registration_ants.py")
    paths = [case_dir / "rigid_syn_1Warp.nii.gz", case_dir / "rigid_syn_0GenericAffine.mat", case_dir / "rigid_syn_1InverseWarp.nii.gz"]
    checked = ants._assert_residual_linear_identity([str(path) for path in paths])
    return {"residual_linear_identity_verified": bool(checked), "checked_paths": checked}


def load_fixed_real_support(contract: FrozenSeriesContract) -> np.ndarray:
    rec = contract.post.record
    # Crop support depends only on the native canvas and frozen G0 bbox, not image intensity.
    source = np.zeros(rec.canvas_shape_yx, dtype=np.uint8)
    return crop_with_border_median_padding(source, rec.expanded_bbox).valid_support.astype(bool)


def _post_lesion(contract: FrozenSeriesContract) -> np.ndarray:
    rec = contract.post.record
    native = resize_whole_canvas(_read_gray(rec.mask_path), rec.canvas_shape_yx, is_mask=True) > 0
    return crop_with_border_median_padding(native.astype(np.uint8), rec.expanded_bbox).image > 0


def rederive_canonical_maps(contract: FrozenSeriesContract, cfg: dict[str, Any]) -> dict[str, Any]:
    """Use V5's canonical InverseWarp determinant function; never registration()."""
    antsmod = load_v5_module(cfg, "registration_ants.py")
    case = contract.g0_case_dir
    with np.load(case / "rigid_maps.npz", allow_pickle=False) as stored:
        stored_logj = stored["logj"].astype(np.float32)
        stored_disp = stored["disp"].astype(np.float32)
        stored_valid = stored["valid"].astype(bool)
    support = load_fixed_real_support(contract)
    if stored_logj.shape != support.shape or stored_disp.shape != support.shape:
        raise ValueError(f"{contract.series_uid}: stored map/support shape mismatch")
    shape = support.shape
    # Pre has already been rigid-resampled into this Post grid by G0, so both image
    # references deliberately share Post geometry here.
    fixed = antsmod._aimg(np.zeros(shape, dtype=np.float32))
    moving = antsmod._aimg(np.zeros(shape, dtype=np.float32))
    forward_warp = case / "rigid_syn_1Warp.nii.gz"
    inverse_warp = case / "rigid_syn_1InverseWarp.nii.gz"
    linear = case / "rigid_syn_0GenericAffine.mat"
    identity = assert_existing_residual_identity(case, cfg)
    maps = antsmod._canonical_inverse_warp_jacobian(
        moving, fixed, str(inverse_warp), str(forward_warp),
        [str(forward_warp), str(linear)], [str(linear), str(inverse_warp)], geom=True,
    )
    valid_positive = np.asarray(maps["canonical_valid"], dtype=bool) & support
    folding = np.asarray(maps["canonical_folding"], dtype=bool) & support
    if np.any(valid_positive & folding):
        raise AssertionError(f"{contract.series_uid}: positive/folding supports overlap")
    logj = np.asarray(maps["canonical_logjac"], dtype=np.float32)
    logj[~valid_positive] = np.nan
    jac = np.full(shape, np.nan, dtype=np.float32)
    jac[valid_positive] = np.exp(logj[valid_positive])
    disp = stored_disp.copy()
    disp[~valid_positive] = np.nan
    comparison = compare_stored_rederived(stored_logj, logj, stored_valid & valid_positive)
    lesion = _post_lesion(contract)
    regions = build_g0_jacobian_regions(lesion, valid_positive)
    return {
        "logj": logj, "jac": jac, "disp": disp, "valid_positive": valid_positive, "folding": folding,
        "fixed_real_support": support, "deformation_support": valid_positive | folding, "lesion": regions["lesion"],
        # Keep the NPZ's historical ``peri`` key while exposing the explicit G0
        # region name used by every descriptor function.
        "peri": regions["peri_lesion"], "peri_lesion": regions["peri_lesion"],
        "whole_valid_local_roi": regions["whole_valid_local_roi"],
        "comparison": comparison, "identity": identity,
        "inverse_consistency_logjac_mae": float(maps["inverse_consistency_logjac_mae"]),
    }


def compare_stored_rederived(stored_logj: np.ndarray, derived_logj: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    support = np.asarray(valid, dtype=bool) & np.isfinite(stored_logj) & np.isfinite(derived_logj)
    if not np.any(support):
        return {"n": 0, "mae": np.nan, "max_abs": np.nan}
    delta = np.abs(np.asarray(stored_logj)[support] - np.asarray(derived_logj)[support])
    return {"n": int(delta.size), "mae": float(np.mean(delta)), "max_abs": float(np.max(delta))}


def build_g0_jacobian_regions(post_lesion: np.ndarray, valid_positive: np.ndarray) -> dict[str, np.ndarray]:
    lesion = np.asarray(post_lesion, dtype=bool) & np.asarray(valid_positive, dtype=bool)
    dilation = cv2.dilate(np.asarray(post_lesion, dtype=np.uint8), np.ones((15, 15), dtype=np.uint8)) > 0
    peri = dilation & ~np.asarray(post_lesion, dtype=bool) & np.asarray(valid_positive, dtype=bool)
    return {"lesion": lesion, "peri_lesion": peri, "whole_valid_local_roi": np.asarray(valid_positive, dtype=bool)}


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else np.nan


def extract_existing42(logj: np.ndarray, disp: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in REGION_NAMES:
        mask = np.asarray(regions[region], dtype=bool) & np.isfinite(logj) & np.isfinite(disp)
        values, moves = np.asarray(logj)[mask], np.asarray(disp)[mask]
        stats = {
            "logJ_mean": float(np.mean(values)) if values.size else np.nan,
            "logJ_median": float(np.median(values)) if values.size else np.nan,
            "logJ_std": float(np.std(values)) if values.size else np.nan,
            "logJ_P10": _q(values, .10), "logJ_P25": _q(values, .25), "logJ_P75": _q(values, .75),
            "logJ_P90": _q(values, .90), "logJ_P95": _q(values, .95),
            "abs_logJ_median": float(np.median(np.abs(values))) if values.size else np.nan,
            "abs_logJ_P90": _q(np.abs(values), .90), "abs_logJ_P95": _q(np.abs(values), .95),
            "disp_median": float(np.median(moves)) if moves.size else np.nan,
            "disp_P90": _q(moves, .90), "disp_P95": _q(moves, .95),
        }
        out.update({f"{region}_{key}": value for key, value in stats.items()})
    if list(out) != existing42_columns():
        raise AssertionError("existing42 schema order changed")
    return out


def _tau_name(tau: float) -> str:
    return f"{float(tau):.3f}".replace(".", "p")


def extract_extended_raw28(logj: np.ndarray, regions: dict[str, np.ndarray], taus: tuple[float, ...] = (.025, .05, .10),
                           component_tau: float = .05) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in REGION_NAMES:
        values = np.asarray(logj)[np.asarray(regions[region], dtype=bool) & np.isfinite(logj)]
        out[f"{region}_positive_logJ_burden"] = float(np.mean(np.maximum(values, 0))) if values.size else np.nan
        out[f"{region}_negative_logJ_burden"] = float(np.mean(np.maximum(-values, 0))) if values.size else np.nan
        for tau in taus:
            suffix = _tau_name(tau)
            out[f"{region}_expansion_fraction_tau_{suffix}"] = float(np.mean(values > tau)) if values.size else np.nan
        for tau in taus:
            suffix = _tau_name(tau)
            out[f"{region}_contraction_fraction_tau_{suffix}"] = float(np.mean(values < -tau)) if values.size else np.nan
    whole = np.asarray(regions["whole_valid_local_roi"], dtype=bool) & np.isfinite(logj)
    denominator = int(whole.sum())
    connected: dict[str, tuple[float, float]] = {}
    for name, mask in (("expansion", whole & (logj > component_tau)), ("contraction", whole & (logj < -component_tau))):
        labels, count = label(mask)  # scipy default 2-D connectivity is frozen by contract.
        largest = int(np.bincount(labels.ravel())[1:].max()) if count else 0
        connected[name] = (float(largest / denominator) if denominator else np.nan, float(count) if denominator else np.nan)
    suffix = _tau_name(component_tau)
    out[f"whole_valid_local_roi_largest_expansion_component_ratio_tau_{suffix}"] = connected["expansion"][0]
    out[f"whole_valid_local_roi_largest_contraction_component_ratio_tau_{suffix}"] = connected["contraction"][0]
    out[f"whole_valid_local_roi_expansion_component_count_tau_{suffix}"] = connected["expansion"][1]
    out[f"whole_valid_local_roi_contraction_component_count_tau_{suffix}"] = connected["contraction"][1]
    if list(out) != extended28_columns():
        raise AssertionError("extended raw28 schema order changed")
    return out


def build_jacobian_qc(contract: FrozenSeriesContract, maps: dict[str, Any], existing42: dict[str, float], cfg: dict[str, Any]) -> dict[str, Any]:
    row = contract.g0_row
    jc = cfg["jacobian_hemo"]["jacobian"]
    positive = maps["valid_positive"]
    deform = maps["deformation_support"]
    abs_p99 = _q(np.abs(maps["logj"][positive]), .99)
    disp_p95 = _q(maps["disp"][positive], .95)
    folding_rate = float(maps["folding"].sum() / deform.sum()) if deform.any() else np.nan
    reasons: list[str] = []
    compare = maps["comparison"]
    if not maps["identity"]["residual_linear_identity_verified"]:
        reasons.append("residual_linear_not_identity")
    if compare["n"] == 0 or not np.isfinite(compare["mae"]) or compare["mae"] > float(jc["stored_rederived_logj_mae_max"]) or compare["max_abs"] > float(jc["stored_rederived_logj_max_abs"]):
        reasons.append("stored_rederived_logj_mismatch")
    if not positive.any(): reasons.append("no_finite_positive_support")
    if not np.isfinite(folding_rate) or folding_rate > float(jc["folding_rate_max"]): reasons.append("folding_rate")
    if not np.isfinite(abs_p99) or abs_p99 > float(jc["abs_logj_p99_max"]): reasons.append("abs_logj_p99")
    if not np.isfinite(maps["inverse_consistency_logjac_mae"]) or maps["inverse_consistency_logjac_mae"] > float(jc["inverse_consistency_logjac_mae_max"]): reasons.append("inverse_consistency")
    if not bool(int(float(row.get("registration_valid", 0)))): reasons.append("frozen_registration_invalid")
    return {
        "series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split,
        "registration_valid_frozen": int(float(row.get("registration_valid", 0))),
        "linear_valid_frozen": int(float(row.get("linear_valid", 0))), "nonrigid_valid_frozen": int(float(row.get("nonrigid_valid", 0))),
        "metric_before": float(row.get("metric_before", np.nan) or np.nan), "metric_after": float(row.get("metric_after", np.nan) or np.nan),
        "metric_gain": float(row.get("metric_gain", np.nan) or np.nan), "structural_similarity_after": float(row.get("structural_similarity_after", np.nan) or np.nan),
        "support_loss": float(row.get("support_loss", np.nan) or np.nan),
        "inverse_consistency_logjac_mae": maps["inverse_consistency_logjac_mae"], "abs_logJ_P99": abs_p99,
        "displacement_P95": disp_p95, "corrected_folding_rate": folding_rate,
        "positive_valid_fraction": float(positive.mean()), "deformation_support_fraction": float(deform.mean()),
        "residual_linear_identity_verified": maps["identity"]["residual_linear_identity_verified"],
        "stored_vs_rederived_logj_mae": compare["mae"], "stored_vs_rederived_logj_max_abs": compare["max_abs"],
        "jacobian_map_valid": not reasons, "jacobian_invalid_reasons": ";".join(reasons),
    }


def _write_sheet(maps: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lesion, peri, whole = maps["lesion"], maps["peri"], maps["whole_valid_local_roi"]
    logj, disp = maps["logj"], maps["disp"]
    exp, con = whole & (logj > .05), whole & (logj < -.05)
    items = [
        ("Post lesion", lesion.astype(float)), ("signed logJ", logj), ("abs(logJ)", np.abs(logj)), ("displacement (native px)", disp),
        ("lesion signed logJ", np.where(lesion, logj, np.nan)), ("peri signed logJ", np.where(peri, logj, np.nan)),
        ("expansion > .05", exp), ("contraction < -.05", con), ("corrected folding", maps["folding"]),
        ("deformation support", maps["deformation_support"]), ("positive valid", maps["valid_positive"]), ("fixed real support", maps["fixed_real_support"]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for axis, (name, image) in zip(axes.ravel(), items):
        axis.imshow(image, cmap="coolwarm" if "logJ" in name else "viridis")
        axis.set_title(name); axis.axis("off")
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def write_jacobian_artifacts(contract: FrozenSeriesContract, maps: dict[str, Any], existing42: dict[str, float], extended28: dict[str, float], qc: dict[str, Any], output_dir: str | Path, *, write_sheet: bool) -> None:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "jacobian_maps_derived.npz", **{key: maps[key] for key in (
        "logj", "jac", "disp", "valid_positive", "folding", "fixed_real_support", "lesion", "peri", "whole_valid_local_roi",
    )})
    (root / "jacobian_features.json").write_text(json.dumps({
        "series_uid": contract.series_uid, "existing42": existing42, "extended_raw28": extended28, "qc": qc,
    }, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (root / "jacobian_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    if write_sheet:
        _write_sheet(maps, root / "jacobian_interpretation_sheet.png", contract.series_uid)
