from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt


def masked_ncc(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    if mask is None:
        mask = np.ones_like(a, bool)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 8:
        return np.nan
    x, y = a[mask].astype(float), b[mask].astype(float)
    x -= x.mean(); y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else np.nan


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    den = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / den) if den else np.nan


def vascular_ncc(vessel_a: np.ndarray, vessel_b: np.ndarray, support=None, sigma_px=2.0) -> float:
    """NCC of smoothed pseudo-vascular occupancy in a valid stable support.

    Inputs are boolean/0--1 vesselness-derived masks in the same 2-D canvas domain.
    Gaussian smoothing makes the metric tolerant to sub-pixel centreline offsets and to
    different vessel widths.  Unlike NCC of arbitrary structural intensity over a sparse
    union support, it has a direct vascular-correspondence meaning and is used for QC
    only (not as a disease-change measurement).  Output is dimensionless in [-1, 1].
    """
    a = gaussian_filter(np.asarray(vessel_a, dtype=np.float32), float(sigma_px))
    b = gaussian_filter(np.asarray(vessel_b, dtype=np.float32), float(sigma_px))
    return masked_ncc(a, b, support)


def symmetric_chamfer(skel_a: np.ndarray, skel_b: np.ndarray) -> float:
    a, b = np.asarray(skel_a, dtype=bool), np.asarray(skel_b, dtype=bool)
    if not np.any(a) or not np.any(b):
        return np.nan
    d_to_b = distance_transform_edt(~b)
    d_to_a = distance_transform_edt(~a)
    return float(0.5 * (d_to_b[a].mean() + d_to_a[b].mean()))


def registration_validity(ncc_global: float, ncc_nonrigid: float,
                          chamfer_global: float, chamfer_nonrigid: float,
                          folding_rate: float, abs_logjac_p99: float,
                          inverse_consistency_logjac_mae: float,
                          stable_pixels: int, cfg: dict,
                          dice_global: float = np.nan, dice_nonrigid: float = np.nan) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    # Sparse pseudo-vessel maps can have negative occupancy NCC despite acceptable
    # centreline proximity.  Retain NCC as a diagnostic but gate correspondence on Dice
    # and Chamfer when configured, which have direct geometric semantics here.
    min_dice = cfg.get("min_vessel_dice_after")
    if min_dice is None:
        if not np.isfinite(ncc_nonrigid) or ncc_nonrigid < float(cfg["min_anchor_ncc_after"]):
            reasons.append("low_nonrigid_anchor_ncc")
    elif not np.isfinite(dice_nonrigid) or dice_nonrigid < float(min_dice):
        reasons.append("low_nonrigid_vessel_dice")
    if np.isfinite(dice_global) and np.isfinite(dice_nonrigid):
        if dice_nonrigid + float(cfg.get("max_vessel_dice_degradation", 0.02)) < dice_global:
            reasons.append("nonrigid_vessel_dice_degraded")
    if np.isfinite(ncc_global) and np.isfinite(ncc_nonrigid):
        if ncc_nonrigid + float(cfg.get("max_ncc_degradation", 0.03)) < ncc_global:
            reasons.append("nonrigid_ncc_degraded")
    if np.isfinite(chamfer_global) and np.isfinite(chamfer_nonrigid):
        if chamfer_nonrigid > chamfer_global * float(cfg.get("max_chamfer_ratio", 1.10)):
            reasons.append("nonrigid_chamfer_degraded")
    if cfg.get("max_centerline_chamfer_after") is not None:
        if not np.isfinite(chamfer_nonrigid) or chamfer_nonrigid > float(cfg["max_centerline_chamfer_after"]):
            reasons.append("nonrigid_chamfer_too_high")
    if not np.isfinite(folding_rate) or folding_rate > float(cfg["fold_bad"]):
        reasons.append("folding_rate_high")
    if not np.isfinite(abs_logjac_p99) or abs_logjac_p99 > float(cfg["max_abs_logjac_p99"]):
        reasons.append("logjac_extreme")
    if np.isfinite(inverse_consistency_logjac_mae):
        if inverse_consistency_logjac_mae > float(cfg.get("max_inverse_consistency_logjac_mae", 0.20)):
            reasons.append("inverse_consistency_bad")
    if int(stable_pixels) < int(cfg.get("min_stable_pixels", 50)):
        reasons.append("stable_support_too_small")
    return len(reasons) == 0, reasons


def initial_qreg(ncc_global: float, ncc_nonrigid: float,
                 chamfer_global: float, chamfer_nonrigid: float,
                 folding_rate: float, abs_logjac_p99: float,
                 inverse_consistency_logjac_mae: float,
                 cfg: dict, dice_global: float = np.nan, dice_nonrigid: float = np.nan) -> float:
    """Heuristic V1 quality score. Calibrate/lock using Train QC only; never labels."""
    min_ncc = float(cfg["min_anchor_ncc_after"])
    if cfg.get("min_vessel_dice_after") is not None:
        dmin = float(cfg["min_vessel_dice_after"])
        ncc_score = np.clip((dice_nonrigid - dmin) / max(1e-6, 1.0 - dmin), 0, 1) if np.isfinite(dice_nonrigid) else 0.0
        gain_score = np.clip((dice_nonrigid - dice_global + .02) / .10, 0, 1) if np.isfinite(dice_global) and np.isfinite(dice_nonrigid) else 0.0
    else:
        ncc_score = np.clip((ncc_nonrigid - min_ncc) / max(1e-6, 1.0 - min_ncc), 0, 1) if np.isfinite(ncc_nonrigid) else 0.0
        gain_score = np.clip((ncc_nonrigid - ncc_global + 0.05) / 0.15, 0, 1) if np.isfinite(ncc_global) and np.isfinite(ncc_nonrigid) else 0.0
    if np.isfinite(chamfer_global) and chamfer_global > 0 and np.isfinite(chamfer_nonrigid):
        ch_score = np.clip(1.0 - chamfer_nonrigid / chamfer_global, 0, 1)
    else:
        ch_score = 0.5 if np.isnan(chamfer_global) or np.isnan(chamfer_nonrigid) else 0.0
    fold_bad = float(cfg["fold_bad"])
    fold_score = np.clip(1.0 - folding_rate / max(fold_bad, 1e-8), 0, 1) if np.isfinite(folding_rate) else 0.0
    jac_score = np.clip(1.0 - abs_logjac_p99 / float(cfg["max_abs_logjac_p99"]), 0, 1) if np.isfinite(abs_logjac_p99) else 0.0
    ic_thr = float(cfg.get("max_inverse_consistency_logjac_mae", 0.20))
    ic_score = np.clip(1.0 - inverse_consistency_logjac_mae / max(ic_thr, 1e-8), 0, 1) if np.isfinite(inverse_consistency_logjac_mae) else 0.5
    return float(0.25*ncc_score + 0.15*gain_score + 0.20*ch_score + 0.20*fold_score + 0.10*jac_score + 0.10*ic_score)


def save_overlay(a: np.ndarray, b: np.ndarray, path: str | Path, title=""):
    def norm(x):
        x = np.asarray(x, dtype=float)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return np.zeros_like(x, dtype=float)
        lo, hi = np.percentile(finite, [1, 99])
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.zeros(a.shape + (3,), np.float32)
    rgb[..., 0] = norm(b)
    rgb[..., 1] = norm(a)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5)); plt.imshow(rgb); plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def save_heatmap(base: np.ndarray, heat: np.ndarray, path: str | Path, title="", mask=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5)); plt.imshow(base, cmap="gray")
    h = np.asarray(heat, dtype=float).copy()
    if mask is not None:
        h[~np.asarray(mask, dtype=bool)] = np.nan
    vmax = np.nanpercentile(np.abs(h), 98) if np.any(np.isfinite(h)) else 1.0
    vmax = max(vmax, 1e-6)
    plt.imshow(h, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.55)
    plt.colorbar(label="canonical log-Jacobian (Pre→Post, expansion +)")
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def save_peak_selection_plot(raw_contrast, norm_contrast, raw_vessel, norm_vessel,
                             combined, selected_index: int, path: str | Path, title=""):
    """Save peak-selection audit curves with raw and comparable normalised scores."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(combined))
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    axes[0].plot(x, raw_contrast, label="raw contrast", color="tab:blue")
    axes[0].plot(x, raw_vessel, label="raw vesselness", color="tab:orange")
    axes[0].set_ylabel("raw score"); axes[0].legend(loc="best")
    axes[1].plot(x, norm_contrast, label="normalised contrast", color="tab:blue")
    axes[1].plot(x, norm_vessel, label="normalised vesselness", color="tab:orange")
    axes[1].plot(x, combined, label="combined", color="tab:green", linewidth=2)
    for ax in axes:
        ax.axvline(int(selected_index), color="black", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
    axes[1].set_xlabel("temporal frame position"); axes[1].set_ylabel("normalised score")
    axes[1].legend(loc="best"); fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_displacement_map(base: np.ndarray, displacement: np.ndarray, path: str | Path, mask=None):
    """Save residual displacement magnitude (pixels in the registration canvas domain)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(displacement, dtype=float).copy()
    if mask is not None:
        d[~np.asarray(mask, dtype=bool)] = np.nan
    vmax = np.nanpercentile(d, 98) if np.any(np.isfinite(d)) else 1.0
    vmax = max(float(vmax), 1e-6)
    plt.figure(figsize=(5, 5)); plt.imshow(base, cmap="gray")
    plt.imshow(d, cmap="magma", vmin=0, vmax=vmax, alpha=.60)
    plt.colorbar(label="residual Pre-global→Post displacement (canvas pixels)")
    plt.title("Residual displacement magnitude"); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
