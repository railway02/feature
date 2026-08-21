from __future__ import annotations
import numpy as np


def time_density_curve(seq: np.ndarray, mask: np.ndarray, baseline_n_frames=3):
    if not np.any(mask):
        return np.full(seq.shape[0], np.nan, np.float32), 1.0
    raw = np.asarray([float(np.mean(f[mask])) for f in seq], dtype=np.float64)
    n0 = max(1, min(int(baseline_n_frames), len(raw)))
    baseline = float(np.median(raw[:n0]))
    centered = raw - baseline
    # DSA exports may encode opacified vessels as brighter or darker values.
    pos = float(np.nanmax(centered))
    neg = float(-np.nanmin(centered))
    polarity = 1.0 if pos >= neg else -1.0
    c = polarity * centered
    return c.astype(np.float32), polarity


def _linear_slope(x, y):
    if len(x) < 2 or not np.all(np.isfinite(y)):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def curve_features(curve: np.ndarray, frame_interval_seconds=None, arrival_fraction=0.10):
    c = np.asarray(curve, dtype=np.float64)
    keys = [
        "peak", "ttp", "auc", "washin", "washout", "arrival", "ttp_from_arrival",
        "mtt", "curve_width", "auc_peaknorm", "washin_peaknorm", "washout_peaknorm",
    ]
    if len(c) == 0 or not np.any(np.isfinite(c)):
        return {k: np.nan for k in keys}

    cp = np.maximum(c, 0)
    peak_i = int(np.nanargmax(cp))
    peak = float(cp[peak_i])
    dt = float(frame_interval_seconds) if frame_interval_seconds is not None else 1.0
    t = np.arange(len(cp), dtype=float) * dt

    if peak <= 0:
        arrival_i = 0
    else:
        candidates = np.where(cp >= arrival_fraction * peak)[0]
        arrival_i = int(candidates[0]) if len(candidates) else 0

    auc = float(np.trapz(cp, t))
    washin = _linear_slope(t[arrival_i:peak_i + 1], cp[arrival_i:peak_i + 1]) if peak_i > arrival_i else np.nan
    washout = _linear_slope(t[peak_i:], cp[peak_i:]) if peak_i < len(cp) - 1 else np.nan
    denom = float(np.sum(cp))
    mtt = float(np.sum(t * cp) / denom) if denom > 0 else np.nan
    half = 0.5 * peak
    above = np.where(cp >= half)[0] if peak > 0 else []
    width = float((above[-1] - above[0]) * dt) if len(above) >= 2 else 0.0

    if peak > 0:
        pn = cp / peak
        auc_pn = float(np.trapz(pn, t))
        washin_pn = _linear_slope(t[arrival_i:peak_i + 1], pn[arrival_i:peak_i + 1]) if peak_i > arrival_i else np.nan
        washout_pn = _linear_slope(t[peak_i:], pn[peak_i:]) if peak_i < len(pn) - 1 else np.nan
    else:
        auc_pn = washin_pn = washout_pn = np.nan

    return {
        "peak": peak,  # acquisition-intensity dependent; keep but interpret cautiously
        "ttp": float(t[peak_i]),
        "auc": auc,    # acquisition-intensity dependent; normalized counterpart is also emitted
        "washin": washin,
        "washout": washout,
        "arrival": float(t[arrival_i]),
        "ttp_from_arrival": float(t[peak_i] - t[arrival_i]),
        "mtt": mtt,
        "curve_width": width,
        "auc_peaknorm": auc_pn,
        "washin_peaknorm": washin_pn,
        "washout_peaknorm": washout_pn,
    }


def normalized_phase_features(curve: np.ndarray, n_samples=32, arrival_fraction=0.10,
                              washout_fraction=0.10):
    """Describe a TDC on a unit contrast phase rather than frame/second time.

    Input is a baseline-centred, polarity-corrected one-dimensional DSA curve.  The
    interval begins at first arrival and ends at the last sample above the washout
    fraction of peak (or the sequence end when washout is not observed).  It is mapped
    linearly to [0, 1] and resampled to ``n_samples``.  Returned timing quantities are
    dimensionless phase units; they must not be reported as seconds.  Empty/non-positive
    curves return NaN features and an all-NaN resampled curve.
    """
    c = np.asarray(curve, dtype=np.float64)
    keys = ["norm_ttp", "norm_auc", "norm_washin", "norm_washout", "norm_mtt", "norm_width"]
    empty = {k: np.nan for k in keys}
    n_samples = max(4, int(n_samples))
    if c.size == 0 or not np.any(np.isfinite(c)):
        return empty, np.full(n_samples, np.nan, np.float32), {"valid": False}
    cp = np.maximum(np.nan_to_num(c, nan=0.0), 0.0)
    peak = float(np.max(cp))
    if peak <= 0:
        return empty, np.full(n_samples, np.nan, np.float32), {"valid": False}
    arrival_candidates = np.flatnonzero(cp >= float(arrival_fraction) * peak)
    start = int(arrival_candidates[0]) if arrival_candidates.size else 0
    wash_candidates = np.flatnonzero(cp >= float(washout_fraction) * peak)
    end = int(wash_candidates[-1]) if wash_candidates.size else int(c.size - 1)
    # A single contrast frame is valid as a curve observation but does not support a
    # slope/time-width estimate.  Use the entire sequence when phase endpoints collapse.
    if end <= start:
        end = int(c.size - 1)
    if end <= start:
        return empty, np.full(n_samples, np.nan, np.float32), {
            "valid": False, "arrival_index": start, "end_index": end,
        }
    source_t = np.arange(start, end + 1, dtype=np.float64)
    unit_t = (source_t - float(start)) / float(end - start)
    y = cp[start:end + 1] / peak
    target_t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    sampled = np.interp(target_t, unit_t, y).astype(np.float32)
    peak_i = int(np.argmax(y))
    ttp = float(unit_t[peak_i])
    auc = float(np.trapz(y, unit_t))
    washin = _linear_slope(unit_t[:peak_i + 1], y[:peak_i + 1]) if peak_i > 0 else np.nan
    washout = _linear_slope(unit_t[peak_i:], y[peak_i:]) if peak_i < len(y) - 1 else np.nan
    denom = float(np.sum(y))
    mtt = float(np.sum(unit_t * y) / denom) if denom > 0 else np.nan
    above = np.flatnonzero(y >= 0.5)
    width = float(unit_t[above[-1]] - unit_t[above[0]]) if above.size >= 2 else 0.0
    return {
        "norm_ttp": ttp, "norm_auc": auc, "norm_washin": washin,
        "norm_washout": washout, "norm_mtt": mtt, "norm_width": width,
    }, sampled, {
        "valid": True, "arrival_index": start, "end_index": end,
        "n_source_samples": int(len(y)), "n_resampled": n_samples,
    }


def delta_features(pre: dict, post: dict, prefix="hemo") -> dict:
    out = {}
    for k in sorted(set(pre) & set(post)):
        a, b = pre[k], post[k]
        if not (np.isscalar(a) and np.isscalar(b)):
            continue
        out[f"{prefix}_{k}_pre"] = a
        out[f"{prefix}_{k}_post"] = b
        out[f"{prefix}_{k}_delta"] = b - a if np.isfinite(a) and np.isfinite(b) else np.nan
        out[f"{prefix}_{k}_reldelta"] = (
            (b - a) / (abs(a) + 1e-6) if np.isfinite(a) and np.isfinite(b) else np.nan
        )
    return out
