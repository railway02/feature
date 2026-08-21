from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt, binary_dilation, label, find_objects
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize


def common_percentile_normalize(seq: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    vals = np.asarray(seq, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(vals, dtype=np.float32)
    out = (vals - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def robust_normalize_2d(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    vals = np.asarray(img, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(vals, dtype=np.float32)
    return np.clip((vals - lo) / (hi - lo), 0, 1).astype(np.float32)


def vesselness(img01: np.ndarray, sigmas=(1, 2, 3)) -> np.ndarray:
    # DSA export polarity differs by phase/site. Keep the stronger bright/dark ridge response.
    bright = frangi(img01, sigmas=sigmas, black_ridges=False)
    dark = frangi(img01, sigmas=sigmas, black_ridges=True)
    v = np.maximum(bright, dark).astype(np.float32)
    m = float(np.nanmax(v)) if np.any(np.isfinite(v)) else 0.0
    return v / m if m > 0 else np.zeros_like(v, dtype=np.float32)


def sequence_vesselness(seq01: np.ndarray, sigmas=(1, 2, 3), workers: int = 1) -> np.ndarray:
    """Compute per-frame vesselness with bounded frame-level CPU parallelism."""
    n_workers = max(1, min(int(workers), len(seq01)))
    if n_workers == 1:
        maps = [vesselness(x, sigmas=sigmas) for x in seq01]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            maps = list(pool.map(lambda x: vesselness(x, sigmas=sigmas), seq01))
    return np.stack(maps, axis=0)


def peak_score(vmap: np.ndarray, top_fraction=0.08, spatial_mask: np.ndarray | None = None) -> float:
    vals = vmap[spatial_mask] if spatial_mask is not None and np.any(spatial_mask) else vmap.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -np.inf
    k = max(1, int(np.ceil(vals.size * float(top_fraction))))
    return float(np.partition(vals, vals.size - k)[-k:].mean())


def choose_peak_index(vseq: np.ndarray, candidate_positions=None, top_fraction=0.08, spatial_mask=None) -> tuple[int, np.ndarray]:
    n = len(vseq)
    candidates = list(range(n)) if not candidate_positions else [i for i in candidate_positions if 0 <= i < n]
    if not candidates:
        candidates = list(range(n))
    scores = np.asarray([peak_score(vseq[i], top_fraction, spatial_mask) for i in range(n)], dtype=np.float32)
    candidate_scores = scores[candidates]
    if not np.any(np.isfinite(candidate_scores)):
        raise RuntimeError("No finite peak-frame vesselness scores")
    best = int(candidates[int(np.nanargmax(candidate_scores))])
    return best, scores


def choose_contrast_peak_index(seq01: np.ndarray, vseq: np.ndarray, candidate_positions=None,
                               top_fraction=0.08, spatial_mask=None, baseline_n_frames=3,
                               vessel_weight=0.25):
    """Choose an angiographic contrast peak with vesselness as a structural tie-breaker."""
    n = len(seq01)
    candidates = list(range(n)) if not candidate_positions else [i for i in candidate_positions if 0 <= i < n]
    if not candidates:
        candidates = list(range(n))
    n0 = max(1, min(int(baseline_n_frames), n))
    baseline = np.median(np.asarray(seq01[:n0], dtype=np.float32), axis=0)
    support = np.asarray(spatial_mask, dtype=bool) if spatial_mask is not None and np.any(spatial_mask) else np.ones_like(baseline, bool)
    contrast_scores = []
    for frame in seq01:
        vals = np.abs(np.asarray(frame, dtype=np.float32) - baseline)[support]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            contrast_scores.append(-np.inf)
            continue
        k = max(1, int(np.ceil(vals.size * float(top_fraction))))
        contrast_scores.append(float(np.partition(vals, vals.size-k)[-k:].mean()))
    contrast_scores = np.asarray(contrast_scores, dtype=np.float32)
    vessel_scores = np.asarray([peak_score(vseq[i], top_fraction, support) for i in range(n)], dtype=np.float32)

    def unit(x):
        out = np.zeros_like(x, dtype=np.float32)
        finite = np.isfinite(x)
        if not np.any(finite):
            return out
        lo, hi = float(np.min(x[finite])), float(np.max(x[finite]))
        if hi > lo:
            out[finite] = (x[finite] - lo) / (hi - lo)
        return out

    w = float(np.clip(vessel_weight, 0.0, 1.0))
    combined = (1.0-w)*unit(contrast_scores) + w*unit(vessel_scores)
    candidate_scores = combined[candidates]
    if not np.any(np.isfinite(candidate_scores)):
        raise RuntimeError("No finite contrast-peak scores")
    best = int(candidates[int(np.nanargmax(candidate_scores))])
    # Keep both raw and within-sequence normalised values.  Raw values are audit
    # information; only the normalised values are commensurate in the weighted peak
    # objective because contrast and Frangi vesselness have different units.
    contrast_norm = unit(contrast_scores)
    vessel_norm = unit(vessel_scores)
    combined = (1.0-w)*contrast_norm + w*vessel_norm
    return (best, combined.astype(np.float32), contrast_scores, vessel_scores,
            contrast_norm.astype(np.float32), vessel_norm.astype(np.float32))


def make_peak_search_mask(lesion_mask: np.ndarray, valid_mask: np.ndarray | None, margin_px: int) -> np.ndarray:
    m = binary_dilation(np.asarray(lesion_mask, dtype=bool), iterations=max(0, int(margin_px)))
    if valid_mask is not None:
        m &= np.asarray(valid_mask, dtype=bool)
    return m


def pseudo_vessel_mask(vmap: np.ndarray, threshold_percentile=82.0, min_size=12,
                       valid_mask: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(vmap, dtype=np.float32)
    support = (v > 0) & np.isfinite(v)
    if valid_mask is not None:
        support &= np.asarray(valid_mask, dtype=bool)
    pos = v[support]
    if pos.size == 0:
        return np.zeros_like(v, dtype=bool)
    thr = np.percentile(pos, threshold_percentile)
    m = support & (v >= thr)
    m = remove_small_objects(m, min_size=int(min_size))
    return m.astype(bool)


def centerline_similarity_map(vessel_mask: np.ndarray, sigma_px=4.0) -> tuple[np.ndarray, np.ndarray]:
    skel = skeletonize(np.asarray(vessel_mask, dtype=bool))
    if not np.any(skel):
        return vessel_mask.astype(np.float32), skel
    d = distance_transform_edt(~skel)
    sim = np.exp(-(d * d) / (2.0 * float(sigma_px) * float(sigma_px))).astype(np.float32)
    return sim, skel


def suppress_linear_border_artifacts(vessel_mask: np.ndarray, border_fraction: float = 0.15,
                                     min_span_fraction: float = 0.25,
                                     max_thickness_fraction: float = 0.03) -> tuple[np.ndarray, list[dict]]:
    """Remove only extremely thin, long components near the acquisition border.

    DSA PNG/JPG exports frequently contain vertical/horizontal frame bars.  This rule is
    deliberately geometric and lesion-independent: a component must lie near a border,
    span a substantial fraction of the canvas, and remain extremely thin.  Central
    vessels/catheters are not removed by this function.  Output stays in the same canvas
    domain and includes an auditable list of removed component bounding boxes.
    """
    m = np.asarray(vessel_mask, dtype=bool).copy()
    h, w = m.shape
    lab, _ = label(m)
    removed = []
    for idx, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        bh, bw = ys.stop - ys.start, xs.stop - xs.start
        near_lr = xs.start < border_fraction * w or xs.stop > (1.0 - border_fraction) * w
        near_tb = ys.start < border_fraction * h or ys.stop > (1.0 - border_fraction) * h
        vertical_bar = near_lr and bh >= min_span_fraction * h and bw <= max_thickness_fraction * w
        horizontal_bar = near_tb and bw >= min_span_fraction * w and bh <= max_thickness_fraction * h
        if vertical_bar or horizontal_bar:
            comp = lab[sl] == idx
            m[sl][comp] = False
            removed.append({
                "component": int(idx), "y0": int(ys.start), "y1": int(ys.stop),
                "x0": int(xs.start), "x1": int(xs.stop), "pixels": int(comp.sum()),
                "orientation": "vertical" if vertical_bar else "horizontal",
            })
    return m, removed


def build_structure_map(vmap: np.ndarray, threshold_percentile=82.0, min_size=12, sigma_px=4.0,
                        valid_mask: np.ndarray | None = None):
    vessel = pseudo_vessel_mask(vmap, threshold_percentile, min_size, valid_mask=valid_mask)
    sim, skel = centerline_similarity_map(vessel, sigma_px)
    structure = (0.75 * sim + 0.25 * np.asarray(vmap, dtype=np.float32)).astype(np.float32)
    if valid_mask is not None:
        structure = structure * np.asarray(valid_mask, dtype=np.float32)
    return structure, vessel, skel


def central_fov_support(valid_mask: np.ndarray, border_px: int = 24) -> np.ndarray:
    """Return a whole-FOV support with export borders suppressed.

    This support is independent of the lesion annotation.  It is used for sequence peak
    selection and temporal vascular aggregation so that a lesion-centred mask cannot
    define the global correspondence problem.
    """
    support = np.asarray(valid_mask, dtype=bool).copy()
    b = max(0, int(border_px))
    if b:
        b = min(b, max(0, min(support.shape) // 4))
        support[:b] = False
        support[-b:] = False
        support[:, :b] = False
        support[:, -b:] = False
    return support


def temporal_vascular_aggregate(corrected_signal: np.ndarray, valid_mask: np.ndarray,
                                baseline_n_frames: int = 3,
                                frame_strength_fraction: float = 0.35,
                                aggregate_percentile: float = 85.0,
                                border_px: int = 24) -> tuple[np.ndarray, dict]:
    """Build a motion-corrected whole-FOV angiographic vascular aggregate.

    Input is the independently rigid-corrected raw DSA sequence ``[T,H,W]``.  A common
    intensity normalisation is followed by baseline subtraction, selection of frames
    carrying appreciable contrast, and a robust temporal percentile.  Static text,
    collimator edges, coils and much of the catheter signal are consequently suppressed
    before vesselness extraction.  Output is a dimensionless [0,1] contrast map in the
    same whole-FOV canvas domain; it is an automatic/pseudo vascular representation, not
    a manual parent-vessel segmentation.
    """
    seq = common_percentile_normalize(np.asarray(corrected_signal, dtype=np.float32))
    n = int(seq.shape[0])
    n0 = max(1, min(int(baseline_n_frames), n))
    baseline = np.median(seq[:n0], axis=0)
    contrast = np.abs(seq - baseline[None, ...])
    support = central_fov_support(valid_mask, border_px)
    if not np.any(support):
        raise ValueError("No valid whole-FOV support for temporal vascular aggregate")
    strengths = np.asarray([
        float(np.percentile(frame[support], 95)) for frame in contrast
    ], dtype=np.float32)
    max_strength = float(np.max(strengths)) if strengths.size else 0.0
    selected = np.flatnonzero(strengths >= max(1e-6, float(frame_strength_fraction) * max_strength))
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(strengths))], dtype=int)
    agg = np.percentile(contrast[selected], float(aggregate_percentile), axis=0).astype(np.float32)
    agg *= support.astype(np.float32)
    pos = agg[support & np.isfinite(agg)]
    if pos.size:
        lo, hi = np.percentile(pos, [5, 99.5])
        agg = np.clip((agg - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
    else:
        agg = np.zeros_like(agg, dtype=np.float32)
    return agg, {
        "baseline_frames": int(n0),
        "selected_frames": selected.astype(int).tolist(),
        "frame_strengths": strengths.astype(float).tolist(),
        "frame_strength_fraction": float(frame_strength_fraction),
        "aggregate_percentile": float(aggregate_percentile),
        "border_px": int(border_px),
    }
