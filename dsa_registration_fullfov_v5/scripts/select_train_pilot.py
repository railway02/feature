#!/usr/bin/env python
"""Deterministically select diverse Train-only series for technical registration QC.

Selection uses only manifest paths, frame counts, source geometry and lesion-mask geometry;
it never reads outcomes.  The output remains a strict subset of the supplied manifest, so
the cohort runner cannot accidentally discover/scan unlisted cases.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from dsa_reg.manifest import load_manifest, parse_frame_paths, apply_path_remap


def shape(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)
    return x.shape


def mask_stats(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)
    m = x > 0
    yy, xx = np.nonzero(m)
    center = (float(np.mean(yy)), float(np.mean(xx))) if len(xx) else (np.nan, np.nan)
    return int(m.sum()), center, x.shape


def take_first(frame, selected, description):
    for _, r in frame.iterrows():
        if str(r.series_uid) not in selected:
            selected[str(r.series_uid)] = description
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5); p.add_argument("--remap-old", default="/root/autodl-tmp")
    p.add_argument("--remap-new", default="/root/autodl-tmp")
    a = p.parse_args()
    d = load_manifest(a.manifest); d = d[d.split.astype(str) == "Train"].copy()
    remap = {a.remap_old: a.remap_new}
    rows = []
    for _, r in d.iterrows():
        pp, qp = parse_frame_paths(r.pre_frame_paths, remap)[0], parse_frame_paths(r.post_frame_paths, remap)[0]
        ps, qs = shape(pp), shape(qp)
        pa, pc, pms = mask_stats(apply_path_remap(r.pre_mask_path, remap))
        qa, qc, qms = mask_stats(apply_path_remap(r.post_mask_path, remap))
        # Selection-only difficulty descriptor in a normalised reference coordinate.
        # It is never supplied to global registration and never used as correspondence.
        centroid_gap = float(np.hypot(pc[0] / pms[0] - qc[0] / qms[0],
                                      pc[1] / pms[1] - qc[1] / qms[1]))
        rows.append({
            "series_uid": str(r.series_uid), "pre_source_shape": str(ps), "post_source_shape": str(qs),
            "source_shape_different": ps != qs,
            "frame_count_gap": abs(int(r.pre_n_frames) - int(r.post_n_frames)),
            "mask_log_area_ratio": float(abs(np.log((qa + 1) / (pa + 1)))),
            "normalised_mask_centroid_gap": centroid_gap,
            "mapping_min": float(min(r.pre_mapping_score, r.post_mapping_score)),
        })
    audit = pd.DataFrame(rows)
    work = d.merge(audit, on="series_uid", validate="one_to_one")
    selected = {}
    take_first(work.sort_values("normalised_mask_centroid_gap", ascending=False), selected,
               "large_prepost_location_difference")
    take_first(work.sort_values("frame_count_gap", ascending=False), selected, "max_frame_count_difference")
    take_first(work[work.source_shape_different].sort_values("frame_count_gap", ascending=False), selected,
               "different_source_resolution")
    take_first(work.sort_values("mask_log_area_ratio", ascending=False), selected, "largest_mask_scale_difference")
    take_first(work.sort_values("mapping_min", ascending=True), selected, "difficult_lowest_mapping")
    # If categories collide, fill deterministically with remaining high-information cases.
    for _, r in work.sort_values(["frame_count_gap", "mask_log_area_ratio"], ascending=False).iterrows():
        if len(selected) >= a.n:
            break
        if str(r.series_uid) not in selected:
            selected[str(r.series_uid)] = "diversity_fill"
    chosen = work[work.series_uid.astype(str).isin(selected)].copy()
    chosen["pilot_selection_reason"] = chosen.series_uid.astype(str).map(selected)
    chosen = chosen.sort_values("pilot_selection_reason")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(a.out, index=False)
    print(chosen[["patient_id", "series_uid", "pilot_selection_reason", "pre_source_shape", "post_source_shape",
                  "frame_count_gap", "mask_log_area_ratio", "normalised_mask_centroid_gap",
                  "mapping_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
