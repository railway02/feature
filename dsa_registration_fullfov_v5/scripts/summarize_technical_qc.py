#!/usr/bin/env python
"""Summarise retained series-level registration QC without reading outcomes.

The CSV labels are technical/model QC only.  They are not clinical adjudication and do not
claim that a clinician verified any registration or Jacobian hotspot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def assess(r):
    reasons = []
    if int(r.get("phase_geometry_valid", 1)) != 1:
        return "FAIL", "phase_geometry_invalid"
    global_status = str(r.get("global_compatibility_status", ""))
    if global_status == "GLOBAL_FAIL":
        return "FAIL", "global_correspondence_failed:" + str(r.get("global_compatibility_reasons", ""))
    if global_status == "GLOBAL_PASS_WITH_CAUTION":
        reasons.append("global_pass_with_caution:" + str(r.get("global_compatibility_reasons", "")))
    if int(r.get("registration_valid", 0)) != 1:
        return "FAIL", str(r.get("registration_invalid_reasons", "registration_invalid"))
    if int(r.get("pre_peak_tdc_warning", 0)) or int(r.get("post_peak_tdc_warning", 0)):
        reasons.append("peak_vs_lesion_tdc_warning")
    if finite(r.get("folding_rate")) and float(r["folding_rate"]) > 0:
        reasons.append("nonzero_folding")
    if finite(r.get("abs_logjac_p99")) and float(r["abs_logjac_p99"]) > 1.0:
        reasons.append("high_logjac_tail")
    return ("PASS_WITH_CAUTION" if reasons else "PASS"), "|".join(reasons)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); rows = []
    for path in Path(a.output_root).glob("*/*/*/features.json"):
        try:
            x = json.loads(path.read_text())
            x["case_output_dir"] = str(path.parent)
            x["technical_qc_status"], x["technical_qc_reasons"] = assess(x)
            rows.append(x)
        except Exception as e:
            rows.append({"case_output_dir": str(path.parent), "technical_qc_status": "FAIL",
                         "technical_qc_reasons": "unreadable_features:" + repr(e)})
    out = pd.DataFrame(rows)
    if len(out):
        sort_cols = [c for c in ["split", "patient_id", "series_uid"] if c in out]
        out = out.sort_values(sort_cols)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.out, index=False)
    print("technical/model QC only; clinician validation has not been performed")
    print(out.technical_qc_status.value_counts(dropna=False).to_dict() if len(out) else {})
    for col, ascending in [("global_similarity_score", True),
                           ("global_similarity_trimmed_chamfer", False),
                           ("global_similarity_coverage_5_moving", True),
                           ("global_similarity_coverage_5_fixed", True),
                           ("nonrigid_anchor_ncc", True), ("nonrigid_centerline_chamfer", False),
                           ("folding_rate", False), ("abs_logjac_p99", False), ("disp_roi_p95", False)]:
        if col in out:
            z = out[pd.to_numeric(out[col], errors="coerce").notna()].copy()
            if len(z):
                z = z.sort_values(col, ascending=ascending).head(5)
                print("OUTLIERS", col, z[[c for c in ["patient_id", "series_uid", col, "technical_qc_status"] if c in z]].to_dict("records"))


if __name__ == "__main__":
    main()
