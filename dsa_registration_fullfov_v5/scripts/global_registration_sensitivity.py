#!/usr/bin/env python
"""Train-only global-registration technical sensitivity runner.

This compares predeclared registration metric/mask alternatives on the same manifest rows
without outcomes.  It is a diagnostic/calibration tool, not per-case model selection; one
algorithm must be fixed before any Valid processing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from dsa_reg.pipeline import process_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--out-root", required=True); p.add_argument("--patient-id", type=int, action="append", required=True)
    a = p.parse_args()
    base = yaml.safe_load(Path(a.config).read_text())
    df = pd.read_csv(a.manifest)
    df = df[(df.split.astype(str) == "Train") & df.patient_id.isin(a.patient_id)]
    if len(df) != len(set(a.patient_id)):
        raise ValueError("Each requested patient_id must have one Train series in supplied manifest")
    variants = {
        "correlation_both_anchor_masks": {"metric": "correlation", "use_moving_mask": True},
        "correlation_fixed_anchor_only": {"metric": "correlation", "use_moving_mask": False},
        "mattes_fixed_anchor_only": {"metric": "mattes", "use_moving_mask": False},
    }
    rows = []
    for name, update in variants.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["paths"]["output_root"] = str(Path(a.out_root) / name)
        cfg["global_registration"].update(update)
        for _, r in df.iterrows():
            out = process_series(r, cfg)
            rows.append({"variant": name, **out})
    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(a.out_root) / "global_sensitivity_features.csv", index=False)
    print(json.dumps(pd.DataFrame(rows)[["variant", "patient_id", "registration_valid", "q_reg",
                                        "global_similarity_ncc", "nonrigid_anchor_ncc",
                                        "nonrigid_centerline_chamfer"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
