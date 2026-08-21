#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.synthetic import run_synthetic_suite
from dsa_local_reg.v5_adapter import v5_core_description


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C: synthetic independent-local geometry/Jacobian tests")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_synthetic_suite()
    failures = [result for result in results if not result["passed"]]
    payload = {
        "status": "fail" if failures else "pass", "stage": "C",
        "scope": "synthetic_only_no_real_patient_registration",
        "reference_primary_geometry": cfg["geometry"]["primary_mode"],
        "v5_core": v5_core_description(cfg), "results": results, "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, args.output_dir / "STAGE_C_SYNTHETIC_RESULTS.json")
    pd.DataFrame(results).to_csv(args.output_dir / "synthetic_check_report.csv", index=False)
    atomic_json(payload, args.output_dir / ("STAGE_C_FAIL.json" if failures else "STAGE_C_PASS.json"))
    if failures:
        raise AssertionError(f"Synthetic failures: {[item['name'] for item in failures]}")
    print(f"PASS: {len(results)} synthetic geometry/Jacobian tests; output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
