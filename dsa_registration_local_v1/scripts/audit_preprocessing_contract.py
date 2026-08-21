#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.preflight import audit_preprocessing_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A: Local Reference preprocessing/data-contract preflight")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true", help="Only validate manifests/hashes/row contracts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary, detail, task_rows, train_folds = audit_preprocessing_contract(
        cfg, verify_input_files=not args.skip_file_check
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not detail.empty:
        detail.to_csv(args.output_dir / "input_asset_audit.csv", index=False)
    task_rows.to_csv(args.output_dir / "local_reference_task_rows_all.csv", index=False)
    task_rows[task_rows["split"].eq("Train")].to_csv(args.output_dir / "local_reference_task_rows_train.csv", index=False)
    task_rows[task_rows["split"].eq("Valid")].to_csv(args.output_dir / "local_reference_task_rows_valid.csv", index=False)
    train_folds.to_csv(args.output_dir / "local_reference_train800_grouped_folds.csv", index=False)
    marker = "STAGE_A_PASS.json" if summary["status"] == "pass" else "STAGE_A_FAIL.json"
    atomic_json(summary, args.output_dir / marker)
    print(f"{summary['status'].upper()}: {summary['pair_records']} Local Reference pairs; output={args.output_dir}")
    if summary["status"] != "pass":
        print("FAILURES: " + "; ".join(summary["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
