#!/usr/bin/env python3
"""Create one isolated, hash-locked Local Reference Jacobian + HEMO RUN_ID."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.hemodynamics_v1 import COMPACT_METRICS, REGIONS, compact36_columns
from dsa_local_reg.jacobian_derived import existing42_columns, extended28_columns
from dsa_local_reg.temporal_contract import (
    build_frozen_contracts, contract_rows, select_smoke10, validate_case_inputs, write_input_lock,
)


def _code_paths() -> list[Path]:
    return [
        PROJECT / "config/default.yaml", *(PROJECT / "dsa_local_reg" / name for name in (
            "temporal_contract.py", "temporal_motion.py", "hemodynamics_v1.py", "jacobian_derived.py", "technical_bank.py", "v5_adapter.py",
        )), *(PROJECT / "scripts" / name for name in (
            "prepare_jacobian_hemo_contract.py", "run_jacobian_hemo.py", "finalize_jacobian_hemo.py", "check_jacobian_hemo_code.py",
        )),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"local_reference_jacobian_hemo_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "default.yaml")
    args = parser.parse_args()
    if not args.run_id.startswith("local_reference_jacobian_hemo_"):
        raise ValueError("RUN_ID must start with local_reference_jacobian_hemo_")
    root = PROJECT / "outputs" / args.run_id
    if root.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite historical/new run directory: {root}")
    for name in ("contracts", "cohort", "smoke10", "cases/train", "cases/valid", "featurebanks", "qc", "contact_sheets", "logs", "state", "reports"):
        (root / name).mkdir(parents=True, exist_ok=False if name == "contracts" else True)
    cfg = load_config(args.config)
    contracts = build_frozen_contracts(cfg)
    validations = [validate_case_inputs(contract, stat_all_frames=True) for contract in contracts]
    invalid = [row for row in validations if not row["valid"]]
    if invalid:
        atomic_json({"status": "FAIL", "invalid_cases": invalid}, root / "contracts" / "INPUT_PRECHECK_FAIL.json")
        raise RuntimeError(f"Input contract precheck failed for {len(invalid)} series; no run may start")
    pd.DataFrame(contract_rows(contracts)).to_csv(root / "cohort" / "paired1011_contract.csv", index=False)
    smoke = select_smoke10(cfg["paths"]["fov50_series"])
    contract_ids = {item.series_uid for item in contracts}
    if not set(smoke["series_uid"]).issubset(contract_ids):
        raise AssertionError("Smoke10 contains series absent from frozen1011 contract")
    smoke.to_csv(root / "cohort" / "smoke10_series.csv", index=False)
    locked = {key: value for key, value in cfg.items() if not key.startswith("_")}
    locked.update({
        "run_id": args.run_id, "outcome_accessed": False, "g0_rigid_or_syn_rerun": False,
        "contract": "LOCAL_REFERENCE_JACOBIAN_HEMO_FINAL_EXECUTION_PLAN_ZH.md",
    })
    (root / "contracts" / "LOCKED_JACOBIAN_HEMO_CONFIG.yaml").write_text(yaml.safe_dump(locked, sort_keys=False, allow_unicode=True), encoding="utf-8")
    atomic_json({
        "name": "HEMO compact36", "dimension": 36, "regions": list(REGIONS), "metrics": list(COMPACT_METRICS),
        "representations": ["pre", "post", "post_minus_pre"], "time_unit": "frame", "columns": compact36_columns(),
        "invalid_policy": "NaN with explicit hemo_valid/reason; never biological zero",
    }, root / "contracts" / "HEMO_FEATURE_SCHEMA.json")
    atomic_json({
        "name": "jacobian_extended_raw_descriptors", "dimension": 28, "taus": [0.025, 0.05, 0.10],
        "component_tau": 0.05, "component_region": "whole_valid_local_roi", "columns": extended28_columns(),
        "existing42_columns": existing42_columns(), "not_a_version": "not JAC_V2",
    }, root / "contracts" / "JACOBIAN_EXTENDED_RAW_SCHEMA.json")
    lock = write_input_lock(cfg, root, new_code_paths=_code_paths())
    atomic_json({
        "status": "READY", "run_id": args.run_id, "paired1011": len(contracts), "train": sum(c.split == "Train" for c in contracts),
        "valid": sum(c.split == "Valid" for c in contracts), "smoke10": len(smoke), "input_precheck_invalid": 0,
        "outcome_accessed": False, "g0_rigid_or_syn_rerun": False, "input_lock_sha256_entries": len(lock["sha256"]),
    }, root / "contracts" / "PREPARE_STATUS.json")
    print(json.dumps({"run_id": args.run_id, "run_root": str(root), "contracts": len(contracts), "smoke10": len(smoke)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
