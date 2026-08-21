#!/usr/bin/env python3
"""Aggregate terminal states, audit banks, and write the technical-only report."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.hemodynamics_v1 import COMPACT_METRICS, REGIONS, compact36_columns
from dsa_local_reg.jacobian_derived import existing42_columns, extended28_columns
from dsa_local_reg.technical_bank import audit_featurebanks, build_technical_master, write_featurebanks
from dsa_local_reg.temporal_contract import build_frozen_contracts


def _locked(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "contracts" / "LOCKED_JACOBIAN_HEMO_CONFIG.yaml").read_text(encoding="utf-8"))


def _write_schemas(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "HEMO_DESCRIPTOR_SCHEMA.md").write_text(
        "# HEMO descriptor schema\n\n"
        "Compact bank contains 36 columns: lesion/peri × six descriptors × pre/post/post-minus-pre. "
        "TTP and MTT are acquisition-frame units. NaN is retained only with explicit validity/reason.\n\n"
        + "\n".join(f"- `{item}`" for item in compact36_columns()) + "\n", encoding="utf-8")
    (root / "reports" / "JACOBIAN_DESCRIPTOR_SCHEMA.md").write_text(
        "# Jacobian descriptor schema\n\n"
        "The frozen G0 42D statistics and 28D raw extensions use a canonical residual Pre→Post field. "
        "Displacement is native pixels; no rigid/affine/global-scale determinant enters clinical Jacobian.\n\n"
        "## Existing 42D\n\n" + "\n".join(f"- `{item}`" for item in existing42_columns()) +
        "\n\n## Extended raw 28D\n\n" + "\n".join(f"- `{item}`" for item in extended28_columns()) + "\n", encoding="utf-8")


def _report(root: Path, audit: dict[str, Any], tables: dict[str, pd.DataFrame], master: pd.DataFrame, smoke_only: bool) -> dict[str, Any]:
    jac_qcs = pd.concat([value for key, value in tables.items() if key.endswith("_jac_qc")], ignore_index=True)
    temporal = pd.concat([value for key, value in tables.items() if key.endswith("_temporal_qc")], ignore_index=True)
    q = lambda column: {
        "median": float(np.nanmedian(pd.to_numeric(jac_qcs[column], errors="coerce"))) if column in jac_qcs else np.nan,
        "max": float(np.nanmax(pd.to_numeric(jac_qcs[column], errors="coerce"))) if column in jac_qcs and pd.to_numeric(jac_qcs[column], errors="coerce").notna().any() else np.nan,
    }
    payload = {
        "report_type": "Local Reference Jacobian + HEMO technical report", "generated_utc": time.strftime("%FT%TZ", time.gmtime()),
        "scope": "smoke10 engineering only" if smoke_only else "Train800 + Valid211 technical banks", "status": audit["status"],
        "coverage": {"total": int(len(master)), "train": int((master.split == "Train").sum()), "valid": int((master.split == "Valid").sum())},
        "technical_only_assertions": {"outcome_accessed": False, "g0_rigid_or_syn_rerun": False, "gpu_tasks_started": False},
        "hemo": {"valid": int(master.hemo_valid.sum()), "invalid": int((~master.hemo_valid.astype(bool)).sum()),
                 "failure_distribution": master.hemo_invalid_reasons.fillna("").value_counts().to_dict(), "compact_dimension": 36},
        "jacobian": {"valid": int(master.jacobian_map_valid.sum()), "invalid": int((~master.jacobian_map_valid.astype(bool)).sum()),
                     "failure_distribution": master.jacobian_invalid_reasons.fillna("").value_counts().to_dict(), "existing_dimension": 42,
                     "extended_raw_dimension": 28, "corrected_folding_rate": q("corrected_folding_rate"),
                     "stored_vs_rederived_logj_mae": q("stored_vs_rederived_logj_mae"), "stored_vs_rederived_logj_max_abs": q("stored_vs_rederived_logj_max_abs")},
        "audit": audit,
        "interpretation_boundary": "Residual local 2-D Pre→Post expansion/contraction in lesion-centred independent local G0 crops; not full-FOV absolute anatomy, physical area, or 3-D volume.",
    }
    root_reports = root / "reports"
    (root_reports / "LOCAL_REFERENCE_JACOBIAN_HEMO_TECHNICAL_REPORT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = ["# Local Reference Jacobian + HEMO technical report", "", f"- Status: `{audit['status']}`", f"- Scope: {payload['scope']}",
             f"- Coverage: {payload['coverage']['train']} Train / {payload['coverage']['valid']} Valid / {payload['coverage']['total']} total",
             f"- HEMO valid: {payload['hemo']['valid']}; Jacobian map valid: {payload['jacobian']['valid']}",
             "- No outcome data, G0 Rigid/SyN rerun, or GPU task was used.", "",
             "## Interpretation boundary", "", payload["interpretation_boundary"], "",
             "## Audit", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", ""]
    (root_reports / "LOCAL_REFERENCE_JACOBIAN_HEMO_TECHNICAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _smoke_gate(master: pd.DataFrame, terminal_by_uid: dict[str, dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    errors = list(audit["errors"])
    if len(master) != 10: errors.append("smoke_coverage_not_10")
    for uid, terminal in terminal_by_uid.items():
        if terminal.get("status") == "FAILED_IMPLEMENTATION": errors.append(f"{uid}:failed_implementation")
        jac = terminal.get("jacobian", {})
        if not jac.get("artifact_written", False): errors.append(f"{uid}:jacobian_assets_not_readable")
        reason = str(jac.get("jacobian_invalid_reasons", ""))
        if "stored_rederived_logj_mismatch" in reason or "residual_linear_not_identity" in reason or "jacobian_exception" in reason:
            errors.append(f"{uid}:canonical_jacobian_contract")
        if not (Path(terminal.get("temporal_artifact_dir", "")) / "temporal_sheet.png").is_file() and terminal.get("hemo", {}).get("artifact_written", False):
            errors.append(f"{uid}:temporal_sheet_missing")
    # A case can fail closed clinically, but systematic all-case temporal failure is an implementation stop.
    if int(master.hemo_valid.sum()) == 0: errors.append("smoke_no_finite_hemo_case")
    return {"status": "PASS_TO_FULL_BANK" if not errors else "STOP", "errors": errors, "hemo_valid": int(master.hemo_valid.sum()), "jacobian_valid": int(master.jacobian_map_valid.sum())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = PROJECT / "outputs" / args.run_id
    cfg = _locked(root)
    contracts = build_frozen_contracts(cfg)
    smoke_only = bool(args.smoke_only)
    if smoke_only:
        ids = set(pd.read_csv(root / "cohort" / "smoke10_series.csv", dtype=str)["series_uid"])
        contracts = [item for item in contracts if item.series_uid in ids]
        train, valid = [item for item in contracts if item.split == "Train"], [item for item in contracts if item.split == "Valid"]
    else:
        train, valid = [item for item in contracts if item.split == "Train"], [item for item in contracts if item.split == "Valid"]
        if len(train) != 800 or len(valid) != 211: raise AssertionError("full contract coverage changed")
    tables = write_featurebanks(root, train, valid)
    terminal_by_uid: dict[str, dict[str, Any]] = {}
    for key, frame in tables.items():
        if key.endswith("_terminals"):
            terminal_by_uid.update({str(row["series_uid"]): row for row in frame.to_dict("records")})
    master = build_technical_master(root, contracts, [terminal_by_uid[item.series_uid] for item in contracts])
    audit = audit_featurebanks(root, tables, master, smoke_only=smoke_only)
    _write_schemas(root)
    report = _report(root, audit, tables, master, smoke_only)
    if smoke_only:
        gate = _smoke_gate(master, terminal_by_uid, audit)
        (root / "smoke10" / "SMOKE10_ENGINEERING_GATE.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, ensure_ascii=False))
        if gate["status"] != "PASS_TO_FULL_BANK": return 2
    else:
        if audit["status"] != "PASS": return 2
    print(json.dumps({"status": audit["status"], "report": str(root / "reports" / "LOCAL_REFERENCE_JACOBIAN_HEMO_TECHNICAL_REPORT.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
