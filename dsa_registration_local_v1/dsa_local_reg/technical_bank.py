"""Final aggregation and fail-closed audit for technical-only feature banks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hemodynamics_v1 import compact36_columns
from .jacobian_derived import existing42_columns, extended28_columns
from .temporal_contract import FrozenSeriesContract


IDENTITY = ["series_uid", "patient_id", "split"]
TERMINAL = {
    "COMPLETE", "COMPLETE_WITH_INVALID_HEMO", "COMPLETE_WITH_INVALID_JACOBIAN", "COMPLETE_WITH_BOTH_INVALID", "FAILED_IMPLEMENTATION",
}


def _status_path(run_root: Path, contract: FrozenSeriesContract) -> Path:
    return run_root / "state" / contract.split.lower() / f"{contract.series_uid}.json"


def _read_terminal(run_root: Path, contract: FrozenSeriesContract) -> dict[str, Any]:
    path = _status_path(run_root, contract)
    if not path.is_file():
        raise RuntimeError(f"worker_crash_unreconciled:{contract.series_uid}")
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"malformed_terminal_state:{contract.series_uid}") from exc
    if row.get("status") not in TERMINAL or row.get("series_uid") != contract.series_uid:
        raise RuntimeError(f"invalid_terminal_state:{contract.series_uid}")
    return row


def collect_terminal_rows(run_root: str | Path, split: str, expected_contracts: list[FrozenSeriesContract]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root)
    compact_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    jac_rows: list[dict[str, Any]] = []
    temporal_qc: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    for contract in expected_contracts:
        item = _read_terminal(root, contract)
        terminals.append(item)
        identity = {"series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split}
        hemo = item.get("hemo", {})
        jac = item.get("jacobian", {})
        compact_rows.append({**identity, "pre_hemo_valid": hemo.get("pre_hemo_valid", False), "post_hemo_valid": hemo.get("post_hemo_valid", False),
                             "hemo_valid": hemo.get("hemo_valid", False), "hemo_invalid_reasons": hemo.get("hemo_invalid_reasons", item.get("hemo_failure_reason", "")),
                             **hemo.get("compact36", {key: np.nan for key in compact36_columns()})})
        raw_rows.append({**identity, "hemo_valid": hemo.get("hemo_valid", False), "hemo_invalid_reasons": hemo.get("hemo_invalid_reasons", item.get("hemo_failure_reason", "")),
                         **hemo.get("raw", {})})
        jac_rows.append({**identity, "jacobian_map_valid": jac.get("jacobian_map_valid", False),
                         "jacobian_invalid_reasons": jac.get("jacobian_invalid_reasons", item.get("jacobian_failure_reason", "")),
                         **jac.get("existing42", {key: np.nan for key in existing42_columns()}),
                         **jac.get("extended_raw28", {key: np.nan for key in extended28_columns()})})
        temporal_qc.append({**identity, "status": item["status"], "pre_hemo_valid": hemo.get("pre_hemo_valid", False),
                            "post_hemo_valid": hemo.get("post_hemo_valid", False), "hemo_valid": hemo.get("hemo_valid", False),
                            "hemo_invalid_reasons": hemo.get("hemo_invalid_reasons", item.get("hemo_failure_reason", "")),
                            "temporal_artifact_dir": item.get("temporal_artifact_dir", "")})
    return (pd.DataFrame(compact_rows), pd.DataFrame(raw_rows), pd.DataFrame(jac_rows), pd.DataFrame(temporal_qc), terminals)


def _ordered(frame: pd.DataFrame, contracts: list[FrozenSeriesContract]) -> pd.DataFrame:
    expected = [item.series_uid for item in contracts]
    if frame["series_uid"].duplicated().any():
        raise AssertionError("duplicate series in terminal aggregation")
    return frame.set_index("series_uid").reindex(expected).reset_index()


def write_featurebanks(run_root: str | Path, train_contracts: list[FrozenSeriesContract], valid_contracts: list[FrozenSeriesContract]) -> dict[str, pd.DataFrame]:
    root = Path(run_root)
    result: dict[str, pd.DataFrame] = {}
    for split, contracts, token in (("Train", train_contracts, "train800"), ("Valid", valid_contracts, "valid211")):
        if contracts:
            compact, raw, jac, tqc, terminals = collect_terminal_rows(root, split, contracts)
            compact, raw, jac, tqc = (_ordered(x, contracts) for x in (compact, raw, jac, tqc))
        else:
            compact = pd.DataFrame(columns=IDENTITY + ["pre_hemo_valid", "post_hemo_valid", "hemo_valid", "hemo_invalid_reasons"] + compact36_columns())
            raw = pd.DataFrame(columns=IDENTITY + ["hemo_valid", "hemo_invalid_reasons"])
            jac = pd.DataFrame(columns=IDENTITY + ["jacobian_map_valid", "jacobian_invalid_reasons"] + existing42_columns() + extended28_columns())
            tqc = pd.DataFrame(columns=IDENTITY + ["status", "pre_hemo_valid", "post_hemo_valid", "hemo_valid", "hemo_invalid_reasons", "temporal_artifact_dir"])
            terminals = []
        existing = jac[IDENTITY + ["jacobian_map_valid", "jacobian_invalid_reasons"] + existing42_columns()]
        extended = jac[IDENTITY + ["jacobian_map_valid", "jacobian_invalid_reasons"] + extended28_columns()]
        jac_qc = pd.DataFrame([item.get("jacobian", {}).get("qc", {**{key: getattr(c, key) for key in IDENTITY}, "jacobian_map_valid": False, "jacobian_invalid_reasons": item.get("jacobian_failure_reason", "")}) for item, c in zip(terminals, contracts)])
        jac_qc = _ordered(jac_qc, contracts) if contracts else pd.DataFrame(columns=IDENTITY + ["jacobian_map_valid", "jacobian_invalid_reasons"])
        paths = {
            "compact": root / "featurebanks" / f"{token}_hemo_compact36.csv",
            "raw": root / "featurebanks" / f"{token}_hemo_raw.csv",
            "existing": root / "featurebanks" / f"{token}_jacobian_existing42.csv",
            "extended": root / "featurebanks" / f"{token}_jacobian_extended_raw28.csv",
            "jac": root / "featurebanks" / f"{token}_jacobian_bank.csv",
            "jac_qc": root / "qc" / f"{token}_jacobian_qc.csv",
            "temporal_qc": root / "qc" / f"{token}_temporal_qc.csv",
        }
        for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
        compact.to_csv(paths["compact"], index=False); raw.to_csv(paths["raw"], index=False)
        existing.to_csv(paths["existing"], index=False); extended.to_csv(paths["extended"], index=False); jac.to_csv(paths["jac"], index=False)
        jac_qc.to_csv(paths["jac_qc"], index=False); tqc.to_csv(paths["temporal_qc"], index=False)
        result.update({f"{split}_compact": compact, f"{split}_raw": raw, f"{split}_jac": jac, f"{split}_jac_qc": jac_qc, f"{split}_temporal_qc": tqc, f"{split}_terminals": pd.DataFrame(terminals)})
    return result


def build_technical_master(run_root: str | Path, contracts: list[FrozenSeriesContract], terminals: list[dict[str, Any]]) -> pd.DataFrame:
    root = Path(run_root)
    rows = []
    for contract, item in zip(contracts, terminals):
        case_rel = Path("cases") / contract.split.lower() / contract.series_uid
        rows.append({
            "series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split,
            "g0_case_dir": str(contract.g0_case_dir), "g0_maps_path": str(contract.g0_case_dir / "rigid_maps.npz"),
            "g0_warp_path": str(contract.g0_case_dir / "rigid_syn_1Warp.nii.gz"), "g0_inverse_warp_path": str(contract.g0_case_dir / "rigid_syn_1InverseWarp.nii.gz"),
            "terminal_status": item.get("status"), "jacobian_map_valid": item.get("jacobian", {}).get("jacobian_map_valid", False),
            "hemo_valid": item.get("hemo", {}).get("hemo_valid", False),
            "jacobian_invalid_reasons": item.get("jacobian", {}).get("jacobian_invalid_reasons", item.get("jacobian_failure_reason", "")),
            "hemo_invalid_reasons": item.get("hemo", {}).get("hemo_invalid_reasons", item.get("hemo_failure_reason", "")),
            "jacobian_feature_row_path": str(case_rel / "jacobian" / "jacobian_features.json"),
            "hemo_feature_row_path": str(case_rel / "hemo" / "hemodynamic_features.json"),
            "jacobian_contact_sheet": str(case_rel / "jacobian" / "jacobian_interpretation_sheet.png"),
            "temporal_contact_sheet": str(case_rel / "temporal" / "temporal_sheet.png"),
        })
    frame = _ordered(pd.DataFrame(rows), contracts)
    path = root / "qc" / "TECHNICAL_BANK_MASTER_1011.csv"; path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)
    return frame


def _finite_violations(frame: pd.DataFrame, columns: list[str], valid_column: str) -> list[str]:
    violations: list[str] = []
    for _, row in frame.iterrows():
        if bool(row.get(valid_column, False)):
            bad = [column for column in columns if column in row and not np.isfinite(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])]
            if bad: violations.append(f"{row['series_uid']}:{valid_column}:" + ",".join(bad))
        elif not str(row.get("hemo_invalid_reasons" if valid_column == "hemo_valid" else "jacobian_invalid_reasons", "")).strip():
            violations.append(f"{row['series_uid']}:{valid_column}:missing_failure_reason")
    return violations


def audit_featurebanks(run_root: str | Path, tables: dict[str, pd.DataFrame], master: pd.DataFrame, *, smoke_only: bool) -> dict[str, Any]:
    all_compact = pd.concat([tables[key] for key in tables if key.endswith("_compact")], ignore_index=True)
    all_jac = pd.concat([tables[key] for key in tables if key.endswith("_jac")], ignore_index=True)
    all_terminal = pd.concat([tables[key] for key in tables if key.endswith("_terminals")], ignore_index=True)
    expected = 10 if smoke_only else 1011
    errors: list[str] = []
    if len(master) != expected: errors.append(f"coverage={len(master)} expected={expected}")
    if master["series_uid"].duplicated().any(): errors.append("duplicate_series_uid")
    if (all_terminal["status"] == "FAILED_IMPLEMENTATION").any(): errors.append("failed_implementation_terminal")
    errors += _finite_violations(all_compact, compact36_columns(), "hemo_valid")
    errors += _finite_violations(all_jac, existing42_columns() + extended28_columns(), "jacobian_map_valid")
    numeric = pd.concat([all_compact.select_dtypes(include=[np.number]), all_jac.select_dtypes(include=[np.number])], axis=1)
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any(): errors.append("infinite_feature_value")
    payload = {
        "status": "PASS" if not errors else "FAIL", "smoke_only": smoke_only, "expected_series": expected,
        "coverage": int(len(master)), "train_rows": int((master["split"] == "Train").sum()), "valid_rows": int((master["split"] == "Valid").sum()),
        "duplicate_series_uid": int(master["series_uid"].duplicated().sum()), "hemo_valid": int(master["hemo_valid"].sum()),
        "jacobian_valid": int(master["jacobian_map_valid"].sum()), "errors": errors,
    }
    root = Path(run_root); (root / "qc").mkdir(parents=True, exist_ok=True)
    (root / "qc" / "FEATUREBANK_AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
