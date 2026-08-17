#!/usr/bin/env python3
"""Inspect whether the record-to-series mapping stage is ready for series V3.

This script does not create or modify mappings. It reports:
- finalized mapping candidates;
- api_record_v1 input-table completeness;
- audit workbook candidates;
- the exact next stage that is possible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_INPUTS = (
    "train_record_table.csv",
    "valid_record_table.csv",
    "train_all_series_manifest.csv",
    "valid_all_series_manifest.csv",
    "train_record_series_suggestions.csv",
    "valid_record_series_suggestions.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/root/autodl-tmp/aneurysm"),
    )
    parser.add_argument("--mapping-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--audit-xlsx", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    manifests = project / "manifests"
    mapping_candidates = sorted(
        {
            path.parent.resolve()
            for path in manifests.rglob("train_record_series_map.csv")
            if (path.parent / "valid_record_series_map.csv").is_file()
        },
        key=str,
    )
    mapping_dir = args.mapping_dir.resolve() if args.mapping_dir else None
    mapping_ready = False
    if mapping_dir is not None:
        mapping_ready = all(
            (mapping_dir / name).is_file()
            for name in (
                "train_record_series_map.csv",
                "valid_record_series_map.csv",
            )
        )
    elif len(mapping_candidates) == 1:
        mapping_dir = mapping_candidates[0]
        mapping_ready = True

    input_dir = (
        args.input_dir.resolve()
        if args.input_dir
        else manifests / "api_record_v1"
    )
    input_status = {
        name: (input_dir / name).is_file() for name in REQUIRED_INPUTS
    }
    inputs_ready = all(input_status.values())

    audit_candidates = sorted(
        {
            path.resolve()
            for root in (project, project.parent)
            for path in root.rglob("current_vs_train_valid_manual_audit.xlsx")
        },
        key=str,
    )
    audit_path = args.audit_xlsx.resolve() if args.audit_xlsx else None
    audit_ready = audit_path.is_file() if audit_path else len(audit_candidates) == 1
    if audit_path is None and len(audit_candidates) == 1:
        audit_path = audit_candidates[0]

    if mapping_ready:
        next_action = "build_formal_series_task_v3"
    elif inputs_ready and audit_ready:
        next_action = "run_16_finalize_api_record_v1_mapping.py"
    elif not inputs_ready:
        next_action = "build_or_restore_api_record_v1_input_tables"
    else:
        next_action = "provide_current_vs_train_valid_manual_audit.xlsx"

    payload = {
        "status": "ready" if mapping_ready else "not_ready",
        "project": str(project),
        "mapping_ready": mapping_ready,
        "resolved_mapping_dir": str(mapping_dir) if mapping_dir else "",
        "mapping_candidates": [str(path) for path in mapping_candidates],
        "input_dir": str(input_dir),
        "input_files": input_status,
        "api_record_v1_inputs_ready": inputs_ready,
        "audit_workbook_ready": audit_ready,
        "resolved_audit_workbook": str(audit_path) if audit_path else "",
        "audit_candidates": [str(path) for path in audit_candidates],
        "next_action": next_action,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if mapping_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
