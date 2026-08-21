#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json


def _read_unique_marker(stage_dir: Path, stage: str) -> tuple[str, dict]:
    pass_path = stage_dir / f"STAGE_{stage}_PASS.json"
    fail_path = stage_dir / f"STAGE_{stage}_FAIL.json"
    found = [path for path in (pass_path, fail_path) if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one Stage {stage} marker in {stage_dir}, found={found}")
    with found[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get("status", "fail")).casefold(), payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Local Reference Stage A–C and stop before 20-case")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stage_payloads = {}
    for stage in ("A", "B", "C"):
        status, payload = _read_unique_marker(args.output_root / f"stage_{stage.casefold()}", stage)
        stage_payloads[stage] = {"status": status, "marker": payload}
    overall = "pass" if all(item["status"] == "pass" for item in stage_payloads.values()) else "fail"
    payload = {
        "status": overall,
        "stages": stage_payloads,
        "terminal_boundary": "Stage C complete; no 20-case registration or Full Train was run",
        "next_authorized_stage": "Stage D 20-case Rigid/Similarity -> SyNOnly -> Jacobian, only in a later turn",
    }
    atomic_json(payload, args.output_root / "STAGE_A_C_FINAL_SUMMARY.json")
    print(f"{overall.upper()}: Stage A={stage_payloads['A']['status']} Stage B={stage_payloads['B']['status']} Stage C={stage_payloads['C']['status']}")
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
