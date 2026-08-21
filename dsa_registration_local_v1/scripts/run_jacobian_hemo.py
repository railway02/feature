#!/usr/bin/env python3
"""Unified resumable runner for temporal/HEMO and read-only Jacobian derivation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json
from dsa_local_reg.hemodynamics_v1 import RAW_METRICS, REGIONS, compact36_columns, extract_phase_hemo, write_hemo_artifacts
from dsa_local_reg.jacobian_derived import (
    existing42_columns, extended28_columns, build_jacobian_qc, extract_existing42, extract_extended_raw28,
    rederive_canonical_maps, write_jacobian_artifacts,
)
from dsa_local_reg.technical_bank import TERMINAL
from dsa_local_reg.temporal_contract import FrozenSeriesContract, build_frozen_contracts, validate_case_inputs
from dsa_local_reg.temporal_motion import CorrectedPhaseSequence, correct_phase_sequence


def _failure_hemo(reason: str) -> dict[str, Any]:
    raw = {f"hemo_{region}_{metric}_{phase}": np.nan for phase in ("pre", "post") for region in REGIONS for metric in RAW_METRICS}
    return {
        "pre_hemo_valid": False, "post_hemo_valid": False, "hemo_valid": False, "hemo_invalid_reasons": reason,
        "compact36": {name: np.nan for name in compact36_columns()}, "raw": raw,
    }


def _failure_jacobian(reason: str, contract: FrozenSeriesContract) -> dict[str, Any]:
    qc = {
        "series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split,
        "jacobian_map_valid": False, "jacobian_invalid_reasons": reason,
    }
    return {"jacobian_map_valid": False, "jacobian_invalid_reasons": reason,
            "existing42": {name: np.nan for name in existing42_columns()},
            "extended_raw28": {name: np.nan for name in extended28_columns()}, "qc": qc}


def _temporal_sheet(pre: CorrectedPhaseSequence, post: CorrectedPhaseSequence, pre_hemo: dict[str, Any], post_hemo: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        ("Pre raw peak", pre.local.raw[pre.local.frozen_peak_index]), ("Pre corrected peak", pre.corrected_signal[np.where(pre.local.source_positions == pre.local.frozen_peak_index)[0][0]]),
        ("Post raw peak", post.local.raw[post.local.frozen_peak_index]), ("Post corrected peak", post.corrected_signal[np.where(post.local.source_positions == post.local.frozen_peak_index)[0][0]]),
        ("Pre stable support", pre.stable_valid), ("Post stable support", post.stable_valid),
    ]
    for axis, (name, image) in zip(axes.ravel()[:6], panels):
        axis.imshow(image, cmap="gray"); axis.set_title(name); axis.axis("off")
    for axis, name, a, b in ((axes[1, 2], "lesion TDC", pre_hemo["curves"]["lesion"], post_hemo["curves"]["lesion"]),
                              (axes[1, 3], "peri TDC", pre_hemo["curves"]["peri"], post_hemo["curves"]["peri"])):
        axis.plot(a, label="Pre"); axis.plot(b, label="Post"); axis.set_title(name); axis.set_xlabel("frame"); axis.legend()
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def _terminal_paths(root: Path, contract: FrozenSeriesContract) -> tuple[Path, Path]:
    case = root / "cases" / contract.split.lower() / contract.series_uid
    return case / "case_status.json", root / "state" / contract.split.lower() / f"{contract.series_uid}.json"


def _is_resumable(root: Path, contract: FrozenSeriesContract) -> dict[str, Any] | None:
    case_status, state = _terminal_paths(root, contract)
    if not case_status.is_file() or not state.is_file(): return None
    try:
        a, b = json.loads(case_status.read_text()), json.loads(state.read_text())
    except Exception:
        return None
    if a != b or a.get("status") not in TERMINAL or a.get("series_uid") != contract.series_uid:
        return None
    case = case_status.parent
    if a.get("jacobian", {}).get("artifact_written") and not (case / "jacobian" / "jacobian_maps_derived.npz").is_file(): return None
    if a.get("hemo", {}).get("artifact_written") and not (case / "hemo" / "hemodynamic_features.json").is_file(): return None
    return a


def _atomic_terminal(root: Path, contract: FrozenSeriesContract, payload: dict[str, Any]) -> None:
    case_status, state = _terminal_paths(root, contract)
    atomic_json(payload, case_status)
    atomic_json(payload, state)


def _process_case(contract: FrozenSeriesContract, run_root: str, cfg: dict[str, Any], smoke: bool, frame_workers: int) -> dict[str, Any]:
    root = Path(run_root)
    cached = _is_resumable(root, contract)
    if cached is not None:
        cached["resumed"] = True
        return cached
    case_root = root / "cases" / contract.split.lower() / contract.series_uid
    case_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = {"series_uid": contract.series_uid, "patient_id": contract.patient_id, "split": contract.split,
            "started_utc": time.strftime("%FT%TZ", time.gmtime(started)), "resumed": False,
            "g0_case_dir": str(contract.g0_case_dir), "outcome_accessed": False, "g0_rigid_or_syn_rerun": False}
    try:
        check = validate_case_inputs(contract, stat_all_frames=False)
        if not check["valid"]:
            raise RuntimeError("contract_input_invalid:" + ";".join(check["reasons"]))
    except Exception as exc:
        payload = {**base, "status": "FAILED_IMPLEMENTATION", "failure_reason": f"input_contract:{type(exc).__name__}:{exc}",
                   "hemo": _failure_hemo(f"input_contract:{exc}"), "jacobian": _failure_jacobian(f"input_contract:{exc}", contract),
                   "finished_utc": time.strftime("%FT%TZ", time.gmtime())}
        _atomic_terminal(root, contract, payload); return payload
    hemo_payload: dict[str, Any]
    try:
        temporal_root = case_root / "temporal"
        pre = correct_phase_sequence(contract.pre, cfg, temporal_root / "pre", frame_workers=frame_workers)
        post = correct_phase_sequence(contract.post, cfg, temporal_root / "post", frame_workers=frame_workers)
        pre_hemo = extract_phase_hemo(pre, contract.pre, cfg)
        post_hemo = extract_phase_hemo(post, contract.post, cfg)
        hemo_payload = write_hemo_artifacts(contract.series_uid, pre_hemo, post_hemo, case_root / "hemo")
        hemo_payload["artifact_written"] = True
        if smoke:
            np.savez_compressed(temporal_root / "pre_corrected_sequence.npz", corrected_signal=pre.corrected_signal, corrected_valid=pre.corrected_valid,
                                stable_valid=pre.stable_valid, source_frame_indices=pre.source_frame_indices, frozen_peak_index=pre.local.frozen_peak_index)
            np.savez_compressed(temporal_root / "post_corrected_sequence.npz", corrected_signal=post.corrected_signal, corrected_valid=post.corrected_valid,
                                stable_valid=post.stable_valid, source_frame_indices=post.source_frame_indices, frozen_peak_index=post.local.frozen_peak_index)
            _temporal_sheet(pre, post, pre_hemo, post_hemo, temporal_root / "temporal_sheet.png", contract.series_uid)
    except Exception as exc:
        hemo_payload = _failure_hemo(f"temporal_or_hemo_exception:{type(exc).__name__}:{exc}")
        hemo_payload["artifact_written"] = False
    try:
        maps = rederive_canonical_maps(contract, cfg)
        regions = {name: maps[name] for name in ("lesion", "peri_lesion", "whole_valid_local_roi")}
        existing = extract_existing42(maps["logj"], maps["disp"], regions)
        jc = cfg["jacobian_hemo"]["jacobian"]
        extended = extract_extended_raw28(maps["logj"], regions, tuple(float(x) for x in jc["taus"]), float(jc["component_tau"]))
        qc = build_jacobian_qc(contract, maps, existing, cfg)
        write_jacobian_artifacts(contract, maps, existing, extended, qc, case_root / "jacobian", write_sheet=smoke)
        jacobian_payload = {"jacobian_map_valid": bool(qc["jacobian_map_valid"]), "jacobian_invalid_reasons": qc["jacobian_invalid_reasons"],
                            "existing42": existing, "extended_raw28": extended, "qc": qc, "artifact_written": True}
    except Exception as exc:
        jacobian_payload = _failure_jacobian(f"jacobian_exception:{type(exc).__name__}:{exc}", contract)
        jacobian_payload["artifact_written"] = False
    hv, jv = bool(hemo_payload["hemo_valid"]), bool(jacobian_payload["jacobian_map_valid"])
    status = "COMPLETE" if hv and jv else "COMPLETE_WITH_INVALID_HEMO" if jv else "COMPLETE_WITH_INVALID_JACOBIAN" if hv else "COMPLETE_WITH_BOTH_INVALID"
    payload = {**base, "status": status, "hemo": hemo_payload, "jacobian": jacobian_payload,
               "temporal_artifact_dir": str(case_root / "temporal"), "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
               "elapsed_seconds": float(time.time() - started)}
    _atomic_terminal(root, contract, payload)
    return payload


def _worker_init(itk_threads: int) -> None:
    # Avoid V5-style nested oversubscription while keeping all CPU lanes usefully busy.
    for name in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(itk_threads)


def _load_locked(root: Path) -> dict[str, Any]:
    path = root / "contracts" / "LOCKED_JACOBIAN_HEMO_CONFIG.yaml"
    if not path.is_file(): raise FileNotFoundError(f"Run not prepared: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("outcome_accessed") or cfg.get("g0_rigid_or_syn_rerun"):
        raise RuntimeError("Invalid or non-frozen technical-only run contract")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("smoke10", "train", "valid"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--itk-threads", type=int, default=2)
    parser.add_argument("--frame-workers", type=int, default=2)
    args = parser.parse_args()
    root = PROJECT / "outputs" / args.run_id
    cfg = _load_locked(root)
    all_contracts = build_frozen_contracts(cfg)
    if args.stage == "smoke10":
        ids = set(pd.read_csv(root / "cohort" / "smoke10_series.csv", dtype=str)["series_uid"])
        contracts = [item for item in all_contracts if item.series_uid in ids]
        if len(contracts) != 10: raise AssertionError("Smoke10 contract selection changed")
    else:
        split = "Train" if args.stage == "train" else "Valid"
        contracts = [item for item in all_contracts if item.split == split]
    workers = max(1, int(args.workers)); threads = max(1, int(args.itk_threads)); frame_workers = max(1, int(args.frame_workers))
    run_info = {"run_id": args.run_id, "stage": args.stage, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
                "pid": os.getpid(), "parent_pid": os.getppid(), "workers": workers, "itk_threads": threads,
                "frame_workers": frame_workers, "gpu_tasks_started": False, "g0_rigid_or_syn_rerun": False,
                "resume_command": f"python scripts/run_jacobian_hemo.py --run-id {args.run_id} --stage {args.stage} --workers {workers} --itk-threads {threads} --frame-workers {frame_workers}"}
    atomic_json(run_info, root / "logs" / f"{args.stage}_run_info.json")
    with (root / "logs" / "resource_monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**run_info, "event": "stage_start"}) + "\n")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(threads,)) as pool:
        futures = {pool.submit(_process_case, item, str(root), cfg, args.stage == "smoke10", frame_workers): item for item in contracts}
        for number, future in enumerate(as_completed(futures), start=1):
            contract = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"series_uid": contract.series_uid, "status": "WORKER_UNCAUGHT", "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(limit=3)}
            results.append(result)
            print(f"[{args.stage}] {number}/{len(contracts)} {contract.series_uid} {result.get('status')}", flush=True)
    pd.DataFrame(results).to_csv(root / "logs" / f"{args.stage}_terminal_summary.csv", index=False)
    uncaught = [item for item in results if item.get("status") == "WORKER_UNCAUGHT"]
    with (root / "logs" / "resource_monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "stage_end", "stage": args.stage, "utc": time.strftime("%FT%TZ", time.gmtime()), "worker_uncaught": len(uncaught)}) + "\n")
    if uncaught:
        raise RuntimeError(f"{len(uncaught)} worker futures crashed before terminal state; finalizer will fail closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
