#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import as_bool, atomic_csv, atomic_json, load_config, normalize_patient_id, parse_pipe, sha256_file


PHASES = ("pre", "post")


def require_columns(frame: pd.DataFrame, required: set[str], path: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} 缺少列：{missing}")


def phase_rows(source: pd.DataFrame, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_order, record in enumerate(source.to_dict("records")):
        for phase_order, phase in enumerate(PHASES):
            if not as_bool(record.get(f"can_run_{phase}")):
                continue
            paths = parse_pipe(record.get(f"{phase}_frame_paths", ""))
            frame_hash = str(record.get(f"{phase}_frame_list_hash", "")).strip()
            if not paths:
                raise ValueError(f"{split}/{record['series_uid']}/{phase} frame_paths 为空")
            if not frame_hash:
                raise ValueError(f"{split}/{record['series_uid']}/{phase} frame_list_hash 为空")
            declared = str(record.get(f"n_{phase}_frames", "")).strip()
            if declared and int(float(declared)) != len(paths):
                raise ValueError(
                    f"{split}/{record['series_uid']}/{phase} 帧数不闭合："
                    f"manifest={declared}, paths={len(paths)}"
                )
            rows.append({
                "phase_uid": f"{record['series_uid']}::{phase}",
                "split": split,
                "source_series_order": series_order,
                "source_phase_order": phase_order,
                "patient_id": normalize_patient_id(record.get("patient_id")),
                "series_uid": str(record["series_uid"]),
                "series_id": str(record.get("series_id", "")),
                "selected_series_id": str(record.get("selected_series_id", "")),
                "source_type": str(record.get("source_type", "")),
                "source_medical_record_root": str(record.get("source_medical_record_root", "")),
                "series_path": str(record.get("series_path", "")),
                "phase": phase,
                "api_dir": str(record.get(f"{phase}_api_dir", "")),
                "frame_paths": str(record.get(f"{phase}_frame_paths", "")),
                "frame_list_hash": frame_hash,
                "frame_indices": str(record.get(f"{phase}_frame_indices", "")),
                "selected_filenames": str(record.get(f"{phase}_selected_filenames", "")),
                "dimensions": str(record.get(f"{phase}_dimensions", "")),
                "n_frames": len(paths),
                "first_frame_path": paths[0],
                "last_frame_path": paths[-1],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    source_locks: dict[str, object] = {}
    all_phases: list[pd.DataFrame] = []
    for split in ("Train", "Valid"):
        path = Path(cfg["source_series_manifests"][split])
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        require_columns(
            source,
            {
                "patient_id", "series_uid", "series_path", "source_medical_record_root",
                "can_run_pre", "can_run_post", "pre_frame_paths", "post_frame_paths",
                "pre_frame_list_hash", "post_frame_list_hash",
            },
            str(path),
        )
        source["patient_id"] = source["patient_id"].map(normalize_patient_id)
        if source["series_uid"].duplicated().any():
            duplicate = source.loc[source["series_uid"].duplicated(False), "series_uid"].iloc[0]
            raise AssertionError(f"{split} source manifest 重复 series_uid：{duplicate}")
        expected = cfg.get("expected_source_counts", {}).get(split, {})
        counts = {
            "series": int(len(source)),
            "patients": int(source["patient_id"].nunique()),
            "pre_series": int(source["can_run_pre"].map(as_bool).sum()),
            "post_series": int(source["can_run_post"].map(as_bool).sum()),
        }
        counts["phases"] = counts["pre_series"] + counts["post_series"]
        if bool(cfg.get("strict_expected_source_counts", True)):
            for key in ("series", "patients", "pre_series", "post_series"):
                if key in expected and int(expected[key]) != counts[key]:
                    raise AssertionError(
                        f"{split} {key} 不符合冻结值：actual={counts[key]}, expected={expected[key]}"
                    )
        rows = pd.DataFrame(phase_rows(source, split))
        if rows["phase_uid"].duplicated().any() or rows["frame_list_hash"].duplicated().any():
            raise AssertionError(f"{split} source phase index 中 phase_uid/frame_list_hash 不唯一")
        atomic_csv(rows, manifests / f"source_phase_index_{split.casefold()}.csv")
        all_phases.append(rows)
        source_locks[split] = {
            **counts,
            "source_manifest": str(path),
            "source_manifest_sha256": sha256_file(path),
            "source_series_uid_order_sha256": __import__("hashlib").sha256(
                "\n".join(source["series_uid"].astype(str)).encode("utf-8")
            ).hexdigest(),
            "phase_index": str(manifests / f"source_phase_index_{split.casefold()}.csv"),
        }

    phase_index = pd.concat(all_phases, ignore_index=True)
    if phase_index["phase_uid"].duplicated().any() or phase_index["frame_list_hash"].duplicated().any():
        raise AssertionError("Train+Valid phase_uid/frame_list_hash 不唯一")
    atomic_csv(phase_index, manifests / "source_phase_index_all.csv")
    summary = {
        "version": cfg["version"],
        "total_series": sum(int(source_locks[s]["series"]) for s in source_locks),
        "total_phases": int(len(phase_index)),
        "splits": source_locks,
    }
    atomic_json(summary, manifests / "source_manifest_lock.json")
    atomic_json(summary, reports / "00_source_phase_index_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
