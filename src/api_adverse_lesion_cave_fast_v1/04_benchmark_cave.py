#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, atomic_text, configure_runtime, load_config


def phase_dirs(root: Path, split: str) -> dict[tuple[str, str, str], Path]:
    output = {}
    base = root / split.casefold()
    if not base.is_dir():
        return output
    for success in base.rglob(".SUCCESS.json"):
        directory = success.parent
        relative = directory.relative_to(base)
        if len(relative.parts) != 3:
            continue
        patient_id, series_uid, phase = relative.parts
        output[(patient_id, series_uid, phase)] = directory
    return output


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def compare_phase(reference: Path, candidate: Path) -> dict[str, float]:
    ref_embedding = np.load(reference / "embedding_5120.npy").astype(np.float32)
    can_embedding = np.load(candidate / "embedding_5120.npy").astype(np.float32)
    if ref_embedding.shape != (5120,) or can_embedding.shape != ref_embedding.shape:
        raise AssertionError("Embedding shape mismatch")
    difference = can_embedding.astype(np.float64) - ref_embedding.astype(np.float64)
    result = {
        "embedding_cosine": cosine(ref_embedding, can_embedding),
        "embedding_max_abs_error": float(np.max(np.abs(difference))),
        "embedding_mean_abs_error": float(np.mean(np.abs(difference))),
        "embedding_relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(ref_embedding), 1e-12)),
    }
    for name, key in (("f4_last_ensemble.fp16.npy", "f4"), ("f5_last_ensemble.fp16.npy", "f5")):
        first = np.load(reference / name).astype(np.float32)
        second = np.load(candidate / name).astype(np.float32)
        if first.shape != second.shape:
            raise AssertionError(f"{key} shape mismatch")
        result[f"{key}_max_abs_error"] = float(np.max(np.abs(second - first)))
        result[f"{key}_mean_abs_error"] = float(np.mean(np.abs(second - first)))
    with np.load(reference / "phase_trajectories_16.fp16.npz") as raw:
        ref_traj = {name: raw[name].astype(np.float32) for name in raw.files}
    with np.load(candidate / "phase_trajectories_16.fp16.npz") as raw:
        can_traj = {name: raw[name].astype(np.float32) for name in raw.files}
    if set(ref_traj) != set(can_traj):
        raise AssertionError("Trajectory schema mismatch")
    result["trajectory_max_abs_error"] = float(max(np.max(np.abs(can_traj[name] - ref_traj[name])) for name in ref_traj))
    ref_scalar = json.loads((reference / "scalar_features.json").read_text(encoding="utf-8"))
    can_scalar = json.loads((candidate / "scalar_features.json").read_text(encoding="utf-8"))
    if list(ref_scalar) != list(can_scalar):
        raise AssertionError("Scalar column order mismatch")
    errors = []
    for name in ref_scalar:
        a, b = ref_scalar[name], can_scalar[name]
        if a is None and b is None:
            continue
        if a is None or b is None:
            errors.append(float("inf"))
        else:
            errors.append(abs(float(a) - float(b)))
    result["scalar_max_abs_error"] = float(max(errors or [0.0]))
    return result


def select_benchmark_series(roi_path: Path, desired_phases: int) -> tuple[list[str], pd.DataFrame]:
    roi = pd.read_csv(roi_path, dtype=str, keep_default_na=False)
    roi = roi[(roi["split"] == "Train") & (~roi["duplicate_excluded"].astype(str).str.casefold().eq("true"))].copy()
    roi["n_frames_numeric"] = pd.to_numeric(roi["n_frames"], errors="coerce").fillna(0)
    roi["has_padding"] = roi["crop_padding"].astype(str).ne("0|0|0|0")
    rows = []
    for uid, group in roi.groupby("series_uid", sort=True):
        rows.append({
            "series_uid": str(uid),
            "patient_id": str(group["patient_id"].iloc[0]),
            "phase_count": int(len(group)),
            "has_pre": bool((group["phase"] == "pre").any()),
            "has_post": bool((group["phase"] == "post").any()),
            "has_padding": bool(group["has_padding"].any()),
            "max_frames": int(group["n_frames_numeric"].max()),
            "min_frames": int(group["n_frames_numeric"].min()),
        })
    frame = pd.DataFrame(rows)
    frame["category"] = (
        frame["has_padding"].map({True: "padding", False: "inside"}) + "|" +
        (frame["max_frames"] > 20).map({True: "long", False: "short"}) + "|" +
        (frame["has_pre"] & frame["has_post"]).map({True: "both", False: "single"})
    )
    groups = {name: group.sort_values(["max_frames", "series_uid"], ascending=[False, True]).to_dict("records") for name, group in frame.groupby("category")}
    selected = []
    phase_total = 0
    while phase_total < desired_phases and any(groups.values()):
        for name in sorted(groups):
            if not groups[name]:
                continue
            row = groups[name].pop(0)
            selected.append(row["series_uid"])
            phase_total += int(row["phase_count"])
            if phase_total >= desired_phases:
                break
    selected_frame = frame[frame["series_uid"].isin(selected)].sort_values("series_uid")
    return selected, selected_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    reports = Path(config["paths"]["reports"])
    outputs = Path(config["paths"]["outputs"])
    manifests = Path(config["paths"]["manifests"])
    benchmark_tag = config["version"].replace(".", "_")
    benchmark_root = outputs / f"benchmark_{benchmark_tag}"
    benchmark_reports = reports / f"benchmark_{benchmark_tag}"
    benchmark_reports.mkdir(parents=True, exist_ok=True)

    selected, selected_frame = select_benchmark_series(
        manifests / "roi_manifest_pred.csv", int(config["benchmark"]["phase_count"])
    )
    series_file = benchmark_reports / "benchmark_series_uids.txt"
    atomic_text("\n".join(selected) + "\n", series_file)
    atomic_csv(selected_frame, benchmark_reports / "benchmark_selection.csv")

    candidates = [
        {"name": "reference_1p_4io", "processes": 1, "io": 4, "no_empty": False},
        {"name": "candidate_1p_8io_noempty", "processes": 1, "io": 8, "no_empty": True},
        {"name": "candidate_2p_4io_noempty", "processes": 2, "io": 4, "no_empty": True},
        {"name": "candidate_2p_8io_noempty", "processes": 2, "io": 8, "no_empty": True},
        {"name": "candidate_4p_4io_noempty", "processes": 4, "io": 4, "no_empty": True},
    ]
    run_rows = []
    for candidate in candidates:
        output_root = benchmark_root / candidate["name"]
        report_root = benchmark_reports / candidate["name"]
        command = [
            config["cave_python"],
            str(Path(config["paths"]["code"]) / "03_run_sharded_extraction.py"),
            "--config", config["_config_path"],
            "--split", "Train",
            "--series-uids-file", str(series_file),
            "--gpu-processes", str(candidate["processes"]),
            "--io-workers", str(candidate["io"]),
            "--output-root", str(output_root),
            "--report-root", str(report_root),
        ]
        if candidate["no_empty"]:
            command.append("--disable-empty-cache")
        log = (benchmark_reports / f"{candidate['name']}.log").open("w", encoding="utf-8")
        result = subprocess.run(command, cwd=config["project_root"], stdout=log, stderr=subprocess.STDOUT)
        log.close()
        row = {**candidate, "exit_code": int(result.returncode)}
        summary_path = report_root / "run_summary.json"
        if summary_path.is_file():
            row.update(json.loads(summary_path.read_text(encoding="utf-8")))
        run_rows.append(row)

    reference_root = benchmark_root / candidates[0]["name"]
    ref_phases = phase_dirs(reference_root, "Train")
    comparison_rows = []
    gate = config["benchmark"]
    for candidate in candidates[1:]:
        candidate_root = benchmark_root / candidate["name"]
        can_phases = phase_dirs(candidate_root, "Train")
        if set(can_phases) != set(ref_phases):
            comparison_rows.append({
                "candidate": candidate["name"],
                "phase_set_equal": False,
                "passed": False,
                "reference_phases": len(ref_phases),
                "candidate_phases": len(can_phases),
            })
            continue
        phase_metrics = []
        for key in sorted(ref_phases):
            values = compare_phase(ref_phases[key], can_phases[key])
            phase_metrics.append({"patient_id": key[0], "series_uid": key[1], "phase": key[2], **values})
        detail = pd.DataFrame(phase_metrics)
        atomic_csv(detail, benchmark_reports / f"{candidate['name']}_numeric_detail.csv")
        summary = {
            "candidate": candidate["name"],
            "phase_set_equal": True,
            "reference_phases": len(ref_phases),
            "candidate_phases": len(can_phases),
            "embedding_cosine_min": float(detail["embedding_cosine"].min()),
            "embedding_max_abs_error_max": float(detail["embedding_max_abs_error"].max()),
            "embedding_mean_abs_error_max": float(detail["embedding_mean_abs_error"].max()),
            "embedding_relative_l2_error_max": float(detail["embedding_relative_l2_error"].max()),
            "f4_max_abs_error_max": float(detail["f4_max_abs_error"].max()),
            "f5_max_abs_error_max": float(detail["f5_max_abs_error"].max()),
            "trajectory_max_abs_error_max": float(detail["trajectory_max_abs_error"].max()),
            "scalar_max_abs_error_max": float(detail["scalar_max_abs_error"].max()),
        }
        summary["passed"] = bool(
            summary["embedding_cosine_min"] >= float(gate["cosine_minimum"]) and
            summary["embedding_max_abs_error_max"] <= float(gate["embedding_max_abs_tolerance"]) and
            summary["embedding_mean_abs_error_max"] <= float(gate["embedding_mean_abs_tolerance"]) and
            summary["embedding_relative_l2_error_max"] <= float(gate["relative_l2_tolerance"]) and
            math.isfinite(summary["scalar_max_abs_error_max"])
        )
        comparison_rows.append(summary)

    runs = pd.DataFrame(run_rows)
    comparisons = pd.DataFrame(comparison_rows)
    atomic_csv(runs, reports / "cave_runtime_benchmark.csv")
    atomic_csv(comparisons, reports / "cave_numeric_equivalence.csv")
    passing = comparisons[comparisons["passed"] == True]["candidate"].tolist() if not comparisons.empty else []
    merged = runs[runs["name"].isin(passing)].copy()
    if merged.empty:
        recommended = {
            "gpu_processes": 1,
            "io_workers": 4,
            "disable_per_view_empty_cache": False,
            "reason": "No optimized candidate passed numeric equivalence; use reference.",
        }
    else:
        best = merged.sort_values(["phases_per_second", "gpu_utilization_mean"], ascending=[False, False]).iloc[0]
        recommended = {
            "gpu_processes": int(best["processes"]),
            "io_workers": int(best["io"]),
            "disable_per_view_empty_cache": bool(best["no_empty"]),
            "candidate": str(best["name"]),
            "phases_per_second": float(best["phases_per_second"]),
            "gpu_utilization_mean": float(best["gpu_utilization_mean"]),
            "gpu_memory_used_peak_mb": float(best["gpu_memory_used_peak_mb"]),
            "reason": "Fastest candidate passing phase identity and numeric equivalence gates.",
        }
    atomic_json(recommended, reports / "recommended_runtime_config.json")
    lines = [
        "# CAVE runtime benchmark", "",
        f"- Benchmark series: {len(selected)}",
        f"- Reference phases: {len(ref_phases)}",
        f"- Recommended: `{json.dumps(recommended, ensure_ascii=False)}`", "",
        "## Runtime", "", runs.to_markdown(index=False), "",
        "## Numeric equivalence", "", comparisons.to_markdown(index=False) if not comparisons.empty else "No comparisons.",
    ]
    atomic_text("\n".join(lines) + "\n", reports / "cave_runtime_benchmark.md")
    print(json.dumps(recommended, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
