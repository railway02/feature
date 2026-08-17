#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from matched_common import atomic_csv, atomic_json, require_new_directory, sha256_file


PROJECT = Path("/root/autodl-tmp/aneurysm")
RELEASE = PROJECT / "releases/adverse_prepost_matched_ablation_v1"


def formal_code_files(code_root: Path) -> list[Path]:
    allowed = {".py", ".sh", ".json", ".md", ".txt"}
    files: list[Path] = []
    for path in code_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(code_root)
        if "matched_ablation_v1" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix not in allowed:
            continue
        if any(token in path.name for token in (".bak", ".orig", ".rej")):
            continue
        files.append(path)
    return sorted(files)


def baseline_paths(project: Path) -> list[tuple[str, Path]]:
    local_root = project / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq"
    task = local_root / "adverse_prepost_series_task_v3"
    tables = local_root / "tables/local_eligible"
    models = local_root / "adverse_prepost_series_formal_models_v31"
    code_root = project / "code/api_gtmask_roi_cave_v5_fullmask_fullseq"

    rows: list[tuple[str, Path]] = [
        ("config", project / "configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json"),
        ("task", task / ".TASK_SUCCESS.json"),
        ("task", task / "task_summary.json"),
        ("task", task / "train_series_samples.csv"),
        ("task", task / "valid_series_samples.csv"),
        ("task", task / "train_grouped_folds.csv"),
        ("task", task / "train_fold_balance_audit.csv"),
        ("task", task / "train_features.npz"),
        ("task", task / "valid_features.npz"),
        ("local_table", tables / "train/series_embeddings_5120.npz"),
        ("local_table", tables / "valid/series_embeddings_5120.npz"),
        ("local_table", tables / "train/series_scalar_features.csv"),
        ("local_table", tables / "valid/series_scalar_features.csv"),
        ("whole", project / "outputs/api_fullseq_cave_v3_featurebank/feature_schema.json"),
        (
            "checkpoint",
            Path(
                "/root/autodl-tmp/CAVE_DSA/checkpoints/"
                "sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt"
            ),
        ),
        ("local_result", models / ".MODELS_SUCCESS.json"),
        ("local_result", models / "summary.json"),
        ("local_result", models / "selected_model_by_train_oof.json"),
        ("local_result", models / "metrics.csv"),
        ("local_result", models / "train_oof_predictions.csv"),
        ("local_result", models / "valid_predictions.csv"),
        (
            "review",
            project
            / "reports/api_gtmask_roi_cave_v5_fullmask_fullseq/"
            "CODEX_REVIEW_LOCAL_CAVE.md",
        ),
    ]
    rows.extend(("formal_code", path) for path in formal_code_files(code_root))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--release-dir", type=Path, default=RELEASE)
    args = parser.parse_args()

    project = args.project.resolve()
    release = require_new_directory(args.release_dir.resolve())
    records: list[dict[str, object]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for category, path in baseline_paths(project):
        path = path.resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            missing.append(key)
            continue
        try:
            relative = str(path.relative_to(project))
        except ValueError:
            relative = ""
        records.append(
            {
                "category": category,
                "path": key,
                "project_relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    if missing:
        raise FileNotFoundError("Baseline files missing:\n" + "\n".join(missing))
    manifest = pd.DataFrame(records).sort_values(
        ["category", "path"], kind="stable"
    ).reset_index(drop=True)
    manifest_path = release / "baseline_manifest.csv"
    atomic_csv(manifest, manifest_path)
    summary = {
        "status": "success",
        "version": "adverse_prepost_matched_ablation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "record_only_no_historical_file_changes",
        "formal_code_excludes": [
            "matched_ablation_v1",
            "__pycache__",
            "*.bak*",
            "*.orig",
            "*.rej",
        ],
        "file_count": int(len(manifest)),
        "total_size_bytes": int(manifest["size_bytes"].sum()),
        "category_counts": {
            str(key): int(value)
            for key, value in manifest["category"].value_counts().items()
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(summary, release / "baseline_summary.json")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

