#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matched_common import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    read_npz,
    require_new_directory,
    sha256_file,
    stable_hash,
)


PROJECT = Path("/root/autodl-tmp/aneurysm")
LOCAL_OUTPUT = PROJECT / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq"
BASE_TASK = LOCAL_OUTPUT / "adverse_prepost_series_task_v3"
OUTPUT_TASK = LOCAL_OUTPUT / "adverse_prepost_matched_ablation_task_v1"
REPORTS = PROJECT / "reports/api_gtmask_roi_cave_v5_fullmask_fullseq"

MORPHOLOGY_BASE_COLUMNS = (
    "mask_area_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_aspect_ratio",
    "bbox_fill_ratio",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "circularity",
    "solidity",
    "component_count",
    "largest_component_ratio",
    "roi_area_ratio",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "prediction",
    "probability",
    "patient",
    "series",
    "record",
    "split",
    "phase_uid",
    "path",
    "sha256",
    "fold",
)


def temporal_signature(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "indices": block.get("indices"),
            "view_indices": block.get("view_indices"),
        }
        for block in metadata.get("blocks", [])
    ]


def morphology_feature_names() -> list[str]:
    return [
        f"{prefix}_{column}"
        for prefix in ("pre", "post", "delta_post_minus_pre")
        for column in MORPHOLOGY_BASE_COLUMNS
    ]


def assert_feature_names_safe(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise AssertionError("Duplicate morphology feature names")
    violations = [
        name
        for name in names
        if any(token in name.casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if violations:
        raise AssertionError(f"Leakage/identity-like morphology names: {violations}")


def build_morphology_matrix(
    pre: np.ndarray,
    post: np.ndarray,
) -> np.ndarray:
    pre = np.asarray(pre, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    if pre.shape != post.shape or pre.ndim != 2:
        raise AssertionError(f"Bad morphology phase shapes: {pre.shape}, {post.shape}")
    output = np.concatenate([pre, post, post - pre], axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise AssertionError("Nonfinite mask morphology features")
    return output


def load_fixed_split(task: Path, split: str) -> dict[str, np.ndarray]:
    payload = read_npz(task / f"{split}_features.npz")
    required = {"deep", "target", "series_uid", "patient_id"}
    if split == "train":
        required.add("fold")
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Base {split} task missing {sorted(missing)}")
    n = len(payload["series_uid"])
    if payload["deep"].shape != (n, 10240):
        raise AssertionError(f"Base Local deep shape changed: {payload['deep'].shape}")
    if not np.isfinite(payload["deep"]).all():
        raise AssertionError(f"Base {split} Local deep contains nonfinite values")
    return payload


def validate_fixed_order(task: Path, split: str, payload: dict[str, np.ndarray]) -> None:
    samples = pd.read_csv(
        task / f"{split}_series_samples.csv",
        dtype={"series_uid": str, "patient_id": str},
        keep_default_na=False,
    )
    if not np.array_equal(
        payload["series_uid"].astype(str), samples["series_uid"].astype(str).to_numpy()
    ):
        raise AssertionError(f"{split}: base NPZ/sample UID order mismatch")
    if not np.array_equal(
        payload["target"].astype(np.int64), samples["target"].to_numpy(np.int64)
    ):
        raise AssertionError(f"{split}: base NPZ/sample target mismatch")
    if split == "train":
        folds = pd.read_csv(
            task / "train_grouped_folds.csv",
            dtype={"series_uid": str, "patient_id": str},
            keep_default_na=False,
        )
        if not np.array_equal(
            payload["series_uid"].astype(str), folds["series_uid"].to_numpy()
        ):
            raise AssertionError("Base Train NPZ/fold UID order mismatch")
        if not np.array_equal(
            payload["fold"].astype(np.int64), folds["fold"].to_numpy(np.int64)
        ):
            raise AssertionError("Base Train NPZ/fold assignment mismatch")


def phase_directory(
    local_featurebank: Path,
    split: str,
    patient_id: str,
    series_uid: str,
    phase: str,
) -> Path:
    return local_featurebank / split / patient_id / series_uid / phase


def common_arrays(payload: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    arrays = {
        "series_uid": payload["series_uid"].astype(str),
        "patient_id": payload["patient_id"].astype(str),
        "target": payload["target"].astype(np.int64),
    }
    if split == "train":
        arrays["fold"] = payload["fold"].astype(np.int64)
    return arrays


def write_experiment(
    root: Path,
    experiment: str,
    train_arrays: dict[str, np.ndarray],
    valid_arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> dict[str, Any]:
    directory = root / experiment
    directory.mkdir(parents=False, exist_ok=False)
    train_path = directory / "train_features.npz"
    valid_path = directory / "valid_features.npz"
    atomic_npz(train_path, **train_arrays)
    atomic_npz(valid_path, **valid_arrays)
    payload = {
        **summary,
        "status": "success",
        "experiment": experiment,
        "train_npz": str(train_path),
        "train_npz_sha256": sha256_file(train_path),
        "valid_npz": str(valid_path),
        "valid_npz_sha256": sha256_file(valid_path),
    }
    atomic_json(payload, directory / "task_summary.json")
    atomic_json(payload, directory / ".TASK_SUCCESS.json")
    return payload


def verify_experiment_identity(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    split: str,
) -> None:
    for key in ("series_uid", "patient_id", "target"):
        if not np.array_equal(base[key].astype(str), candidate[key].astype(str)):
            raise AssertionError(f"{split}: candidate {key} differs from fixed task")
    if split == "train" and not np.array_equal(
        base["fold"].astype(np.int64), candidate["fold"].astype(np.int64)
    ):
        raise AssertionError("Train candidate folds differ from fixed task")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--base-task", type=Path, default=BASE_TASK)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_TASK)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    args = parser.parse_args()

    project = args.project.resolve()
    base_task = args.base_task.resolve()
    output = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    config_path = project / "configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    whole_root = Path(config["whole_featurebank"]).resolve()
    expected_whole = project / "outputs/api_fullseq_cave_v3_featurebank"
    if whole_root != expected_whole.resolve():
        raise AssertionError(
            f"Configured Whole root differs from expected metadata candidate: {whole_root}"
        )
    if not whole_root.is_dir() or not (whole_root / "feature_schema.json").is_file():
        raise FileNotFoundError(f"Whole featurebank/schema missing: {whole_root}")

    manifests = Path(config["paths"]["manifests"])
    roi = pd.read_csv(
        manifests / "roi_phase_manifest_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    morphology = pd.read_csv(
        manifests / "mask_morphology_phase_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    for name, frame in (("ROI", roi), ("morphology", morphology)):
        if frame["phase_uid"].duplicated().any():
            raise AssertionError(f"{name} has duplicate phase_uid")
    roi_by_phase = roi.set_index("phase_uid", drop=False)
    morphology_by_phase = morphology.set_index("phase_uid", drop=False)
    local_featurebank = Path(config["paths"]["outputs"]) / "cave_local_eligible_featurebank"

    names = morphology_feature_names()
    assert_feature_names_safe(names)
    print(f"M0_FEATURE_DIM={len(names)}")
    print("M0_FEATURE_COLUMNS=" + "|".join(names))

    split_features: dict[str, dict[str, np.ndarray]] = {}
    coverage_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for split in ("train", "valid"):
        base = load_fixed_split(base_task, split)
        validate_fixed_order(base_task, split, base)
        series_uid = base["series_uid"].astype(str)
        patient_id = base["patient_id"].astype(str)
        n = len(series_uid)
        whole_phase = np.empty((n, 2, 5120), dtype=np.float32)
        morph_phase = np.empty(
            (n, 2, len(MORPHOLOGY_BASE_COLUMNS)), dtype=np.float32
        )

        for index, (uid, pid) in enumerate(zip(series_uid, patient_id)):
            for phase_index, phase in enumerate(("pre", "post")):
                phase_uid = f"{uid}::{phase}"
                audit: dict[str, Any] = {
                    "split": split,
                    "series_index": index,
                    "patient_id": pid,
                    "series_uid": uid,
                    "phase": phase,
                    "phase_uid": phase_uid,
                    "morphology_available": 0,
                    "whole_metadata_available": 0,
                    "whole_embedding_available": 0,
                    "local_metadata_available": 0,
                    "frame_list_hash_equal": 0,
                    "temporal_indices_equal": 0,
                    "status": "failed",
                    "reason": "",
                }
                try:
                    if phase_uid not in roi_by_phase.index:
                        raise KeyError("missing_roi_phase")
                    if phase_uid not in morphology_by_phase.index:
                        raise KeyError("missing_morphology_phase")
                    roi_row = roi_by_phase.loc[phase_uid]
                    morph_row = morphology_by_phase.loc[phase_uid]
                    if (
                        str(roi_row["series_uid"]) != uid
                        or str(roi_row["patient_id"]) != pid
                        or str(roi_row["phase"]) != phase
                    ):
                        raise AssertionError("roi_identity_mismatch")
                    if (
                        str(morph_row["series_uid"]) != uid
                        or str(morph_row["patient_id"]) != pid
                        or str(morph_row["phase"]) != phase
                    ):
                        raise AssertionError("morphology_identity_mismatch")
                    values = pd.to_numeric(
                        morph_row[list(MORPHOLOGY_BASE_COLUMNS)], errors="coerce"
                    ).to_numpy(np.float32)
                    if values.shape != (len(MORPHOLOGY_BASE_COLUMNS),) or not np.isfinite(
                        values
                    ).all():
                        raise AssertionError("nonfinite_morphology")
                    morph_phase[index, phase_index] = values
                    audit["morphology_available"] = 1

                    whole_metadata_path = Path(roi_row["whole_metadata_path"]).resolve()
                    try:
                        whole_metadata_path.relative_to(whole_root)
                    except ValueError as exc:
                        raise AssertionError("whole_path_outside_configured_root") from exc
                    whole_embedding_path = whole_metadata_path.parent / "embedding_5120.npy"
                    if not whole_metadata_path.is_file():
                        raise FileNotFoundError("missing_whole_metadata")
                    audit["whole_metadata_available"] = 1
                    if not whole_embedding_path.is_file():
                        raise FileNotFoundError("missing_whole_embedding")
                    whole_embedding = np.load(whole_embedding_path, allow_pickle=False)
                    if whole_embedding.shape != (5120,) or not np.isfinite(
                        whole_embedding
                    ).all():
                        raise AssertionError("bad_whole_embedding")
                    whole_phase[index, phase_index] = whole_embedding.astype(np.float32)
                    audit["whole_embedding_available"] = 1

                    local_directory = phase_directory(
                        local_featurebank, split, pid, uid, phase
                    )
                    local_metadata_path = local_directory / "metadata.json"
                    if not local_metadata_path.is_file():
                        raise FileNotFoundError("missing_local_metadata")
                    audit["local_metadata_available"] = 1
                    whole_metadata = json.loads(
                        whole_metadata_path.read_text(encoding="utf-8")
                    )
                    local_metadata = json.loads(
                        local_metadata_path.read_text(encoding="utf-8")
                    )
                    expected_hash = str(roi_row["frame_list_hash"])
                    if (
                        str(whole_metadata.get("frame_list_hash", "")) != expected_hash
                        or str(local_metadata.get("frame_list_hash", "")) != expected_hash
                    ):
                        raise AssertionError("frame_list_hash_mismatch")
                    audit["frame_list_hash_equal"] = 1
                    if temporal_signature(whole_metadata) != temporal_signature(
                        local_metadata
                    ):
                        raise AssertionError("temporal_indices_mismatch")
                    audit["temporal_indices_equal"] = 1
                    audit["whole_metadata_path"] = str(whole_metadata_path)
                    audit["whole_embedding_path"] = str(whole_embedding_path)
                    audit["local_metadata_path"] = str(local_metadata_path)
                    audit["status"] = "ok"
                except Exception as exc:
                    audit["reason"] = f"{type(exc).__name__}:{exc}"
                    failures.append(dict(audit))
                coverage_rows.append(audit)

        mask = build_morphology_matrix(morph_phase[:, 0], morph_phase[:, 1])
        whole_deep = whole_phase.reshape(n, 10240).astype(np.float32)
        local_deep = np.asarray(base["deep"], dtype=np.float32)
        split_features[split] = {
            **common_arrays(base, split),
            "mask_features": mask,
            "whole_deep": whole_deep,
            "local_deep": local_deep,
        }

    coverage = pd.DataFrame(coverage_rows)
    coverage_path = report_dir / "matched_ablation_v1_coverage.csv"
    failure_path = report_dir / "matched_ablation_v1_missing_or_mismatch.csv"
    atomic_csv(coverage, coverage_path)
    atomic_csv(pd.DataFrame(failures, columns=coverage.columns), failure_path)
    split_counts = {
        split: {
            "series": int(
                len(split_features[split]["series_uid"])
            ),
            "phases": int((coverage["split"] == split).sum()),
            "ok_phases": int(
                (
                    (coverage["split"] == split)
                    & (coverage["status"] == "ok")
                ).sum()
            ),
        }
        for split in ("train", "valid")
    }
    coverage_summary = {
        "status": "failed" if failures else "success",
        "configured_whole_featurebank": str(whole_root),
        "whole_feature_schema": str(whole_root / "feature_schema.json"),
        "whole_feature_schema_sha256": sha256_file(whole_root / "feature_schema.json"),
        "fixed_base_task": str(base_task),
        "fixed_base_task_success_sha256": sha256_file(base_task / ".TASK_SUCCESS.json"),
        "splits": split_counts,
        "failure_count": len(failures),
        "coverage_csv": str(coverage_path),
        "missing_or_mismatch_csv": str(failure_path),
        "frame_list_hash_mismatch_count": int(
            (coverage["frame_list_hash_equal"] != 1).sum()
        ),
        "temporal_indices_mismatch_count": int(
            (coverage["temporal_indices_equal"] != 1).sum()
        ),
    }
    atomic_json(
        coverage_summary, report_dir / "matched_ablation_v1_coverage.json"
    )
    if failures:
        raise RuntimeError(
            f"Coverage failed for {len(failures)} phase rows; tasks were not created"
        )

    output = require_new_directory(output)
    atomic_csv(coverage, output / "coverage.csv")
    atomic_csv(pd.DataFrame(failures, columns=coverage.columns), output / "missing_or_mismatch.csv")
    atomic_json(coverage_summary, output / "coverage_summary.json")
    morphology_schema = {
        "source": str(manifests / "mask_morphology_phase_eligible.csv"),
        "source_sha256": sha256_file(
            manifests / "mask_morphology_phase_eligible.csv"
        ),
        "base_numeric_columns": list(MORPHOLOGY_BASE_COLUMNS),
        "construction": "Pre + Post + (Post - Pre)",
        "feature_names": names,
        "feature_dimension": len(names),
        "forbidden_tokens_checked": list(FORBIDDEN_FEATURE_TOKENS),
        "contains_labels_or_predictions": False,
        "contains_identity_fields": False,
    }
    atomic_json(morphology_schema, output / "m0_feature_schema.json")

    experiments: dict[str, Any] = {}
    for experiment in ("M0", "W0", "WL"):
        train_base = split_features["train"]
        valid_base = split_features["valid"]
        common_train = {
            key: train_base[key]
            for key in ("series_uid", "patient_id", "target", "fold")
        }
        common_valid = {
            key: valid_base[key]
            for key in ("series_uid", "patient_id", "target")
        }
        if experiment == "M0":
            train_arrays = {
                **common_train,
                "mask_features": train_base["mask_features"],
                "feature_names": np.asarray(names, dtype=str),
            }
            valid_arrays = {
                **common_valid,
                "mask_features": valid_base["mask_features"],
                "feature_names": np.asarray(names, dtype=str),
            }
            feature_summary = {
                "input": "Mask-only Pre+Post+delta numeric morphology",
                "train_feature_shape": list(train_base["mask_features"].shape),
                "valid_feature_shape": list(valid_base["mask_features"].shape),
                "logical_feature_dimension": len(names),
                "contains_local_embedding": False,
                "contains_whole_embedding": False,
            }
        elif experiment == "W0":
            train_arrays = {**common_train, "whole_deep": train_base["whole_deep"]}
            valid_arrays = {**common_valid, "whole_deep": valid_base["whole_deep"]}
            feature_summary = {
                "input": "Whole Pre 5120 + Whole Post 5120",
                "train_feature_shape": list(train_base["whole_deep"].shape),
                "valid_feature_shape": list(valid_base["whole_deep"].shape),
                "logical_feature_dimension": 10240,
                "contains_local_embedding": False,
                "contains_whole_embedding": True,
            }
        else:
            train_arrays = {
                **common_train,
                "whole_deep": train_base["whole_deep"],
                "local_deep": train_base["local_deep"],
            }
            valid_arrays = {
                **common_valid,
                "whole_deep": valid_base["whole_deep"],
                "local_deep": valid_base["local_deep"],
            }
            feature_summary = {
                "input": "Whole deep + Local deep as two independently preprocessed branches",
                "train_branch_shapes": {
                    "whole": list(train_base["whole_deep"].shape),
                    "local": list(train_base["local_deep"].shape),
                },
                "valid_branch_shapes": {
                    "whole": list(valid_base["whole_deep"].shape),
                    "local": list(valid_base["local_deep"].shape),
                },
                "logical_feature_shape_train": [len(train_base["series_uid"]), 20480],
                "logical_feature_shape_valid": [len(valid_base["series_uid"]), 20480],
                "logical_feature_dimension": 20480,
                "preprocessing_contract": (
                    "Whole and Local scaler/PCA fit separately inside each Train fold; "
                    "never prefit or fit on Valid"
                ),
                "contains_local_embedding": True,
                "contains_whole_embedding": True,
            }
        verify_experiment_identity(train_base, train_arrays, "train")
        verify_experiment_identity(valid_base, valid_arrays, "valid")
        experiments[experiment] = write_experiment(
            output,
            experiment,
            train_arrays,
            valid_arrays,
            {
                "version": "adverse_prepost_matched_ablation_task_v1",
                "fixed_cohort": True,
                "fixed_train_series": 908,
                "fixed_valid_series": 239,
                "prediction_unit": "series_uid",
                "grouping_unit": "patient_id",
                "fixed_outer_fold_assignments": True,
                "valid_used_for_feature_selection": False,
                **feature_summary,
            },
        )

    root_summary = {
        "status": "success",
        "version": "adverse_prepost_matched_ablation_task_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_base_task": str(base_task),
        "fixed_base_task_success_sha256": sha256_file(base_task / ".TASK_SUCCESS.json"),
        "configured_whole_featurebank": str(whole_root),
        "whole_feature_schema_sha256": sha256_file(whole_root / "feature_schema.json"),
        "coverage": coverage_summary,
        "m0_schema": morphology_schema,
        "experiments": experiments,
        "identity_contract": {
            "train_series": 908,
            "valid_series": 239,
            "uid_order": "byte-for-byte inherited from current task NPZ",
            "patient_id": "inherited from current task NPZ",
            "target": "inherited from current task NPZ",
            "train_fold": "inherited from current task NPZ",
            "official_split": "unchanged",
        },
    }
    root_summary["scientific_contract_sha256"] = stable_hash(
        {
            "identity_contract": root_summary["identity_contract"],
            "m0_schema": morphology_schema,
            "whole_schema_sha256": root_summary["whole_feature_schema_sha256"],
            "experiment_npz_hashes": {
                key: {
                    "train": value["train_npz_sha256"],
                    "valid": value["valid_npz_sha256"],
                }
                for key, value in experiments.items()
            },
        }
    )
    atomic_json(root_summary, output / "task_summary.json")
    atomic_json(root_summary, output / ".TASK_SUCCESS.json")
    print(json.dumps(root_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
