#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_runtime_exclusions(path: Path, split: str) -> pd.DataFrame:
    columns = [
        "phase_uid",
        "split",
        "stage",
        "reason",
        "active_pixels",
        "decision",
    ]
    if not path.is_file():
        return pd.DataFrame(columns=columns)

    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"phase_uid", "split", "reason", "decision"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"runtime_feature_exclusions.csv 缺少列：{missing}")

    table = table[table["split"] == split].copy()
    if table["phase_uid"].duplicated().any():
        duplicate = sorted(
            table.loc[
                table["phase_uid"].duplicated(keep=False),
                "phase_uid",
            ].astype(str).unique().tolist()
        )
        raise AssertionError(f"runtime exclusions 重复：{duplicate[:20]}")
    if not table["decision"].str.casefold().eq("exclude").all():
        raise AssertionError("runtime exclusions 存在非 exclude decision")
    return table


def build_feature_available_manifest(
    source: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    result = source.copy()
    row_by_series = {
        str(uid): index
        for index, uid in enumerate(result["series_uid"].astype(str).tolist())
    }

    for record in exclusions.to_dict("records"):
        phase_uid = str(record["phase_uid"])
        try:
            series_uid, phase = phase_uid.rsplit("::", 1)
        except ValueError as exc:
            raise ValueError(f"非法 phase_uid：{phase_uid}") from exc
        if phase not in {"pre", "post"}:
            raise ValueError(f"非法 phase：{phase_uid}")
        if series_uid not in row_by_series:
            raise AssertionError(
                f"runtime exclusion 不属于 {split} Local manifest：{phase_uid}"
            )

        row_index = row_by_series[series_uid]
        can_run_column = f"can_run_{phase}"
        if can_run_column not in result.columns:
            raise KeyError(f"manifest 缺少 {can_run_column}")
        if not as_bool(result.at[row_index, can_run_column]):
            raise AssertionError(
                f"runtime exclusion 指向本来就不可运行的 phase：{phase_uid}"
            )

        result.at[row_index, can_run_column] = "False"

        # api_fullseq_cave_v3.manifest._validate_phase requires every
        # frozen-frame field and every expected count to be empty/zero when
        # can_run_<phase> is false.  In particular, leaving
        # n_<phase>_contiguous_pairs non-zero triggers the generic
        # "non-runnable phase contains frames" assertion.
        zero_columns = (
            f"n_{phase}_frames",
            f"n_{phase}_contiguous_pairs",
        )
        empty_columns = (
            f"{phase}_frame_paths",
            f"{phase}_frame_list_hash",
            f"{phase}_frame_indices",
            f"{phase}_selected_filenames",
            f"{phase}_frame_gaps",
        )
        for column in zero_columns:
            if column in result.columns:
                result.at[row_index, column] = "0"
        for column in empty_columns:
            if column in result.columns:
                result.at[row_index, column] = ""

        # Keep this broad cleanup for compatible future manifest columns, but
        # do not use it in place of the explicit count reset above.
        phase_prefix = f"{phase}_"
        tokens = ("frame", "path", "hash", "indice", "selected", "filename")
        for column in result.columns:
            name = column.casefold()
            if name.startswith(phase_prefix) and any(token in name for token in tokens):
                result.at[row_index, column] = ""

        if "can_run_prepost" in result.columns:
            result.at[row_index, "can_run_prepost"] = str(
                as_bool(result.at[row_index, "can_run_pre"])
                and as_bool(result.at[row_index, "can_run_post"])
            )

    # Fail here with a precise message instead of letting the frozen CAVE
    # manifest loader fail later with a generic assertion.
    for record in exclusions.to_dict("records"):
        phase_uid = str(record["phase_uid"])
        series_uid, phase = phase_uid.rsplit("::", 1)
        row_index = row_by_series[series_uid]
        if as_bool(result.at[row_index, f"can_run_{phase}"]):
            raise AssertionError(f"runtime exclusion 仍被标记为 runnable：{phase_uid}")
        for column in (
            f"{phase}_frame_paths",
            f"{phase}_frame_indices",
            f"{phase}_frame_list_hash",
            f"{phase}_selected_filenames",
            f"{phase}_frame_gaps",
        ):
            if column in result.columns and str(result.at[row_index, column]).strip():
                raise AssertionError(
                    f"runtime exclusion 仍含帧字段：{phase_uid}:{column}"
                )
        for column in (f"n_{phase}_frames", f"n_{phase}_contiguous_pairs"):
            if column in result.columns:
                value = str(result.at[row_index, column]).strip()
                if value not in {"", "0", "0.0"}:
                    raise AssertionError(
                        f"runtime exclusion 仍含非零计数：{phase_uid}:{column}={value}"
                    )

    return result


def combine_series_availability(canonical_dir: Path) -> Path:
    split_paths = [
        canonical_dir / "local_series_availability_train.csv",
        canonical_dir / "local_series_availability_valid.csv",
    ]
    present = [path for path in split_paths if path.is_file()]
    combined_path = canonical_dir / "local_series_availability.csv"
    if present:
        combined = pd.concat(
            [
                pd.read_csv(path, dtype=str, keep_default_na=False)
                for path in present
            ],
            ignore_index=True,
        )
        atomic_csv(combined, combined_path)
    return combined_path


def prepare_split_builder_feature_root(
    feature_root: Path,
    reports: Path,
    split: str,
) -> tuple[Path, Path]:
    split_key = split.casefold()
    split_schema = feature_root / f"feature_schema_{split_key}.json"
    if not split_schema.is_file():
        raise FileNotFoundError(f"缺少 split-specific schema：{split_schema}")

    source_split_dir = feature_root / split_key
    if not source_split_dir.is_dir():
        raise FileNotFoundError(f"缺少 split feature directory：{source_split_dir}")

    builder_root = reports / f"09_builder_feature_root_{split_key}"
    if builder_root.is_symlink() or builder_root.is_file():
        builder_root.unlink()
    elif builder_root.is_dir():
        shutil.rmtree(builder_root)
    builder_root.mkdir(parents=True, exist_ok=False)

    os.symlink(
        source_split_dir,
        builder_root / split_key,
        target_is_directory=True,
    )
    shutil.copy2(split_schema, builder_root / "feature_schema.json")
    return builder_root, split_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    outputs = Path(cfg["paths"]["outputs"])
    reports = Path(cfg["paths"]["reports"])
    feature_root = outputs / "cave_local_eligible_featurebank"
    success_lock = feature_root / ".ELIGIBLE_FEATUREBANK_SUCCESS.json"
    if not success_lock.is_file():
        raise RuntimeError("eligible featurebank 尚未通过 All validation")

    validation_report = reports / "08_featurebank_validation_all.json"
    if not validation_report.is_file():
        raise RuntimeError("缺少 All featurebank validation 报告")
    validation_summary = json.loads(
        validation_report.read_text(encoding="utf-8")
    )
    if str(validation_summary.get("status", "")) != "success":
        raise RuntimeError("eligible featurebank validation 报告状态异常")
    if int(validation_summary.get("closed_phase_count", -1)) != int(
        validation_summary.get("eligible_phase_count", -2)
    ):
        raise RuntimeError("eligible featurebank 尚未按 success+exclusion 闭合")

    manifest_path = (
        manifests
        / f"cave_manifest_local_{args.split.casefold()}_eligible.csv"
    )
    original_manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    exclusion_path = manifests / "runtime_feature_exclusions.csv"
    exclusions = load_runtime_exclusions(exclusion_path, args.split)
    feature_available_manifest = build_feature_available_manifest(
        original_manifest,
        exclusions,
        args.split,
    )

    output_dir = (
        outputs
        / "tables"
        / "local_eligible"
        / args.split.casefold()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_manifest_path = (
        reports
        / f"09_cave_manifest_local_{args.split.casefold()}_feature_available.csv"
    )
    atomic_csv(feature_available_manifest, filtered_manifest_path)

    builder_feature_root, split_schema_path = (
        prepare_split_builder_feature_root(
            feature_root,
            reports,
            args.split,
        )
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = cfg["cave_code_root"]
    command = [
        cfg["cave_python"],
        str(Path(cfg["cave_code_root"]) / "build_feature_tables.py"),
        "--manifest",
        str(filtered_manifest_path),
        "--feature-root",
        str(builder_feature_root),
        "--output-dir",
        str(output_dir),
        "--expected-split",
        args.split,
        "--verify-files",
    ]
    subprocess.run(
        command,
        cwd=cfg["project_root"],
        env=env,
        check=True,
    )

    npz_path = output_dir / "series_embeddings_5120.npz"
    with np.load(npz_path, allow_pickle=False) as payload:
        output_uids = payload["series_uid"].astype(str).tolist()
        feature_key = (
            "embeddings"
            if "embeddings" in payload.files
            else "embedding"
            if "embedding" in payload.files
            else ""
        )
        if not feature_key:
            raise KeyError(f"{npz_path} 缺少 embeddings/embedding 数组")
        embeddings = np.asarray(payload[feature_key])
        embedding_shape = list(embeddings.shape)

    # build_feature_tables intentionally omits rows for which neither phase is
    # runnable.  The eligible manifest keeps those rows for availability and
    # source-order auditing, so the table must be compared with the ordered
    # runnable subset rather than with every manifest row.
    runnable_mask = (
        feature_available_manifest["can_run_pre"].map(as_bool)
        | feature_available_manifest["can_run_post"].map(as_bool)
    )
    expected_table_manifest = feature_available_manifest.loc[runnable_mask].copy()
    expected_output_uids = (
        expected_table_manifest["series_uid"].astype(str).tolist()
    )

    if len(expected_output_uids) != len(set(expected_output_uids)):
        duplicates = expected_table_manifest.loc[
            expected_table_manifest["series_uid"].astype(str).duplicated(keep=False),
            "series_uid",
        ].astype(str).unique().tolist()
        raise AssertionError(
            f"feature-available manifest 存在重复 series_uid：{duplicates[:20]}"
        )

    # The frozen builder may aggregate/group rows in a deterministic order that
    # is different from the input manifest order.  Order is therefore not a
    # validity condition.  The scientific contract is key equality and unique
    # series_uid; downstream joins must always be key-based, never positional.
    if len(output_uids) != len(set(output_uids)):
        duplicate_output_uids = sorted({
            uid for uid in output_uids if output_uids.count(uid) > 1
        })
        raise AssertionError(
            f"builder 输出存在重复 series_uid：{duplicate_output_uids[:20]}"
        )

    output_set = set(output_uids)
    expected_set = set(expected_output_uids)
    missing = [uid for uid in expected_output_uids if uid not in output_set]
    extras = [uid for uid in output_uids if uid not in expected_set]
    if missing or extras or len(output_uids) != len(expected_output_uids):
        raise AssertionError(
            "Local series table 的 series_uid 集合错误："
            f"expected={len(expected_output_uids)}, actual={len(output_uids)}, "
            f"missing={missing[:20]}, extras={extras[:20]}"
        )

    order_matches_manifest = output_uids == expected_output_uids
    expected_index = {uid: index for index, uid in enumerate(expected_output_uids)}
    builder_index = {uid: index for index, uid in enumerate(output_uids)}
    order_audit = expected_table_manifest[[
        "split", "patient_id", "series_uid"
    ]].copy()
    order_audit["expected_manifest_index"] = [
        expected_index[str(uid)] for uid in order_audit["series_uid"]
    ]
    order_audit["builder_output_index"] = [
        builder_index[str(uid)] for uid in order_audit["series_uid"]
    ]
    order_audit["order_matches_at_index"] = (
        order_audit["expected_manifest_index"]
        == order_audit["builder_output_index"]
    ).astype(int)
    order_audit_path = (
        reports / f"09_series_order_audit_{args.split.casefold()}.csv"
    )
    atomic_csv(order_audit, order_audit_path)
    first_mismatch_index = next(
        (
            index
            for index, (actual, expected) in enumerate(
                zip(output_uids, expected_output_uids)
            )
            if actual != expected
        ),
        None,
    )

    if embeddings.ndim != 3 or embeddings.shape[1] != 2:
        raise AssertionError(
            f"series embedding 预期 [series,2,D]，实际 {embeddings.shape}"
        )
    if embeddings.shape[0] != len(expected_output_uids):
        raise AssertionError(
            "series embedding 第一维与可运行 series 数不一致："
            f"{embeddings.shape[0]}!={len(expected_output_uids)}"
        )

    uid_to_index = {
        uid: index
        for index, uid in enumerate(output_uids)
    }
    manifest_by_series = feature_available_manifest.set_index(
        feature_available_manifest["series_uid"].astype(str),
        drop=False,
    )
    for record in exclusions.to_dict("records"):
        series_uid, phase = str(record["phase_uid"]).rsplit("::", 1)
        phase_index = 0 if phase == "pre" else 1

        # A series disappears from the NPZ only when both phases are
        # non-runnable.  Otherwise the excluded phase must be represented by
        # an all-NaN vector while the other phase remains available.
        if series_uid not in uid_to_index:
            manifest_row = manifest_by_series.loc[series_uid]
            if (
                as_bool(manifest_row["can_run_pre"])
                or as_bool(manifest_row["can_run_post"])
            ):
                raise AssertionError(
                    f"仍有可运行 phase 的 series 未进入 table：{series_uid}"
                )
            continue

        vector = embeddings[uid_to_index[series_uid], phase_index]
        if not np.isnan(vector).all():
            raise AssertionError(
                f"运行期排除 phase 未在 series table 中表示为全 NaN："
                f"{record['phase_uid']}"
            )

    canonical_dir = outputs / "tables" / "local_eligible"
    availability_path = canonical_dir / "local_phase_availability.csv"
    if not availability_path.is_file():
        availability_path = reports / "08_local_phase_availability.csv"

    series_availability_path = ""
    combined_availability_path = ""
    if availability_path.is_file():
        availability = pd.read_csv(
            availability_path,
            dtype=str,
            keep_default_na=False,
        )
        availability = availability[
            availability["split"] == args.split
        ].copy()
        available = availability[
            availability["local_feature_available"].astype(int) == 1
        ].copy()

        reason_by_uid = dict(zip(
            availability["phase_uid"].astype(str),
            availability["feature_exclusion_reason"].astype(str),
        ))
        full_source = pd.read_csv(
            cfg["source_series_manifests"][args.split],
            dtype=str,
            keep_default_na=False,
        )
        rows: list[dict[str, object]] = []
        for record in full_source.to_dict("records"):
            uid = str(record["series_uid"])
            group = available[available["series_uid"] == uid]
            pre = bool((group["phase"] == "pre").any())
            post = bool((group["phase"] == "post").any())
            rows.append({
                "split": args.split,
                "patient_id": record["patient_id"],
                "series_uid": uid,
                "local_pre_available": int(pre),
                "local_post_available": int(post),
                "local_both_available": int(pre and post),
                "local_any_available": int(pre or post),
                "local_pre_exclusion_reason": (
                    "" if pre else reason_by_uid.get(f"{uid}::pre", "")
                ),
                "local_post_exclusion_reason": (
                    "" if post else reason_by_uid.get(f"{uid}::post", "")
                ),
                "source_can_run_pre": record.get("can_run_pre", ""),
                "source_can_run_post": record.get("can_run_post", ""),
            })

        split_availability = (
            canonical_dir
            / f"local_series_availability_{args.split.casefold()}.csv"
        )
        atomic_csv(pd.DataFrame(rows), split_availability)
        series_availability_path = str(split_availability)
        combined_availability_path = str(
            combine_series_availability(canonical_dir)
        )

    summary = {
        "status": "success",
        "split": args.split,
        "source_manifest_series": int(len(original_manifest)),
        "feature_available_series": int(len(expected_output_uids)),
        "fully_non_runnable_series": int((~runnable_mask).sum()),
        "series": len(output_uids),
        "series_uid_set_equal": True,
        "builder_order_matches_manifest": bool(order_matches_manifest),
        "first_order_mismatch_index": first_mismatch_index,
        "series_order_audit": str(order_audit_path),
        "join_contract": "keyed_by_series_uid_not_position",
        "embedding_shape": embedding_shape,
        "original_manifest": str(manifest_path),
        "original_manifest_sha256": sha256_file(manifest_path),
        "feature_available_manifest": str(filtered_manifest_path),
        "feature_available_manifest_sha256": sha256_file(
            filtered_manifest_path
        ),
        "runtime_feature_exclusions": str(exclusion_path),
        "runtime_exclusion_count": int(len(exclusions)),
        "feature_root": str(feature_root),
        "builder_feature_root": str(builder_feature_root),
        "split_feature_schema": str(split_schema_path),
        "split_feature_schema_sha256": sha256_file(split_schema_path),
        "featurebank_success_lock": str(success_lock),
        "featurebank_success_lock_sha256": sha256_file(success_lock),
        "featurebank_validation_report": str(validation_report),
        "featurebank_validation_report_sha256": sha256_file(
            validation_report
        ),
        "output_dir": str(output_dir),
        "npz": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "excluded_phases_encoded_as_nan": True,
        "local_series_availability": series_availability_path,
        "combined_local_series_availability": combined_availability_path,
    }
    atomic_json(
        summary,
        reports / f"09_local_table_{args.split.casefold()}.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
