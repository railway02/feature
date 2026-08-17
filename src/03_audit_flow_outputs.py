# 统一审计
# 检查Train和Valid光流是否完整、干净、无重复
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
MANIFEST_DIR = PROJECT / "manifests"
OUTPUT_DIR = PROJECT / "outputs"
REPORT_DIR = PROJECT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def safe_read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path, dtype={"patient_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def audit_split(split):
    manifest_path = (
        MANIFEST_DIR / f"flow_{split}_manifest.csv"
    )
    result_dir = OUTPUT_DIR / f"full_{split}"

    pair_path = (
        result_dir
        / f"flow_{split}_manifest_pair_features.csv"
    )
    sequence_path = (
        result_dir
        / f"flow_{split}_manifest_sequence_features.csv"
    )
    failure_path = (
        result_dir
        / f"flow_{split}_manifest_failures.csv"
    )

    manifest = pd.read_csv(
        manifest_path,
        dtype={"patient_id": str},
    )

    manifest["patient_id"] = (
        manifest["patient_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # 与运行时 --exclude-patch 一致
    if "patch_case" in manifest.columns:
        manifest = manifest[
            ~as_bool(manifest["patch_case"])
        ].copy()

    expected_rows = []

    for _, row in manifest.iterrows():
        patient_id = row["patient_id"]

        if bool(as_bool(pd.Series([row["run_pre"]])).iloc[0]):
            expected_rows.append({
                "patient_id": patient_id,
                "phase": "pre",
                "expected_pairs": int(row["pre_pair_count"]),
            })

        if bool(as_bool(pd.Series([row["run_post"]])).iloc[0]):
            expected_rows.append({
                "patient_id": patient_id,
                "phase": "post",
                "expected_pairs": int(row["post_pair_count"]),
            })

    expected = pd.DataFrame(expected_rows)

    pairs = safe_read_csv(pair_path)
    sequences = safe_read_csv(sequence_path)
    failures = safe_read_csv(failure_path)

    if pairs.empty:
        raise RuntimeError(
            f"{split}: 帧对结果文件为空或不存在：{pair_path}"
        )

    pairs["patient_id"] = (
        pairs["patient_id"].astype(str).str.strip()
    )
    pairs["phase"] = (
        pairs["phase"].astype(str).str.lower()
    )

    if not sequences.empty:
        sequences["patient_id"] = (
            sequences["patient_id"]
            .astype(str)
            .str.strip()
        )
        sequences["phase"] = (
            sequences["phase"]
            .astype(str)
            .str.lower()
        )

    actual = (
        pairs.groupby(["patient_id", "phase"])
        .size()
        .rename("actual_pairs")
        .reset_index()
    )

    comparison = expected.merge(
        actual,
        on=["patient_id", "phase"],
        how="outer",
    )

    comparison["expected_pairs"] = (
        comparison["expected_pairs"]
        .fillna(0)
        .astype(int)
    )
    comparison["actual_pairs"] = (
        comparison["actual_pairs"]
        .fillna(0)
        .astype(int)
    )

    comparison["difference"] = (
        comparison["actual_pairs"]
        - comparison["expected_pairs"]
    )

    comparison["status"] = np.where(
        comparison["difference"] == 0,
        "OK",
        "MISMATCH",
    )

    mismatch = comparison[
        comparison["status"] != "OK"
    ].copy()

    pair_key = [
        "patient_id",
        "phase",
        "pair_index",
    ]

    duplicate_pairs = 0
    if set(pair_key).issubset(pairs.columns):
        duplicate_pairs = int(
            pairs.duplicated(pair_key).sum()
        )

    duplicate_sequences = 0
    if not sequences.empty:
        duplicate_sequences = int(
            sequences.duplicated(
                ["patient_id", "phase"]
            ).sum()
        )

    numeric_columns = [
        "mag_mean",
        "mag_median",
        "mag_std",
        "mag_p90",
        "mag_p95",
        "mag_max",
        "mag_norm_mean",
        "mag_norm_p90",
        "u_mean",
        "u_std",
        "v_mean",
        "v_std",
        "direction_entropy",
        "uncertainty_mean",
        "uncertainty_std",
        "uncertainty_p90",
        "runtime_s",
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if column in pairs.columns
    ]

    numeric = pairs[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    nan_count = int(numeric.isna().sum().sum())
    nonfinite_count = int(
        (~np.isfinite(numeric.to_numpy())).sum()
    )

    expected_keys = set(
        zip(expected["patient_id"], expected["phase"])
    )

    sequence_keys = set()
    if not sequences.empty:
        sequence_keys = set(
            zip(
                sequences["patient_id"],
                sequences["phase"],
            )
        )

    missing_sequence_keys = (
        expected_keys - sequence_keys
    )
    unexpected_sequence_keys = (
        sequence_keys - expected_keys
    )

    expected_pairs = int(
        expected["expected_pairs"].sum()
    )
    expected_phases = len(expected)
    expected_patients = expected[
        "patient_id"
    ].nunique()

    hard_pass = all([
        len(pairs) == expected_pairs,
        len(sequences) == expected_phases,
        pairs["patient_id"].nunique()
        == expected_patients,
        len(mismatch) == 0,
        len(failures) == 0,
        duplicate_pairs == 0,
        duplicate_sequences == 0,
        nan_count == 0,
        nonfinite_count == 0,
        len(missing_sequence_keys) == 0,
        len(unexpected_sequence_keys) == 0,
    ])

    summary = {
        "split": split,
        "manifest_patients": len(manifest),
        "expected_result_patients": expected_patients,
        "actual_result_patients": (
            pairs["patient_id"].nunique()
        ),
        "expected_phases": expected_phases,
        "actual_phases": len(sequences),
        "expected_pairs": expected_pairs,
        "actual_pairs": len(pairs),
        "failures": len(failures),
        "pair_count_mismatches": len(mismatch),
        "duplicate_pairs": duplicate_pairs,
        "duplicate_sequences": duplicate_sequences,
        "nan_count": nan_count,
        "nonfinite_count": nonfinite_count,
        "missing_sequence_phases": (
            len(missing_sequence_keys)
        ),
        "unexpected_sequence_phases": (
            len(unexpected_sequence_keys)
        ),
        "average_runtime_s": float(
            pairs["runtime_s"].mean()
        ),
        "audit_pass": hard_pass,
    }

    comparison.to_csv(
        REPORT_DIR
        / f"{split}_phase_pair_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mismatch.to_csv(
        REPORT_DIR
        / f"{split}_phase_mismatches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n========== {split.upper()}审计 ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


summaries = [
    audit_split("train"),
    audit_split("valid"),
]

summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    REPORT_DIR / "flow_output_audit_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== 总结 ==========")
print(summary_df.to_string(index=False))

if not summary_df["audit_pass"].all():
    raise RuntimeError(
        "审计未通过，请先检查reports目录中的mismatch文件"
    )

print("\n[PASS] Train和Valid光流结果审计全部通过")
