# 对Train全量光流结果做详细验收和异常值检查

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
MANIFEST_PATH = PROJECT / "manifests/flow_train_manifest.csv"
OUTPUT_DIR = PROJECT / "outputs/full_train"
REPORT_DIR = PROJECT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_PATH = (
    OUTPUT_DIR
    / "flow_train_manifest_pair_features.csv"
)
SEQUENCE_PATH = (
    OUTPUT_DIR
    / "flow_train_manifest_sequence_features.csv"
)
FAILURE_PATH = (
    OUTPUT_DIR
    / "flow_train_manifest_failures.csv"
)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def read_failure_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


required_files = [
    MANIFEST_PATH,
    PAIR_PATH,
    SEQUENCE_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "缺少验收所需文件：\n"
        + "\n".join(missing_files)
    )


# ----------------------------
# 1. 读取Manifest并计算预期工作量
# ----------------------------
manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"patient_id": str},
)

manifest["patient_id"] = (
    manifest["patient_id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

patch = as_bool(manifest["patch_case"])

work = manifest.loc[~patch].copy()
work["run_pre_bool"] = as_bool(work["run_pre"])
work["run_post_bool"] = as_bool(work["run_post"])

for column in [
    "pre_pair_count",
    "post_pair_count",
]:
    work[column] = pd.to_numeric(
        work[column],
        errors="coerce",
    ).fillna(0).astype(int)


expected_rows = []

for _, row in work.iterrows():
    patient_id = row["patient_id"]

    if row["run_pre_bool"]:
        expected_rows.append({
            "patient_id": patient_id,
            "phase": "pre",
            "expected_pairs": int(
                row["pre_pair_count"]
            ),
        })

    if row["run_post_bool"]:
        expected_rows.append({
            "patient_id": patient_id,
            "phase": "post",
            "expected_pairs": int(
                row["post_pair_count"]
            ),
        })

expected = pd.DataFrame(expected_rows)

expected_patients = (
    expected["patient_id"].nunique()
)

expected_phases = len(expected)
expected_pairs = int(
    expected["expected_pairs"].sum()
)


# ----------------------------
# 2. 读取实际结果
# ----------------------------
pairs = pd.read_csv(
    PAIR_PATH,
    dtype={"patient_id": str},
)

sequences = pd.read_csv(
    SEQUENCE_PATH,
    dtype={"patient_id": str},
)

failures = read_failure_csv(
    FAILURE_PATH
)

for df in [pairs, sequences]:
    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

pairs["phase"] = (
    pairs["phase"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sequences["phase"] = (
    sequences["phase"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ----------------------------
# 3. 检查每个患者每个阶段帧对数
# ----------------------------
actual_counts = (
    pairs.groupby(
        ["patient_id", "phase"]
    )
    .size()
    .reset_index(name="actual_pairs")
)

coverage = expected.merge(
    actual_counts,
    on=["patient_id", "phase"],
    how="outer",
    indicator=True,
)

coverage["expected_pairs"] = (
    pd.to_numeric(
        coverage["expected_pairs"],
        errors="coerce",
    ).fillna(0).astype(int)
)

coverage["actual_pairs"] = (
    pd.to_numeric(
        coverage["actual_pairs"],
        errors="coerce",
    ).fillna(0).astype(int)
)

coverage["pair_difference"] = (
    coverage["actual_pairs"]
    - coverage["expected_pairs"]
)

coverage_issues = coverage[
    (coverage["_merge"] != "both")
    | (coverage["pair_difference"] != 0)
].copy()

coverage.to_csv(
    REPORT_DIR / "train_phase_coverage.csv",
    index=False,
    encoding="utf-8-sig",
)

coverage_issues.to_csv(
    REPORT_DIR / "train_phase_coverage_issues.csv",
    index=False,
    encoding="utf-8-sig",
)


# ----------------------------
# 4. 检查重复帧对和重复阶段
# ----------------------------
pair_key = [
    "patient_id",
    "phase",
    "pair_index",
]

duplicate_pairs = int(
    pairs.duplicated(
        subset=pair_key,
        keep=False,
    ).sum()
)

duplicate_sequences = int(
    sequences.duplicated(
        subset=["patient_id", "phase"],
        keep=False,
    ).sum()
)


# ----------------------------
# 5. 检查NaN和Inf
# ----------------------------
feature_columns = [
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

missing_feature_columns = [
    column
    for column in feature_columns
    if column not in pairs.columns
]

if missing_feature_columns:
    raise KeyError(
        "帧对结果缺少特征列："
        f"{missing_feature_columns}"
    )

numeric = pairs[
    feature_columns
].apply(
    pd.to_numeric,
    errors="coerce",
)

nan_count = int(
    numeric.isna().sum().sum()
)

inf_count = int(
    (~np.isfinite(
        numeric.to_numpy()
    )).sum()
)


# ----------------------------
# 6. 检查pair_index是否连续
# ----------------------------
index_issues = []

for (
    patient_id,
    phase,
), group in pairs.groupby(
    ["patient_id", "phase"]
):
    indices = sorted(
        pd.to_numeric(
            group["pair_index"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    expected_indices = list(
        range(len(indices))
    )

    if indices != expected_indices:
        index_issues.append({
            "patient_id": patient_id,
            "phase": phase,
            "actual_indices": str(indices),
            "expected_indices": str(
                expected_indices
            ),
        })

index_issue_df = pd.DataFrame(
    index_issues
)

index_issue_df.to_csv(
    REPORT_DIR / "train_pair_index_issues.csv",
    index=False,
    encoding="utf-8-sig",
)


# ----------------------------
# 7. 输出异常值清单，用于医学QC
# ----------------------------
for column in [
    "mag_norm_mean",
    "uncertainty_mean",
]:
    threshold = numeric[
        column
    ].quantile(0.99)

    pairs[f"qc_{column}_p99"] = (
        pd.to_numeric(
            pairs[column],
            errors="coerce",
        ) > threshold
    )

outliers = pairs[
    pairs["qc_mag_norm_mean_p99"]
    | pairs["qc_uncertainty_mean_p99"]
].copy()

outliers.to_csv(
    REPORT_DIR / "train_flow_outliers_p99.csv",
    index=False,
    encoding="utf-8-sig",
)


# ----------------------------
# 8. 汇总验收结果
# ----------------------------
actual_patients = (
    pairs["patient_id"].nunique()
)

actual_phases = len(sequences)
actual_pairs = len(pairs)
failure_count = len(failures)

checks = {
    "patient_count_match":
        actual_patients == expected_patients,

    "phase_count_match":
        actual_phases == expected_phases,

    "pair_count_match":
        actual_pairs == expected_pairs,

    "no_coverage_issues":
        len(coverage_issues) == 0,

    "no_failures":
        failure_count == 0,

    "no_nan":
        nan_count == 0,

    "no_inf":
        inf_count == 0,

    "no_duplicate_pairs":
        duplicate_pairs == 0,

    "no_duplicate_sequences":
        duplicate_sequences == 0,

    "continuous_pair_indices":
        len(index_issues) == 0,
}

passed = all(checks.values())

summary = {
    "expected_patients": int(expected_patients),
    "actual_patients": int(actual_patients),

    "expected_phases": int(expected_phases),
    "actual_phases": int(actual_phases),

    "expected_pairs": int(expected_pairs),
    "actual_pairs": int(actual_pairs),

    "failure_count": int(failure_count),
    "coverage_issue_count": int(
        len(coverage_issues)
    ),

    "nan_count": int(nan_count),
    "inf_count": int(inf_count),

    "duplicate_pair_rows": int(
        duplicate_pairs
    ),

    "duplicate_sequence_rows": int(
        duplicate_sequences
    ),

    "pair_index_issue_count": int(
        len(index_issues)
    ),

    "mean_runtime_s": float(
        numeric["runtime_s"].mean()
    ),

    "total_runtime_minutes": float(
        numeric["runtime_s"].sum() / 60
    ),

    "checks": checks,
    "passed": passed,
}

with open(
    REPORT_DIR / "train_acceptance_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("========== FULL TRAIN 验收 ==========")
print(
    f"预计有结果患者："
    f"{expected_patients}"
)
print(
    f"实际有结果患者："
    f"{actual_patients}"
)

print(
    f"\n预计阶段数：{expected_phases}"
)
print(
    f"实际阶段数：{actual_phases}"
)

print(
    f"\n预计帧对数：{expected_pairs}"
)
print(
    f"实际帧对数：{actual_pairs}"
)

print(
    f"\n失败数：{failure_count}"
)
print(
    f"覆盖问题数：{len(coverage_issues)}"
)
print(
    f"NaN数量：{nan_count}"
)
print(
    f"Inf数量：{inf_count}"
)
print(
    f"重复帧对行数：{duplicate_pairs}"
)
print(
    f"重复阶段行数：{duplicate_sequences}"
)
print(
    f"帧对索引问题：{len(index_issues)}"
)

print(
    "\n平均每帧对耗时："
    f"{numeric['runtime_s'].mean():.4f} 秒"
)

print(
    "累计推理耗时："
    f"{numeric['runtime_s'].sum() / 60:.2f} 分钟"
)

print("\n各项检查：")

for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"{status:4s}  {name}")

print("\n最终验收：", "通过" if passed else "未通过")
print("报告目录：", REPORT_DIR)

if not passed:
    print(
        "\n请优先检查："
        "\n- train_phase_coverage_issues.csv"
        "\n- train_pair_index_issues.csv"
        "\n- flow_train_manifest_failures.csv"
    )
