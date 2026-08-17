# 把Pre/Post序列特征整理成一位患者一行，并计算Delta
# outputs/features/train_patient_flow_features.csv
# outputs/features/valid_patient_flow_features.csv
# outputs/features/patient_flow_features.csv

from pathlib import Path
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
MANIFEST_DIR = PROJECT / "manifests"
OUTPUT_DIR = PROJECT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


all_splits = []

for split in ["train", "valid"]:
    manifest = pd.read_csv(
        MANIFEST_DIR / f"flow_{split}_manifest.csv",
        dtype={"patient_id": str},
    )

    if "patch_case" in manifest.columns:
        manifest = manifest[
            ~as_bool(manifest["patch_case"])
        ].copy()

    base = (
        manifest[["patient_id"]]
        .drop_duplicates()
        .copy()
    )
    base["patient_id"] = (
        base["patient_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    base["split"] = split

    sequence_path = (
        OUTPUT_DIR
        / f"full_{split}"
        / f"flow_{split}_manifest_sequence_features.csv"
    )

    sequence = pd.read_csv(
        sequence_path,
        dtype={"patient_id": str},
    )

    sequence["patient_id"] = (
        sequence["patient_id"]
        .astype(str)
        .str.strip()
    )
    sequence["phase"] = (
        sequence["phase"]
        .astype(str)
        .str.lower()
    )

    feature_columns = [
        column
        for column in sequence.columns
        if column not in {
            "patient_id",
            "phase",
        }
    ]

    wide = sequence.pivot(
        index="patient_id",
        columns="phase",
        values=feature_columns,
    )

    # MultiIndex格式：(特征名, phase)
    wide.columns = [
        f"{phase}_{feature}"
        for feature, phase in wide.columns
    ]

    wide = wide.reset_index()

    patient = base.merge(
        wide,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )

    patient["missing_pre"] = (
        patient.get(
            "pre_n_pairs",
            pd.Series(index=patient.index, dtype=float),
        ).isna().astype(int)
    )

    patient["missing_post"] = (
        patient.get(
            "post_n_pairs",
            pd.Series(index=patient.index, dtype=float),
        ).isna().astype(int)
    )

    # 构建Post - Pre变化量
    pre_columns = [
        column
        for column in patient.columns
        if column.startswith("pre_")
        and column not in {"pre_n_pairs"}
        and "runtime_s" not in column
    ]

    for pre_column in pre_columns:
        feature = pre_column[len("pre_"):]
        post_column = f"post_{feature}"

        if post_column in patient.columns:
            patient[f"delta_{feature}"] = (
                patient[post_column]
                - patient[pre_column]
            )

    all_splits.append(patient)

    split_output = (
        FEATURE_DIR
        / f"{split}_patient_flow_features.csv"
    )

    patient.to_csv(
        split_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"{split}: 患者{len(patient)}，"
        f"有Pre={(patient['missing_pre'] == 0).sum()}，"
        f"有Post={(patient['missing_post'] == 0).sum()}"
    )

combined = pd.concat(
    all_splits,
    ignore_index=True,
)

output = (
    FEATURE_DIR
    / "patient_flow_features.csv"
)

combined.to_csv(
    output,
    index=False,
    encoding="utf-8-sig",
)

print("\n========== 患者级二维特征 ==========")
print("总患者数:", len(combined))
print("Train:", (combined["split"] == "train").sum())
print("Valid:", (combined["split"] == "valid").sum())
print("总列数:", len(combined.columns))
print("输出:", output)
