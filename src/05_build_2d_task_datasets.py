#  将患者特征与临床标签合并，生成四个建模任务

# outputs/task_datasets/adverse_pre_train.csv
# outputs/task_datasets/adverse_pre_valid.csv

# outputs/task_datasets/immediate_pre_train.csv
# outputs/task_datasets/immediate_pre_valid.csv

# outputs/task_datasets/immediate_post_train.csv
# outputs/task_datasets/immediate_post_valid.csv

# outputs/task_datasets/followup_prepost_train.csv
# outputs/task_datasets/followup_prepost_valid.csv

from pathlib import Path
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
META = PROJECT / "metadata"
FEATURE_DIR = PROJECT / "outputs/features"
OUTPUT_DIR = PROJECT / "outputs/task_datasets"
REPORT_DIR = PROJECT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "病案号"
LABELS = {
    "adverse": "不良转归：1是；0否",
    "immediate": "术后即刻RROC",
    "followup": "随访RROC123",
}


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def build_patient_labels(path, split):
    raw = pd.read_excel(path)

    required = [ID_COLUMN, *LABELS.values()]

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]

    if missing:
        raise KeyError(
            f"{path.name}缺少列：{missing}"
        )

    raw["patient_id"] = raw[
        ID_COLUMN
    ].map(normalize_id)

    raw = raw.dropna(
        subset=["patient_id"]
    ).copy()

    conflict_rows = []
    patient_rows = []

    for patient_id, group in raw.groupby(
        "patient_id",
        sort=False,
    ):
        row = {
            "patient_id": patient_id,
            "split": split,
            "source_record_count": len(group),
        }

        conflict = False

        for short_name, label_column in LABELS.items():
            values = (
                group[label_column]
                .dropna()
                .unique()
                .tolist()
            )

            if len(values) != 1:
                conflict = True

                conflict_rows.append({
                    "patient_id": patient_id,
                    "split": split,
                    "label": label_column,
                    "values": "|".join(
                        map(str, values)
                    ),
                    "record_count": len(group),
                })
            else:
                row[short_name] = values[0]

        if not conflict:
            patient_rows.append(row)

    patients = pd.DataFrame(patient_rows)
    conflicts = pd.DataFrame(conflict_rows)

    return patients, conflicts


def load_features(path, expected_split):
    df = pd.read_csv(
        path,
        dtype={"patient_id": str},
    )

    df["patient_id"] = df[
        "patient_id"
    ].map(normalize_id)

    if "split" in df.columns:
        wrong = df[
            df["split"].astype(str).str.lower()
            != expected_split
        ]

        if not wrong.empty:
            raise RuntimeError(
                f"{path.name}中存在错误split"
            )

    if df["patient_id"].duplicated().any():
        duplicated = df.loc[
            df["patient_id"].duplicated(False),
            "patient_id",
        ].tolist()

        raise RuntimeError(
            f"特征表存在重复患者：{duplicated[:20]}"
        )

    return df


def select_features(df, mode):
    excluded_common = {
        "patient_id",
        "split",
        "missing_pre",
        "missing_post",
    }

    columns = []

    for column in df.columns:
        if column in excluded_common:
            continue

        # GPU运行时间不是医学特征
        if "runtime_s" in column:
            continue

        # 主分析先排除采集帧数，后面单独消融
        if column in {
            "pre_n_pairs",
            "post_n_pairs",
        }:
            continue

        if mode == "pre":
            if column.startswith("pre_"):
                columns.append(column)

        elif mode == "post":
            if column.startswith("post_"):
                columns.append(column)

        elif mode == "prepost":
            if column.startswith(
                ("pre_", "post_", "delta_")
            ):
                columns.append(column)

        else:
            raise ValueError(
                f"未知mode：{mode}"
            )

    return columns


def create_task(
    train_merged,
    valid_merged,
    task_name,
    label_column,
    mode,
):
    if mode == "pre":
        train = train_merged[
            train_merged["missing_pre"] == 0
        ].copy()

        valid = valid_merged[
            valid_merged["missing_pre"] == 0
        ].copy()

    elif mode == "post":
        train = train_merged[
            train_merged["missing_post"] == 0
        ].copy()

        valid = valid_merged[
            valid_merged["missing_post"] == 0
        ].copy()

    elif mode == "prepost":
        train = train_merged[
            (train_merged["missing_pre"] == 0)
            & (train_merged["missing_post"] == 0)
        ].copy()

        valid = valid_merged[
            (valid_merged["missing_pre"] == 0)
            & (valid_merged["missing_post"] == 0)
        ].copy()

    feature_columns = select_features(
        train,
        mode,
    )

    output_columns = [
        "patient_id",
        "split",
        label_column,
        *feature_columns,
    ]

    train_output = train[
        output_columns
    ].copy()

    valid_output = valid[
        output_columns
    ].copy()

    train_path = (
        OUTPUT_DIR
        / f"{task_name}_train.csv"
    )

    valid_path = (
        OUTPUT_DIR
        / f"{task_name}_valid.csv"
    )

    train_output.to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig",
    )

    valid_output.to_csv(
        valid_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n========== {task_name} =========="
    )
    print("特征数:", len(feature_columns))
    print("Train患者:", len(train_output))
    print("Valid患者:", len(valid_output))

    print("\nTrain标签分布:")
    print(
        train_output[label_column]
        .value_counts()
        .sort_index()
    )

    print("\nValid标签分布:")
    print(
        valid_output[label_column]
        .value_counts()
        .sort_index()
    )


train_labels, train_conflicts = (
    build_patient_labels(
        META / "Train.xlsx",
        "train",
    )
)

valid_labels, valid_conflicts = (
    build_patient_labels(
        META / "valid.xlsx",
        "valid",
    )
)

conflicts = pd.concat(
    [train_conflicts, valid_conflicts],
    ignore_index=True,
)

conflicts.to_csv(
    REPORT_DIR / "patient_label_conflicts.csv",
    index=False,
    encoding="utf-8-sig",
)

train_features = load_features(
    FEATURE_DIR
    / "train_patient_flow_features.csv",
    "train",
)

valid_features = load_features(
    FEATURE_DIR
    / "valid_patient_flow_features.csv",
    "valid",
)

train_merged = train_features.merge(
    train_labels,
    on=["patient_id", "split"],
    how="left",
    validate="one_to_one",
)

valid_merged = valid_features.merge(
    valid_labels,
    on=["patient_id", "split"],
    how="left",
    validate="one_to_one",
)

for split, merged in [
    ("train", train_merged),
    ("valid", valid_merged),
]:
    missing_label = merged[
        list(LABELS.keys())
    ].isna().any(axis=1)

    print(
        f"{split}没有唯一标签的患者:",
        int(missing_label.sum()),
    )

    # 当前有真实二维影像的患者必须全部有唯一标签
    has_flow = (
        (merged["missing_pre"] == 0)
        | (merged["missing_post"] == 0)
    )

    problematic = merged[
        has_flow & missing_label
    ]

    if not problematic.empty:
        raise RuntimeError(
            f"{split}存在有二维特征但标签冲突的患者："
            f"{problematic['patient_id'].tolist()[:20]}"
        )

create_task(
    train_merged,
    valid_merged,
    task_name="adverse_pre",
    label_column="adverse",
    mode="pre",
)

create_task(
    train_merged,
    valid_merged,
    task_name="immediate_pre",
    label_column="immediate",
    mode="pre",
)

create_task(
    train_merged,
    valid_merged,
    task_name="immediate_post",
    label_column="immediate",
    mode="post",
)

create_task(
    train_merged,
    valid_merged,
    task_name="followup_prepost",
    label_column="followup",
    mode="prepost",
)

print("\n输出目录:", OUTPUT_DIR)
print("冲突审计:", REPORT_DIR / "patient_label_conflicts.csv")
