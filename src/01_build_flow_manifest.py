# 用于合并：

# Train.xlsx
# valid.xlsx
# 患者名单1331人.csv
# 纳入动脉瘤1467个.csv
# 标注相关CSV
# filesystem_inventory.csv

# 输出：

# patient_manifest.csv
# record_manifest.csv
# excluded_patients.csv
# lesion_count_mismatch.csv
# annotation_list_file_mismatch.csv

# Train 和 Valid 中的患者，分别对应哪个 Pre/Post 影像目录，是否可以运行 SEA-RAFT。


from pathlib import Path
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
META = PROJECT / "metadata"
MANIFESTS = PROJECT / "manifests"

INVENTORY_PATH = MANIFESTS / "filesystem_inventory.csv"
OUTPUT_PATH = MANIFESTS / "flow_manifest.csv"

PATCH_PATIENTS = {
    "726527", "696044", "667483", "640779", "593174",
    "585192", "565733", "554148", "549117", "458123",
}


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def find_id_column(df):
    for column in df.columns:
        name = str(column).strip()
        if "病案号" in name:
            return column

    raise KeyError(
        f"没有找到病案号列，现有列：{list(df.columns)}"
    )


def load_split(path, split):
    df = pd.read_excel(path, dtype=str)

    id_column = find_id_column(df)

    result = pd.DataFrame({
        "patient_id": df[id_column].map(normalize_id),
        "split": split,
    })

    result = (
        result
        .dropna(subset=["patient_id"])
        .drop_duplicates(subset=["patient_id"])
    )

    return result


train = load_split(META / "Train.xlsx", "train")
valid = load_split(META / "valid.xlsx", "valid")

train_ids = set(train["patient_id"])
valid_ids = set(valid["patient_id"])

overlap = train_ids & valid_ids

if overlap:
    raise RuntimeError(
        f"Train和Valid存在患者交集：{sorted(overlap)[:20]}"
    )

split_df = pd.concat(
    [train, valid],
    ignore_index=True,
)

inventory = pd.read_csv(
    INVENTORY_PATH,
    dtype={"patient_id": str},
)

inventory["patient_id"] = (
    inventory["patient_id"]
    .map(normalize_id)
)

for column in [
    "pre_frame_count",
    "post_frame_count",
    "pre_pair_count",
    "post_pair_count",
]:
    inventory[column] = pd.to_numeric(
        inventory[column],
        errors="coerce",
    ).fillna(0).astype(int)

manifest = split_df.merge(
    inventory,
    on="patient_id",
    how="left",
    validate="one_to_one",
)

manifest["directory_matched"] = (
    manifest["patient_dir"].notna()
)

manifest["run_pre"] = (
    manifest["pre_frame_count"] >= 2
)

manifest["run_post"] = (
    manifest["post_frame_count"] >= 2
)

manifest["run_any"] = (
    manifest["run_pre"]
    | manifest["run_post"]
)

manifest["patch_case"] = (
    manifest["patient_id"].isin(PATCH_PATIENTS)
)

manifest["pilot_eligible"] = (
    (manifest["split"] == "train")
    & manifest["run_pre"]
    & manifest["run_post"]
    & (~manifest["patch_case"])
)

manifest.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("========== Flow Manifest ==========")
print("Train唯一患者:", len(train))
print("Valid唯一患者:", len(valid))
print("目标患者总数:", len(manifest))
print("Train/Valid患者交集:", len(overlap))
print(
    "未匹配患者目录:",
    int((~manifest["directory_matched"]).sum()),
)
print(
    "可运行Pre:",
    int(manifest["run_pre"].sum()),
)
print(
    "可运行Post:",
    int(manifest["run_post"].sum()),
)
print(
    "Pre/Post均可运行:",
    int(
        (
            manifest["run_pre"]
            & manifest["run_post"]
        ).sum()
    ),
)
print(
    "至少一个阶段可运行:",
    int(manifest["run_any"].sum()),
)
print(
    "修正10例中的目标患者:",
    int(manifest["patch_case"].sum()),
)
print(
    "Pilot候选患者:",
    int(manifest["pilot_eligible"].sum()),
)
print("输出:", OUTPUT_PATH)
