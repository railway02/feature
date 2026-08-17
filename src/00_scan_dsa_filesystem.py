
# 实际有多少患者目录；
# 哪些患者有 Pre-Image；
# 哪些患者有 Post-Image；
# 每个序列多少帧；
# 一共需要计算多少相邻帧对；
# 图像尺寸是否统一；
# Pre/Post分割文件是否真实存在；
# 全量保存原始光流大概要多少空间。




from pathlib import Path
import re
import cv2
import pandas as pd

DATA_ROOT = Path("/root/autodl-tmp/tiantanDSA")
OUT_DIR = Path("/root/autodl-tmp/aneurysm/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff"
}

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [
        int(x) if x.isdigit() else x.lower()
        for x in parts
    ]

def image_files(directory: Path):
    if not directory.is_dir():
        return []

    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=natural_key)

def image_shape(path):
    if path is None:
        return None, None, None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        return None, None, None

    if image.ndim == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    return h, w, c

rows = []

patient_dirs = sorted(
    [
        p for p in DATA_ROOT.iterdir()
        if p.is_dir() and p.name.isdigit()
    ],
    key=lambda p: int(p.name),
)

for patient_dir in patient_dirs:
    pre_dir = patient_dir / "Pre-Image"
    post_dir = patient_dir / "Post-Image"
    pre_api_dir = patient_dir / "Pre-API"
    post_api_dir = patient_dir / "Post-API"

    pre_files = image_files(pre_dir)
    post_files = image_files(post_dir)
    pre_api_files = image_files(pre_api_dir)
    post_api_files = image_files(post_api_dir)

    pre_h, pre_w, pre_c = image_shape(
        pre_files[0] if pre_files else None
    )
    post_h, post_w, post_c = image_shape(
        post_files[0] if post_files else None
    )

    all_file_names = [
        p.name.lower()
        for p in patient_dir.rglob("*")
        if p.is_file()
    ]

    def has_keyword(keyword):
        keyword = keyword.lower()
        return any(keyword in name for name in all_file_names)

    rows.append({
        "patient_id": patient_dir.name,
        "patient_dir": str(patient_dir),

        "pre_image_exists": pre_dir.is_dir(),
        "pre_frame_count": len(pre_files),
        "pre_pair_count": max(len(pre_files) - 1, 0),
        "pre_first_frame": str(pre_files[0]) if pre_files else None,
        "pre_last_frame": str(pre_files[-1]) if pre_files else None,
        "pre_height": pre_h,
        "pre_width": pre_w,
        "pre_channels": pre_c,

        "post_image_exists": post_dir.is_dir(),
        "post_frame_count": len(post_files),
        "post_pair_count": max(len(post_files) - 1, 0),
        "post_first_frame": str(post_files[0]) if post_files else None,
        "post_last_frame": str(post_files[-1]) if post_files else None,
        "post_height": post_h,
        "post_width": post_w,
        "post_channels": post_c,

        "pre_api_dir_exists": pre_api_dir.is_dir(),
        "pre_api_image_count": len(pre_api_files),
        "post_api_dir_exists": post_api_dir.is_dir(),
        "post_api_image_count": len(post_api_files),

        "has_cbf_file": has_keyword("cbf"),
        "has_cbv_file": has_keyword("cbv"),
        "has_mtt_file": has_keyword("mtt"),
        "has_ttp_file": has_keyword("ttp"),

        "pre_seg_exists": (
            patient_dir / "Pre-Segmentation.nii.gz"
        ).is_file(),

        "post_seg_exists": (
            patient_dir / "Post-Segmentation.nii.gz"
        ).is_file(),
    })

df = pd.DataFrame(rows)

output = OUT_DIR / "filesystem_inventory.csv"
df.to_csv(output, index=False, encoding="utf-8-sig")

total_pairs = (
    df["pre_pair_count"].sum()
    + df["post_pair_count"].sum()
)

both_valid = (
    (df["pre_frame_count"] >= 2)
    & (df["post_frame_count"] >= 2)
)

print("========== DSA文件系统审计 ==========")
print("患者目录数:", len(df))
print("Pre至少2帧:", int((df["pre_frame_count"] >= 2).sum()))
print("Post至少2帧:", int((df["post_frame_count"] >= 2).sum()))
print("Pre和Post均至少2帧:", int(both_valid.sum()))
print("Pre完全缺失:", int((df["pre_frame_count"] == 0).sum()))
print("Post完全缺失:", int((df["post_frame_count"] == 0).sum()))
print("Pre分割存在:", int(df["pre_seg_exists"].sum()))
print("Post分割存在:", int(df["post_seg_exists"].sum()))
print("总相邻帧对:", int(total_pairs))
print("\nPre尺寸分布:")
print(
    df.groupby(["pre_height", "pre_width"], dropna=False)
    .size()
    .sort_values(ascending=False)
    .head(20)
)
print("\nPost尺寸分布:")
print(
    df.groupby(["post_height", "post_width"], dropna=False)
    .size()
    .sort_values(ascending=False)
    .head(20)
)
print("\n输出文件:", output)
