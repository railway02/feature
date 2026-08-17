# Train/Valid 与影像目录清单
# 输出：
# patient_manifest.csv
# record_manifest.csv
# duplicate_patient_records.csv
# missing_patient_dirs.csv
# incomplete_sequences.csv
# pilot_10.csv


from pathlib import Path
import re
import cv2
import pandas as pd


PATCH_ROOT = Path(
    "/root/autodl-tmp/aneurysm/"
    "staging/updated_10_cases"
)

OUT_DIR = Path(
    "/root/autodl-tmp/aneurysm/manifests"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff",
}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [
        int(part) if part.isdigit()
        else part.lower()
        for part in parts
    ]


def sequence_images(directory: Path):
    if not directory.is_dir():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTS
                and not any(
                    key in path.name.lower()
                    for key in (
                        "cbf", "cbv",
                        "mtt", "ttp",
                    )
                )
            )
        ],
        key=natural_key,
    )


def image_shape(path):
    if path is None:
        return None, None, None

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        return None, None, None

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    return height, width, channels


def find_api_map(directory: Path, key: str):
    if not directory.is_dir():
        return None

    matches = [
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
            and key.lower() in path.name.lower()
        )
    ]

    return str(matches[0]) if matches else None


def find_segmentation(lesion_dir: Path, phase: str):
    # 结构1：
    # Pre-Segmentation.nii.gz
    direct = lesion_dir / f"{phase}-Segmentation.nii.gz"

    if direct.is_file():
        return str(direct)

    # 结构2：
    # Pre-biaozhu/Segmentation.nii.gz
    nested = (
        lesion_dir
        / f"{phase}-biaozhu"
        / "Segmentation.nii.gz"
    )

    if nested.is_file():
        return str(nested)

    return None


rows = []

for patient_dir in sorted(
    [
        path
        for path in PATCH_ROOT.iterdir()
        if path.is_dir()
    ],
    key=lambda path: int(path.name),
):
    for lesion_dir in sorted(
        [
            path
            for path in patient_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name,
    ):
        pre_dir = lesion_dir / "Pre-Image"
        post_dir = lesion_dir / "Post-Image"

        pre_api_dir = lesion_dir / "Pre-API"
        post_api_dir = lesion_dir / "Post-API"

        pre_frames = sequence_images(pre_dir)
        post_frames = sequence_images(post_dir)

        pre_h, pre_w, pre_c = image_shape(
            pre_frames[0] if pre_frames else None
        )

        post_h, post_w, post_c = image_shape(
            post_frames[0] if post_frames else None
        )

        rows.append({
            "sample_id": (
                f"{patient_dir.name}_"
                f"{lesion_dir.name}"
            ),
            "patient_id": patient_dir.name,
            "lesion_folder": lesion_dir.name,
            "data_source": "updated_10",
            "lesion_dir": str(lesion_dir),

            "pre_image_dir": str(pre_dir),
            "pre_frame_count": len(pre_frames),
            "pre_pair_count": max(
                len(pre_frames) - 1,
                0,
            ),
            "pre_first_frame": (
                str(pre_frames[0])
                if pre_frames else None
            ),
            "pre_last_frame": (
                str(pre_frames[-1])
                if pre_frames else None
            ),
            "pre_height": pre_h,
            "pre_width": pre_w,
            "pre_channels": pre_c,

            "post_image_dir": str(post_dir),
            "post_frame_count": len(post_frames),
            "post_pair_count": max(
                len(post_frames) - 1,
                0,
            ),
            "post_first_frame": (
                str(post_frames[0])
                if post_frames else None
            ),
            "post_last_frame": (
                str(post_frames[-1])
                if post_frames else None
            ),
            "post_height": post_h,
            "post_width": post_w,
            "post_channels": post_c,

            "pre_cbf": find_api_map(
                pre_api_dir, "CBF"
            ),
            "pre_cbv": find_api_map(
                pre_api_dir, "CBV"
            ),
            "pre_mtt": find_api_map(
                pre_api_dir, "MTT"
            ),
            "pre_ttp": find_api_map(
                pre_api_dir, "TTP"
            ),

            "post_cbf": find_api_map(
                post_api_dir, "CBF"
            ),
            "post_cbv": find_api_map(
                post_api_dir, "CBV"
            ),
            "post_mtt": find_api_map(
                post_api_dir, "MTT"
            ),
            "post_ttp": find_api_map(
                post_api_dir, "TTP"
            ),

            "pre_seg_path": find_segmentation(
                lesion_dir, "Pre"
            ),
            "post_seg_path": find_segmentation(
                lesion_dir, "Post"
            ),
        })


df = pd.DataFrame(rows)

df["pre_api_complete"] = df[
    ["pre_cbf", "pre_cbv", "pre_mtt", "pre_ttp"]
].notna().all(axis=1)

df["post_api_complete"] = df[
    ["post_cbf", "post_cbv", "post_mtt", "post_ttp"]
].notna().all(axis=1)

df["pre_seg_exists"] = df[
    "pre_seg_path"
].notna()

df["post_seg_exists"] = df[
    "post_seg_path"
].notna()

df["ready_for_flow"] = (
    (df["pre_frame_count"] >= 2)
    & (df["post_frame_count"] >= 2)
)

output = OUT_DIR / "updated_10_inventory.csv"

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig",
)

print("========== 修改10例审计 ==========")
print("患者数:", df["patient_id"].nunique())
print("病灶目录数:", len(df))
print(
    "Pre至少2帧:",
    int((df["pre_frame_count"] >= 2).sum()),
)
print(
    "Post至少2帧:",
    int((df["post_frame_count"] >= 2).sum()),
)
print(
    "Pre/Post都可跑:",
    int(df["ready_for_flow"].sum()),
)
print(
    "Pre API完整:",
    int(df["pre_api_complete"].sum()),
)
print(
    "Post API完整:",
    int(df["post_api_complete"].sum()),
)
print(
    "Pre分割存在:",
    int(df["pre_seg_exists"].sum()),
)
print(
    "Post分割存在:",
    int(df["post_seg_exists"].sum()),
)
print(
    "总相邻帧对:",
    int(
        df["pre_pair_count"].sum()
        + df["post_pair_count"].sum()
    ),
)

print("\n病灶明细：")
print(
    df[
        [
            "sample_id",
            "pre_frame_count",
            "post_frame_count",
            "pre_api_complete",
            "post_api_complete",
            "pre_seg_exists",
            "post_seg_exists",
            "ready_for_flow",
        ]
    ].to_string(index=False)
)

print("\n输出:", output)
