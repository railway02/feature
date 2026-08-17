# Manifest 与输入输出规范

## 核心主键

| 层级 | 主键 | 用途 |
|---|---|---|
| series | `series_uid` | 原 CAVE Pre/Post 聚合 |
| phase | `phase_uid = series_uid::phase` | Mask、ROI、逐帧裁切、CAVE phase 输出 |
| 帧列表 | `frame_list_hash` | 在 worker 中把 CAVE 当前读取的帧精确关联到 ROI |

## 输入

### 原始 series manifest

必须保留：

```text
patient_id
split
series_uid
series_path
source_medical_record_root
can_run_pre
can_run_post
n_pre_frames
n_post_frames
pre_frame_paths
post_frame_paths
pre_frame_list_hash
post_frame_list_hash
```

### 标注文件

```text
Image.nii.gz + Segmentation.nii.gz
或
Pre-Segmentation.nii.gz / Post-Segmentation.nii.gz
```

### 原 Whole-CAVE metadata

```text
whole_featurebank/<split>/<patient>/<series_uid>/<phase>/metadata.json
```

用于读取固定的时间 views。

## 关键中间 manifest

### `source_phase_index_all.csv`

一行一个原始 runnable phase；预期 2622 行。

### `mask_inventory.csv`

一行一个发现的 Mask 路径，包含文件 hash、reference 路径、phase/series 路径提示。

### `source_phase_with_mask_map.csv`

一行一个原始 runnable phase；仍为 2622 行。字段 `phase_mapping_status` 必须全部为 `accepted`。

### `roi_phase_manifest_full.csv`

CAVE worker 的空间裁切真值表，必须包含：

```text
phase_uid
series_uid
phase
frame_paths
frame_list_hash
mask_path
mask_sha256
orientation_transform
original_bbox
expanded_bbox
fallback_bbox
whole_metadata_path
whole_metadata_sha256
```

### `cave_manifest_local_{train,valid}_full.csv`

直接继承原 all-series manifest，不缩小、不重排、不修改 Pre/Post availability。

## featurebank 输出

```text
cave_local_full_featurebank/<split>/<patient_id>/<series_uid>/<phase>/
```

每个 phase 必须有：

```text
.SUCCESS.json
metadata.json
原 CAVE feature 文件
```

## 完成标记

```text
manifests/.../.STAGE1_FULLMASK_SUCCESS.json
outputs/.../cave_local_full_featurebank/.FULL_FEATUREBANK_SUCCESS.json
```

前者证明 Mask→ROI→manifest 已全量闭合；后者证明 2622 个 phase 特征已全量闭合。
