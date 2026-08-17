# GT-Mask ROI-CAVE v5：真实 Mask、eligible 全量 Local-CAVE

## 这版做什么

不训练分割模型，不生成 Pred Mask，不为凑数复用/改名/生成任何 Mask。数据中已有的 `.nii/.nii.gz` 人工 Mask 是唯一输入：

```text
已有 Image.nii.gz + Segmentation.nii.gz
或已有 Pre/Post-Segmentation.nii.gz
        ↓
每个 Mask 只映射到其自身的 series_uid + phase
        ↓
所有非零标签合并（labels != 0），只用于定位
        ↓
tight bbox → 正方形 1.5× padding（最小 96 px，向上对齐 32）
        ↓
同一 ROI 坐标裁该 phase 的全部帧（内存中，不落盘）
        ↓
原 CAVE 全量重新提取 Local 特征（网络/checkpoint/归一化/pooling 不变）
```

**coverage 政策是 `eligible_only`，不再要求 2622/2622 强行闭合**：

```text
source_phase_index_all.csv 永远包含全部 2622 个 source phase
= eligible_phase_count（有可靠 Mask，进入提取）
+ excluded_phase_count（每个都有明确 local_exclusion_reason）
```

缺失或不可靠的 phase 只保留排除记录，最后预测训练时按任务过滤。

## local_eligible 的定义

一个 source phase 进入 Local-CAVE 提取，当且仅当：

- Mask 映射到**自身正确**的 `series_uid + phase`（不跨 phase、不跨 series 复用）；
- Mask 可读取；
- Mask 与帧尺寸匹配（`allow_mask_resize=false`，不自动 resize）；
- foreground 非空；
- ROI 合法（不裁病灶）；
- Whole-CAVE temporal metadata 存在。

排除原因示例：`no_mask_mapping`、`mapping_needs_review`、`mapping_conflict`、
`mask_unreadable:*`、`shape_mismatch:*`、`empty_foreground`、`roi_build_error:*`、
`whole_metadata_missing:*`。

## 方向（orientation）优先级

```text
1. upstream authoritative orientation_transform
2. paired Image.nii 与真实帧匹配得到的 orientation
3. manual mapping 显式指定
4. path_exact / unique 系列方法：identity 默认值，orientation_status=default_identity_unverified
```

没有自动方向猜测；审计不改变方向。所有非 upstream/manual 映射的 phase 全部生成 QA 叠加图供人工核查。

## 两种 Mask 输入

- A. 配对标注：`Pre-biaozhu/Image.nii.gz + Segmentation.nii.gz`。Image 只用于确定 series/phase/帧/方向，不作为 CAVE 输入。
- B. 独立 phase Mask：`Pre-Segmentation.nii.gz / Post-Segmentation.nii.gz`。按 series 目录与 phase 文件名映射。

## 安装与配置

```bash
cd /root/autodl-tmp
PROJECT=/root/autodl-tmp/aneurysm \
  bash api_gtmask_roi_cave_v5_fullmask_fullseq/api_gtmask_roi_cave_v5_fullmask_fullseq/install.sh
```

检查 `/root/autodl-tmp/aneurysm/configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json`：
源 manifest、whole featurebank、CAVE 代码/仓库/checkpoint 路径、`runtime`（gpu_processes/io_workers）。

## 执行顺序

```bash
export PYTHON=/root/autodl-tmp/envs/aneurysm-ml/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_gtmask_roi_cave_v5_fullmask_fullseq

bash $CODE/run_pipeline.sh stage1                     # index→discover→map→roi-eligible→qa→validate-stage1-eligible
# 检查 reports/.../roi_qa/、02_source_phase_mapping_gaps.csv、03_local_feature_exclusion.csv
# 需要时在 manual_mask_mapping.csv 补人工映射（只允许映射到其自身 series+phase），重跑 map→roi-eligible→qa→validate

bash $CODE/run_pipeline.sh smoke-train                # 真实 GPU smoke + smoke_verify 断言
bash $CODE/run_pipeline.sh smoke-valid

bash $CODE/run_pipeline.sh extract-all-eligible       # 全量 eligible 提取，支持断点续跑
bash $CODE/run_pipeline.sh validate-features-eligible
bash $CODE/run_pipeline.sh build-tables-eligible
```

## 关键输出

```text
manifests/.../source_phase_index_all.csv              # 全部 2622 phase
manifests/.../local_phase_coverage_all.csv            # 每 phase 的 eligible 判定与排除原因
manifests/.../roi_phase_manifest_eligible.csv         # eligible ROI（裁切真值表）
manifests/.../cave_manifest_local_{train,valid}_eligible.csv
manifests/.../.STAGE1_ELIGIBLE_SUCCESS.json           # 阶段一锁（coverage_policy=eligible_only）
reports/.../roi_qa/                                   # 叠加质控图（造影代表帧）
outputs/.../cave_local_eligible_featurebank/          # 独立 Local featurebank
outputs/.../cave_local_eligible_featurebank/.ELIGIBLE_FEATUREBANK_SUCCESS.json
outputs/.../tables/local_eligible/{train,valid}/series_embeddings_5120.npz
outputs/.../tables/local_eligible/local_series_availability_{train,valid}.csv
reports/.../08_local_phase_availability.csv
reports/.../08_local_feature_exclusion.csv
```

## QA 帧选取

QA 不再只用第一帧（首帧可能无造影）。每个 phase 从冻结的 Whole-CAVE
`contrast_core20` 时间索引中取代表帧（0/25/50/75/100 分位，去重后最多 3 帧），
叠加 Mask 轮廓、tight bbox、expanded ROI，并给出局部裁剪预览。

## 已知数据修复（provenance 保留）

- `tiantanDSA/651374/Ach/Pre-Segmentation.nii.gz` 原是未压缩的裸 NIfTI（错误 .gz 后缀），
  已在保持字节内容不变的情况下补 gzip 容器。原始备份与新旧 sha256 见
  `staging/file_repairs/Valid__651374__Ach__pre__provenance.json`。
- `tiantanDSA/663779/Post-Sgmentation.nii.gz` 文件名笔误已改为 `Post-Segmentation.nii.gz`。

## 下游过滤字段（记录级训练用）

`local_series_availability_*.csv` 提供：

```text
local_pre_available / local_post_available / local_both_available / local_any_available
```

- Pre 任务：`local_pre_available == 1`
- Post 任务：`local_post_available == 1`
- Pre+Post 任务：`local_pre_available == 1 and local_post_available == 1`

后续实验：Whole-full / Whole-matched / Local / Whole-matched+Local / Mask-only；
Whole-matched 必须与 Local 使用完全相同的记录。
