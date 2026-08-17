# api_png2d_segresnet_cave_fusion_v4_teacher_aligned

## Teacher-alignment patch note

Default spatial representation is now `global_only`: `[G_pre,G_post]`. Use `spatial.representation=global_gt_roi` or `global_pred_roi` only as S1/S2 ablations. `roi_source` is retained for backward compatibility and is not the representation selector.

`pilot_single` Train scores are explicitly pilot/not representation-cross-fit; use `strict_crossfit` for formal OOF.

## 这版为什么是最终收敛版

v4 只保留老师图真正规定的主路径，不再额外改融合数学：

```text
z2D_raw -> P2D -> z2D [256]
zT_raw  -> PT  -> zT  [256]

        ↓

双向条件门控

        ↓

hmain = [
  z2D_hat,
  zT_hat,
  z2D_hat * zT_hat,
  |z2D_hat-zT_hat|
]

hmain [1024]
        ↓
Fusion
        ↓
zmain [256]
        ↓
main_head
        ↓
adverse logit
```

投影层严格是：

```text
Linear(input_dim,256)
LayerNorm(256)
GELU
Dropout(0.2)
```

**主模型彻底删除 PCA。**

---

# 一、数据上的最重要简化

直接读取：

```text
/root/autodl-tmp/2D/image/{patient_id}_Pre.png
/root/autodl-tmp/2D/image/{patient_id}_Post.png

/root/autodl-tmp/2D/mask/{patient_id}_Pre.png
/root/autodl-tmp/2D/mask/{patient_id}_Post.png
```

这些已经是专门准备好的均值图和一一对应的 mask。

因此删除：

```text
原始时序 -> 重新求 mean
NIfTI mask
orientation
frame_paths
phase mapping
```

CAVE 也不重新跑，直接读取已有：

```text
train_features.npz -> deep
valid_features.npz -> deep
```

---

# 二、z2D_raw 怎么定义

每个 phase：

```text
mean PNG
  ↓
SegResNet encoder
  ↓
feature map F
```

提两个表示：

## global

```text
GAP(F)
```

## ROI

有两种模式。

### roi_source = gt（默认）

直接使用你已经有的对应 mask：

```text
GT mask
 ↓
resize to feature map
 ↓
masked pooling(F)
```

这是最贴合你当前“图像已经有对应分割”的数据条件。

### roi_source = pred

使用 SegResNet 自己预测：

```text
image
 ↓
SegResNet decoder
 ↓
pred mask probability
 ↓
masked pooling(F)
```

这是没有 GT mask 时也可部署的版本。

最终：

```text
z2D_raw = [
  Pre-global,
  Pre-ROI,
  Post-global,
  Post-ROI
]
```

默认 deepest channels=256 时：

```text
z2D_raw = 1024-D
```

然后严格：

```text
Linear(1024,256)
LayerNorm
GELU
Dropout(0.2)
```

---

# 三、SegResNet 到底需不需要重新训练

v4 不再强迫一个答案，而是支持三个明确模式。

## 1. pilot_single（默认，推荐你现在先跑）

目的：

> 先按老师说的“做个融合看看效果”。

流程：

```text
Train image+mask
       ↓
patient-level inner split
       ↓
选择 SegResNet best epoch
       ↓
重新初始化
       ↓
全部 Train image+mask refit
       ↓
得到一个 SegResNet checkpoint
       ↓
冻结
       ↓
提 Train / Valid spatial feature
```

然后只训练融合层。

优点：

```text
简单
快
最适合先给老师结果
```

限制：

```text
Train OOF 的 spatial representation learner 见过全部 Train 图像/mask，
所以该 OOF 是 pilot OOF，不是最终论文级无泄漏 OOF。
```

代码会在 metrics 中自动标：

```text
pilot_representation_warning = true
```

---

## 2. strict_crossfit（正式论文版）

对 outcome Fold 1：

```text
只用 Fold2-5 image+mask 训练 SegResNet
        ↓
给 Fold1 提 spatial feature
```

依次 5 folds。

这样：

```text
OOF patient 的 image/mask
也没有被对应 SegResNet 训练见过
```

这才是最严格版。

---

## 3. external_checkpoint

如果高老师后面直接给你一个真正匹配的 DSA SegResNet checkpoint：

```json
"spatial": {
  "strategy": "external_checkpoint",
  "external_checkpoint": "/path/to/model.pt"
}
```

此时：

```text
不训练 SegResNet
直接 load -> freeze -> feature extraction
```

程序不会使用随机初始化权重。

---

# 四、为什么现在不硬找一个网上 checkpoint

你这批数据是：

```text
2D grayscale cerebral DSA mean image
+
对应 lesion/vessel mask
```

最匹配的 supervision 已经在你自己手里。

因此默认：

```text
external_checkpoint = ""
```

让 MONAI SegResNet architecture 在你的 image-mask pairs 上学习。

如果以后拿到同域 DSA checkpoint，可以 warm start 或 external 模式。

---

# 五、融合主路径

## P2D

```text
z2D_raw [B,D2D]
 ↓
Linear(D2D,256)
 ↓
LayerNorm
 ↓
GELU
 ↓
Dropout(0.2)
 ↓
z2D [B,256]
```

## PT

```text
CAVE deep [B,10240]
 ↓
Linear(10240,256)
 ↓
LayerNorm
 ↓
GELU
 ↓
Dropout(0.2)
 ↓
zT [B,256]
```

没有 PCA。

---

# 六、双向条件门控

```text
a2D = sigmoid(f2D([z2D,zT]))
aT  = sigmoid(fT([zT,z2D]))
```

然后：

```text
z2D_hat =
(1-a2D)*z2D
+
a2D*phi_T_to_2D(zT)

zT_hat =
(1-aT)*zT
+
aT*phi_2D_to_T(z2D)
```

---

# 七、交互特征

```text
hmain = [
 z2D_hat,
 zT_hat,
 z2D_hat * zT_hat,
 abs(z2D_hat-zT_hat)
]
```

每个 256：

```text
hmain = 1024-D
```

再：

```text
1024
 ↓
512
 ↓
256
```

得到：

```text
zmain [B,256]
```

最后：

```text
Linear(256,1)
```

输出 adverse logit。

---

# 八、实验只保留 5 个

```text
E0 cave_only
E1 spatial_only
E2 concat
E3 interaction
E4 gated_interaction
```

不要一开始再塞 morphology、3D、TabPFN、各种 feature ban。

先把老师这个问题回答干净：

```text
CAVE
vs
SegResNet
vs
CAVE + SegResNet
vs
老师的双向门控
```

---

# 九、你现在先用 pilot_single

配置：

```json
"spatial": {
  "strategy": "pilot_single",
  "roi_source": "gt"
}
```

这是我现在最推荐你第一轮跑的。

因为：

```text
你已经有 GT mask
老师现在只是想看融合效果
```

所以不必一上来训练五个 SegResNet。

---

# 十、运行顺序

## 1. preflight

```bash
cd /root/autodl-tmp/aneurysm

bash \
code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh \
preflight
```

确认：

```text
PREFLIGHT_OK
```

重点看：

```text
reports/.../00_image_mask_audit.csv
```

尤其：

```text
mask_area_ratio
```

因为 Windows 缩略图里 mask 看起来几乎全黑，很可能只是 ROI 很小。

---

## 2. 训练一个 pilot SegResNet

```bash
bash \
code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh \
train-spatial
```

看：

```text
segmentation/pilot/search_history.csv
```

重点看：

```text
valid_dice
```

---

## 3. 提空间特征

```bash
bash \
code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh \
extract
```

输出：

```text
seg_features/pilot/train.npz
seg_features/pilot/valid.npz
```

其中包括：

```text
global
gt_roi
pred_roi
gt_combined
pred_combined
```

默认融合：

```text
roi_source = gt
```

所以使用：

```text
gt_combined
=
[
Pre-global,
Pre-GT-ROI,
Post-global,
Post-GT-ROI
]
```

---

## 4. 先跑三个关键结果

```bash
bash code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh fusion cave_only

bash code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh fusion spatial_only

bash code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh fusion concat
```

先看：

```text
CAVE only
Spatial only
Concat
```

如果 concat 确实提升，再：

```bash
bash code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh fusion interaction

bash code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh fusion gated_interaction
```

---

## 5. 汇总

```bash
bash \
code/api_png2d_segresnet_cave_fusion_v4_teacher_aligned/run_pipeline.sh \
summarize
```

最终表：

```text
mode
OOF AUROC
OOF AUPRC
OOF Brier
Valid AUROC
Valid AUPRC
Valid Brier
ΔOOF AUPRC vs CAVE
folds improved vs CAVE
```

---

# 十一、pilot 有结果后再切 formal

只改：

```json
"strategy": "strict_crossfit"
```

然后：

```bash
train-spatial all
extract all
fusion all
summarize
```

这时才是严格 representation-level OOF。

---

# 十二、这版与 v3 的关键差异

```text
v3:
CAVE -> PCA -> 256        ❌ 主路径偏离老师图

v4:
CAVE raw -> Linear ->256  ✅
```

```text
v3:
一上来默认严格 5-fold SegResNet

v4:
pilot_single 默认         ✅ 先看老师要求的效果
strict_crossfit 可选       ✅ 最后正式实验
```

```text
v3:
pred-mask ROI 更偏部署逻辑

v4:
GT ROI 默认                ✅ 利用你已经有的 mask
pred ROI 同时保留           ✅ 后续部署/消融
```

```text
v4:
z2D_raw = [
 Pre-global,
 Pre-ROI,
 Post-global,
 Post-ROI
]
```

这就是我现在建议正式锁住的定义。
