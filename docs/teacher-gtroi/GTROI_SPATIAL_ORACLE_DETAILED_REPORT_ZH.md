# GTROI Spatial Oracle：完整流程、结果与“最佳版本”定位

**日期：2026-08-09**  
**状态：已有实验的只读总结；未训练新模型、未重新提取特征、未重新运行 outcome。**

---

## 1. 先回答“这是不是最佳版本”

答案需要分成“实验指标最佳”和“实际部署最佳”两个层面。

### 1.1 如果按 strict Train OOF AUPRC

当前统一七模型中：

| 模型 | OOF AUPRC |
|:--|--:|
| GTROI_2D_only | **0.245125** |
| GTROI_CAVE_gated_oracle | 0.229501 |
| PredROI+CAVE gated primary | 0.209765 |
| PredROI_2D_only | 0.199404 |
| Historical CAVE Logistic | 0.186670 |
| PredROI+CAVE Logistic | 0.173391 |
| Global+CAVE gated | 0.172532 |

因此，按本任务最重视的 strict OOF AUPRC，**GTROI_2D_only 是当前已完成模型中最高的版本**。

### 1.2 如果按 strict Train OOF AUROC

| 模型 | OOF AUROC |
|:--|--:|
| GTROI_CAVE_gated_oracle | **0.588590** |
| GTROI_2D_only | 0.576197 |
| PredROI+CAVE gated primary | 0.546053 |
| PredROI_2D_only | 0.542038 |
| Historical CAVE Logistic | 0.529878 |
| PredROI+CAVE Logistic | 0.506393 |
| Global+CAVE gated | 0.494256 |

因此，按 strict OOF AUROC，**GTROI_CAVE_gated_oracle 是当前最高版本**。

### 1.3 为什么不能直接把它叫作最终部署主模型

GTROI 使用人工 GT Mask：

    完整 Mean PNG
      -> SegResNet encoder fmap
      -> 人工 GT Mask 引导 ROI pooling

实际部署时通常没有人工 GT Mask，因此 GTROI 属于 oracle 条件。它回答的是：

> 如果病灶定位完全正确，现有 SegResNet encoder 特征能够达到什么水平？

它不能回答：

> 在完全自动推理时，模型能达到什么水平？

所以当前正确命名是：

| 层面 | 模型 |
|:--|:--|
| 当前最高 OOF AUPRC 的 spatial oracle | GTROI_2D_only |
| 当前最高 OOF AUROC 的 oracle fusion | GTROI_CAVE_gated_oracle |
| 当前冻结的可自动推理主模型 | PredROI-CAVE-Deep Gated Fusion |
| 可自动推理的 2D-only 分支 | PredROI_2D_only |

---

## 2. 数据与 strict fold 边界

Outcome cohort：

| Split | Series | Positive | Negative |
|:--|--:|--:|--:|
| Train | 781 | 133 | 648 |
| Valid | 207 | 37 | 170 |

Train 使用固定 patient-grouped 五折：

| Fold | Holdout series |
|--:|--:|
| 1 | 157 |
| 2 | 156 |
| 3 | 156 |
| 4 | 156 |
| 5 | 156 |

每个 series 有：

    Pre Mean PNG
    Post Mean PNG
    Pre GT Mask
    Post GT Mask

对 outcome outer fold k：

    SegResNet fold-k development = Train fold != k
    SegResNet fold-k excluded    = Train fold k + all Valid

后续 outcome fold-k 必须读取：

    seg_features/fold_k/train.npz
    seg_features/fold_k/valid.npz

不能把 fold-1 outcome holdout 与 fold-2 featurebank 混用。fold-k featurebank 才对应排除了 fold-k 图像和 Mask 的 strict SegResNet。

---

## 3. SegResNet 是怎样训练出来的

实际 architecture：

| 参数 | 值 |
|:--|:--|
| spatial_dims | 2 |
| in_channels | 1 |
| out_channels | 1 |
| init_filters | 32 |
| blocks_down | [1,2,2,4] |
| blocks_up | [1,1,1] |
| dropout_prob | None |

训练输入：

    image [B,1,768,768]
    binary GT Mask [B,1,768,768]

分割损失：

    Lseg =
      0.8 * DiceLoss
      + 0.2 * weighted BCEWithLogits

优化设置：

| 项目 | 值 |
|:--|:--|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 4 |
| AMP | enabled |
| Max epochs | 80 |
| Min epochs | 10 |
| Patience | 12 |
| Inner validation fraction | 0.15 |
| Epoch selection metric | segmentation Dice |

每折训练过程：

1. 在 outer development 患者中再划分 segmentation inner-train 和 inner-valid。
2. 使用 inner-valid Dice 搜索最佳 epoch。
3. 丢弃 epoch-search model。
4. 重新随机初始化相同 SegResNet。
5. 在完整 outer development 上训练选定的 epoch 数。
6. 保存最终 model.pt。
7. 最终 model.pt 才用于 feature extraction。

五折实际 epoch：

| Fold | Development series | Development phases | Selected epoch | Best inner-valid Dice |
|--:|--:|--:|--:|--:|
| 1 | 624 | 1248 | 56 | 0.566275 |
| 2 | 625 | 1250 | 63 | 0.629571 |
| 3 | 625 | 1250 | 55 | 0.572622 |
| 4 | 625 | 1250 | 20 | 0.548377 |
| 5 | 625 | 1250 | 40 | 0.629966 |

五个 checkpoint 全部为随机初始化训练。strict config 的 external checkpoint 为空，五个 SUCCESS 和 model.pt 的 init_info 均为 used=false，没有加载 pretrained weights。

---

## 4. 完整 Mean PNG 如何进入 SegResNet

输入不是先按病灶裁剪。

实际过程：

    完整灰度 Mean PNG
      -> OpenCV grayscale read
      -> image 1st/99th percentile normalization
      -> clip to [0,1]
      -> keep aspect ratio
      -> letterbox to 768x768
      -> tensor [1,768,768]

Mask 处理：

    original Mask
      -> mask > 0
      -> binary foreground/background
      -> same geometric letterbox
      -> [1,768,768]

如果人工 Mask 原始像素值包含 0/1/2：

    0   -> background
    1   -> foreground
    2   -> foreground

代码不区分 1 和 2 的病灶类别，所有大于 0 的像素统一视为 ROI foreground。

Feature extraction 阶段不使用 brightness、contrast 或 Gaussian noise augmentation。

---

## 5. encoder 的真实层与 tensor shape

使用真实 Mean PNG：

    /root/autodl-tmp/2D/image/232848_Pre.png

以及 strict fold-1 model.pt，在 model.eval() 与 torch.no_grad() 下实际 forward：

| Tensor/module | Shape |
|:--|:--|
| Input | [1,1,768,768] |
| convInit | [1,32,768,768] |
| down_layers[0] | [1,32,768,768] |
| down_layers[1] | [1,64,384,384] |
| down_layers[2] | [1,128,192,192] |
| down_layers[3] | [1,256,96,96] |
| fmap | [1,256,96,96] |
| segmentation logits | [1,1,768,768] |

准确代码路径：

    02_extract_spatial_features.extract_phase_features()
      -> segresnet_model.encode_and_decode(model,x)
      -> segresnet_model.encode(model,x)
      -> model.encode(x)[0]
      -> model.down_layers[3] output

因此：

    F = fmap = final encoder down-stage output
    F shape per image = [256,96,96]

F 不是：

- decoder/up layer feature；
- segmentation logits；
- logits sigmoid probability Mask；
- 单独添加的 projection 或 bottleneck；
- CAVE feature。

---

## 6. 同一个 fmap 上的三种空间特征

令：

    F in R^(256 x 96 x 96)

### 6.1 Global feature G

实现：

    G = adaptive_avg_pool2d(F,1).flatten()

逐 channel：

    G[c] = mean over all h,w of F[c,h,w]

输出：

    G_pre  [256]
    G_post [256]

它概括整张 Mean PNG 的全局空间上下文。

### 6.2 Predicted ROI feature

首先从 decoder logits 得到概率 Mask：

    P = sigmoid(logits / temperature)

配置：

    temperature = 1.0

P 的初始 shape：

    [1,768,768]

然后：

    P96 = bilinear_resize(P,96x96)

PredROI weighted pooling：

    ROI_pred[c]
      = sum_h,w F[c,h,w] * P96[h,w]
        /
        max(sum_h,w P96[h,w], 1e-6)

输出：

    ROI_pred_pre  [256]
    ROI_pred_post [256]

这里不对 P 做 0.5 hard threshold，而是使用连续预测概率。

### 6.3 GT ROI feature

人工 Mask 先完成：

    mask > 0

然后：

    M96 = area_resize(binary_GT_mask,96x96)

因为使用 area interpolation，M96 的边缘可以是 0 到 1 之间的面积覆盖权重，而不一定仍是纯 0/1。

GT ROI pooling：

    ROI_gt[c]
      = sum_h,w F[c,h,w] * M96[h,w]
        /
        max(sum_h,w M96[h,w], 1e-6)

代码要求：

    sum(M96) > 1e-6

如果 GT ROI 在 fmap 尺度消失，提取立即失败。现有 strict featurebank 已全部成功生成。

输出：

    ROI_gt_pre  [256]
    ROI_gt_post [256]

---

## 7. 为什么要同时保留 Global 和 ROI

只使用 ROI 可能丢失：

- 整体血管分布；
- 全局造影强度；
- 非病灶区域结构；
- 全幅背景和采集上下文；
- Pre/Post 整体差异。

只使用 Global 又可能弱化小病灶的局部信号。

因此当前 combined representation 同时包含：

    global context
    lesion-localized context
    Pre/Post phase information

GT oracle 的最终顺序是：

    gt_combined =
    [
      G_pre,
      ROI_gt_pre,
      G_post,
      ROI_gt_post
    ]

维度：

    256 + 256 + 256 + 256 = 1024

PredROI 对应：

    pred_combined =
    [
      G_pre,
      ROI_pred_pre,
      G_post,
      ROI_pred_post
    ]
    = 1024-D

---

## 8. featurebank 的实际文件与 shape

对每个 strict fold，保存：

    seg_features/fold_k/train.npz
    seg_features/fold_k/valid.npz

NPZ keys：

| Key | Train shape | Valid shape | 定义 |
|:--|:--|:--|:--|
| global | [781,512] | [207,512] | [G_pre,G_post] |
| gt_roi | [781,512] | [207,512] | [ROI_gt_pre,ROI_gt_post] |
| pred_roi | [781,512] | [207,512] | [ROI_pred_pre,ROI_pred_post] |
| gt_combined | [781,1024] | [207,1024] | [G_pre,ROI_gt_pre,G_post,ROI_gt_post] |
| pred_combined | [781,1024] | [207,1024] | [G_pre,ROI_pred_pre,G_post,ROI_pred_post] |
| series_uid | [781] | [207] | row primary key |
| patient_id | [781] | [207] | patient grouping |
| target | [781] | [207] | adverse outcome |

每个 fold 的 featurebank 覆盖相同 781 Train 与 207 Valid，但 tensor 来自不同的 fold-specific SegResNet。

---

## 9. GTROI_2D_only 的 outcome 训练

输入：

    x2D_raw = gt_combined [1024]

不连接 CAVE。

模型：

    Linear(1024,256)
      -> LayerNorm(256)
      -> GELU
      -> Dropout(0.2)
      -> Linear(256,1)
      -> adverse outcome logit

训练设置：

| 项目 | 值 |
|:--|:--|
| Outer folds | fixed strict patient-grouped 5 folds |
| Inner validation | outer development 内 patient grouped/stratified 18% |
| Epoch selection | inner-valid AUPRC |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-3 |
| Batch size | 128 |
| Max/min epoch | 160/15 |
| Patience | 20 |
| Seed | 20260818 |
| Loss | weighted BCEWithLogits |

每折：

1. development = Train fold != k。
2. holdout = Train fold == k。
3. 读取 seg_features/fold_k 的 gt_combined。
4. development 内选择 epoch。
5. fresh initialization。
6. 在完整 development refit。
7. 只预测 fold-k holdout。
8. 五个 holdout 拼成 781 OOF。
9. 五个 models 都预测 207 Valid。
10. Valid probability 按例平均。

### 9.1 Fold-level 结果

| Fold | Best epoch | Inner AUPRC | Holdout AUROC | Holdout AUPRC | Holdout Brier |
|--:|--:|--:|--:|--:|--:|
| 1 | 37 | 0.297728 | 0.568383 | 0.234650 | 0.181542 |
| 2 | 25 | 0.310792 | 0.555268 | 0.236870 | 0.186898 |
| 3 | 43 | 0.501605 | 0.599770 | 0.295091 | 0.231129 |
| 4 | 114 | 0.334261 | 0.638166 | 0.316886 | 0.174120 |
| 5 | 8 | 0.245223 | 0.526107 | 0.297702 | 0.220013 |

### 9.2 Pooled 结果

| Split | AUROC | AUPRC | Brier |
|:--|--:|--:|--:|
| Train strict OOF | **0.576197** | **0.245125** | **0.198718** |
| Valid | **0.663752** | **0.303179** | **0.171596** |

这是当前七模型中最高的 strict OOF AUPRC，也是最低的 strict OOF Brier。

---

## 10. GTROI_CAVE_gated_oracle 的 outcome 训练

Spatial：

    gt_combined [1024]

Temporal：

    CAVE deep [10240]
    = Pre deep [5120] + Post deep [5120]

两个 raw projections：

    gt_combined
      -> Linear(1024,256)
      -> LayerNorm
      -> GELU
      -> Dropout(0.2)
      -> z2D

    CAVE deep
      -> Linear(10240,256)
      -> LayerNorm
      -> GELU
      -> Dropout(0.2)
      -> zT

固定老师双向门控：

    a2D = sigmoid(f2D([z2D,zT]))
    aT  = sigmoid(fT([zT,z2D]))

    z2D_hat =
      (1-a2D)*z2D
      + a2D*phi_T_to_2D(zT)

    zT_hat =
      (1-aT)*zT
      + aT*phi_2D_to_T(z2D)

四项 interaction：

    hmain =
    [
      z2D_hat,
      zT_hat,
      z2D_hat*zT_hat,
      abs(z2D_hat-zT_hat)
    ]
    = 1024-D

Fusion：

    1024
      -> 512
      -> LayerNorm
      -> GELU
      -> Dropout
      -> 256
      -> LayerNorm
      -> GELU
      -> main head
      -> adverse logit

没有比较 concat、ordinary interaction 或 gated necessity。门控结构按老师要求固定。

### 10.1 Fold-level 结果

| Fold | Best epoch | Inner AUPRC | Holdout AUROC | Holdout AUPRC | Holdout Brier |
|--:|--:|--:|--:|--:|--:|
| 1 | 9 | 0.318485 | 0.587209 | 0.239094 | 0.308942 |
| 2 | 10 | 0.304149 | 0.587712 | 0.230530 | 0.197215 |
| 3 | 25 | 0.456121 | 0.618719 | 0.285166 | 0.204176 |
| 4 | 47 | 0.334044 | 0.657692 | 0.287602 | 0.201417 |
| 5 | 10 | 0.234576 | 0.570382 | 0.275192 | 0.177117 |

### 10.2 Pooled 结果

| Split | AUROC | AUPRC | Brier |
|:--|--:|--:|--:|
| Train strict OOF | **0.588590** | **0.229501** | **0.217890** |
| Valid | **0.616057** | **0.251398** | **0.176937** |

它是当前七模型中 strict OOF AUROC 最高的版本，但 OOF AUPRC 低于 GTROI_2D_only。

---

## 11. GTROI 与 PredROI 的直接对照

### 11.1 2D-only

GTROI_2D_only 减去 PredROI_2D_only：

| Split | Delta AUROC | Delta AUPRC | Delta Brier |
|:--|--:|--:|--:|
| Train strict OOF | +0.034159 | +0.045721 | -0.004244 |
| Valid | -0.000318 | +0.015366 | -0.010850 |

strict OOF 上三项均支持 GTROI：AUROC 和 AUPRC 更高，Brier 更低。

这说明现有预测 ROI 与人工 GT ROI 之间仍有性能差距，空间定位质量可能是当前自动路径的限制因素。

### 11.2 加 CAVE gated

GTROI_CAVE_gated_oracle 减去 PredROI_CAVE gated primary：

| Split | Delta AUROC | Delta AUPRC | Delta Brier |
|:--|--:|--:|--:|
| Train strict OOF | +0.042537 | +0.019735 | -0.004748 |
| Valid | -0.005723 | -0.007557 | +0.002633 |

strict OOF 上 GTROI fusion 更好；Valid 没有保持同方向。因此应报告为当前 cohort 的 oracle 上限，不应写成已经证明 GTROI fusion 稳定优于 PredROI fusion。

---

## 12. 2D-only 与 2D+CAVE 的关系

### 12.1 GTROI 路径

GTROI_CAVE_gated_oracle 减去 GTROI_2D_only：

| Split | Delta AUROC | Delta AUPRC | Delta Brier |
|:--|--:|--:|--:|
| Train strict OOF | +0.012392 | -0.015624 | +0.019172 |
| Valid | -0.047695 | -0.051781 | +0.005341 |

加入 CAVE 后：

- strict OOF AUROC 提高；
- strict OOF AUPRC 下降；
- strict OOF Brier 变差；
- Valid 三项未超过 GTROI_2D_only。

所以当前不能简单说“加 CAVE 一定比 2D-only 更好”。它提升了一个排序维度，但没有改善最重视的 OOF AUPRC。

### 12.2 PredROI 路径

PredROI+CAVE gated primary 减去 PredROI_2D_only：

| Split | Delta AUROC | Delta AUPRC | Delta Brier |
|:--|--:|--:|--:|
| Train strict OOF | +0.004015 | +0.010362 | +0.019675 |
| Valid | -0.042289 | -0.028857 | -0.008142 |

在自动 PredROI 路径中，加入 CAVE 对 strict OOF AUROC/AUPRC 有小幅提升，因此 A 仍保留为已冻结 teacher gated primary；但 PredROI_2D_only 也是重要、简洁且表现接近的空间分支。

---

## 13. Outer segmentation 质量

五个 strict SegResNet 分别只评估其未见 outer holdout：

| Fold | Overall Dice | Overall IoU | Empty prediction rate | Mean pred/GT area ratio |
|--:|--:|--:|--:|--:|
| 1 | 0.591044 | 0.477920 | 0 | 1.425760 |
| 2 | 0.615578 | 0.505637 | 0 | 1.469510 |
| 3 | 0.582990 | 0.472039 | 0 | 1.517610 |
| 4 | 0.546186 | 0.431064 | 0 | 1.229060 |
| 5 | 0.587254 | 0.474212 | 0 | 1.627260 |

GTROI_2D_only 高于 PredROI_2D_only，与当前 Dice 仍处于约 0.55–0.62、预测面积总体偏大的现象一致：PredROI 仍可能受概率 Mask 范围和定位精度限制。

这也是为什么后续“在独立无标注 DSA 上预训练 encoder，再严格 fine-tune segmentation”在方法上值得考虑。但该方向尚未训练，不属于当前结果。

---

## 14. 汇报时推荐的准确表述

建议对老师这样说：

> 我们先按 patient-grouped strict five-fold 训练五个 SegResNet。对每个 outer fold，SegResNet 只使用其余 Train folds 的人工 Mask，完全排除该 outer holdout 和 Valid。完整 Pre/Post Mean PNG 输入 encoder 后，实际从最后一个 down stage 提取 [256,96,96] feature map；分别做全局平均池化，以及用人工 GT Mask area-resize 后做归一化 ROI 加权池化。最终形成 [G_pre, ROI_gt_pre, G_post, ROI_gt_post] 1024-D spatial representation。

> 在 adverse outcome strict OOF 中，GTROI_2D_only 得到 AUROC 0.5762、AUPRC 0.2451、Brier 0.1987，是当前已完成七模型中 OOF AUPRC 最高、Brier 最低的结果。将 GTROI 与 CAVE deep 接入固定 teacher gated fusion 后，OOF AUROC 提高到 0.5886，为当前最高 AUROC，但 AUPRC 为 0.2295，低于 GTROI_2D_only。

> 因为 GTROI 依赖人工 Mask，它代表空间 oracle，而不是完全自动部署模型。自动部署主路径仍使用 SegResNet probability Mask 形成 PredROI。GTROI 与 PredROI 的差距提示进一步改善 ROI localization 或 encoder representation 可能有价值。

不建议说：

- “GTROI 是最终临床部署模型。”
- “GTROI+CAVE 在所有指标上最佳。”
- “加入 CAVE 一定提高 GTROI。”
- “GT Mask oracle 证明了模型已经可泛化。”
- “Valid 被用于选择 GTROI 分支。”

---

## 15. 当前模型层级的最终定位

| 地位 | 模型 | 原因 |
|:--|:--|:--|
| Primary deployable teacher model | PredROI-CAVE-Deep Gated Fusion | 自动概率 Mask，不需要 GT；此前按固定 2x2 strict OOF 冻结 |
| Best strict OOF AUPRC oracle | GTROI_2D_only | OOF AUPRC 0.245125，为当前最高 |
| Best strict OOF AUROC oracle | GTROI_CAVE_gated_oracle | OOF AUROC 0.588590，为当前最高 |
| Deployable spatial reference | PredROI_2D_only | 不需要 CAVE 或 GT，OOF AUPRC 0.199404 |
| Historical temporal baseline | CAVE-Deep Logistic | 历史正式 B0 |

---

## 16. 对应结果文件

GTROI_2D_only：

    /root/autodl-tmp/aneurysm/outputs/
    api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only/
      fusion/spatial_only/
        metrics.json
        fold_metrics.csv
        train_oof_predictions.csv
        valid_predictions.csv

GTROI_CAVE_gated_oracle：

    /root/autodl-tmp/aneurysm/outputs/
    api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle/
      fusion/gated_interaction/
        metrics.json
        fold_metrics.csv
        train_oof_predictions.csv
        valid_predictions.csv

Spatial featurebank：

    /root/autodl-tmp/aneurysm/outputs/
    api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/
      seg_features/fold_1..fold_5/
        train.npz
        valid.npz
        train_roi_pool_audit.csv
        valid_roi_pool_audit.csv

Feature extraction 审计：

    /root/autodl-tmp/aneurysm/reports/
    CURRENT_SEGRESNET_FEATURE_EXTRACTION_AUDIT.md

Teacher featurebank package：

    /root/autodl-tmp/
    TEACHER_STRICT_SPATIAL_FEATUREBANKS_20260809.zip

---

## 17. 最终结论

如果“最佳版本”指当前已完成实验中 strict OOF AUPRC 最佳，那么答案是：

    GTROI_2D_only
    OOF AUROC = 0.576197
    OOF AUPRC = 0.245125
    OOF Brier = 0.198718

如果“最佳版本”指 strict OOF AUROC 最佳，那么答案是：

    GTROI_CAVE_gated_oracle
    OOF AUROC = 0.588590
    OOF AUPRC = 0.229501
    OOF Brier = 0.217890

如果“最佳版本”指无需人工 Mask、能够完全自动推理且保持老师门控结构的正式主模型，那么仍然是：

    PredROI-CAVE-Deep Gated Fusion
    OOF AUROC = 0.546053
    OOF AUPRC = 0.209765
    OOF Brier = 0.222638

因此最严谨的总结是：**GTROI_2D_only 是当前表现最好的空间 oracle；GTROI_CAVE gated 是 AUROC 最好的 oracle fusion；PredROI-CAVE-Deep Gated Fusion 是当前冻结的自动推理主模型。**

