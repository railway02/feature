# CURRENT STRICT SEGRESNET FEATURE EXTRACTION AUDIT

审计日期：2026-08-09  
审计性质：只读。读取 strict config、五个现有 checkpoint、fold-1 featurebank，并在 model.eval() 和 torch.no_grad() 下用一张真实 Mean PNG forward。没有训练、没有重新提取 featurebank、没有运行 outcome、没有改写 checkpoint 或结果。

## 1. 直接结论

现有 GTROI 1024-D 特征来自 strict fold 对应 SegResNet 的最终 encoder down stage。准确路径是：

    segresnet_model.encode_and_decode(model, x)
      -> segresnet_model.encode(model, x)
      -> model.encode(x)[0]
      -> output of model.down_layers[3]

对真实 768 x 768 Mean PNG，最终实际 fmap shape 为 [1,256,96,96]。它不是 decoder feature，不是任何 up layer 输出，不是 segmentation logits，也不是外加 bottleneck。GTROI 的 1024-D 排列为：

    [G_pre, ROI_gt_pre, G_post, ROI_gt_post]

其中四项均为 256-D。

## 2. 实际 SegResNet architecture

模型由 segresnet_model.build_segresnet() 直接创建 MONAI SegResNet。

| 参数 | 实际值 |
|:--|:--|
| spatial_dims | 2 |
| in_channels | 1 |
| out_channels | 1 |
| init_filters | 32 |
| blocks_down | [1, 2, 2, 4] |
| blocks_up | [1, 1, 1] |
| dropout_prob | None |

实际构造参数为 SegResNet(spatial_dims, in_channels, out_channels, init_filters, blocks_down, blocks_up, dropout_prob)。

## 3. 五个 strict checkpoint 的初始化来源

| fold | checkpoint top-level keys | state tensors | selected refit epochs | checkpoint init_info | SUCCESS init_info | config equals strict config |
|--:|:--|--:|--:|:--|:--|:--:|
| 1 | ['init_info', 'segresnet_config', 'selected_epochs', 'state_dict'] | 83 | 56 | {'used': False} | {'used': False} | True |
| 2 | ['init_info', 'segresnet_config', 'selected_epochs', 'state_dict'] | 83 | 63 | {'used': False} | {'used': False} | True |
| 3 | ['init_info', 'segresnet_config', 'selected_epochs', 'state_dict'] | 83 | 55 | {'used': False} | {'used': False} | True |
| 4 | ['init_info', 'segresnet_config', 'selected_epochs', 'state_dict'] | 83 | 20 | {'used': False} | {'used': False} | True |
| 5 | ['init_info', 'segresnet_config', 'selected_epochs', 'state_dict'] | 83 | 40 | {'used': False} | {'used': False} | True |

结论：五个 strict SegResNet 都从随机初始化训练，不是 pretrained fine-tuning。依据是：

1. strict config 的 spatial.external_checkpoint 为空字符串。
2. 训练函数 fresh_model() 先调用 build_segresnet(cfg)，再调用 maybe_load_external_checkpoint(model,cfg)。
3. 外部路径为空时 maybe_load_external_checkpoint 立即返回 used=false，不调用 torch.load 或 model.load_state_dict。
4. 五个 SUCCESS 记录的 external_initialization 都是 used=false。
5. 五个最终 model.pt 的 init_info 也都是 used=false。
6. 每次 fresh construction 前调用 set_seed，因此是可复现的伪随机默认 MONAI/PyTorch 初始化，随后用各 fold development 数据训练。

Feature extraction 阶段会 load_state_dict(final model.pt state_dict, strict=true)，但这是加载该 fold 已训练好的本地 checkpoint 做 eval，不是 pretrained initialization。

## 4. fmap 的准确代码路径

02_extract_spatial_features.py 的 extract_phase_features() 执行：

    fmap, logits = encode_and_decode(model, x)
    zg = global_pool(fmap)

segresnet_model.encode_and_decode() 内部执行：

    fmap, down = encode(model, x)
    logits = model.decode(fmap, list(reversed(down)))
    return fmap, logits

encode() 优先调用 MONAI model.encode(x)。对当前模型，model.encode(x) 返回的第一个对象是完成 convInit 和全部四个 down_layers 后的 x。因此：

    fmap = model.encode(x)[0]
         = output of model.down_layers[3]
         = final encoder down-stage feature map

decode 在之后利用 fmap 和 reversed down skip list 生成 logits。没有 decoder tensor 被用于 pooling。

## 5. 真实 Mean PNG forward

| 项目 | 值 |
|:--|:--|
| checkpoint | strict segmentation/fold_1/model.pt |
| device | cuda:0 |
| series_uid | Train__232848__main__b28b7af693 |
| phase | Pre |
| Mean PNG | /root/autodl-tmp/2D/image/232848_Pre.png |
| GT Mask | /root/autodl-tmp/2D/mask/232848_Pre.png |

预处理严格复用 data.prepare_pair()：灰度读取，图像自身有限像素 1st/99th percentile 归一化，Mask 二值化，保宽高比 letterbox 到 768x768。Feature extraction 不使用训练期 brightness/contrast/noise augmentation。

| tensor 或 module | 实际 observed shape |
|:--|:--|
| input x | [1, 1, 768, 768] |
| model.convInit | [1, 32, 768, 768] |
| model.down_layers[0] | [1, 32, 768, 768] |
| model.down_layers[1] | [1, 64, 384, 384] |
| model.down_layers[2] | [1, 128, 192, 192] |
| model.down_layers[3] | [1, 256, 96, 96] |
| down list returned by model.encode | [[1, 32, 768, 768], [1, 64, 384, 384], [1, 128, 192, 192], [1, 256, 96, 96]] |
| final fmap from encode | [1, 256, 96, 96] |
| fmap returned by encode_and_decode | [1, 256, 96, 96] |
| decoder logits | [1, 1, 768, 768] |
| sigmoid probability Mask | [1, 1, 768, 768] |
| global pooled G | [1, 256] |
| predicted ROI pooled | [1, 256] |
| GT ROI pooled | [1, 256] |
| predicted ROI mass | [1, 1] |
| GT ROI mass | [1, 1] |

输入是 [1,1,768,768]，encoder 实际将分辨率经过 stages 变为 768、384、192、96，最终 fmap 为 [1,256,96,96]；decoder logits 恢复到 [1,1,768,768]。

## 6. G、ROI_pred、ROI_gt 的实际计算

令 F=fmap，其实际 shape 是 [B,256,96,96]。

### 6.1 G

实现是 global_pool(fmap) = adaptive average pool 到 1x1 后 flatten：

    G[b,c] = mean over h,w of F[b,c,h,w]

G batch shape 是 [B,256]，单相位 G 是 [256]。

### 6.2 ROI_pred

    pred_prob = sigmoid(logits / 1.0)                     # [B,1,768,768]
    weight = bilinear interpolate(pred_prob, size=(96,96))
    numerator = sum over h,w of F * weight                # [B,256]
    mass = sum over h,w of weight                          # [B,1]
    ROI_pred = numerator / clamp_min(mass, 1e-6)           # [B,256]

没有 hard threshold。每个 96x96 feature-map 位置按预测概率连续加权。

### 6.3 ROI_gt

    weight = area interpolate(gt_mask, size=(96,96))
    numerator = sum over h,w of F * weight                 # [B,256]
    mass = sum over h,w of weight                          # [B,1]
    assert mass > 1e-6
    ROI_gt = numerator / clamp_min(mass, 1e-6)             # [B,256]

GT Mask 使用 area resize 从 768x768 到 96x96。若 GT ROI 在 feature-map 尺度消失，现有 extractor 会报错；strict featurebank 已完成，表示此检查通过。

## 7. Pre/Post 和 gt_combined 的最终 shape

pack_series() 的严格顺序：

    G_pre       = zg[pre]       -> [256]
    ROI_gt_pre  = zgt[pre]      -> [256]
    G_post      = zg[post]      -> [256]
    ROI_gt_post = zgt[post]     -> [256]
    gt_combined = concat([G_pre, ROI_gt_pre, G_post, ROI_gt_post]) -> [1024]

| 项目 | 单个 phase 或 series shape | fold-1 Train featurebank full-array shape |
|:--|:--|:--|
| G_pre | [256] | global [781, 512]; each row [G_pre,G_post] |
| ROI_gt_pre | [256] | gt_roi [781, 512]; each row [ROI_gt_pre,ROI_gt_post] |
| G_post | [256] | included in global row [512] |
| ROI_gt_post | [256] | included in gt_roi row [512] |
| gt_combined | [1024] | [781, 1024] |

fold-1 train.npz 单行 shape 验证：{'global': [512], 'gt_roi': [512], 'pred_roi': [512], 'gt_combined': [1024], 'pred_combined': [1024]}。

## 8. 现有 featurebank arrays 的实际 shape

| NPZ key | fold-1 Train shape | 含义 |
|:--|:--|:--|
| global | [781, 512] | [G_pre,G_post] |
| gt_roi | [781, 512] | [ROI_gt_pre,ROI_gt_post] |
| pred_roi | [781, 512] | [ROI_pred_pre,ROI_pred_post] |
| gt_combined | [781, 1024] | [G_pre,ROI_gt_pre,G_post,ROI_gt_post] |
| pred_combined | [781, 1024] | [G_pre,ROI_pred_pre,G_post,ROI_pred_post] |
| series_uid | [781] | feature row primary key |
| patient_id | [781] | patient grouping identity |
| target | [781] | adverse outcome target |

## 9. 最终回答

1. 实际 architecture 是 2D SegResNet，1 input channel，1 output channel，init_filters=32，blocks_down=[1,2,2,4]，blocks_up=[1,1,1]。
2. 五个 strict checkpoint 都从随机初始化开始；代码、config、checkpoint init_info 和 SUCCESS 记录均排除 pretrained/external weight load。
3. fmap 精确来自 model.encode(x)[0]，也就是 model.down_layers[3] 输出。
4. 真实 Mean PNG forward 的 fmap 是 [1,256,96,96]，logits 是 [1,1,768,768]。
5. G 是 fmap 全局平均池化；ROI_pred 是 logits 概率图 bilinear resize 后的加权池化；ROI_gt 是原始 GT Mask area resize 后的加权池化。
6. G_pre、ROI_gt_pre、G_post、ROI_gt_post 各为 [256]；gt_combined 为 [1024]，顺序固定为 [G_pre,ROI_gt_pre,G_post,ROI_gt_post]。

## 10. 证据文件

- Model architecture and encode/decode helpers: /root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/segresnet_model.py
- SegResNet training initialization: /root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/01_train_spatial_encoder.py
- Spatial extraction and packing: /root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/02_extract_spatial_features.py
- Strict config: /root/autodl-tmp/aneurysm/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json
- Strict checkpoints: /root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/segmentation/fold_1..fold_5/model.pt
- Strict featurebank audited: /root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/seg_features/fold_1/train.npz
