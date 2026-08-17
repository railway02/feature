# 2D Spatial Feature Interface Specification

Version: `dsa_2d_spatial_v1_segresnet_strict_soft_predroi`  
Current formal encoder: frozen strict MONAI 2D SegResNet  
Downstream field name: `z_2d_raw`

## 1. Scope

This package is the formal 2-D input interface for multimodal fusion. It does
not retrain segmentation, select a model, train an outcome classifier, or
project the feature to 256 dimensions. DeepLabV3+ is intentionally not a
dependency of this version. A future encoder may implement the same semantic
field `z_2d_raw` without changing the fusion model API.

## 2. SegResNet input and frozen preprocessing

For each adverse-outcome case, the interface reads the complete Pre and Post
Mean PNGs using OpenCV grayscale mode. It does not crop by GT mask.

Each phase image is processed exactly as the frozen strict segmentation run:

1. per-image p1/p99 percentile normalization;
2. clip to `[0,1]`;
3. preserve aspect ratio and resize/letterbox to `768×768`;
4. image downsampling uses `INTER_AREA`, upsampling uses `INTER_LINEAR`;
5. no training augmentation during feature export.

The tensor presented to SegResNet is `[B,1,768,768]`.

## 3. Feature source and actual tensor shapes

The model is MONAI 2D SegResNet with `init_filters=32`,
`blocks_down=(1,2,2,4)` and `blocks_up=(1,1,1)`. The feature is returned by
the frozen wrapper's `model.encode(x)` final output, i.e. the output of the
fourth/down-most encoder level before decoder reconstruction.

Actual smoke-tested shapes:

```text
input                         [B,   1, 768, 768]
final encoder feature map     [B, 256,  96,  96]
segmentation logits           [B,   1, 768, 768]
continuous probability map    [B,   1, 768, 768]
Global                        [B, 256]
PredROI                       [B, 256]
case-level z_2d_raw           [B,1024]
```

## 4. Global and PredROI definitions

For feature map `F ∈ R^(256×96×96)`:

```text
Global_c = mean_(h,w) F_(c,h,w)
```

PredROI uses the segmentation probability continuously:

```text
P_768 = sigmoid(segmentation_logits)
P_96  = bilinear_resize(P_768, 96×96)

PredROI_c = Σ_(h,w) F_(c,h,w) P_96_(h,w)
            --------------------------------
                   Σ_(h,w) P_96_(h,w)
```

PredROI therefore uses a soft, normalized probability-weighted average. The
formal interface does **not** use a GT mask, GTROI, or a hard threshold such
as `probability >= 0.5`.

GTROI remains an internal oracle-analysis concept and is absent from every
public NPZ file.

## 5. Fixed Pre/Post concatenation

The component order is immutable:

```text
z_2d_raw = [G_pre, PredROI_pre, G_post, PredROI_post]
```

Slices:

| Component | Slice | Dimension |
|---|---:|---:|
| `G_pre` | `[0:256]` | 256 |
| `PredROI_pre` | `[256:512]` | 256 |
| `G_post` | `[512:768]` | 256 |
| `PredROI_post` | `[768:1024]` | 256 |

The resulting dtype and shape are `float32 [N,1024]`.

## 6. Identifier alignment

The authoritative case order comes from the frozen `case_manifest.csv`.
Public rows contain:

```text
series_uid
patient_id
split
outer_fold
source_model_fold
model_family
feature_version
z_2d_raw
```

No outcome target is loaded or exported. `series_uid` is the primary row key;
`patient_id` is retained for patient-level grouping and audits. The exporter
checks exact row-by-row identifier equality after writing each file.

## 7. Strict Train OOF routing

Train contains 781 cases. A Train case is exported only by its own outer-fold
checkpoint:

```text
outer_fold=1 → frozen SegResNet fold_1
outer_fold=2 → frozen SegResNet fold_2
outer_fold=3 → frozen SegResNet fold_3
outer_fold=4 → frozen SegResNet fold_4
outer_fold=5 → frozen SegResNet fold_5
```

For every fold, the exporter reads the frozen segmentation legal-manifest and
asserts that the OOF patients have zero overlap with the checkpoint's legal
training patients. It also verifies checkpoint fold metadata and SHA256.

There is no five-model averaging and no cross-fold feature substitution for
Train. In `train_oof_z_2d_raw.npz`, `source_model_fold` must equal
`outer_fold` for every row.

## 8. Independent Valid routing

Valid contains 207 outcome series / 206 patients and has no Train outer-fold
assignment. Public Valid rows therefore use:

```text
outer_fold = 0
source_model_fold = k
```

Five separate files are retained:

```text
valid_fold_1_z_2d_raw.npz
valid_fold_2_z_2d_raw.npz
valid_fold_3_z_2d_raw.npz
valid_fold_4_z_2d_raw.npz
valid_fold_5_z_2d_raw.npz
```

Fusion fold `k` must read `valid_fold_k_z_2d_raw.npz`. The aggregate
`valid_z_2d_raw_by_fold.npz` is provided only as a convenient indexed view
with shape `[207,5,1024]`; it must not be averaged before fold-specific
fusion inference.

Valid is not used for segmentation selection, feature selection, or fusion
epoch selection.

## 9. Fusion-side loading

Train OOF:

```python
import numpy as np

with np.load("train_oof_z_2d_raw.npz", allow_pickle=False) as data:
    series_uid = data["series_uid"].astype(str)
    patient_id = data["patient_id"].astype(str)
    outer_fold = data["outer_fold"].astype(np.int64)
    z_2d_raw = data["z_2d_raw"].astype(np.float32)

assert z_2d_raw.shape == (781, 1024)
assert np.isfinite(z_2d_raw).all()
```

Valid for fusion fold `k`:

```python
with np.load(f"valid_fold_{k}_z_2d_raw.npz", allow_pickle=False) as data:
    assert np.all(data["source_model_fold"] == k)
    z_2d_raw_valid = data["z_2d_raw"].astype(np.float32)

assert z_2d_raw_valid.shape == (207, 1024)
```

The fusion model, not the 2-D exporter, performs:

```text
Linear(1024,256)
LayerNorm(256)
GELU
Dropout(0.2)
```

The 2-D package must not pre-project or normalize away the raw 1024-D
component semantics.

## 10. Public package and integrity files

The exporter writes:

```text
train_oof_z_2d_raw.npz
valid_fold_1_z_2d_raw.npz ... valid_fold_5_z_2d_raw.npz
valid_z_2d_raw_by_fold.npz
train_oof_manifest.csv
valid_by_fold_manifest.csv
fold_routing_audit.csv
interface_metadata.json
SMOKE_TEST.json
SHA256SUMS.txt
SUCCESS.json
```

`SMOKE_TEST.json` verifies dimensions, dtype, finite values, identifiers,
fold routing, zero patient leakage, absence of GTROI/mask/target fields, and
absence of latent fold averaging.

## 11. Reproduction command

```bash
/root/autodl-tmp/envs/png2d-spatial-v6/bin/python \
  /root/autodl-tmp/aneurysm/code/api_png2d_spatial_backbones_v6_strict/export_2d_interface.py \
  --config /root/autodl-tmp/aneurysm/configs/api_png2d_spatial_backbones_v6_strict.json \
  --device cuda:0
```

This command performs frozen inference only. It does not retrain SegResNet,
run DeepLabV3+, modify strict splits, or train an outcome fusion model.
