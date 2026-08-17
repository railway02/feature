# 2D DSA Spatial V6 expanded strict protocol

This directory is independent from corrected v5. The active protocol is the
frozen expanded strict segmentation protocol. Both backbones are predeclared
confirmatory models; there is no fold-1/fold-4 development contest and no
`PROMOTION_DECISION.json`.

- `segresnet`: corrected MONAI 2D SegResNet, random initialization.
- `deeplabv3plus_resnet50_imagenet`: SMP DeepLabV3+ with an ImageNet-pretrained
  ResNet50 encoder, grayscale-to-RGB repeat, and ImageNet normalization.

Both models use the same 768x768 image/mask preprocessing, synchronized affine
and intensity augmentation, `0.8 Dice + 0.2 weighted BCE`, approximately 14
positive weight, threshold 0.5, and patient-level inner validation. Only model-
specific physical batch size and AdamW weight decay differ as frozen in config.

For each outer fold, the segmentation pool starts from 1,780 image/mask rows
whose patients appear in `metadata/Train.xlsx`; patients in the adverse outer
holdout are removed as a group. `metadata/valid.xlsx` contributes 453 image/
mask rows and is prohibited from segmentation training, pretraining, and epoch
selection. The legal fold counts are 1464/818, 1468/819, 1468/819, 1468/819,
and 1466/819 rows/patients for folds 1–5 respectively.

Each fold selects its epoch on a patient-level inner validation set composed
only of adverse development patients. The 214 segmentation-only rows (194
patients) are inner-train only. The final fold checkpoint is a fresh refit on
every legal row. The four extra adverse series rows inherit their patient's
outer and inner split.

Key outputs:

- `reports/.../FROZEN_TWO_BACKBONE_PROTOCOL.json`
- `reports/.../expanded_strict_preflight/SUCCESS.json`
- `reports/.../expanded_strict_smoke/SUCCESS.json`
- `run_expanded_strict.sh {preflight|smoke|start-full-strict|audit|extract|featurebanks|fusion|report}`

`10_preflight_all2d_segmentation.py` and
`11_train_all2d_segmentation.py` are retained solely for traceability and exit
with `superseded_not_for_strict`. A full 2,233-pair encoder is deployment-only
after strict segmentation, strict outcome OOF, and independent Valid reporting
are complete.

ImageNet weight provenance:

- URL: `https://download.pytorch.org/models/resnet50-19c8e357.pth`
- SHA256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`
- V6 cache: `/root/autodl-tmp/envs/png2d-spatial-v6/torch_cache/hub/checkpoints/resnet50-19c8e357.pth`

The DeepLab decoder map is `[B,256,192,192]`; it is deliberately not projected
or resized to the SegResNet `[B,256,96,96]` latent grid. Later ROI pooling must
adapt masks to each model's native feature-map spatial size.
