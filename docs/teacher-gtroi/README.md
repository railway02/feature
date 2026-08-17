# GTROI and Teacher Gated Code Map

This package contains the exact existing code used for spatial extraction and adverse prediction. No code was rewritten for this handoff.

## Core gated architecture

File: fusion_models.py

- Lines 7-25: FeatureProjection
  - Linear(input_dim,256)
  - LayerNorm(256)
  - GELU
  - Dropout(0.2)

- Lines 28-101: SpatialTemporalFusion
  - gate_2d and gate_t
  - phi_t_to_2d and phi_2d_to_t
  - bidirectional gated updates
  - four-way interaction
  - 1024 -> 512 -> 256 fusion

- Lines 104-210: OutcomeModel
  - creates spatial and temporal projections
  - selects gated_interaction
  - applies main_head Linear(256,1)

## Strict outcome training flow

File: 03_train_fusion.py

- Lines 135-168: DataLoader and model construction
- Lines 171-199: weighted BCE training epoch
- Lines 202-239: eval/no_grad prediction
- Lines 242-315: inner-valid AUPRC epoch selection
- Lines 318-366: fresh refit on full outer development
- Lines 369-521: strict fold loop, fold-k featurebank, holdout OOF and five Valid predictions
- Lines 528-575: metrics and prediction outputs

## Spatial extraction flow

Files:
- data.py: Mean PNG/Mask preprocessing and FeaturePhaseDataset
- segresnet_model.py: SegResNet construction, encode/decode, global_pool and mask_pool
- 02_extract_spatial_features.py: fold checkpoint loading, phase extraction and Pre/Post packing

GT combined definition:

    gt_combined =
    [G_pre, ROI_gt_pre, G_post, ROI_gt_post]
    = 1024-D

## Configs

- api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only.json
- api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle.json

The gated oracle config fixes:
- spatial representation = global_gt_roi
- temporal representation = deep_only
- mode = gated_interaction
- hidden_dim = 256
- fusion_mid_dim = 512
- dropout = 0.2
- batch_size = 128
- seed = 20260818

## Important

The files in this package are the exact code used for the completed runs. The separate input ZIPs contain the fold-specific NPZ arrays. The exported minimal NPZ paths differ from the original project paths, so a recipient should either preserve the original directory layout/config or adapt only the file loader paths without changing the architecture or fold logic.

