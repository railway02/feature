# Teacher Alignment Audit

> **Correction (2026-08-08):** The earlier missing-file conclusion was caused by an invalid patient-only filename template. It is superseded by `SERIES_MAPPING_RUN_READY.md`. The corrected pipeline joins `roi_phase_manifest_eligible.csv` on `series_uid+phase`; all 988 adverse series and all 1,976 required phase mappings are present.


Date: 2026-08-08

Scope: read-only audit of the supplied `api_png2d_segresnet_cave_fusion_v4_teacher_aligned.zip` and the configured existing PNG/Mask and CAVE NPZ inputs. No training, mean-image generation, CAVE extraction, or source-data modification was performed.

## Evidence inspected

- ZIP source: `00_preflight.py`, `01_train_spatial_encoder.py`, `02_extract_spatial_features.py`, `03_train_fusion.py`, `data.py`, `segresnet_model.py`, `fusion_models.py`, configuration and tests.
- CAVE inputs:
  - Train: `.../adverse_prepost_series_task_v3/train_features.npz`
  - Valid: `.../adverse_prepost_series_task_v3/valid_features.npz`
- Existing direct PNG roots: `/root/autodl-tmp/2D/image` and `/root/autodl-tmp/2D/mask`.

## Required answers

| Question | Read-only finding | Alignment status before patch |
|---|---|---|
| Current real SegResNet input | `prepare_pair` directly reads `{patient_id}_{Pre|Post}.png` grayscale Mean PNG, percentile normalizes it, then letterboxes/resizes it to `1 x 768 x 768`. SegResNet is configured as MONAI `spatial_dims=2`, `in_channels=1`, `out_channels=1`. It does not recompute a mean image or use NIfTI/frame mapping. | Aligned |
| Current real GT Mask use | Same-basename mask PNG is the segmentation target for Dice+BCE training. At feature extraction it is additionally used for GT ROI pooling. It is not used as the adverse-outcome label or as a CAVE input. | Aligned, except pooling risk below |
| Current `z2D_raw` | Extraction already saved `global=[G_pre,G_post]`, but fusion selected only `gt_combined` or `pred_combined`: `[G_pre, ROI_pre, G_post, ROI_post]`. Thus the operational default was ROI-enhanced 1024-D rather than pure global. | Needs patch |
| Pure `global_only` support | Feature bank contained a `global` array but no fusion configuration/key path could select it. | Needs patch |
| Tiny-mask vanish risk | GT ROI pooling resized a hard mask with nearest-neighbor to deepest feature-map scale, then used `clamp_min(eps)`. A tiny lesion can become empty after downsampling and produce a silent zero-like ROI feature. No raw-pixel / feature-mass / vanished-count audit existed. | Needs patch |
| `zT_raw` provenance | `load_temporal` loads only existing NPZ key `deep` as `float32`; no CAVE encoder is run. Real shapes: Train `(781, 10240)`, Valid `(207, 10240)`. | Aligned |
| P2D/PT | `FeatureProjection` is exactly `Linear(input_dim,256) -> LayerNorm(256) -> GELU -> Dropout(0.2)` in the supplied configuration. No PCA/scaler is in the main path. | Aligned |
| Bidirectional gate | The code implements `a2D=sigmoid(f2D([z2D,zT]))`, `aT=sigmoid(fT([zT,z2D]))`, then the specified convex exchanges through `phi_T_to_2D` and `phi_2D_to_T`. It does not use `z+a*z`, Transformer, or cross-attention. | Aligned |
| `hmain` dimension | For `hidden_dim=256`, concatenation is `[z2D_hat,zT_hat,product,absdiff]`, hence `(B,1024)`; existing model smoke test asserts this. | Aligned |
| `pilot_single` cross-fit | A pilot SegResNet is refit using every Train image/mask, then used to create Train spatial features. Fusion outcome folds therefore receive representations whose encoder saw their holdout images/masks. Existing metrics flag `pilot_representation_warning=true`; it is not formal representation-level OOF. | Correctly warned, report label needs strengthening |
| UID/patient/record alignment | Manifest task rows are ordered by `task_row` and extraction asserts exact `series_uid` order against each CAVE NPZ. Train has 781 unique patients/UIDs. Valid has 207 unique UIDs but only 206 patients: patient `719585` has two series (`R-C6`, target 1; `L-MCA`, target 0) and shares the patient-level filenames. There is no Train/Valid UID or patient overlap. The supplied NPZ has `source_record_count`, but no per-record identifier is loaded or asserted; strict record-level alignment therefore cannot be claimed. | UID and patient checks partially aligned; record-level provenance unavailable |
| Train/Valid actual counts | Train: 781 rows, 133 positives, 781 unique patients, folds 1--5. Valid: 207 rows, 37 positives, 206 unique patients, 207 unique UIDs. | Counts match configuration |

## Corrected PNG/Mask and series-mapping audit

- The all-2D segmentation inventory contains 2,233 unique image/mask pairs.
- The adverse task contains 988 unique `series_uid` rows.
- Filtering the phase mapping by those task UIDs yields exactly 1,976 rows: one Pre and one Post per series.
- All mapped image/mask paths exist and have identical basenames.
- `496194`, `513318`, and `719585` are correctly resolved with their series suffixes; no patient-level filename fallback is used.
- Patient `719585` has two distinct series and two distinct PNG pairs (`L-MCA` and `R-C6`), so there is no image reuse ambiguity.

## Actual tensor checks

- CAVE `deep`: Train `(781,10240)` and Valid `(207,10240)`, finite check implemented by preflight.
- The intended SegResNet input is `(B,1,768,768)` and feature pooling is GAP over its deepest 2-D map. The isolated runtime now provides MONAI 1.3.2 with PyTorch 2.1.2+cu118. SegResNet initialization has been verified without running a forward pass or pipeline stage.
- Existing fusion smoke test confirms `logit=(B,1)`, `zmain=(B,256)`, gates `(B,256)`, and gated `hmain=(B,1024)` without MONAI.

## What was already fully teacher-aligned

1. Direct use of the prepared 2-D PNG Mean images and same-name GT PNG masks.
2. MONAI SegResNet architectural configuration and segmentation supervision objective.
3. Existing CAVE deep-feature provenance and no CAVE rerun.
4. Separate learned 256-D projections with no PCA.
5. The prescribed convex, bidirectional conditional gate and 1024-to-256 main fusion path.
6. The requested E0--E4 outcome-mode definitions.
7. Patient-grouped outer outcome folds and an available `strict_crossfit` SegResNet mode.

## Existing additions beyond the teacher baseline

- GT ROI and predicted-probability ROI pooling.
- Segmentation augmentation, inner epoch selection/refit, optional external checkpoint loading.
- Pilot strategy and outcome fusion ablations (`concat`, ungated interaction).

These are acceptable ablations, but GT/Pred ROI must be explicit S1/S2 representations rather than hiding the required S0 global-only baseline.

## Minimal patch decision

P0/P1 only:

1. Add explicit `global_only`, `global_gt_roi`, and `global_pred_roi` representation modes, with `global_only` as default.
2. Replace GT hard-nearest deepest-map pooling with area-preserving soft occupancy pooling; emit raw foreground, prepared foreground, deepest-map mass, and vanished-count reports; never silently fall back through an empty GT ROI.
3. Preserve direct PNG/CAVE paths and strict UID-order assertions. Make patient-image reuse and missing PNG/Mask failures visible in preflight reports.
4. Add real SegResNet and fusion shape probes, improve checkpoint-load validation, and mark pilot versus strict representation OOF unambiguously.

Deferred by request: probability-mask embedding, morphology features, multi-task heads, and any additional architecture. No PCA, Transformer, cross-attention, mean recomputation, CAVE rerun, source-data change, or historical-output overwrite is included.
