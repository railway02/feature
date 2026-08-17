# Strict SegResNet—CAVE main fusion interface

Version: `dsa_2d_cave_main_fusion_v6_strict`. This package implements teacher scheme IV.2 only. It does not retrain segmentation, depend on DeepLabV3+, use 3D/TabPFN, use GTROI, or concatenate CAVE's engineered scalar features.

## Inputs and alignment

The immutable upstream sources are the unified SegResNet featurebanks and CAVE `deep` NPZs. There are exactly 781 Train and 207 Valid rows. Alignment is fail-closed: exact `series_uid`, `patient_id`, target, and Train fold equality are asserted before training. Train uses CAVE's `fold`; it must equal SegResNet's `outer_fold`.

`z_2d_raw` is **only** `pred_combined_by_fold[:, k-1, :]`, `float32 [N,1024]`; its order is `[G_pre, PredROI_pre, G_post, PredROI_post]`. `z_time_raw` is CAVE `deep`, `float32 [N,10240]`, `[CAVE_pre 5120, CAVE_post 5120]`. GTROI, masks, target fields, scalar features and any latent fold averaging are prohibited in the model input.

For fusion outer fold `k`, development is outcome folds `!= k`, holdout is outcome fold `k`, and both Train rows and Valid use the SegResNet `k` coordinate view. The development/holdout patient sets must be disjoint. Valid never enters selection. The inner split is patient-level and stratified by patient maximum target. After inner AUPRC selects an epoch under min/max/patience bounds, a newly initialized model is trained on all outer development rows for precisely that epoch count.

## Network output contract

The network applies independently `Linear → LayerNorm → GELU → Dropout(0.2)` to raw 2D and time inputs, yielding `z_2d`, `z_time` (`[B,256]`). It uses sigmoid bidirectional conditional gates, convex candidate mixing, and `h_main=[z_2d_interacted,z_time_interacted,product,abs_difference]` (`[B,1024]`), followed by `1024→512→256` fusion and a linear head. Returned `float32` fields are:

`z_2d`, `z_time`, `spatial_gate`, `temporal_gate`, `z_2d_interacted`, `z_time_interacted` (`[B,256]`); `h_main` (`[B,1024]`); `z_main` (`[B,256]`); and `main_logit`, `main_prob` (`[B,1]`). Gates and probabilities are in `[0,1]`.

## Outputs

`raw_modalities/cave_{train,valid}_z_time_raw.npz` exclude target labels. SegResNet raw banks are referenced rather than copied and are SHA256-recorded. `alignment/` contains row-level CSVs and `alignment_audit.json`.

`train_oof_main_outputs.npz` contains only OOF Train `z_main`, logits, probabilities, and gates, plus identifiers and source fold. It is the authoritative artifact for strict main-path evaluation.

`train_main_outputs_by_fold.npz` contains frozen outputs from every fusion fold for every Train row: `z_main_by_fold [781,5,256]`, logits/probabilities `[781,5,1]`, and gates `[781,5,256]`. This file is for fold-specific downstream modelling. Downstream outer fold `k` must read `[:,k-1,:]` for both its development and holdout rows so that all rows within that downstream model use the same fusion latent coordinate system. Selecting each row's `outer_fold` column reconstructs the strict OOF artifact (verified numerically); the other columns are not OOF predictions and must not be used to report main-path Train performance.

`valid_main_outputs_by_fold.npz` retains all five `[207,5,*]` fold-specific representations. Valid metrics average probabilities only; no latent or `z_main` averaging is allowed. Downstream outer fold `k` must use Train and Valid column `k-1`; it must never average `z_main` across folds.

`downstream_loader.py` is the recommended fail-closed reader. It returns the fold-specific Train/Valid coordinate views plus Train development/holdout masks, rejects label fields, checks identifiers and fold labels, and verifies that the fold holdout equals the strict OOF output. These banks guarantee correct outer-fold routing. They do not by themselves make a newly introduced downstream inner-validation search leakage-free, because the frozen fold-k main model was refit on all fold-k outer development rows. A downstream stage that searches new epochs/hyperparameters must pre-specify them or implement a suitably nested upstream/downstream protocol.

## Commands

```bash
PY=/root/autodl-tmp/envs/aneurysm-ml/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_main_fusion_v6_strict
$PY $CODE/package_raw_modalities.py
$PY $CODE/train_main_fusion.py --smoke-only --device cuda:0
$PY $CODE/train_main_fusion.py --device cuda:0
$PY $CODE/export_train_by_fold.py --device cuda:0
$PY $CODE/downstream_loader.py
$PY $CODE/build_delivery_manifest.py
$PY $CODE/audit_outputs.py
```
