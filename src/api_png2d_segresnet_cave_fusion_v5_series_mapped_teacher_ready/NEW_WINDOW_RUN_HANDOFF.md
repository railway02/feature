# New Window Run Handoff: SegResNet + CAVE Teacher-Aligned Adverse Pipeline

Date: 2026-08-08

## New-window objective

Continue from the prepared state and execute the corrected v5 pipeline to
completion. Run the pilot protocol first, then the formal strict-crossfit
protocol. Stay with long-running commands until they finish; do not launch a
job and immediately return while it is still running.

The corrected v5 pipeline has **not** yet run preflight, training, extraction,
fusion, or summarization. Static compilation, environment imports, MONAI
SegResNet initialization, and in-memory mapping-contract tests have passed.

## Read these files first

1. `/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/SERIES_MAPPING_RUN_READY.md`
2. `/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/READINESS_STATUS.json`
3. This handoff document.

## Frozen runtime and entry points

```text
Python:
/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python

Code:
/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready

Runner:
/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/run_pipeline.sh

Pilot config:
/root/autodl-tmp/aneurysm/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot.json

Strict config:
/root/autodl-tmp/aneurysm/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json
```

Runtime versions are recorded in `RUNTIME_ENVIRONMENT.json`. The existing
`/root/autodl-tmp/envs/aneurysm-ml` environment was not modified and must not
be modified for this run.

## Correct data contract

Do not derive filenames from `patient_id`.

- Segmentation pilot population: all 2,233 rows in
  `png_annotation_inventory.csv`.
- Adverse task: 988 `series_uid` rows, consisting of Train 781 and Valid 207.
- Task image mapping: join `roi_phase_manifest_eligible.csv` on
  `series_uid + phase`.
- Every task series has exactly one Pre and one Post mapping: 1,976 phase rows.
- Patient ID is only for grouping/leakage checks. Series UID is the adverse
  sample key.

Examples such as `496194L_Pre.png`, `513318Acom_Post.png`, and the distinct
`719585L-MCA_*` / `719585R-C6_*` pairs are expected and valid.

The four frozen upstream files and expected SHA-256 values are already in both
configs. A mismatch must stop the run.

## Architecture that must not change

```text
Pre Mean  -> trained 2D SegResNet encoder -> deepest map -> GAP -> G_pre
Post Mean -> trained 2D SegResNet encoder -> deepest map -> GAP -> G_post

z2D_raw = [G_pre,G_post]                         # global_only, normally 512-D
z2D = Linear(raw,256) -> LayerNorm -> GELU -> Dropout(0.2)

zT_raw = existing CAVE deep                     # 10240-D
zT = Linear(10240,256) -> LayerNorm -> GELU -> Dropout(0.2)

a2D = sigmoid(f2D([z2D,zT]))
aT  = sigmoid(fT([zT,z2D]))

z2D_hat = (1-a2D)*z2D + a2D*phi_T_to_2D(zT)
zT_hat  = (1-aT)*zT  + aT*phi_2D_to_T(z2D)

hmain = [z2D_hat,zT_hat,z2D_hat*zT_hat,abs(z2D_hat-zT_hat)]  # 1024-D
hmain -> fusion -> zmain 256 -> adverse binary logit
```

Do not add PCA, Transformer, attention, morphology, multitask heads, mean-image
recalculation, CAVE reruns, or silent case dropping.

This execution is the S0 `global_only` main experiment only. Run E0-E4:

```text
cave_only
spatial_only
concat
interaction
gated_interaction
```

Do not change to `global_gt_roi` or `global_pred_roi` in this run. The current
fusion result directories are mode-based and an in-place representation change
would overwrite S0 results. ROI ablations need a separate follow-up output
layout.

## Phase A: pilot end-to-end

The runner defaults to the pilot config. Execute stepwise so failures and
resumes are visible:

```bash
RUN=/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/run_pipeline.sh

bash "$RUN" preflight
bash "$RUN" train-spatial all
bash "$RUN" extract all
bash "$RUN" fusion all
bash "$RUN" summarize
```

Pilot semantics:

- SegResNet trains on all 2,233 image/mask pairs.
- Its inner split is patient-grouped and uses no adverse label.
- Feature extraction selects only the 988 adverse series through the mapping.
- Because the all-2D inventory includes Valid image/mask, pilot Valid
  representation is annotation-informed. Never present pilot metrics as formal
  external/generalization results.

Pilot output root:

```text
/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot
```

Pilot report root:

```text
/root/autodl-tmp/aneurysm/reports/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot
```

Pilot acceptance checks:

```text
00_preflight.json                         status=PASS
segmentation/pilot/.SUCCESS.json
seg_features/pilot/.SUCCESS.json
fusion/{each E0-E4}/metrics.json
reports/.../04_summary_metrics.csv
reports/.../04_summary.json
reports/.../04_summary.md
```

Confirm metrics label pilot results as:

```text
representation_oof_status = pilot_not_representation_crossfit
valid_representation_status = pilot_all_2d_includes_valid_image_mask
```

## Phase B: formal strict-crossfit end-to-end

Start only after Phase A completes and its artifacts pass the acceptance
checks. Use the strict config explicitly:

```bash
RUN=/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/run_pipeline.sh
STRICT=/root/autodl-tmp/aneurysm/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json

CONFIG="$STRICT" bash "$RUN" preflight
CONFIG="$STRICT" bash "$RUN" train-spatial all
CONFIG="$STRICT" bash "$RUN" extract all
CONFIG="$STRICT" bash "$RUN" fusion all
CONFIG="$STRICT" bash "$RUN" summarize
```

Strict semantics for outer fold k:

- Train fold-k SegResNet only on other Train outcome folds.
- Holdout-k image, mask, and outcome do not enter that representation learner.
- Valid image, mask, and outcome do not enter representation training.
- Outcome fusion trains on other folds and predicts holdout-k.
- Valid prediction is the mean of the five fold models.

Strict output root:

```text
/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict
```

Strict acceptance checks:

```text
00_preflight.json                         status=PASS
segmentation/fold_{1..5}/.SUCCESS.json
seg_features/fold_{1..5}/.SUCCESS.json
fusion/{each E0-E4}/metrics.json
fusion/{each E0-E4}/train_oof_predictions.csv
fusion/{each E0-E4}/valid_predictions.csv
reports/.../04_summary_metrics.csv
reports/.../04_summary.json
reports/.../04_summary.md
```

Confirm formal metrics label results as:

```text
representation_oof_status = strict_crossfit
valid_representation_status = strict_train_only_representation
```

## Long-run and resume rules

- Keep the active command/session attached and monitor it until completion.
- Do not start duplicate pilot or fold jobs against the same output root.
- `train-spatial` skips a model only when its `.SUCCESS.json` exists.
- After interruption, inspect the exact stage/fold artifacts and resume that
  stage; do not delete successful folds or reset output directories.
- Do not treat a partially written history/model as success without the
  corresponding `.SUCCESS.json`.
- Never switch pilot and strict configs while reusing an output root.
- Do not use Valid results for epoch selection, hyperparameter changes, or
  mode selection.

## Failure handling

Preflight failure:

- Read the generated JSON/CSV audit first.
- Do not bypass hash, UID, patient, phase, basename, shape, foreground, or fold
  checks.
- Do not silently drop a series.

CUDA OOM:

- First reduce only `segresnet.batch_size` from 4 to 2, then 1 if needed.
- Reduce `feature_extraction.batch_size` from 8 to 4/2 if extraction OOMs.
- Keep input size, model architecture, projection, fusion formula, and data
  population unchanged.
- Record any batch-size-only config change in the final run report.

Dependency issue:

- Use only the isolated v5 environment.
- Do not install into or alter `aneurysm-ml`.
- Do not upgrade PyTorch/CUDA during the run.

## Final handoff expected from the new window

Report:

1. Pilot and strict completion status.
2. Exact preflight counts and whether all input locks matched.
3. SegResNet selected epochs and segmentation Dice per pilot/fold.
4. E0-E4 Train OOF and Valid AUROC/AUPRC/Brier tables.
5. Gated-interaction gate statistics.
6. Any batch-size-only operational change.
7. Paths to all summary reports and checkpoints.
8. A clear distinction between pilot and formal strict results.

Do not claim completion while a required command is still running.
