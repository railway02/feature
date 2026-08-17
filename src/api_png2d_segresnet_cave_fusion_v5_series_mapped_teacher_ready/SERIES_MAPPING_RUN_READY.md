# Series-Mapped Teacher Pipeline: Ready, Not Executed

Date: 2026-08-08

## Corrected data contract

The adverse task is keyed by `series_uid`, not by `patient_id` alone.
`roi_phase_manifest_eligible.csv` is the only source of the task-specific
`series_uid + phase -> Mean PNG + GT Mask PNG` mapping.

Verified task coverage:

- 988 adverse series in total.
- Train: 781 series / 1,562 phase rows.
- Valid: 207 series / 414 phase rows.
- Every task series has exactly one Pre and one Post mapping.
- All 1,976 task image/mask paths exist and use identical basenames.
- Examples include `496194L_Pre.png`, `513318Acom_Post.png`, and the distinct
  `719585L-MCA_*` / `719585R-C6_*` pairs.

Patient ID is used for grouping and leakage checks. Series UID is used for
adverse feature alignment and prediction rows.

## Two spatial-learning protocols

### Pilot

`pilot_single` trains the segmentation learner with all 2,233 phase-level
image/mask pairs frozen in `png_annotation_inventory.csv`. Its inner split is
patient-grouped and does not use the adverse label. Spatial feature extraction
then selects only the 988 adverse series through the phase mapping manifest.

Because the all-2D inventory includes Valid image/mask pairs, pilot Valid
representations are annotation-informed. Pilot metrics must not be presented
as formal external/generalization results.

### Formal

`strict_crossfit` ignores the all-2D pilot population for representation
training. For outer fold k, it trains SegResNet only on task Train folds other
than k. Holdout fold image/mask/outcome and all Valid image/mask/outcome are
excluded from that fold's representation learner.

## Environment isolation

The runtime is isolated at:

`/root/autodl-tmp/envs/segresnet-cave-teacher-v5`

It reuses the existing CUDA PyTorch installation and installs MONAI and the
small fusion/report dependencies only inside the new environment. The existing
`/root/autodl-tmp/envs/aneurysm-ml` environment is unchanged.

## Execution status

No preflight, SegResNet training, feature extraction, fusion training, or
summary stage has been executed for this corrected pipeline. Run preflight
first when execution is authorized.
