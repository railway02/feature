#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/aneurysm}
CODE=${CODE:-$ROOT/code/api_adverse_lesion_cave_v1}
CONFIG=${CONFIG:-$ROOT/configs/api_adverse_lesion_cave_v1.json}
CAVE_PY=${CAVE_PY:-/root/autodl-tmp/envs/cave-dsa/bin/python}
PRED_PY=${PRED_PY:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}
REPORTS=${REPORTS:-$ROOT/reports/api_adverse_lesion_cave_v1}
OUTPUTS=${OUTPUTS:-$ROOT/outputs/api_adverse_lesion_cave_v1}
LOGS=${LOGS:-$ROOT/logs}

export ROOT
export RAW_ROOT=${RAW_ROOT:-/root/autodl-tmp/tiantanDSA}
export UPDATED_ROOT=${UPDATED_ROOT:-$ROOT/staging/updated_10_cases}
export CAVE_PY PRED_PY
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export ADVERSE_LESION_CONFIG=$CONFIG

mkdir -p "$REPORTS" "$OUTPUTS" "$LOGS"

run_step() {
  local name=$1
  shift
  local start end
  start=$(date +%s)
  echo "[START] $name $(date -Is)"
  "$@"
  end=$(date +%s)
  echo "[PASS] $name elapsed_seconds=$((end-start)) $(date -Is)"
}

run_step_resume() {
  local name=$1
  local marker=$2
  local required=$3
  shift 3
  local guard=("$PRED_PY" "$CODE/resume_guard.py" --config "$CONFIG" --marker "$marker")
  local item
  if [[ -n "$required" ]]; then
    IFS='|' read -r -a required_items <<< "$required"
    for item in "${required_items[@]}"; do
      if [[ "$item" == sha:* ]]; then
        guard+=(--input-file "${item#sha:}")
      else
        guard+=(--required "$item")
      fi
    done
  fi
  if "${guard[@]}"; then
    echo "[SKIP] $name current marker and outputs match"
  else
    run_step "$name" "$@"
  fi
}

usage() {
  printf '%s\n' \
    "Usage: bash run_pipeline.sh <command>" \
    "Commands:" \
    "  check scan-assets build-manifest align build-seg-data" \
    "  smoke-seg train-seg-oof infer-masks build-roi extract-mask-features" \
    "  smoke-cave extract-cave <branch> <split> compact-cave <branch> <split> extract-cave-train extract-cave-valid build-tables" \
    "  build-adverse-tasks train-models ablations summarize full-auto status"
}

cmd=${1:-}
case "$cmd" in
  check)
    "$PRED_PY" -m py_compile "$CODE"/*.py
    "$CAVE_PY" "$CODE/test_synthetic.py"
    "$PRED_PY" "$CODE/00_static_check.py" --config "$CONFIG"
    ;;
  scan-assets) "$CAVE_PY" "$CODE/01_scan_current_assets.py" --config "$CONFIG" ;;
  build-manifest) "$CAVE_PY" "$CODE/02_build_authoritative_roi_manifest.py" --config "$CONFIG" ;;
  align) "$CAVE_PY" "$CODE/03_infer_reference_rule_and_alignment.py" --config "$CONFIG" ;;
  build-seg-data) "$CAVE_PY" "$CODE/04_build_segmentation_dataset.py" --config "$CONFIG" ;;
  smoke-seg)
    "$PRED_PY" "$CODE/05_train_segmentation_oof.py" --config "$CONFIG" --max-samples 10 --epochs 1 --output-root "$OUTPUTS/smoke_segmentation"
    ;;
  train-seg-oof) "$PRED_PY" "$CODE/05_train_segmentation_oof.py" --config "$CONFIG" ;;
  infer-masks) "$PRED_PY" "$CODE/06_infer_train_oof_valid_masks.py" --config "$CONFIG" ;;
  build-roi) "$CAVE_PY" "$CODE/07_build_roi_manifests.py" --config "$CONFIG" ;;
  extract-mask-features) "$PRED_PY" "$CODE/08_extract_mask_morphology.py" --config "$CONFIG" ;;
  smoke-cave)
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch pred --split Train --max-series 5
    ;;
  extract-cave) "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch "${2:?branch required}" --split "${3:?split required}" ;;
  compact-cave) "$PRED_PY" "$CODE/compact_cave_featurebank.py" --config "$CONFIG" --branch "${2:?branch required}" --split "${3:?split required}" ;;
  extract-cave-train)
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch pred --split Train
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch gt --split Train
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch all_nonzero --split Train
    ;;
  extract-cave-valid)
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch pred --split Valid
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch all_nonzero --split Valid
    "$CAVE_PY" "$CODE/09_extract_roi_cave_featurebank.py" --config "$CONFIG" --branch gt --split Valid
    ;;
  build-tables) "$PRED_PY" "$CODE/10_build_roi_cave_tables.py" --config "$CONFIG" ;;
  build-adverse-tasks) "$PRED_PY" "$CODE/11_build_adverse_tasks.py" --config "$CONFIG" ;;
  train-models) "$PRED_PY" "$CODE/12_train_adverse_models_fixed.py" --config "$CONFIG" --variant pred_roi ;;
  ablations) "$PRED_PY" "$CODE/13_run_ablations.py" --config "$CONFIG" ;;
  summarize) "$PRED_PY" "$CODE/14_summarize_pipeline.py" --config "$CONFIG" ;;
  full-auto)
    run_step_resume static_check "$REPORTS/.STATIC_SUCCESS" "$REPORTS/static_check.json" bash "$0" check
    run_step_resume scan_current_assets "$REPORTS/.ASSET_SCAN_SUCCESS" "$ROOT/manifests/api_adverse_lesion_cave_v1/physical_asset_inventory.csv" bash "$0" scan-assets
    run_step_resume build_manifest "$REPORTS/.MANIFEST_CANDIDATES_SUCCESS" "$ROOT/manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_candidates.csv" bash "$0" build-manifest
    run_step_resume infer_alignment "$REPORTS/.ASSET_SUCCESS" "$ROOT/manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_primary.csv|sha:source_manifest_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_candidates.csv" bash "$0" align
    run_step_resume build_segmentation_dataset "$REPORTS/.SEG_DATA_SUCCESS" "$ROOT/manifests/api_adverse_lesion_cave_v1/segmentation_dataset_index.csv|sha:source_manifest_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_primary.csv" bash "$0" build-seg-data
    run_step segmentation_smoke bash "$0" smoke-seg
    run_step_resume train_segmentation_oof "$REPORTS/.SEGMENTATION_OOF_SUCCESS" "$REPORTS/segmentation_oof_training_summary.json|sha:source_index_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/segmentation_dataset_index.csv" bash "$0" train-seg-oof
    run_step_resume infer_masks "$REPORTS/.SEGMENTATION_COMPLETE" "$ROOT/manifests/api_adverse_lesion_cave_v1/segmentation_prediction_index.csv|sha:oof_index_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/segmentation_train_oof_predictions.csv" bash "$0" infer-masks
    run_step_resume build_roi_manifests "$REPORTS/.ROI_SUCCESS" "$ROOT/manifests/api_adverse_lesion_cave_v1/roi_manifest_pred.csv|$ROOT/manifests/api_adverse_lesion_cave_v1/roi_manifest_gt.csv|sha:aligned_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/authoritative_roi_manifest_primary.csv|sha:prediction_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/segmentation_prediction_index.csv" bash "$0" build-roi
    run_step_resume extract_mask_features "$REPORTS/.MASK_FEATURES_SUCCESS" "$OUTPUTS/mask_morphology/pred_patient_median.csv|sha:roi_manifest_sha256=$ROOT/manifests/api_adverse_lesion_cave_v1/roi_manifest_all_branches.csv" bash "$0" extract-mask-features
    run_step cave_smoke bash "$0" smoke-cave
    run_step_resume cave_pred_train "$REPORTS/.CAVE_PRED_TRAIN_SUCCESS" "$OUTPUTS/cave_pred_roi_featurebank/feature_schema.json" bash "$0" extract-cave pred Train
    run_step_resume cave_pred_valid "$REPORTS/.CAVE_PRED_VALID_SUCCESS" "$OUTPUTS/cave_pred_roi_featurebank/feature_schema.json" bash "$0" extract-cave pred Valid
    run_step_resume compact_pred_train "$REPORTS/.CAVE_PRED_TRAIN_COMPACT_SUCCESS" "$REPORTS/cave_pred_train_compaction.json" bash "$0" compact-cave pred Train
    run_step_resume compact_pred_valid "$REPORTS/.CAVE_PRED_VALID_COMPACT_SUCCESS" "$REPORTS/cave_pred_valid_compaction.json" bash "$0" compact-cave pred Valid
    run_step_resume cave_gt_train "$REPORTS/.CAVE_GT_TRAIN_SUCCESS" "$OUTPUTS/cave_gt_roi_featurebank/feature_schema.json" bash "$0" extract-cave gt Train
    run_step_resume cave_gt_valid "$REPORTS/.CAVE_GT_VALID_SUCCESS" "$OUTPUTS/cave_gt_roi_featurebank/feature_schema.json" bash "$0" extract-cave gt Valid
    run_step_resume compact_gt_train "$REPORTS/.CAVE_GT_TRAIN_COMPACT_SUCCESS" "$REPORTS/cave_gt_train_compaction.json" bash "$0" compact-cave gt Train
    run_step_resume compact_gt_valid "$REPORTS/.CAVE_GT_VALID_COMPACT_SUCCESS" "$REPORTS/cave_gt_valid_compaction.json" bash "$0" compact-cave gt Valid
    run_step_resume cave_all_nonzero_train "$REPORTS/.CAVE_ALL_NONZERO_TRAIN_SUCCESS" "$OUTPUTS/cave_all_nonzero_roi_featurebank/feature_schema.json" bash "$0" extract-cave all_nonzero Train
    run_step_resume cave_all_nonzero_valid "$REPORTS/.CAVE_ALL_NONZERO_VALID_SUCCESS" "$OUTPUTS/cave_all_nonzero_roi_featurebank/feature_schema.json" bash "$0" extract-cave all_nonzero Valid
    run_step_resume compact_all_nonzero_train "$REPORTS/.CAVE_ALL_NONZERO_TRAIN_COMPACT_SUCCESS" "$REPORTS/cave_all_nonzero_train_compaction.json" bash "$0" compact-cave all_nonzero Train
    run_step_resume compact_all_nonzero_valid "$REPORTS/.CAVE_ALL_NONZERO_VALID_COMPACT_SUCCESS" "$REPORTS/cave_all_nonzero_valid_compaction.json" bash "$0" compact-cave all_nonzero Valid
    run_step_resume build_roi_cave_tables "$REPORTS/.CAVE_FEATURES_SUCCESS" "$REPORTS/roi_cave_table_summary.json" bash "$0" build-tables
    run_step_resume build_adverse_tasks "$REPORTS/.ADVERSE_TASKS_SUCCESS" "$REPORTS/adverse_task_audit.json" bash "$0" build-adverse-tasks
    if [[ -f "$REPORTS/.STOPPED_INSUFFICIENT_COHORT" && ! -f "$REPORTS/.ADVERSE_TASKS_SUCCESS" ]]; then
      run_step summarize_failed_cohort bash "$0" summarize
      exit 0
    fi
    run_step train_main_models bash "$0" train-models
    run_step_resume ablations "$REPORTS/.ABLATIONS_SUCCESS" "$REPORTS/all_ablation_metrics.csv" bash "$0" ablations
    run_step_resume summarize "$REPORTS/.PIPELINE_COMPLETE" "$REPORTS/final_summary.json" bash "$0" summarize
    ;;
  status)
    printf '%s\n' "Markers:"
    find "$REPORTS" -maxdepth 1 -type f -name '.*SUCCESS' -o -name '.PIPELINE*' 2>/dev/null | sort || true
    printf '%s\n' "Segmentation checkpoints:"
    find "$OUTPUTS/segmentation_models" -type f -name '*.pt' 2>/dev/null | wc -l || true
    printf '%s\n' "CAVE phase markers:"
    find "$OUTPUTS" -path '*cave_*_roi_featurebank*' -name '.SUCCESS.json' 2>/dev/null | wc -l || true
    printf '%s\n' "Active processes:"
    pgrep -af 'api_adverse_lesion_cave_v1|extract_cave_featurebank|train_cave_prediction_models_fixed' || true
    ;;
  *) usage; exit 2 ;;
esac
