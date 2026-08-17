#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/aneurysm}
CODE=${CODE:-$ROOT/code/api_adverse_lesion_cave_fast_v1}
CONFIG=${CONFIG:-$ROOT/configs/api_adverse_lesion_cave_fast_v1.json}
V1_CODE=${V1_CODE:-$ROOT/code/api_adverse_lesion_cave_v1}
PRED_PY=${PRED_PY:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}
CAVE_PY=${CAVE_PY:-/root/autodl-tmp/envs/cave-dsa/bin/python}
REPORTS=${REPORTS:-$ROOT/reports/api_adverse_lesion_cave_fast_v1}
OUTPUTS=${OUTPUTS:-$ROOT/outputs/api_adverse_lesion_cave_fast_v1}
V1_REPORTS=${V1_REPORTS:-$ROOT/reports/api_adverse_lesion_cave_v1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=42

mkdir -p "$REPORTS" "$OUTPUTS" "$ROOT/logs"

wait_segmentation() {
  while [[ ! -f "$V1_REPORTS/.SEGMENTATION_OOF_SUCCESS" ]]; do
    if ! pgrep -f "$V1_CODE/05_train_segmentation_oof.py" >/dev/null; then
      printf '%s\n' "Segmentation process ended without .SEGMENTATION_OOF_SUCCESS" >&2
      exit 1
    fi
    printf '%s\n' "[WAIT] formal segmentation still running $(date -Is)"
    sleep 30
  done
  printf '%s\n' "[PASS] formal segmentation OOF complete $(date -Is)"
}

prepare_upstream() {
  [[ -f "$V1_REPORTS/.SEGMENTATION_OOF_SUCCESS" ]]
  if [[ ! -f "$V1_REPORTS/.SEGMENTATION_COMPLETE" ]]; then
    bash "$V1_CODE/run_pipeline.sh" infer-masks
  fi
  [[ -f "$V1_REPORTS/.SEGMENTATION_COMPLETE" ]]
  if [[ -f "$V1_REPORTS/.FAILED_SEGMENTATION_GATE" ]]; then
    printf '%s\n' "[WARN] Segmentation quality gate failed; continuing the mandatory hospital ROI-CAVE workflow with explicit warning provenance."
  elif [[ -f "$V1_REPORTS/.SEGMENTATION_SUCCESS" ]]; then
    printf '%s\n' "[PASS] Segmentation quality gate passed."
  else
    printf '%s\n' "[ERROR] Segmentation completed without a recognized quality marker." >&2
    exit 1
  fi
  if [[ ! -f "$V1_REPORTS/.ROI_SUCCESS" ]]; then
    bash "$V1_CODE/run_pipeline.sh" build-roi
  fi
  if [[ ! -f "$V1_REPORTS/.MASK_FEATURES_SUCCESS" ]]; then
    bash "$V1_CODE/run_pipeline.sh" extract-mask-features
  fi
  "$PRED_PY" "$CODE/01_import_upstream.py" --config "$CONFIG"
}

extract_split() {
  local split=$1
  "$CAVE_PY" "$CODE/10_formal_extract.py" --config "$CONFIG" --split "$split"
  "$PRED_PY" "$CODE/09_verify_featurebank.py" --config "$CONFIG" --split "$split"
  "$PRED_PY" "$V1_CODE/compact_cave_featurebank.py" --config "$CONFIG" --branch pred --split "$split"
  "$PRED_PY" "$CODE/05_build_split_table.py" --config "$CONFIG" --split "$split"
}

status() {
  printf '%s\n' "Original orchestrator and segmentation:"
  ps -eo pid,ppid,stat,etime,args | grep -E '77378|05_train_segmentation_oof.py' | grep -v grep || true
  printf '%s\n' "Original markers:"
  find "$V1_REPORTS" -maxdepth 1 -type f \( -name '.SEGMENTATION*' -o -name '.ROI_SUCCESS' \) -printf '%f\n' | sort || true
  printf '%s\n' "Fast markers:"
  find "$REPORTS" -maxdepth 1 -type f -name '.*' -printf '%f\n' | sort || true
  printf '%s\n' "Fast processes:"
  pgrep -af 'api_adverse_lesion_cave_fast_v1' || true
}

cmd=${1:-}
case "$cmd" in
  audit)
    "$PRED_PY" "$CODE/00_readonly_audit.py" --config "$CONFIG"
    ;;
  wait-segmentation)
    wait_segmentation
    ;;
  prepare-upstream)
    prepare_upstream
    ;;
  benchmark)
    "$CAVE_PY" "$CODE/04_benchmark_cave.py" --config "$CONFIG"
    ;;
  extract-train)
    extract_split Train
    ;;
  train-oof)
    "$PRED_PY" "$CODE/06_train_oof_gate_fixed.py" --config "$CONFIG"
    "$PRED_PY" "$CODE/07_train_morphology_oof_fixed.py" --config "$CONFIG"
    ;;
  extract-valid)
    [[ -f "$REPORTS/.OOF_AUDIT_COMPLETE" ]]
    extract_split Valid
    "$PRED_PY" "$CODE/08_evaluate_valid.py" --config "$CONFIG"
    ;;
  train-mlp)
    "$PRED_PY" "$CODE/11_train_mlp_fusion.py" --config "$CONFIG"
    ;;
  fast-main)
    wait_segmentation
    prepare_upstream
    if [[ ! -f "$REPORTS/recommended_runtime_config.json" ]]; then
      "$CAVE_PY" "$CODE/04_benchmark_cave.py" --config "$CONFIG"
    else
      printf '%s\n' "[SKIP] Current CAVE benchmark recommendation already exists."
    fi
    extract_split Train
    "$PRED_PY" "$CODE/06_train_oof_gate_fixed.py" --config "$CONFIG"
    "$PRED_PY" "$CODE/07_train_morphology_oof_fixed.py" --config "$CONFIG"
    printf '%s\n' "[CONTINUE] OOF comparison is an audit, not a stop gate; extracting formal Pred ROI-CAVE Valid."
    extract_split Valid
    "$PRED_PY" "$CODE/08_evaluate_valid.py" --config "$CONFIG"
    ;;
  status)
    status
    ;;
  *)
    printf '%s\n' "Usage: bash run_pipeline.sh {audit|wait-segmentation|prepare-upstream|benchmark|extract-train|train-oof|extract-valid|train-mlp|fast-main|status}"
    exit 2
    ;;
esac
