#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/aneurysm}
CODE=$ROOT/code/api_adverse_lesion_record_v2
CONFIG=$ROOT/configs/api_adverse_lesion_record_v2/config.json
PRED_PY=${PRED_PY:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}
REPORTS=$ROOT/reports/api_adverse_lesion_record_v2
LOG=$ROOT/logs/api_adverse_lesion_record_v2_gt_oracle.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=42

cmd=${1:-status}
case "$cmd" in
  prepare)
    "$PRED_PY" "$CODE/01_prepare_record_oracle.py" --config "$CONFIG"
    ;;
  smoke)
    "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale 30 --split Train --gpu-processes 1 --io-workers 4 --max-series 2 --tag smoke
    "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale 40 --split Train --gpu-processes 1 --io-workers 4 --max-series 2 --tag smoke
    ;;
  gt-oracle)
    test -f "$REPORTS/.RECORD_MANIFEST_PASS"
    test -f "$REPORTS/.TEMPORAL_VIEW_PASS"
    "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale 30 --split Train --tag formal
    "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale 40 --split Train --tag formal
    "$PRED_PY" "$CODE/06_build_gt_local_tables.py" --config "$CONFIG" --scale 30 --split Train
    "$PRED_PY" "$CODE/06_build_gt_local_tables.py" --config "$CONFIG" --scale 40 --split Train
    "$PRED_PY" "$CODE/07_train_record_gt_oracle_oof.py" --config "$CONFIG"
    if [[ -f "$REPORTS/.GT_ORACLE_PASS" ]]; then
      scale=$($PRED_PY -c "import json; print(json.load(open('$REPORTS/record_gt_oracle_oof_summary.json'))['selected_scale'])")
      "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale "$scale" --split Valid --tag formal
      "$PRED_PY" "$CODE/06_build_gt_local_tables.py" --config "$CONFIG" --scale "$scale" --split Valid
      "$PRED_PY" "$CODE/08_evaluate_record_gt_oracle_valid.py" --config "$CONFIG"
    fi
    "$PRED_PY" "$CODE/09_summarize_record_v2.py"
    ;;
  oracle-oof-resume)
    test -f "$ROOT/outputs/api_adverse_lesion_record_v2/cave_gt_context30_tables/train/build_audit.json"
    test -f "$ROOT/outputs/api_adverse_lesion_record_v2/cave_gt_context40_tables/train/build_audit.json"
    "$PRED_PY" "$CODE/07_train_record_gt_oracle_oof.py" --config "$CONFIG"
    if [[ -f "$REPORTS/.GT_ORACLE_PASS" ]]; then
      scale=$($PRED_PY -c "import json; print(json.load(open('$REPORTS/record_gt_oracle_oof_summary.json'))['selected_scale'])")
      "$PRED_PY" "$CODE/05_run_gt_local_sharded_v2.py" --config "$CONFIG" --scale "$scale" --split Valid --tag formal
      "$PRED_PY" "$CODE/06_build_gt_local_tables.py" --config "$CONFIG" --scale "$scale" --split Valid
      "$PRED_PY" "$CODE/08_evaluate_record_gt_oracle_valid.py" --config "$CONFIG"
    fi
    "$PRED_PY" "$CODE/09_summarize_record_v2.py"
    ;;
  status)
    printf '%s\n' "RUN_MANIFEST=$REPORTS/RUN_MANIFEST.json" "LOG=$LOG"
    find "$REPORTS" -maxdepth 1 -type f -name '.*' -printf '%f\n' 2>/dev/null | sort || true
    ps -eo pid,ppid,stat,etime,args | grep -E 'api_adverse_lesion_record_v2|04_extract_gt_local_worker_v2' | grep -v grep || true
    ;;
  *)
    printf '%s\n' "Usage: bash $0 {prepare|smoke|gt-oracle|oracle-oof-resume|status}" >&2
    exit 2
    ;;
esac
