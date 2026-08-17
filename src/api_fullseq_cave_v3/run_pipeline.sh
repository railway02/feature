#!/usr/bin/env bash
set -euo pipefail

# libgomp/OpenBLAS require plain integer values.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
for name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  value=${!name}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    export "$name"=8
  fi
done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}

ROOT=${ROOT:-/root/autodl-tmp/aneurysm}
CODE=${CODE:-$ROOT/code/api_fullseq_cave_v3}
PY=${PY:-/root/autodl-tmp/envs/cave-dsa/bin/python}
PRED_PY=${PRED_PY:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}
CAVE_REPO=${CAVE_REPO:-/root/autodl-tmp/CAVE_DSA}
CKPT=${CKPT:-$CAVE_REPO/checkpoints/sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$ROOT/manifests/api_fullseq_v3_train_all_series_frozen.csv}
VALID_MANIFEST=${VALID_MANIFEST:-$ROOT/manifests/api_fullseq_v3_valid_all_series_frozen.csv}
V3_EXTRACTOR=${V3_EXTRACTOR:-$ROOT/code/api_fullseq_v3/extract_pairdata.py}
V3_BASE_CONFIG=${V3_BASE_CONFIG:-$ROOT/configs/api_fullseq_v2_full_train_valid_config.json}
V3_OVERRIDE_CONFIG=${V3_OVERRIDE_CONFIG:-$ROOT/configs/api_fullseq_v3_improved_overrides.json}
FROZEN_CONFIG=${FROZEN_CONFIG:-$ROOT/configs/api_fullseq_cave_v3_frozen.json}
FEATURE_ROOT=${FEATURE_ROOT:-$ROOT/outputs/api_fullseq_cave_v3_featurebank}
TABLE_ROOT=${TABLE_ROOT:-$ROOT/outputs/api_fullseq_cave_v3_tables}
TASK_ROOT=${TASK_ROOT:-$ROOT/outputs/api_fullseq_cave_v3_tasks}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/outputs/api_fullseq_cave_v3_models}
REPORT_ROOT=${REPORT_ROOT:-$ROOT/reports/api_fullseq_cave_v3}
RELEASE=${RELEASE:-$REPORT_ROOT/train_release_freeze.json}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}

usage() {
  cat <<EOF
Usage: bash run_pipeline.sh <command>
Commands:
  install          create environment, pin CAVE, download checkpoint, run tests
  smoke-real       extract up to 5 real Train series
  train            full Train feature extraction (resumable)
  audit-train      audit and build Train phase/series/patient tables
  freeze           freeze successful Train release
  valid            full Valid extraction; requires frozen Train release
  audit-valid      audit and build Valid phase/series/patient tables
  build-tasks      build adverse/immediate/follow-up CAVE task arrays
  train-models     Train Dummy, Logistic_deep/scalar/fusion and MLP_fusion
  summarize        write final full-pipeline summary
  all-train        train + audit-train + freeze
  full-auto        Train extraction -> Valid extraction -> tasks -> prediction
  status           print markers and active processes
EOF
}

common_args=(
  --cave-repo "$CAVE_REPO"
  --checkpoint "$CKPT"
  --v3-extractor "$V3_EXTRACTOR"
  --v3-base-config "$V3_BASE_CONFIG"
  --v3-override-config "$V3_OVERRIDE_CONFIG"
  --output-root "$FEATURE_ROOT"
  --report-root "$REPORT_ROOT"
  --frozen-config "$FROZEN_CONFIG"
)

run_step() {
  local name=$1
  shift
  local start end
  start=$(date +%s)
  echo "=================================================="
  echo "[START] $name: $(date -Is)"
  "$@"
  end=$(date +%s)
  echo "[PASS] $name elapsed_seconds=$((end-start)): $(date -Is)"
  echo "=================================================="
}

cmd=${1:-}
case "$cmd" in
  install)
    bash "$CODE/install_and_smoke.sh"
    "$PRED_PY" - <<'PY'
import sklearn, torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
print("prediction torch:", torch.__version__)
print("sklearn:", sklearn.__version__)
print("[PASS] prediction environment")
PY
    ;;
  smoke-real)
    "$PY" "$CODE/extract_cave_featurebank.py" \
      --mode smoke --manifest "$TRAIN_MANIFEST" "${common_args[@]}" --max-series 5
    ;;
  train)
    mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
    "$PY" "$CODE/extract_cave_featurebank.py" \
      --mode full_train --manifest "$TRAIN_MANIFEST" "${common_args[@]}"
    ;;
  audit-train)
    "$PY" "$CODE/audit_featurebank.py" \
      --manifest "$TRAIN_MANIFEST" --feature-root "$FEATURE_ROOT" \
      --output "$REPORT_ROOT/train_audit.json" --expected-split Train \
      --expected-series 1147 --expected-patients 1055 --expected-pre 940 --expected-post 1147
    "$PY" "$CODE/build_feature_tables.py" \
      --manifest "$TRAIN_MANIFEST" --feature-root "$FEATURE_ROOT" \
      --output-dir "$TABLE_ROOT/train" --expected-split Train \
      --expected-series 1147 --expected-patients 1055
    ;;
  freeze)
    "$PY" "$CODE/freeze_release.py" \
      --package-root "$CODE" --frozen-config "$FROZEN_CONFIG" \
      --train-manifest "$TRAIN_MANIFEST" --valid-manifest "$VALID_MANIFEST" \
      --cave-repo "$CAVE_REPO" --checkpoint "$CKPT" \
      --v3-extractor "$V3_EXTRACTOR" --v3-base-config "$V3_BASE_CONFIG" \
      --v3-override-config "$V3_OVERRIDE_CONFIG" --feature-root "$FEATURE_ROOT" \
      --train-table-root "$TABLE_ROOT/train" --train-audit "$REPORT_ROOT/train_audit.json" \
      --output "$RELEASE"
    ;;
  valid)
    "$PY" "$CODE/extract_cave_featurebank.py" \
      --mode full_valid --manifest "$VALID_MANIFEST" "${common_args[@]}" \
      --release-freeze "$RELEASE"
    ;;
  audit-valid)
    "$PY" "$CODE/audit_featurebank.py" \
      --manifest "$VALID_MANIFEST" --feature-root "$FEATURE_ROOT" \
      --output "$REPORT_ROOT/valid_audit.json" --expected-split Valid \
      --expected-series 287 --expected-patients 264 --expected-pre 248 --expected-post 287
    "$PY" "$CODE/build_feature_tables.py" \
      --manifest "$VALID_MANIFEST" --feature-root "$FEATURE_ROOT" \
      --output-dir "$TABLE_ROOT/valid" --expected-split Valid \
      --expected-series 287 --expected-patients 264
    ;;
  build-tasks)
    test -f "$TABLE_ROOT/train/build_audit.json"
    test -f "$TABLE_ROOT/valid/build_audit.json"
    "$PRED_PY" "$CODE/build_cave_prediction_tasks.py" \
      --project "$ROOT" \
      --train-table-dir "$TABLE_ROOT/train" \
      --valid-table-dir "$TABLE_ROOT/valid" \
      --output-dir "$TASK_ROOT" --overwrite
    ;;
  train-models)
    test -f "$TASK_ROOT/.TASKS_SUCCESS"
    "$PRED_PY" "$CODE/train_cave_prediction_models.py" \
      --task-root "$TASK_ROOT" --output-dir "$MODEL_ROOT" \
      --device cuda:0 --overwrite
    ;;
  summarize)
    test -f "$MODEL_ROOT/.MODELS_SUCCESS"
    "$PRED_PY" "$CODE/summarize_cave_pipeline.py" \
      --project "$ROOT" --feature-root "$FEATURE_ROOT" \
      --table-root "$TABLE_ROOT" --task-root "$TASK_ROOT" \
      --model-root "$MODEL_ROOT" --report-root "$REPORT_ROOT"
    ;;
  all-train)
    run_step full_train bash "$0" train
    run_step audit_train bash "$0" audit-train
    run_step freeze_train_release bash "$0" freeze
    ;;
  full-auto)
    mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
    PIPELINE_START=$(date +%s)
    run_step full_train bash "$0" train
    run_step audit_train bash "$0" audit-train
    run_step freeze_train_release bash "$0" freeze
    run_step full_valid bash "$0" valid
    run_step audit_valid bash "$0" audit-valid
    run_step build_prediction_tasks bash "$0" build-tasks
    run_step train_and_evaluate_models bash "$0" train-models
    run_step summarize_pipeline bash "$0" summarize
    PIPELINE_END=$(date +%s)
    echo "[PASS] FULL AUTO elapsed_seconds=$((PIPELINE_END-PIPELINE_START))"
    ;;
  status)
    echo "Train feature markers:"; find "$FEATURE_ROOT/train" -name '.SUCCESS.json' 2>/dev/null | wc -l || true
    echo "Valid feature markers:"; find "$FEATURE_ROOT/valid" -name '.SUCCESS.json' 2>/dev/null | wc -l || true
    echo "Release:"; ls -l "$RELEASE" 2>/dev/null || true
    echo "Tasks/models:"; ls -l "$TASK_ROOT/.TASKS_SUCCESS" "$MODEL_ROOT/.MODELS_SUCCESS" "$REPORT_ROOT/.FULL_AUTO_WITH_MODELS_SUCCESS" 2>/dev/null || true
    pgrep -af 'api_fullseq_cave_v3/(extract_cave_featurebank|build_feature_tables|build_cave_prediction_tasks|train_cave_prediction_models)' || true
    ;;
  *)
    usage
    exit 2
    ;;
esac
