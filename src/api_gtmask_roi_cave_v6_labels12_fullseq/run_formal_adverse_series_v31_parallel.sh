#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/aneurysm}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}"
CODE_DIR="${CODE_DIR:-$PROJECT/code/api_gtmask_roi_cave_v6_labels12_fullseq}"

TRAIN_XLSX="${TRAIN_XLSX:-$PROJECT/metadata/Train.xlsx}"
VALID_XLSX="${VALID_XLSX:-$PROJECT/metadata/valid.xlsx}"
TABLE_ROOT="${TABLE_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v6_labels12_fullseq/tables/local_eligible}"
MAPPING_DIR="${MAPPING_DIR:-$PROJECT/manifests/api_record_v1/final_mapping}"

TASK_ROOT="${TASK_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v6_labels12_fullseq/adverse_prepost_series_task_v3}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v6_labels12_fullseq/adverse_prepost_series_formal_models_v31}"

DEVICE="${DEVICE:-cuda:0}"
LOGISTIC_CPU_THREADS="${LOGISTIC_CPU_THREADS:-4}"
MLP_CPU_THREADS="${MLP_CPU_THREADS:-2}"
FINALIZE_CPU_THREADS="${FINALIZE_CPU_THREADS:-8}"
MLP_SEARCH_SEEDS="${MLP_SEARCH_SEEDS:-2}"
MLP_SEEDS="${MLP_SEEDS:-3}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-2000}"
DISABLE_AMP="${DISABLE_AMP:-0}"

CORE_RUNNER="$CODE_DIR/run_formal_adverse_series_v3.sh"
TRAINER="$CODE_DIR/11_train_adverse_prepost_series_formal_v3.py"
LOGISTIC_WORKER="$CODE_DIR/13_train_logistic_base_v31.py"
MLP_WORKER="$CODE_DIR/14_train_mlp_base_v31.py"
SUMMARY_SCRIPT="$CODE_DIR/12_summarize_adverse_series_results_v3.py"

usage() {
  cat <<EOF
Usage:
  bash $0 mapping-preflight
  bash $0 mapping-finalize
  bash $0 preflight
  bash $0 test
  bash $0 build [--overwrite]
  bash $0 cohort-check
  bash $0 train-parallel [--overwrite]
  bash $0 finalize
  bash $0 status
  bash $0 summarize

Parallel layout:
  CPU worker: Logistic_Deep + Logistic_Fusion
  GPU worker 1: MLP_Deep
  GPU worker 2: MLP_Fusion

The two GPU workers share one GPU but own disjoint model directories.
EOF
}

export_common() {
  export PROJECT PYTHON CODE_DIR TRAIN_XLSX VALID_XLSX TABLE_ROOT
  export MAPPING_DIR TASK_ROOT
}

delegate() {
  export_common
  MODEL_ROOT="$MODEL_ROOT" \
    DEVICE="$DEVICE" \
    CPU_THREADS="$FINALIZE_CPU_THREADS" \
    MLP_SEARCH_SEEDS="$MLP_SEARCH_SEEDS" \
    MLP_SEEDS="$MLP_SEEDS" \
    BOOTSTRAP_REPEATS="$BOOTSTRAP_REPEATS" \
    DISABLE_AMP="$DISABLE_AMP" \
    bash "$CORE_RUNNER" "$@"
}

parallel_preflight() {
  delegate preflight
  for path in "$LOGISTIC_WORKER" "$MLP_WORKER" "$TRAINER"; do
    echo "PARALLEL_FILE: exists=$([[ -f "$path" ]] && echo true || echo false) path=$path"
    [[ -f "$path" ]] || {
      echo "Missing parallel file: $path" >&2
      exit 2
    }
  done
  [[ -f "$TASK_ROOT/.TASK_SUCCESS.json" ]] || {
    echo "Task has not been built: $TASK_ROOT/.TASK_SUCCESS.json" >&2
    echo "Run build and cohort-check first." >&2
    exit 2
  }
  echo "PARALLEL_PREFLIGHT_OK"
}

check_no_active_parallel_run() {
  local active
  active="$(
    pgrep -af \
      '13_train_logistic_base_v31.py|14_train_mlp_base_v31.py|11_train_adverse_prepost_series_formal_v3.py' \
      | grep -v "$$" \
      | grep -v "pgrep -af" || true
  )"
  if [[ -n "$active" ]]; then
    echo "An existing V3.1 parallel run appears active:" >&2
    echo "$active" >&2
    exit 3
  fi
}

LAST_PID=""
run_worker() {
  local name="$1"
  shift
  local log_dir="$MODEL_ROOT/parallel_logs"
  mkdir -p "$log_dir" "$MODEL_ROOT/workers"
  echo "Starting $name"
  "$@" > "$log_dir/$name.log" 2>&1 &
  LAST_PID="$!"
}

finalize_models() {
  parallel_preflight
  echo "Checking 20 base model fold caches..."
  local count
  count="$( { find "$MODEL_ROOT/folds" -name ".SUCCESS.json" 2>/dev/null || true; } | wc -l )"
  echo "BASE_FOLD_SUCCESS=$count/20"
  if [[ "$count" -ne 20 ]]; then
    echo "Cannot finalize: expected 20 completed base-model folds." >&2
    exit 4
  fi

  extra=()
  if [[ "$DISABLE_AMP" == "1" ]]; then
    extra+=(--disable-amp)
  fi

  OMP_NUM_THREADS="$FINALIZE_CPU_THREADS" \
  MKL_NUM_THREADS="$FINALIZE_CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$FINALIZE_CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$FINALIZE_CPU_THREADS" \
  "$PYTHON" "$TRAINER" \
    --task-root "$TASK_ROOT" \
    --output-dir "$MODEL_ROOT" \
    --device "$DEVICE" \
    --cpu-threads "$FINALIZE_CPU_THREADS" \
    --mlp-seeds "$MLP_SEEDS" \
    --mlp-search-seeds "$MLP_SEARCH_SEEDS" \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    "${extra[@]}"

  echo "PARALLEL_FINALIZE_OK"
}

train_parallel() {
  local overwrite=0
  if [[ "${1:-}" == "--overwrite" ]]; then
    overwrite=1
    shift
  fi
  if [[ "$#" -ne 0 ]]; then
    echo "Unexpected arguments: $*" >&2
    exit 2
  fi

  parallel_preflight
  check_no_active_parallel_run

  if [[ "$overwrite" == "1" ]]; then
    echo "Removing previous V3.1 model output: $MODEL_ROOT"
    rm -rf "$MODEL_ROOT"
  fi
  mkdir -p "$MODEL_ROOT/parallel_logs" "$MODEL_ROOT/workers"

  amp_args=()
  if [[ "$DISABLE_AMP" == "1" ]]; then
    amp_args+=(--disable-amp)
  fi

  run_worker logistic_cpu \
    env \
      OMP_NUM_THREADS="$LOGISTIC_CPU_THREADS" \
      MKL_NUM_THREADS="$LOGISTIC_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$LOGISTIC_CPU_THREADS" \
      NUMEXPR_NUM_THREADS="$LOGISTIC_CPU_THREADS" \
    "$PYTHON" "$LOGISTIC_WORKER" \
      --task-root "$TASK_ROOT" \
      --output-dir "$MODEL_ROOT" \
      --trainer "$TRAINER" \
      --cpu-threads "$LOGISTIC_CPU_THREADS"
  logistic_pid="$LAST_PID"

  run_worker mlp_deep_gpu \
    env \
      OMP_NUM_THREADS="$MLP_CPU_THREADS" \
      MKL_NUM_THREADS="$MLP_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$MLP_CPU_THREADS" \
      NUMEXPR_NUM_THREADS="$MLP_CPU_THREADS" \
    "$PYTHON" "$MLP_WORKER" \
      --task-root "$TASK_ROOT" \
      --output-dir "$MODEL_ROOT" \
      --trainer "$TRAINER" \
      --model MLP_Deep \
      --device "$DEVICE" \
      --cpu-threads "$MLP_CPU_THREADS" \
      --mlp-seeds "$MLP_SEEDS" \
      --mlp-search-seeds "$MLP_SEARCH_SEEDS" \
      "${amp_args[@]}"
  mlp_deep_pid="$LAST_PID"

  run_worker mlp_fusion_gpu \
    env \
      OMP_NUM_THREADS="$MLP_CPU_THREADS" \
      MKL_NUM_THREADS="$MLP_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$MLP_CPU_THREADS" \
      NUMEXPR_NUM_THREADS="$MLP_CPU_THREADS" \
    "$PYTHON" "$MLP_WORKER" \
      --task-root "$TASK_ROOT" \
      --output-dir "$MODEL_ROOT" \
      --trainer "$TRAINER" \
      --model MLP_Fusion \
      --device "$DEVICE" \
      --cpu-threads "$MLP_CPU_THREADS" \
      --mlp-seeds "$MLP_SEEDS" \
      --mlp-search-seeds "$MLP_SEARCH_SEEDS" \
      "${amp_args[@]}"
  mlp_fusion_pid="$LAST_PID"

  cat > "$MODEL_ROOT/workers/parallel_pids.txt" <<EOF
logistic_cpu=$logistic_pid
mlp_deep_gpu=$mlp_deep_pid
mlp_fusion_gpu=$mlp_fusion_pid
EOF

  echo "logistic_cpu PID=$logistic_pid"
  echo "mlp_deep_gpu PID=$mlp_deep_pid"
  echo "mlp_fusion_gpu PID=$mlp_fusion_pid"
  echo "Worker logs: $MODEL_ROOT/parallel_logs"

  set +e
  wait "$logistic_pid"
  logistic_rc=$?
  wait "$mlp_deep_pid"
  mlp_deep_rc=$?
  wait "$mlp_fusion_pid"
  mlp_fusion_rc=$?
  set -e

  echo "Worker return codes:"
  echo "  logistic_cpu=$logistic_rc"
  echo "  mlp_deep_gpu=$mlp_deep_rc"
  echo "  mlp_fusion_gpu=$mlp_fusion_rc"

  if [[ "$logistic_rc" -ne 0 || "$mlp_deep_rc" -ne 0 || "$mlp_fusion_rc" -ne 0 ]]; then
    echo "At least one worker failed. Inspect:" >&2
    echo "  $MODEL_ROOT/parallel_logs/logistic_cpu.log" >&2
    echo "  $MODEL_ROOT/parallel_logs/mlp_deep_gpu.log" >&2
    echo "  $MODEL_ROOT/parallel_logs/mlp_fusion_gpu.log" >&2
    exit 5
  fi

  finalize_models
}

status() {
  echo "===== RUNNING PROCESSES ====="
  pgrep -af \
    '13_train_logistic_base_v31.py|14_train_mlp_base_v31.py|11_train_adverse_prepost_series_formal_v3.py|run_formal_adverse_series_v31_parallel.sh' \
    || true

  echo
  echo "===== BASE FOLD CACHES ====="
  count="$( { find "$MODEL_ROOT/folds" -name ".SUCCESS.json" 2>/dev/null || true; } | wc -l )"
  echo "$count/20"

  echo
  echo "===== WORKER SUCCESS ====="
  find "$MODEL_ROOT/workers" -name "*.SUCCESS.json" -maxdepth 1 -print 2>/dev/null | sort || true

  echo
  echo "===== LATEST WORKER LOG LINES ====="
  for name in logistic_cpu mlp_deep_gpu mlp_fusion_gpu; do
    log="$MODEL_ROOT/parallel_logs/$name.log"
    echo "--- $name ---"
    if [[ -f "$log" ]]; then
      tail -n 8 "$log"
    else
      echo "not started"
    fi
  done

  echo
  echo "===== FINAL STATUS ====="
  if [[ -f "$MODEL_ROOT/.MODELS_SUCCESS.json" ]]; then
    cat "$MODEL_ROOT/.MODELS_SUCCESS.json"
  else
    echo "not finalized"
  fi
}

summarize() {
  "$PYTHON" "$SUMMARY_SCRIPT" \
    --task-root "$TASK_ROOT" \
    --model-root "$MODEL_ROOT"
}

command="${1:-}"
shift || true
case "$command" in
  mapping-preflight|mapping-finalize|preflight|test|build|cohort-check)
    delegate "$command" "$@"
    ;;
  train-parallel)
    train_parallel "$@"
    ;;
  finalize)
    finalize_models
    ;;
  status)
    status
    ;;
  summarize)
    summarize
    ;;
  *)
    usage
    exit 2
    ;;
esac
