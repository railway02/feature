#!/usr/bin/env bash
set -euo pipefail

PROJECT="/root/autodl-tmp/aneurysm"
PYTHON="/root/miniconda3/bin/python"
CONFIG="$PROJECT/configs/api_fullseq_v2_full_train_valid_config.json"
TRAIN_MANIFEST="$PROJECT/manifests/api_fullseq_v2_train_manifest.csv"
VALID_MANIFEST="$PROJECT/manifests/api_fullseq_v2_valid_manifest.csv"
TRAIN_PAIRDATA="$PROJECT/outputs/api_fullseq_v2_pairdata/full/train"
VALID_PAIRDATA="$PROJECT/outputs/api_fullseq_v2_pairdata/full/valid"
REPORT_ROOT="$PROJECT/reports/api_fullseq_v2_feature_full"
STATE_FILE="$PROJECT/logs/latest_api_fullseq_v2_full_train_valid_run.txt"
NOHUP_LOG="$PROJECT/logs/api_fullseq_v2_full_train_valid_nohup.log"
TRAIN_SUCCESS="$REPORT_ROOT/.FULL_TRAIN_SUCCESS"
VALID_SUCCESS="$REPORT_ROOT/.FULL_VALID_SUCCESS"
FINAL_SUCCESS="$REPORT_ROOT/.FULL_TRAIN_VALID_SUCCESS"
FAILURE_REPORT="$REPORT_ROOT/unattended_failure.md"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CURRENT_STAGE="runner_initializing"

mkdir -p "$PROJECT/logs" "$REPORT_ROOT"
cd "$PROJECT"
export OMP_NUM_THREADS=1

write_state() {
  local status="$1"
  local attempt="${2:-0}"
  {
    printf 'runner_pid=%s\n' "$$"
    printf 'started_utc=%s\n' "$STARTED_UTC"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'status=%s\n' "$status"
    printf 'current_stage=%s\n' "$CURRENT_STAGE"
    printf 'attempt=%s\n' "$attempt"
    printf 'log_path=%s\n' "$NOHUP_LOG"
    printf 'train_success=%s\n' "$TRAIN_SUCCESS"
    printf 'valid_success=%s\n' "$VALID_SUCCESS"
    printf 'final_success=%s\n' "$FINAL_SUCCESS"
  } > "$STATE_FILE"
}

log_event() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

record_failure() {
  local exit_code="$1"
  if [[ ! -e "$FAILURE_REPORT" ]]; then
    {
      printf '# api_fullseq_v2 unattended runner failure\n\n'
      printf -- '- Failed UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf -- '- Runner PID: %s\n' "$$"
      printf -- '- Stage: %s\n' "$CURRENT_STAGE"
      printf -- '- Exit code: %s\n' "$exit_code"
      printf -- '- Existing successful phase outputs were preserved.\n'
      printf -- '- Resume is supported by rerunning this same runner.\n'
      printf -- '- Historical promotion evidence was not modified.\n'
      printf -- '- Labels read: no.\n'
      printf -- '- Model training: no.\n'
    } > "$FAILURE_REPORT"
  fi
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -eq 0 ]]; then
    CURRENT_STAGE="complete"
    write_state "success" 0
    log_event "RUNNER SUCCESS train+valid complete"
  else
    write_state "failed" 0
    record_failure "$exit_code"
    log_event "RUNNER FAILED stage=$CURRENT_STAGE exit_code=$exit_code"
  fi
}
trap 'on_exit $?' EXIT

run_hard_stage() {
  local stage="$1"
  shift
  CURRENT_STAGE="$stage"
  write_state "running" 1
  log_event "START stage=$stage"
  "$@"
  log_event "SUCCESS stage=$stage"
}

run_retry_stage() {
  local stage="$1"
  shift
  local attempt exit_code
  CURRENT_STAGE="$stage"
  for attempt in 1 2 3; do
    write_state "running" "$attempt"
    log_event "START stage=$stage attempt=$attempt/3"
    set +e
    "$@"
    exit_code=$?
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
      log_event "SUCCESS stage=$stage attempt=$attempt/3"
      return 0
    fi
    if [[ "$exit_code" -eq 42 ]]; then
      log_event "HARD FAILURE stage=$stage attempt=$attempt/3; no retry"
      return 42
    fi
    if [[ "$attempt" -lt 3 ]]; then
      log_event "TRANSIENT FAILURE stage=$stage exit_code=$exit_code; waiting 60 seconds before --resume"
      sleep 60
    fi
  done
  log_event "RETRY EXHAUSTED stage=$stage"
  return "$exit_code"
}

write_state "starting" 0
log_event "RUNNER START pid=$$"

run_hard_stage "production_release_preflight" \
  "$PYTHON" code/22_audit_api_fullseq_v2_train_valid.py --mode preflight

run_retry_stage "full_train_pairdata" \
  "$PYTHON" code/20_extract_api_fullseq_v2_train_valid_pairdata.py \
  --mode full_train --manifest "$TRAIN_MANIFEST" --config "$CONFIG" \
  --device cuda:0 --resume --max-visual-pairs 0 --output-root "$TRAIN_PAIRDATA"

run_retry_stage "full_train_features" \
  "$PYTHON" code/21_build_api_fullseq_v2_train_valid_features.py \
  --mode full_train --pairdata-root "$TRAIN_PAIRDATA" --resume

run_hard_stage "full_train_audit" \
  "$PYTHON" code/22_audit_api_fullseq_v2_train_valid.py --mode audit-train

if [[ ! -f "$TRAIN_SUCCESS" ]]; then
  log_event "HARD FAILURE .FULL_TRAIN_SUCCESS missing; Full Valid forbidden"
  exit 42
fi

run_hard_stage "snapshot_before_valid" \
  "$PYTHON" code/22_audit_api_fullseq_v2_train_valid.py --mode snapshot-before-valid

run_retry_stage "full_valid_pairdata" \
  "$PYTHON" code/20_extract_api_fullseq_v2_train_valid_pairdata.py \
  --mode full_valid --manifest "$VALID_MANIFEST" --config "$CONFIG" \
  --device cuda:0 --resume --max-visual-pairs 0 --output-root "$VALID_PAIRDATA"

run_retry_stage "full_valid_features" \
  "$PYTHON" code/21_build_api_fullseq_v2_train_valid_features.py \
  --mode full_valid --pairdata-root "$VALID_PAIRDATA" --resume

run_hard_stage "full_valid_audit" \
  "$PYTHON" code/22_audit_api_fullseq_v2_train_valid.py --mode audit-valid

run_hard_stage "train_valid_finalize" \
  "$PYTHON" code/22_audit_api_fullseq_v2_train_valid.py --mode finalize

if [[ ! -f "$VALID_SUCCESS" || ! -f "$FINAL_SUCCESS" ]]; then
  log_event "HARD FAILURE final success markers missing"
  exit 42
fi

CURRENT_STAGE="complete"
exit 0
