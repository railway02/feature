#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/aneurysm"
CODEX="/root/autodl-tmp/tools/codex/bin/codex"
PY="/root/autodl-tmp/envs/aneurysm-ml/bin/python"
PROMPT="$ROOT/prompts/api_fullseq_v1_cpu_overnight.txt"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/logs/codex_api_fullseq_cpu_overnight_${STAMP}"

mkdir -p "$RUN_DIR"

exec > "$RUN_DIR/launcher.log" 2>&1

echo "START=$(date -Is)"
echo "RUN_DIR=$RUN_DIR"

fail() {
    echo "PREFLIGHT_FAILURE=$1"
    echo "$1" > "$RUN_DIR/preflight_failure.txt"
    echo "1" > "$RUN_DIR/exit_status.txt"
    exit 1
}

test -x "$CODEX" || fail "Codex binary missing"
test -x "$PY" || fail "CPU Python missing"
test -f "$PROMPT" || fail "Prompt missing"
test -f "$ROOT/AGENTS.override.md" || fail "AGENTS.override.md missing"
test -d /root/autodl-tmp/tiantanDSA || fail "tiantanDSA missing"
test -d "$ROOT/staging/updated_10_cases" || fail "updated_10_cases missing"

"$PY" - <<'PY' || exit 31
import pandas
import numpy
import cv2
import matplotlib
import openpyxl
print("CPU dependencies OK")
PY

if pgrep -f '09_train_.*\.py|run_searaft.*\.py|SEA-RAFT.*custom\.py' >/dev/null; then
    fail "Training or SEA-RAFT process already running"
fi

TARGETS=(
  "$ROOT/code/13_build_api_fullseq_manifests.py"
  "$ROOT/code/14_build_api_fullseq_cpu_pilot.py"
  "$ROOT/code/15_run_searaft_api_fullseq_pilot.py"
  "$ROOT/code/16_run_searaft_api_fullseq_batch.py"
  "$ROOT/code/17_build_patient_api_fullseq_features.py"
  "$ROOT/code/18_build_task_datasets_api_fullseq_v1.py"
  "$ROOT/code/19_train_adverse_pre_api_fullseq_v1.py"
  "$ROOT/code/api_fullseq_v1"
  "$ROOT/code/20_build_temporal_v2a_exploratory.py"
  "$ROOT/code/21_audit_global_motion_cpu.py"
  "$ROOT/code/22_prepare_searaft_v2_components.py"
  "$ROOT/reports/api_fullseq_v1"
  "$ROOT/outputs/api_fullseq_cpu_pilot"
  "$ROOT/outputs/temporal_v2a_exploratory"
)

for target in "${TARGETS[@]}"; do
    if [ -f "$target" ]; then
        fail "Target file already exists: $target"
    fi

    if [ -d "$target" ] && find "$target" -mindepth 1 -print -quit | grep -q .; then
        fail "Target directory is non-empty: $target"
    fi
done

echo "Creating environment snapshots"

"$PY" -m pip freeze | sort > "$RUN_DIR/aneurysm_ml_before.txt" 2>&1 || true
/root/miniconda3/bin/python -m pip freeze | sort > "$RUN_DIR/base_before.txt" 2>&1 || true

find \
  "$ROOT/code" \
  "$ROOT/metadata" \
  "$ROOT/outputs/task_datasets" \
  "$ROOT/outputs/baselines" \
  "$ROOT/outputs/features" \
  -type f \
  -printf '%p\t%s\t%T@\n' \
  2>/dev/null \
  | sort > "$RUN_DIR/protected_before.tsv"

cd "$ROOT"

set +e

env \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES="" \
  PYTHONHASHSEED=42 \
  timeout --signal=INT --kill-after=300s 8h \
  "$CODEX" \
    --ask-for-approval never \
    exec \
    --skip-git-repo-check \
    -C "$ROOT" \
    -m gpt-5.6-sol \
    --sandbox danger-full-access \
    --json \
    --output-last-message "$RUN_DIR/final_message.txt" \
    - \
    < "$PROMPT" \
    > "$RUN_DIR/events.jsonl" \
    2> "$RUN_DIR/codex_stderr.log"

STATUS=$?

set -e

echo "$STATUS" > "$RUN_DIR/exit_status.txt"

"$PY" -m pip freeze | sort > "$RUN_DIR/aneurysm_ml_after.txt" 2>&1 || true
/root/miniconda3/bin/python -m pip freeze | sort > "$RUN_DIR/base_after.txt" 2>&1 || true

find \
  "$ROOT/code" \
  "$ROOT/metadata" \
  "$ROOT/outputs/task_datasets" \
  "$ROOT/outputs/baselines" \
  "$ROOT/outputs/features" \
  -type f \
  -printf '%p\t%s\t%T@\n' \
  2>/dev/null \
  | sort > "$RUN_DIR/protected_after.tsv"

diff -u \
  "$RUN_DIR/aneurysm_ml_before.txt" \
  "$RUN_DIR/aneurysm_ml_after.txt" \
  > "$RUN_DIR/aneurysm_ml_environment.diff" || true

diff -u \
  "$RUN_DIR/base_before.txt" \
  "$RUN_DIR/base_after.txt" \
  > "$RUN_DIR/base_environment.diff" || true

diff -u \
  "$RUN_DIR/protected_before.tsv" \
  "$RUN_DIR/protected_after.tsv" \
  > "$RUN_DIR/protected_files.diff" || true

if [ -s "$RUN_DIR/aneurysm_ml_environment.diff" ]; then
    echo "WARNING: aneurysm-ml environment changed"
fi

if [ -s "$RUN_DIR/base_environment.diff" ]; then
    echo "WARNING: base environment changed"
fi

if [ -s "$RUN_DIR/protected_files.diff" ]; then
    echo "WARNING: protected file inventory changed"
fi

echo "END=$(date -Is)"
echo "CODEX_STATUS=$STATUS"

exit "$STATUS"
