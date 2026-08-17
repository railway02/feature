#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/aneurysm}"
PY="${PY:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$PROJECT/manifests/api_fullseq_v3_source_train_all_series.csv}"
VALID_SOURCE="${VALID_SOURCE:-$PROJECT/manifests/api_fullseq_v3_source_valid_all_series.csv}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT/configs/api_fullseq_v2_full_train_valid_config.json}"
OVERRIDE_CONFIG="${OVERRIDE_CONFIG:-$PROJECT/configs/api_fullseq_v3_improved_overrides.json}"
CACHE_SIZE="${CACHE_SIZE:-96}"

CODE="$PROJECT/code/api_fullseq_v3"
MANIFESTS="$PROJECT/manifests"
PILOT_PAIRDATA="$PROJECT/outputs/api_fullseq_v3_pairdata/pilot/improved"
PILOT_FEATURES="$PROJECT/outputs/api_fullseq_v3_features/pilot/improved"
PILOT_REPORT="$PROJECT/reports/api_fullseq_v3_reextract/pilot"
TRAIN_PAIRDATA="$PROJECT/outputs/api_fullseq_v3_pairdata/full/train"
VALID_PAIRDATA="$PROJECT/outputs/api_fullseq_v3_pairdata/full/valid"
TRAIN_FEATURES="$PROJECT/outputs/api_fullseq_v3_features/full/train"
VALID_FEATURES="$PROJECT/outputs/api_fullseq_v3_features/full/valid"
REPORT="$PROJECT/reports/api_fullseq_v3_reextract"
TASKS="$PROJECT/outputs/api_fullseq_v3_tasks"
MODELS="$PROJECT/outputs/api_fullseq_v3_models"
LOGS="$PROJECT/logs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
mkdir -p "$MANIFESTS" "$REPORT" "$LOGS"
cd "$PROJECT"

usage() {
  cat <<EOF
Usage: bash run_api_fullseq_v3_reextract.sh STAGE

Stages:
  check                 Static checks and synthetic semantic tests
  freeze_manifests      Audit/freeze current all-series Train/Valid manifests
  pilot                 Run 40-series improved SEA-RAFT pilot
  pilot_features        Build 106/phase, 212 Pre+Post pilot features
  pilot_audit           Hard engineering/cache audit (manual visual gate remains)
  full_train            Run all 1147 Train series with --resume
  full_train_features   Build Full Train series/patient features
  freeze_release        Freeze code/config/model/schema/manifests after Train
  full_valid            Run all 287 Valid series after frozen Train gate
  full_valid_features   Build Full Valid features and final success marker
  build_tasks           Build adverse/immediate/follow-up prediction tables
  train_models          Train Dummy, Logistic and MLP; evaluate official Valid
  full_auto             Check -> freeze -> Full Train -> Full Valid -> models
  status                Show current progress/status
EOF
}

stage="${1:-}"
case "$stage" in
  check)
    "$PY" -m py_compile "$CODE"/*.py
    bash -n "$PROJECT/run_api_fullseq_v3_reextract.sh"
    "$PY" "$CODE/test_synthetic.py"
    ;;

  freeze_manifests)
    "$PY" "$CODE/prepare_manifests.py" \
      --train-manifest "$TRAIN_SOURCE" \
      --valid-manifest "$VALID_SOURCE" \
      --output-dir "$MANIFESTS" \
      --pilot-count 40 \
      --verify-files \
      --overwrite
    ;;

  pilot)
    test -f "$MANIFESTS/.MANIFESTS_FROZEN_SUCCESS"
    if [[ -d "$PILOT_PAIRDATA" && ! -f "$PILOT_PAIRDATA/.SUCCESS" ]]; then
      resume=(--resume)
    elif [[ -f "$PILOT_PAIRDATA/.SUCCESS" ]]; then
      echo "Pilot pairdata already complete: $PILOT_PAIRDATA"
      exit 0
    else
      resume=()
    fi
    "$PY" "$CODE/extract_pairdata.py" \
      --mode pilot_train \
      --manifest "$MANIFESTS/api_fullseq_v3_pilot_train_all_series.csv" \
      --base-config "$BASE_CONFIG" \
      --override-config "$OVERRIDE_CONFIG" \
      --output-root "$PILOT_PAIRDATA" \
      --device cuda:0 \
      --cache-size "$CACHE_SIZE" \
      --num-workers 4 \
      --max-visual-pairs 1 \
      "${resume[@]}"
    ;;

  pilot_features)
    test -f "$PILOT_PAIRDATA/.SUCCESS"
    "$PY" "$CODE/build_features.py" \
      --mode pilot_train \
      --manifest "$MANIFESTS/api_fullseq_v3_pilot_train_all_series.csv" \
      --pairdata-root "$PILOT_PAIRDATA" \
      --output-dir "$PILOT_FEATURES" \
      --overwrite
    ;;

  pilot_audit)
    test -f "$PILOT_FEATURES/.FEATURES_SUCCESS"
    "$PY" "$CODE/audit_pilot.py" \
      --manifest "$MANIFESTS/api_fullseq_v3_pilot_train_all_series.csv" \
      --pairdata-root "$PILOT_PAIRDATA" \
      --feature-dir "$PILOT_FEATURES" \
      --output-dir "$PILOT_REPORT" \
      --max-projected-cache-gib 30 \
      --overwrite
    echo
    echo "MANUAL GATE: inspect representative images under:"
    echo "  $PILOT_PAIRDATA/*/*/*/visualizations/"
    echo "and read:"
    echo "  $PILOT_REPORT/pilot_engineering_audit.md"
    ;;

  full_train)
    test -f "$MANIFESTS/.MANIFESTS_FROZEN_SUCCESS"
    if [[ -f "$PILOT_REPORT/.PILOT_ENGINEERING_SUCCESS" && "${PILOT_VISUAL_APPROVED:-NO}" = "YES" ]]; then
      echo "Using completed Pilot gate."
    elif [[ "${ALLOW_FULL_WITHOUT_PILOT:-NO}" = "YES" ]]; then
      echo "[NOTICE] Full extraction explicitly authorized without real-image Pilot."
      "$PY" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
path=Path("$REPORT/.FULL_WITHOUT_PILOT_AUTHORIZATION.json")
path.write_text(json.dumps({
  "authorized_utc": datetime.now(timezone.utc).isoformat(),
  "allow_full_without_pilot": True,
  "reason": "User explicitly requested immediate Full Train and Full Valid.",
  "synthetic_check_required": True,
  "manifest_freeze_required": True
}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
PY
    else
      echo "Need Pilot approval or ALLOW_FULL_WITHOUT_PILOT=YES." >&2
      exit 3
    fi
    resume=()
    [[ -d "$TRAIN_PAIRDATA" ]] && resume=(--resume)
    "$PY" "$CODE/extract_pairdata.py" \
      --mode full_train \
      --manifest "$MANIFESTS/api_fullseq_v3_train_all_series_frozen.csv" \
      --base-config "$BASE_CONFIG" \
      --override-config "$OVERRIDE_CONFIG" \
      --output-root "$TRAIN_PAIRDATA" \
      --device cuda:0 \
      --cache-size "$CACHE_SIZE" \
      --num-workers 4 \
      --max-visual-pairs 0 \
      "${resume[@]}"
    ;;

  full_train_features)
    test -f "$TRAIN_PAIRDATA/.SUCCESS"
    "$PY" "$CODE/build_features.py" \
      --mode full_train \
      --manifest "$MANIFESTS/api_fullseq_v3_train_all_series_frozen.csv" \
      --pairdata-root "$TRAIN_PAIRDATA" \
      --output-dir "$TRAIN_FEATURES" \
      --overwrite
    ;;

  freeze_release)
    test -f "$TRAIN_FEATURES/.FEATURES_SUCCESS"
    "$PY" "$CODE/freeze_release.py" \
      --project "$PROJECT" \
      --extractor code/api_fullseq_v3/extract_pairdata.py \
      --builder code/api_fullseq_v3/build_features.py \
      --base-config "${BASE_CONFIG#$PROJECT/}" \
      --override-config "${OVERRIDE_CONFIG#$PROJECT/}" \
      --train-manifest manifests/api_fullseq_v3_train_all_series_frozen.csv \
      --valid-manifest manifests/api_fullseq_v3_valid_all_series_frozen.csv \
      --train-pairdata outputs/api_fullseq_v3_pairdata/full/train \
      --train-features outputs/api_fullseq_v3_features/full/train \
      --report-dir reports/api_fullseq_v3_reextract \
      --overwrite
    ;;

  full_valid)
    test -f "$REPORT/.FULL_TRAIN_FEATURES_SUCCESS"
    resume=()
    [[ -d "$VALID_PAIRDATA" ]] && resume=(--resume)
    "$PY" "$CODE/extract_pairdata.py" \
      --mode full_valid \
      --manifest "$MANIFESTS/api_fullseq_v3_valid_all_series_frozen.csv" \
      --base-config "$BASE_CONFIG" \
      --override-config "$OVERRIDE_CONFIG" \
      --output-root "$VALID_PAIRDATA" \
      --device cuda:0 \
      --cache-size "$CACHE_SIZE" \
      --num-workers 4 \
      --max-visual-pairs 0 \
      "${resume[@]}"
    ;;

  full_valid_features)
    test -f "$VALID_PAIRDATA/.SUCCESS"
    "$PY" "$CODE/build_features.py" \
      --mode full_valid \
      --manifest "$MANIFESTS/api_fullseq_v3_valid_all_series_frozen.csv" \
      --pairdata-root "$VALID_PAIRDATA" \
      --output-dir "$VALID_FEATURES" \
      --overwrite
    "$PY" - <<PY
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
root=Path("$REPORT")
valid=Path("$VALID_FEATURES")
payload={
  "finished_utc": datetime.now(timezone.utc).isoformat(),
  "valid_feature_success": str(valid/".FEATURES_SUCCESS"),
  "valid_feature_success_sha256": hashlib.sha256((valid/".FEATURES_SUCCESS").read_bytes()).hexdigest(),
  "train_release_freeze": str(root/"train_release_freeze.json"),
}
(root/".FULL_VALID_FEATURES_SUCCESS").write_text(json.dumps(payload,indent=2)+"\n")
(root/".FULL_TRAIN_VALID_SUCCESS").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
PY
    ;;

  build_tasks)
    test -f "$TRAIN_FEATURES/.FEATURES_SUCCESS"
    test -f "$VALID_FEATURES/.FEATURES_SUCCESS"
    "$PY" "$CODE/build_prediction_tasks.py" \
      --project "$PROJECT" \
      --train-feature-dir "$TRAIN_FEATURES" \
      --valid-feature-dir "$VALID_FEATURES" \
      --output-dir "$TASKS" \
      --overwrite
    ;;

  train_models)
    test -f "$TASKS/.TASKS_SUCCESS"
    "$PY" "$CODE/train_prediction_models.py" \
      --task-root "$TASKS" \
      --output-dir "$MODELS" \
      --device cuda:0 \
      --overwrite
    ;;

  full_auto)
    export ALLOW_FULL_WITHOUT_PILOT=YES
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" check
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" freeze_manifests
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" full_train
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" full_train_features
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" freeze_release
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" full_valid
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" full_valid_features
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" build_tasks
    bash "$PROJECT/run_api_fullseq_v3_reextract.sh" train_models
    "$PY" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
report=Path("$REPORT")
payload={
  "finished_utc": datetime.now(timezone.utc).isoformat(),
  "full_train_valid_success": str(report/".FULL_TRAIN_VALID_SUCCESS"),
  "tasks_success": str(Path("$TASKS")/".TASKS_SUCCESS"),
  "models_success": str(Path("$MODELS")/".MODELS_SUCCESS"),
  "manual_pilot_skipped": True,
  "valid_used_for_training": False
}
(report/".FULL_AUTO_WITH_MODELS_SUCCESS").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
    ;;

  status)
    echo "Manifest gate:"; ls -l "$MANIFESTS/.MANIFESTS_FROZEN_SUCCESS" 2>/dev/null || true
    echo "Pilot:"; ls -l "$PILOT_PAIRDATA/.SUCCESS" "$PILOT_FEATURES/.FEATURES_SUCCESS" "$PILOT_REPORT/.PILOT_ENGINEERING_SUCCESS" 2>/dev/null || true
    echo "Train:"; ls -l "$TRAIN_PAIRDATA/.SUCCESS" "$TRAIN_FEATURES/.FEATURES_SUCCESS" "$REPORT/.FULL_TRAIN_FEATURES_SUCCESS" 2>/dev/null || true
    echo "Valid:"; ls -l "$VALID_PAIRDATA/.SUCCESS" "$VALID_FEATURES/.FEATURES_SUCCESS" "$REPORT/.FULL_TRAIN_VALID_SUCCESS" 2>/dev/null || true
    echo "Tasks/Models:"; ls -l "$TASKS/.TASKS_SUCCESS" "$MODELS/.MODELS_SUCCESS" "$REPORT/.FULL_AUTO_WITH_MODELS_SUCCESS" 2>/dev/null || true
    pgrep -af 'api_fullseq_v3/(extract_pairdata|build_features|build_prediction_tasks|train_prediction_models)' || true
    ;;

  *) usage; exit 2 ;;
esac
