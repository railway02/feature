#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT:-/root/autodl-tmp/aneurysm}
CODE=${CODE:-$ROOT/code/api_png2d_gtmask_roi_cave_v1_fullseq}
CONFIG=${CONFIG:-$ROOT/configs/api_png2d_gtmask_roi_cave_v1_fullseq.json}
PY=${PYTHON:-python}
CAVE_PY=${CAVE_PYTHON:-/root/autodl-tmp/envs/cave-dsa/bin/python}

stage=${1:-}
case "$stage" in
  index)
    "$PY" "$CODE/00_index_source_phases.py" --config "$CONFIG"
    ;;
  discover)
    "$PY" "$CODE/01_audit_png2d_mapping.py" --config "$CONFIG"
    ;;
  map)
    "$PY" "$CODE/02_import_png2d_mapping.py" --config "$CONFIG"
    ;;
  roi-eligible)
    "$PY" "$CODE/03_build_png2d_roi_manifests.py" --config "$CONFIG"
    ;;
  qa)
    "$PY" "$CODE/04_make_png2d_roi_qa.py" --config "$CONFIG"
    ;;
  validate-stage1-eligible)
    "$PY" "$CODE/05_validate_png2d_stage1.py" --config "$CONFIG"
    ;;
  stage1)
    "$PY" "$CODE/00_index_source_phases.py" --config "$CONFIG"
    "$PY" "$CODE/01_audit_png2d_mapping.py" --config "$CONFIG"
    "$PY" "$CODE/02_import_png2d_mapping.py" --config "$CONFIG"
    "$PY" "$CODE/03_build_png2d_roi_manifests.py" --config "$CONFIG"
    "$PY" "$CODE/04_make_png2d_roi_qa.py" --config "$CONFIG"
    "$PY" "$CODE/05_validate_png2d_stage1.py" --config "$CONFIG"
    ;;
  smoke-train)
    extra=()
    [[ -n "${SMOKE_UIDS_TRAIN:-}" ]] && extra+=(--series-uids-file "$SMOKE_UIDS_TRAIN")
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Train --smoke "${extra[@]}"
    "$PY" "$CODE/smoke_verify.py" --config "$CONFIG" --split Train
    ;;
  smoke-valid)
    extra=()
    [[ -n "${SMOKE_UIDS_VALID:-}" ]] && extra+=(--series-uids-file "$SMOKE_UIDS_VALID")
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Valid --smoke "${extra[@]}"
    "$PY" "$CODE/smoke_verify.py" --config "$CONFIG" --split Valid
    ;;
  extract-train-eligible)
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Train
    ;;
  extract-valid-eligible)
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Valid
    ;;
  extract-all-eligible)
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Train
    "$CAVE_PY" "$CODE/07_run_full_extraction.py" --config "$CONFIG" --split Valid
    ;;
  validate-features-eligible)
    "$PY" "$CODE/08_finalize_runtime_exclusions.py" --config "$CONFIG"
    "$PY" "$CODE/08_validate_full_featurebank.py" --config "$CONFIG" --split All
    ;;
  table-train-eligible)
    "$CAVE_PY" "$CODE/09_build_full_local_tables.py" --config "$CONFIG" --split Train
    ;;
  table-valid-eligible)
    "$CAVE_PY" "$CODE/09_build_full_local_tables.py" --config "$CONFIG" --split Valid
    ;;
  build-tables-eligible)
    "$CAVE_PY" "$CODE/09_build_full_local_tables.py" --config "$CONFIG" --split Train
    "$CAVE_PY" "$CODE/09_build_full_local_tables.py" --config "$CONFIG" --split Valid
    ;;
  all)
    "$0" stage1
    "$0" smoke-train
    "$0" smoke-valid
    "$0" extract-all-eligible
    "$0" validate-features-eligible
    "$0" build-tables-eligible
    ;;
  status)
    echo "== stage1 eligible =="
    ls -l "$ROOT/manifests/api_png2d_gtmask_roi_cave_v1_fullseq/.STAGE1_ELIGIBLE_SUCCESS.json" 2>/dev/null || true
    echo "== eligible featurebank =="
    ls -l "$ROOT/outputs/api_png2d_gtmask_roi_cave_v1_fullseq/cave_local_eligible_featurebank/.ELIGIBLE_FEATUREBANK_SUCCESS.json" 2>/dev/null || true
    echo "== running processes =="
    pgrep -af 'api_png2d_gtmask_roi_cave_v1_fullseq' || true
    ;;
  *)
    echo "Usage: $0 {index|discover|map|roi-eligible|qa|validate-stage1-eligible|stage1|smoke-train|smoke-valid|extract-train-eligible|extract-valid-eligible|extract-all-eligible|validate-features-eligible|table-train-eligible|table-valid-eligible|build-tables-eligible|all|status}" >&2
    exit 2
    ;;
esac
