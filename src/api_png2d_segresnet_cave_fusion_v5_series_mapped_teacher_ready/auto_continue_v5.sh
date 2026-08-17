#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/aneurysm
RUN="$PROJECT/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready/run_pipeline.sh"
PILOT_OUT="$PROJECT/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot"
PILOT_REPORT="$PROJECT/reports/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot"
STRICT="$PROJECT/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json"
STRICT_OUT="$PROJECT/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
STRICT_REPORT="$PROJECT/reports/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
STATE="$PROJECT/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_runtime"

mkdir -p "$STATE/logs" "$STATE/pids"

run_stage() {
  local name="$1"
  shift
  local log="$STATE/logs/${name}.log"
  local pid_file="$STATE/pids/${name}.pid"

  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$name" >> "$STATE/supervisor.log"
  setsid nohup "$@" > "$log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  if ! wait "$pid"; then
    printf '%s FAIL %s pid=%s\n' "$(date -u +%FT%TZ)" "$name" "$pid" >> "$STATE/supervisor.log"
    exit 1
  fi
  printf '%s DONE %s pid=%s\n' "$(date -u +%FT%TZ)" "$name" "$pid" >> "$STATE/supervisor.log"
}

while [[ ! -f "$PILOT_OUT/segmentation/pilot/.SUCCESS.json" ]]; do
  sleep 60
done

"/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python" - "$PILOT_REPORT/00_preflight.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload['status'] == 'PASS'
assert all(item['match'] for item in payload['input_locks'].values())
PY

run_stage pilot_extract bash "$RUN" extract all
run_stage pilot_fusion bash "$RUN" fusion all
run_stage pilot_summarize bash "$RUN" summarize

test -f "$PILOT_OUT/seg_features/pilot/.SUCCESS.json"
for mode in cave_only spatial_only concat interaction gated_interaction; do
  test -f "$PILOT_OUT/fusion/$mode/metrics.json"
done
test -f "$PILOT_REPORT/04_summary_metrics.csv"
test -f "$PILOT_REPORT/04_summary.json"
test -f "$PILOT_REPORT/04_summary.md"
"/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python" - "$PILOT_OUT/fusion/gated_interaction/metrics.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload['representation_oof_status'] == 'pilot_not_representation_crossfit'
assert payload['valid_representation_status'] == 'pilot_all_2d_includes_valid_image_mask'
PY

run_stage strict_preflight env CONFIG="$STRICT" bash "$RUN" preflight
run_stage strict_train_spatial env CONFIG="$STRICT" bash "$RUN" train-spatial all
run_stage strict_extract env CONFIG="$STRICT" bash "$RUN" extract all
run_stage strict_fusion env CONFIG="$STRICT" bash "$RUN" fusion all
run_stage strict_summarize env CONFIG="$STRICT" bash "$RUN" summarize

for fold in 1 2 3 4 5; do
  test -f "$STRICT_OUT/segmentation/fold_$fold/.SUCCESS.json"
  test -f "$STRICT_OUT/seg_features/fold_$fold/.SUCCESS.json"
done
for mode in cave_only spatial_only concat interaction gated_interaction; do
  test -f "$STRICT_OUT/fusion/$mode/metrics.json"
  test -f "$STRICT_OUT/fusion/$mode/train_oof_predictions.csv"
  test -f "$STRICT_OUT/fusion/$mode/valid_predictions.csv"
done
test -f "$STRICT_REPORT/04_summary_metrics.csv"
test -f "$STRICT_REPORT/04_summary.json"
test -f "$STRICT_REPORT/04_summary.md"
"/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python" - "$STRICT_OUT/fusion/gated_interaction/metrics.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload['representation_oof_status'] == 'strict_crossfit'
assert payload['valid_representation_status'] == 'strict_train_only_representation'
PY
printf '%s COMPLETE\n' "$(date -u +%FT%TZ)" >> "$STATE/supervisor.log"
