#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/dsa_registration_local_reference_v1
RUN_ID=${1:?run id required}
OUT="$ROOT/outputs/$RUN_ID"
while true; do
  now=$(date -u +%FT%TZ)
  pilot=$(find "$OUT/stage_d/cases" -name '*.json' 2>/dev/null | wc -l || true)
  train=$(find "$OUT/train/cases" -name "${2:-rigid}.json" 2>/dev/null | wc -l || true)
  valid=$(find "$OUT/valid/cases" -name "${2:-rigid}.json" 2>/dev/null | wc -l || true)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 || true)
  { echo "# Local Reference V1 progress"; echo; echo "- UTC: $now"; echo "- Stage D method-case artifacts: $pilot / 40"; echo "- Train primary completed: $train / 800"; echo "- Valid primary completed: $valid / 211"; echo "- GPU: ${gpu:-unavailable}"; echo "- Registration log: outputs/$RUN_ID/logs/registration.log"; } > "$OUT/PROGRESS_LATEST.md"
  sleep 30
done
