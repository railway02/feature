#!/usr/bin/env bash
# Resumable formal execution through the strict SegResNet/feature stage.
set -euo pipefail
RUN_ID=${1:?RUN_ID}; FOV_ID=${2:?FOV_ID}
ROOT=/root/autodl-tmp
OUT=$ROOT/aneurysm/outputs/$RUN_ID
FOV=$ROOT/dsa_registration_local_reference_v1/outputs/$FOV_ID
GPU_PY=$ROOT/envs/png2d-spatial-v6/bin/python
CPU_PY=/root/miniconda3/bin/python
mkdir -p "$OUT/logs" "$FOV/logs"
"$GPU_PY" "$ROOT/dsa_registration_local_reference_v1/scripts/prepare_strict796_contract.py" --outcome-dir "$OUT" --fov-dir "$FOV"
"$CPU_PY" "$ROOT/dsa_registration_local_reference_v1/scripts/run_fov50_expanded.py" --out "$FOV" --workers 4 >"$FOV/logs/fov50_expanded.log" 2>&1 &
echo $! >"$FOV/logs/fov50_expanded.pid"
"$GPU_PY" "$ROOT/dsa_registration_local_reference_v1/scripts/strict796_segresnet.py" train --out "$OUT" --device cuda:0
"$GPU_PY" "$ROOT/dsa_registration_local_reference_v1/scripts/strict796_segresnet.py" extract --out "$OUT" --device cuda:0
wait "$(cat "$FOV/logs/fov50_expanded.pid")"
