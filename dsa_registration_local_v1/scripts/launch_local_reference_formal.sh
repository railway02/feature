#!/usr/bin/env bash
# Single launcher for the new formal run.  It does not touch the completed G0 run.
set -euo pipefail
RUN_ID=${1:?usage: launch_local_reference_formal.sh local_reference_outcome_strict796_211_TIMESTAMP}
FOV_ID=${2:?usage: launch_local_reference_formal.sh OUTCOME_RUN_ID fov_sensitivity_TIMESTAMP}
ROOT=/root/autodl-tmp
OUT=$ROOT/aneurysm/outputs/$RUN_ID
FOV=$ROOT/dsa_registration_local_reference_v1/outputs/$FOV_ID
mkdir -p "$OUT/logs" "$FOV/logs"
PY=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
nohup setsid "$PY" "$ROOT/dsa_registration_local_reference_v1/scripts/prepare_strict796_contract.py" --outcome-dir "$OUT" --fov-dir "$FOV" >"$OUT/logs/prepare.log" 2>&1 < /dev/null &
pid=$!; printf '%s\n' "$pid" > "$OUT/logs/prepare.pid"
printf 'prepare_pid=%s\nprepare_log=%s\n' "$pid" "$OUT/logs/prepare.log"
