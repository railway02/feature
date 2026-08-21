#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/dsa_registration_local_reference_v1
RUN_ID=${1:-local_reference_v1_20260819_overnight}
OUT="$ROOT/outputs/$RUN_ID"
mkdir -p "$OUT/logs"
export PYTHONUNBUFFERED=1
# registration_guard holds an OS flock for its complete lifetime and owns the
# only resource monitor and supervisor child tree for this RUN_ID.
setsid python "$ROOT/scripts/registration_guard.py" --run-id "$RUN_ID" --workers 4 > "$OUT/logs/registration.log" 2>&1 < /dev/null &
REG_PID=$!
sleep 1
if ! kill -0 "$REG_PID" 2>/dev/null; then
  echo "Registration RUN_ID is already locked: $RUN_ID" >&2
  exit 73
fi
MON_PID=$(pgrep -P "$REG_PID" -f 'resource_monitor.sh' || true)
setsid python "$ROOT/scripts/gpu_3167_new_audit.py" "$RUN_ID" > "$OUT/logs/gpu_3167_new.log" 2>&1 < /dev/null &
python - "$OUT/BACKGROUND_RUN_INFO.json" "$REG_PID" "$MON_PID" "$RUN_ID" <<'PY'
import json,sys
p=sys.argv[1]
try: d=json.load(open(p))
except: d={}
d.update({'run_id':sys.argv[4],'registration_pid':int(sys.argv[2]),'resource_monitor_pid':int(sys.argv[3]),'registration_command':'scripts/overnight_supervisor.py --run-id '+sys.argv[4]+' --workers 8','resume_command':'scripts/launch_overnight_registration.sh '+sys.argv[4]})
open(p,'w').write(json.dumps(d,indent=2)+'\n')
PY
echo "$REG_PID $MON_PID"
