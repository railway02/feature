#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/aneurysm
CODE="$ROOT/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready"
PY=/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python
RUNTIME="$ROOT/outputs/api_png2d_segresnet_cave_fusion_v5_strict_2d_oracles_runtime"

mkdir -p "$RUNTIME/logs"

run_experiment() {
  local name="$1"
  local config="$2"
  local output="$3"
  local report="$4"

  printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$RUNTIME/supervisor.log"
  "$PY" "$CODE/03_train_fusion.py" --config "$config" --mode all --device cuda:0 \
    2>&1 | tee "$RUNTIME/logs/${name}.log"
  "$PY" "$CODE/04_summarize.py" --config "$config" \
    2>&1 | tee -a "$RUNTIME/logs/${name}.log"
  "$PY" -c "
import json
from pathlib import Path
import pandas as pd
root = Path('$output') / 'fusion'
mode = next(root.iterdir())
m = json.loads((mode / 'metrics.json').read_text())
oof = pd.read_csv(mode / 'train_oof_predictions.csv')
valid = pd.read_csv(mode / 'valid_predictions.csv')
fold = pd.read_csv(mode / 'fold_metrics.csv')
assert len(oof) == 781 and oof.series_uid.nunique() == 781 and oof.probability.notna().all()
assert len(valid) == 207 and valid.series_uid.nunique() == 207 and valid.probability.notna().all()
assert len(fold) == 5 and set(fold.fold) == {1,2,3,4,5}
Path('$output/.SUCCESS.json').write_text(json.dumps({'status':'success','experiment':'$name','mode':m['mode'],'oof_rows':len(oof),'valid_rows':len(valid),'folds':5}, indent=2)+'\\n')
Path('$report/.SUCCESS.json').write_text(json.dumps({'status':'success','experiment':'$name'}, indent=2)+'\\n')
"
  printf '[%s] COMPLETE %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$RUNTIME/supervisor.log"
}

run_experiment \
  predroi_2d_only \
  "$ROOT/configs/api_png2d_segresnet_cave_fusion_v5_strict_predroi_2d_only.json" \
  "$ROOT/outputs/api_png2d_segresnet_cave_fusion_v5_strict_predroi_2d_only" \
  "$ROOT/reports/api_png2d_segresnet_cave_fusion_v5_strict_predroi_2d_only"

run_experiment \
  gtroi_2d_only \
  "$ROOT/configs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only.json" \
  "$ROOT/outputs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only" \
  "$ROOT/reports/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only"

run_experiment \
  gtroi_cave_gated_oracle \
  "$ROOT/configs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle.json" \
  "$ROOT/outputs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle" \
  "$ROOT/reports/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle"

printf '[%s] ALL_COMPLETE\n' "$(date -u +%FT%TZ)" | tee -a "$RUNTIME/supervisor.log"
