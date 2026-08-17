#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/aneurysm}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}"
CODE_DIR="${CODE_DIR:-$PROJECT/code/api_gtmask_roi_cave_v5_fullmask_fullseq}"
TRAIN_XLSX="${TRAIN_XLSX:-$PROJECT/metadata/Train.xlsx}"
VALID_XLSX="${VALID_XLSX:-$PROJECT/metadata/valid.xlsx}"
TABLE_ROOT="${TABLE_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/tables/local_eligible}"
TASK_ROOT="${TASK_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_record_task}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_formal_models}"
DEVICE="${DEVICE:-cuda:0}"
MLP_SEEDS="${MLP_SEEDS:-3}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-2000}"
MAPPING_DIR="${MAPPING_DIR:-}"

BUILD_SCRIPT="$CODE_DIR/10_build_adverse_prepost_record_task.py"
TRAIN_SCRIPT="$CODE_DIR/11_train_adverse_prepost_formal.py"

usage() {
  cat <<EOF
Usage:
  bash $0 preflight
  bash $0 build [--overwrite]
  bash $0 train [--overwrite]
  bash $0 all [--overwrite]
  bash $0 status

Environment overrides:
  PROJECT, PYTHON, CODE_DIR, TRAIN_XLSX, VALID_XLSX, TABLE_ROOT,
  TASK_ROOT, MODEL_ROOT, DEVICE, MLP_SEEDS, BOOTSTRAP_REPEATS,
  MAPPING_DIR.

For a formal run, set MAPPING_DIR explicitly to the directory containing:
  train_record_series_map.csv
  valid_record_series_map.csv
EOF
}

preflight() {
  "$PYTHON" - <<PY
import importlib
from pathlib import Path
for module in ("numpy", "pandas", "sklearn", "torch", "openpyxl", "joblib"):
    importlib.import_module(module)
print("PYTHON_IMPORTS_OK")

paths = {
    "BUILD_SCRIPT": Path(r"$BUILD_SCRIPT"),
    "TRAIN_SCRIPT": Path(r"$TRAIN_SCRIPT"),
    "TRAIN_XLSX": Path(r"$TRAIN_XLSX"),
    "VALID_XLSX": Path(r"$VALID_XLSX"),
    "TRAIN_TABLE": Path(r"$TABLE_ROOT/train/series_embeddings_5120.npz"),
    "VALID_TABLE": Path(r"$TABLE_ROOT/valid/series_embeddings_5120.npz"),
}
for name, path in paths.items():
    print(f"{name}: exists={path.exists()} path={path}")
    if not path.exists():
        raise SystemExit(f"MISSING: {name}: {path}")

mapping = r"$MAPPING_DIR".strip()
if mapping:
    root = Path(mapping)
    for name in ("train_record_series_map.csv", "valid_record_series_map.csv"):
        path = root / name
        print(f"MAPPING: exists={path.exists()} path={path}")
        if not path.exists():
            raise SystemExit(f"MISSING mapping: {path}")
else:
    candidates = []
    for path in Path(r"$PROJECT/manifests").rglob("train_record_series_map.csv"):
        if (path.parent / "valid_record_series_map.csv").is_file():
            candidates.append(path.parent)
    print("MAPPING_DIR not set. Candidates:")
    for candidate in sorted(set(candidates)):
        print("  ", candidate)
    if len(set(candidates)) != 1:
        raise SystemExit("Set MAPPING_DIR explicitly before the formal build.")
print("PREFLIGHT_OK")
PY
}

build_task() {
  extra=()
  if [[ -n "$MAPPING_DIR" ]]; then
    extra+=(--mapping-dir "$MAPPING_DIR")
  fi
  "$PYTHON" "$BUILD_SCRIPT" \
    --project "$PROJECT" \
    --train-xlsx "$TRAIN_XLSX" \
    --valid-xlsx "$VALID_XLSX" \
    --table-root "$TABLE_ROOT" \
    --output-dir "$TASK_ROOT" \
    "${extra[@]}" \
    "$@"
}

train_models() {
  "$PYTHON" "$TRAIN_SCRIPT" \
    --task-root "$TASK_ROOT" \
    --output-dir "$MODEL_ROOT" \
    --device "$DEVICE" \
    --mlp-seeds "$MLP_SEEDS" \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    "$@"
}

status() {
  echo "===== TASK ====="
  if [[ -f "$TASK_ROOT/task_summary.json" ]]; then
    cat "$TASK_ROOT/task_summary.json"
  else
    echo "not built: $TASK_ROOT"
  fi
  echo
  echo "===== MODELS ====="
  if [[ -f "$MODEL_ROOT/summary.json" ]]; then
    cat "$MODEL_ROOT/summary.json"
  else
    echo "not complete: $MODEL_ROOT"
  fi
  echo
  echo "===== RUNNING ====="
  pgrep -af "11_train_adverse_prepost_formal.py" || true
}

command="${1:-}"
shift || true
case "$command" in
  preflight)
    preflight
    ;;
  build)
    preflight
    build_task "$@"
    ;;
  train)
    preflight
    train_models "$@"
    ;;
  all)
    preflight
    build_task "$@"
    train_models "$@"
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
