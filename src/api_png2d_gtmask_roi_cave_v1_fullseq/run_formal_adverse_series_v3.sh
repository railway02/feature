#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/aneurysm}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}"
CODE_DIR="${CODE_DIR:-$PROJECT/code/api_png2d_gtmask_roi_cave_v1_fullseq}"
TRAIN_XLSX="${TRAIN_XLSX:-$PROJECT/metadata/Train.xlsx}"
VALID_XLSX="${VALID_XLSX:-$PROJECT/metadata/valid.xlsx}"
TABLE_ROOT="${TABLE_ROOT:-$PROJECT/outputs/api_png2d_gtmask_roi_cave_v1_fullseq/tables/local_eligible}"
TASK_ROOT="${TASK_ROOT:-$PROJECT/outputs/api_png2d_gtmask_roi_cave_v1_fullseq/adverse_prepost_series_task_v3}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT/outputs/api_png2d_gtmask_roi_cave_v1_fullseq/adverse_prepost_series_formal_models_v3}"
MAPPING_DIR="${MAPPING_DIR:-$PROJECT/manifests/api_record_v1_all_series_14e/final_mapping}"
MAPPING_INPUT_DIR="${MAPPING_INPUT_DIR:-$PROJECT/manifests/api_record_v1}"
MAPPING_FINALIZER="${MAPPING_FINALIZER:-}"
AUDIT_XLSX="${AUDIT_XLSX:-}"
REQUIRE_COMPLETE_MAPPING="${REQUIRE_COMPLETE_MAPPING:-0}"
DEVICE="${DEVICE:-cuda:0}"
CPU_THREADS="${CPU_THREADS:-8}"
MLP_SEEDS="${MLP_SEEDS:-3}"
MLP_SEARCH_SEEDS="${MLP_SEARCH_SEEDS:-2}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-2000}"
DISABLE_AMP="${DISABLE_AMP:-0}"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

MAPPING_PREFLIGHT="$CODE_DIR/00_mapping_preflight_v3.py"
BUILD_SCRIPT="$CODE_DIR/10_build_adverse_prepost_series_task_v3.py"
TRAIN_SCRIPT="$CODE_DIR/11_train_adverse_prepost_series_formal_v3.py"
SUMMARY_SCRIPT="$CODE_DIR/12_summarize_adverse_series_results_v3.py"
TEST_SCRIPT="$CODE_DIR/test_formal_adverse_series_pipeline_v3.py"

usage() {
  cat <<EOF
Usage:
  bash $0 mapping-preflight
  bash $0 mapping-finalize
  bash $0 preflight
  bash $0 test
  bash $0 build [--overwrite]
  bash $0 cohort-check
  bash $0 train [--overwrite]
  bash $0 status
  bash $0 summarize

Prediction unit: series_uid
Leakage-control group: patient_id
Strict image policy: both Pre and Post required
EOF
}

mapping_preflight() {
  extra=()
  if [[ -n "$MAPPING_DIR" ]]; then
    extra+=(--mapping-dir "$MAPPING_DIR")
  fi
  "$PYTHON" "$MAPPING_PREFLIGHT" \
    --project "$PROJECT" \
    "${extra[@]}"
}

mapping_finalize() {
  required=(
    train_record_table.csv
    valid_record_table.csv
    train_all_series_manifest.csv
    valid_all_series_manifest.csv
    train_record_series_suggestions.csv
    valid_record_series_suggestions.csv
  )
  for name in "${required[@]}"; do
    file="$MAPPING_INPUT_DIR/$name"
    echo "MAPPING_INPUT: exists=$([[ -f "$file" ]] && echo true || echo false) path=$file"
    [[ -f "$file" ]] || {
      echo "Missing mapping input: $file" >&2
      exit 2
    }
  done

  finalizer="$MAPPING_FINALIZER"
  if [[ -z "$finalizer" ]]; then
    mapfile -t candidates < <(
      find "$PROJECT/code" /root/autodl-tmp \
        -maxdepth 8 \
        -type f \
        -name "16_finalize_api_record_v1_mapping.py" \
        -print 2>/dev/null | sort -u
    )
    if [[ "${#candidates[@]}" -ne 1 ]]; then
      printf 'Mapping finalizer candidates:\n'
      printf '  %s\n' "${candidates[@]:-<none>}"
      echo "Set MAPPING_FINALIZER explicitly." >&2
      exit 2
    fi
    finalizer="${candidates[0]}"
  fi

  audit="$AUDIT_XLSX"
  if [[ -z "$audit" ]]; then
    mapfile -t candidates < <(
      find "$PROJECT" /root/autodl-tmp \
        -maxdepth 8 \
        -type f \
        -name "current_vs_train_valid_manual_audit.xlsx" \
        -print 2>/dev/null | sort -u
    )
    if [[ "${#candidates[@]}" -ne 1 ]]; then
      printf 'Audit workbook candidates:\n'
      printf '  %s\n' "${candidates[@]:-<none>}"
      echo "Set AUDIT_XLSX explicitly." >&2
      exit 2
    fi
    audit="${candidates[0]}"
  fi

  echo "MAPPING_FINALIZER=$finalizer"
  echo "AUDIT_XLSX=$audit"
  echo "MAPPING_INPUT_DIR=$MAPPING_INPUT_DIR"
  echo "MAPPING_DIR=$MAPPING_DIR"

  mkdir -p "$MAPPING_DIR"
  extra=()
  if [[ "$REQUIRE_COMPLETE_MAPPING" == "1" ]]; then
    extra+=(--require-complete)
  fi

  "$PYTHON" "$finalizer" \
    --input-dir "$MAPPING_INPUT_DIR" \
    --audit-xlsx "$audit" \
    --output-dir "$MAPPING_DIR" \
    --overwrite \
    "${extra[@]}"

  test -f "$MAPPING_DIR/train_record_series_map.csv"
  test -f "$MAPPING_DIR/valid_record_series_map.csv"
  echo "MAPPING_FINALIZE_OK"
}

preflight() {
  "$PYTHON" - <<PY
import importlib
import os
from pathlib import Path
for module in ("numpy", "pandas", "sklearn", "torch", "openpyxl", "joblib"):
    importlib.import_module(module)
from sklearn.model_selection import StratifiedGroupKFold
import torch

print("PYTHON_IMPORTS_OK")
print("CPU_COUNT:", os.cpu_count())
print("CPU_THREADS:", r"$CPU_THREADS")
print("TORCH:", torch.__version__)
print("CUDA_AVAILABLE:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU_MEMORY_GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )

paths = {
    "MAPPING_PREFLIGHT": Path(r"$MAPPING_PREFLIGHT"),
    "BUILD_SCRIPT": Path(r"$BUILD_SCRIPT"),
    "TRAIN_SCRIPT": Path(r"$TRAIN_SCRIPT"),
    "SUMMARY_SCRIPT": Path(r"$SUMMARY_SCRIPT"),
    "TEST_SCRIPT": Path(r"$TEST_SCRIPT"),
    "TRAIN_XLSX": Path(r"$TRAIN_XLSX"),
    "VALID_XLSX": Path(r"$VALID_XLSX"),
    "TRAIN_EMBEDDING": Path(r"$TABLE_ROOT/train/series_embeddings_5120.npz"),
    "VALID_EMBEDDING": Path(r"$TABLE_ROOT/valid/series_embeddings_5120.npz"),
    "TRAIN_SCALAR": Path(r"$TABLE_ROOT/train/series_scalar_features.csv"),
    "VALID_SCALAR": Path(r"$TABLE_ROOT/valid/series_scalar_features.csv"),
}
for name, path in paths.items():
    print(f"{name}: exists={path.exists()} path={path}")
    if not path.exists():
        raise SystemExit(f"MISSING: {name}: {path}")

mapping = Path(r"$MAPPING_DIR") if r"$MAPPING_DIR".strip() else None
if mapping is None:
    candidates = sorted({
        path.parent
        for path in Path(r"$PROJECT/manifests").rglob("train_record_series_map.csv")
        if (path.parent / "valid_record_series_map.csv").is_file()
    })
    print("MAPPING candidates:")
    for candidate in candidates:
        print(" ", candidate)
    if len(candidates) != 1:
        raise SystemExit("Set MAPPING_DIR to the finalized mapping directory.")
    mapping = candidates[0]

for name in ("train_record_series_map.csv", "valid_record_series_map.csv"):
    path = mapping / name
    print(f"MAPPING: exists={path.exists()} path={path}")
    if not path.exists():
        raise SystemExit(f"MISSING mapping: {path}")

if r"$DEVICE".startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA device requested but CUDA is unavailable")
print("PREFLIGHT_OK")
PY
}

run_test() {
  "$PYTHON" "$TEST_SCRIPT"
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

cohort_check() {
  "$PYTHON" - <<PY
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(r"$TASK_ROOT")
summary_path = root / "task_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Task not built: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print(json.dumps(summary, ensure_ascii=False, indent=2))

for split in ("train", "valid"):
    audit = pd.read_csv(
        root / f"{split}_series_inclusion_audit.csv",
        dtype=str,
        keep_default_na=False,
    )
    samples = pd.read_csv(
        root / f"{split}_series_samples.csv",
        dtype=str,
        keep_default_na=False,
    )
    with np.load(root / f"{split}_features.npz", allow_pickle=False) as raw:
        deep = raw["deep"]
        target = raw["target"]
        uid = raw["series_uid"].astype(str)
        patient = raw["patient_id"].astype(str)
    print("\\n", split.upper())
    print("series audit:", len(audit))
    print("included samples:", len(samples))
    print("deep shape:", deep.shape)
    print("positive:", int(target.sum()))
    print("patients:", len(set(patient.tolist())))
    print("unique series:", len(set(uid.tolist())))
    print("deep finite:", bool(np.isfinite(deep).all()))
    print("series exclusions:")
    excluded = audit[audit["series_status"] == "excluded"]
    print(excluded["series_exclusion_reason"].value_counts().to_string())
    assert deep.shape == (len(samples), 10240)
    assert len(set(uid.tolist())) == len(uid)
    assert np.isfinite(deep).all()

folds = pd.read_csv(root / "train_grouped_folds.csv", dtype={"patient_id": str})
print("\\nFOLD BALANCE")
print(
    folds.groupby("fold").agg(
        series=("series_uid", "size"),
        patients=("patient_id", "nunique"),
        positive=("target", "sum"),
    ).to_string()
)
assert folds.groupby("patient_id")["fold"].nunique().max() == 1
print("COHORT_CHECK_OK")
PY
}

train_models() {
  extra=()
  if [[ "$DISABLE_AMP" == "1" ]]; then
    extra+=(--disable-amp)
  fi
  "$PYTHON" "$TRAIN_SCRIPT" \
    --task-root "$TASK_ROOT" \
    --output-dir "$MODEL_ROOT" \
    --device "$DEVICE" \
    --cpu-threads "$CPU_THREADS" \
    --mlp-seeds "$MLP_SEEDS" \
    --mlp-search-seeds "$MLP_SEARCH_SEEDS" \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    "${extra[@]}" \
    "$@"
}

status() {
  echo "===== MAPPING ====="
  mapping_preflight || true
  echo
  echo "===== TASK ====="
  if [[ -f "$TASK_ROOT/task_summary.json" ]]; then
    cat "$TASK_ROOT/task_summary.json"
  else
    echo "not built: $TASK_ROOT"
  fi
  echo
  echo "===== BASE MODEL FOLDS ====="
  find "$MODEL_ROOT/folds" -name ".SUCCESS.json" 2>/dev/null | wc -l
  echo
  echo "===== MODELS ====="
  if [[ -f "$MODEL_ROOT/summary.json" ]]; then
    cat "$MODEL_ROOT/summary.json"
  else
    echo "not complete: $MODEL_ROOT"
  fi
  echo
  echo "===== RUNNING ====="
  pgrep -af "11_train_adverse_prepost_series_formal_v3.py" || true
}

summarize() {
  "$PYTHON" "$SUMMARY_SCRIPT" \
    --task-root "$TASK_ROOT" \
    --model-root "$MODEL_ROOT"
}

command="${1:-}"
shift || true
case "$command" in
  mapping-preflight)
    mapping_preflight
    ;;
  mapping-finalize)
    mapping_finalize
    ;;
  preflight)
    preflight
    ;;
  test)
    preflight
    run_test
    ;;
  build)
    preflight
    build_task "$@"
    ;;
  cohort-check)
    cohort_check
    ;;
  train)
    preflight
    train_models "$@"
    ;;
  status)
    status
    ;;
  summarize)
    summarize
    ;;
  *)
    usage
    exit 2
    ;;
esac
