#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/aneurysm

# ============================================================
# 1. 固定绝对路径
# ============================================================

PY=/root/autodl-tmp/envs/aneurysm-ml/bin/python

TRAIN_FEATURES=/root/autodl-tmp/aneurysm/outputs/api_fullseq_v2_features/full/train_patient_features.csv
VALID_FEATURES=/root/autodl-tmp/aneurysm/outputs/api_fullseq_v2_features/full/valid_patient_features.csv
SCHEMA=/root/autodl-tmp/aneurysm/configs/api_fullseq_v2_feature_schema.json

TRAIN_LABELS=/root/autodl-tmp/aneurysm/metadata/Train.xlsx
VALID_LABELS=/root/autodl-tmp/aneurysm/metadata/valid.xlsx

BUILD_SCRIPT=/root/autodl-tmp/aneurysm/code/23_build_adverse_prepost_fullseq_v2.py
TRAIN_SCRIPT=/root/autodl-tmp/aneurysm/code/24_train_adverse_prepost_fullseq_v2_mlp.py

TASK_DIR=/root/autodl-tmp/aneurysm/outputs/tasks/adverse_prepost_fullseq_v2
RUN_DIR=/root/autodl-tmp/aneurysm/outputs/baselines/adverse_prepost_fullseq_v2_mlp

export TRAIN_FEATURES VALID_FEATURES SCHEMA
export TRAIN_LABELS VALID_LABELS TASK_DIR RUN_DIR

echo "============================================================"
echo "adverse_prepost_fullseq_v2 pipeline"
echo "Started: $(date -Is)"
echo "============================================================"

# ============================================================
# 2. 安全保护
# ============================================================

[[ "$TASK_DIR" == /root/autodl-tmp/aneurysm/outputs/tasks/* ]] || {
    echo "[FATAL] TASK_DIR 不安全：$TASK_DIR"
    exit 1
}

[[ "$RUN_DIR" == /root/autodl-tmp/aneurysm/outputs/baselines/* ]] || {
    echo "[FATAL] RUN_DIR 不安全：$RUN_DIR"
    exit 1
}

if pgrep -af \
  '[2]3_build_adverse_prepost_fullseq_v2.py|[2]4_train_adverse_prepost_fullseq_v2_mlp.py'
then
    echo "[FATAL] 已存在同类构建或训练进程，禁止重复启动"
    exit 1
fi

# ============================================================
# 3. 检查所有输入文件
# ============================================================

echo
echo "========== 输入文件检查 =========="

for file in \
    "$TRAIN_FEATURES" \
    "$VALID_FEATURES" \
    "$SCHEMA" \
    "$TRAIN_LABELS" \
    "$VALID_LABELS" \
    "$BUILD_SCRIPT" \
    "$TRAIN_SCRIPT"
do
    if [[ ! -f "$file" ]]; then
        echo "[FATAL] 文件不存在：$file"
        exit 1
    fi

    if [[ ! -s "$file" ]]; then
        echo "[FATAL] 文件为空：$file"
        exit 1
    fi

    stat -c '[PASS] %n | %s bytes | %y' "$file"
done

# ============================================================
# 4. 语法检查
# ============================================================

echo
echo "========== Python语法检查 =========="

"$PY" -m py_compile \
    "$BUILD_SCRIPT" \
    "$TRAIN_SCRIPT"

echo "[PASS] 两个脚本语法检查通过"

# ============================================================
# 5. 输入规模、Excel表头与schema预检
# ============================================================

echo
echo "========== 输入内容预检 =========="

"$PY" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

train_features_path = Path(os.environ["TRAIN_FEATURES"])
valid_features_path = Path(os.environ["VALID_FEATURES"])
schema_path = Path(os.environ["SCHEMA"])
train_labels_path = Path(os.environ["TRAIN_LABELS"])
valid_labels_path = Path(os.environ["VALID_LABELS"])

train_features = pd.read_csv(
    train_features_path,
    dtype={"patient_id": str},
)
valid_features = pd.read_csv(
    valid_features_path,
    dtype={"patient_id": str},
)

train_excel = pd.read_excel(train_labels_path)
valid_excel = pd.read_excel(valid_labels_path)

schema = json.loads(schema_path.read_text(encoding="utf-8"))

phase_candidates = [
    str(item["feature_name"])
    for item in schema["phase_features"]
    if bool(item.get("model_candidate", False))
]

assert len(train_features) == 1055, train_features.shape
assert len(valid_features) == 264, valid_features.shape

assert train_features["patient_id"].is_unique
assert valid_features["patient_id"].is_unique

assert train_features["split"].astype(str).eq("Train").all()
assert valid_features["split"].astype(str).eq("Valid").all()

assert set(train_features["patient_id"]).isdisjoint(
    set(valid_features["patient_id"])
)

assert len(phase_candidates) == 147, len(phase_candidates)
assert len(set(phase_candidates)) == 147

for split, frame in [
    ("Train", train_excel),
    ("Valid", valid_excel),
]:
    assert "病案号" in frame.columns, (
        split,
        "病案号",
        frame.columns.tolist(),
    )

    adverse_columns = [
        column
        for column in frame.columns
        if "不良转归" in str(column)
    ]

    assert len(adverse_columns) == 1, (
        split,
        adverse_columns,
    )

    print(
        f"{split} Excel: rows={len(frame)}, "
        f"unique patient IDs={frame['病案号'].nunique()}, "
        f"label column={adverse_columns[0]!r}"
    )

print(
    "Train patient features:",
    train_features.shape,
)
print(
    "Valid patient features:",
    valid_features.shape,
)
print(
    "Phase model candidates:",
    len(phase_candidates),
)
print("[PASS] 输入路径、规模、表头和schema预检通过")
PY

# ============================================================
# 6. 备份旧任务目录
# ============================================================

echo
echo "========== 准备任务目录 =========="

if [[ -e "$TASK_DIR" ]]; then
    TASK_BACKUP="${TASK_DIR}.backup.$(date +%Y%m%d-%H%M%S)"
    mv "$TASK_DIR" "$TASK_BACKUP"
    echo "[INFO] 旧任务目录已备份：$TASK_BACKUP"
fi

mkdir -p "$(dirname "$TASK_DIR")"

# ============================================================
# 7. 构建任务CSV
# ============================================================

echo
echo "========== 构建Pre+Post不良转归任务 =========="

"$PY" "$BUILD_SCRIPT" \
    --train-features "$TRAIN_FEATURES" \
    --valid-features "$VALID_FEATURES" \
    --train-labels "$TRAIN_LABELS" \
    --valid-labels "$VALID_LABELS" \
    --schema "$SCHEMA" \
    --output-dir "$TASK_DIR"

# ============================================================
# 8. 任务构建硬验收
# ============================================================

echo
echo "========== 任务构建硬验收 =========="

"$PY" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

task_dir = Path(os.environ["TASK_DIR"])
schema_path = Path(os.environ["SCHEMA"])

required_files = [
    "adverse_prepost_fullseq_v2_train.csv",
    "adverse_prepost_fullseq_v2_valid.csv",
    "adverse_label_conflicts.csv",
    "train_label_records_audit.csv",
    "valid_label_records_audit.csv",
    "task_build_summary.json",
    "task_build_report.md",
    ".SUCCESS",
]

for name in required_files:
    path = task_dir / name
    assert path.is_file(), f"Missing output: {path}"

    if name != ".SUCCESS":
        assert path.stat().st_size > 0, f"Empty output: {path}"

train = pd.read_csv(
    task_dir / "adverse_prepost_fullseq_v2_train.csv",
    dtype={"patient_id": str},
)
valid = pd.read_csv(
    task_dir / "adverse_prepost_fullseq_v2_valid.csv",
    dtype={"patient_id": str},
)
conflicts = pd.read_csv(
    task_dir / "adverse_label_conflicts.csv",
    dtype={"patient_id": str},
)

schema = json.loads(schema_path.read_text(encoding="utf-8"))

phase_candidates = [
    str(item["feature_name"])
    for item in schema["phase_features"]
    if bool(item.get("model_candidate", False))
]

expected_features = (
    [f"pre_{name}" for name in phase_candidates]
    + [f"post_{name}" for name in phase_candidates]
)

assert train.shape == (855, 297), train.shape
assert valid.shape == (226, 297), valid.shape

assert train.columns[:3].tolist() == [
    "patient_id",
    "task_split",
    "adverse",
]
assert valid.columns[:3].tolist() == [
    "patient_id",
    "task_split",
    "adverse",
]

assert train.columns[3:].tolist() == expected_features
assert valid.columns[3:].tolist() == expected_features

assert len(phase_candidates) == 147
assert len(expected_features) == 294

assert train["task_split"].eq("Train").all()
assert valid["task_split"].eq("Valid").all()

assert int(train["adverse"].sum()) == 137
assert int(valid["adverse"].sum()) == 38

assert train["patient_id"].is_unique
assert valid["patient_id"].is_unique

assert set(train["patient_id"]).isdisjoint(
    set(valid["patient_id"])
)

assert not np.isinf(
    train[expected_features].to_numpy(dtype=float)
).any()
assert not np.isinf(
    valid[expected_features].to_numpy(dtype=float)
).any()

conflict_counts = (
    conflicts.groupby("split")["patient_id"]
    .nunique()
    .to_dict()
)

assert conflict_counts.get("Train", 0) == 13, conflict_counts
assert conflict_counts.get("Valid", 0) == 3, conflict_counts

print("[PASS] Pre+Post任务构建硬验收通过")
print(
    "Train:",
    train.shape,
    "positive:",
    int(train["adverse"].sum()),
)
print(
    "Valid:",
    valid.shape,
    "positive:",
    int(valid["adverse"].sum()),
)
print("Feature count:", len(expected_features))
print(
    "Train NaN cells:",
    int(train[expected_features].isna().sum().sum()),
)
print(
    "Valid NaN cells:",
    int(valid[expected_features].isna().sum().sum()),
)
print("Conflicts:", conflict_counts)
PY

cat "$TASK_DIR/task_build_report.md"

# ============================================================
# 9. 备份旧训练目录
# ============================================================

echo
echo "========== 准备MLP输出目录 =========="

if [[ -e "$RUN_DIR" ]]; then
    RUN_BACKUP="${RUN_DIR}.backup.$(date +%Y%m%d-%H%M%S)"
    mv "$RUN_DIR" "$RUN_BACKUP"
    echo "[INFO] 旧训练目录已备份：$RUN_BACKUP"
fi

mkdir -p "$(dirname "$RUN_DIR")"

# ============================================================
# 10. 训练环境
# ============================================================

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42

echo
echo "========== CUDA检查 =========="

"$PY" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

assert torch.cuda.is_available(), "CUDA unavailable"

print("GPU:", torch.cuda.get_device_name(0))
print("[PASS] CUDA环境正常")
PY

# ============================================================
# 11. 正式训练
# ============================================================

echo
echo "========== 正式运行五折OOF MLP =========="

"$PY" "$TRAIN_SCRIPT" \
    --train-csv "$TASK_DIR/adverse_prepost_fullseq_v2_train.csv" \
    --valid-csv "$TASK_DIR/adverse_prepost_fullseq_v2_valid.csv" \
    --output-dir "$RUN_DIR" \
    --device cuda:0

# ============================================================
# 12. 训练硬验收
# ============================================================

echo
echo "========== MLP训练硬验收 =========="

test -f "$RUN_DIR/.SUCCESS" || {
    echo "[FATAL] .SUCCESS missing"
    exit 1
}

test ! -f "$RUN_DIR/.RUNNING" || {
    echo "[FATAL] .RUNNING remains"
    exit 1
}

test -f "$RUN_DIR/exit_status.txt" || {
    echo "[FATAL] exit_status.txt missing"
    exit 1
}

[[ "$(tr -d '[:space:]' < "$RUN_DIR/exit_status.txt")" == "0" ]] || {
    echo "[FATAL] exit_status is not 0"
    exit 1
}

if grep -Eiq 'Traceback|ERROR|FATAL' "$RUN_DIR/run.log"; then
    echo "[FATAL] run.log contains an error"
    grep -Ein 'Traceback|ERROR|FATAL' "$RUN_DIR/run.log"
    exit 1
fi

for file in \
    train_oof_predictions.csv \
    valid_predictions.csv \
    fold_metrics.csv \
    train_oof_metrics.csv \
    valid_metrics.csv \
    bootstrap_confidence_intervals.csv \
    paired_bootstrap_comparisons.csv \
    training_history.csv \
    threshold_search.csv \
    thresholds.json \
    frozen_protocol.json \
    configuration.json \
    reports/report.md
do
    test -s "$RUN_DIR/$file" || {
        echo "[FATAL] Missing or empty: $RUN_DIR/$file"
        exit 1
    }
done

for fold in 1 2 3 4 5
do
    test -s "$RUN_DIR/models/mlp_fold_${fold}.pt" || {
        echo "[FATAL] Missing MLP fold $fold"
        exit 1
    }

    test -s "$RUN_DIR/preprocessors/fold_${fold}_preprocessor.joblib" || {
        echo "[FATAL] Missing preprocessor fold $fold"
        exit 1
    }

    test -s "$RUN_DIR/preprocessors/fold_${fold}_feature_audit.json" || {
        echo "[FATAL] Missing feature audit fold $fold"
        exit 1
    }
done

"$PY" - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

run_dir = Path(os.environ["RUN_DIR"])

oof = pd.read_csv(run_dir / "train_oof_predictions.csv")
valid = pd.read_csv(run_dir / "valid_predictions.csv")
folds = pd.read_csv(run_dir / "fold_metrics.csv")
train_metrics = pd.read_csv(run_dir / "train_oof_metrics.csv")
valid_metrics = pd.read_csv(run_dir / "valid_metrics.csv")

assert len(oof) == 855
assert len(valid) == 226
assert len(folds) == 5

probability_columns_oof = [
    column
    for column in oof.columns
    if column.endswith("_probability")
]
probability_columns_valid = [
    column
    for column in valid.columns
    if column.endswith("_probability")
]

assert probability_columns_oof
assert probability_columns_valid

assert np.isfinite(
    oof[probability_columns_oof].to_numpy(dtype=float)
).all()
assert np.isfinite(
    valid[probability_columns_valid].to_numpy(dtype=float)
).all()

assert set(train_metrics["model"]) == {
    "Dummy",
    "Logistic",
    "Pre+Post MLP",
}
assert set(valid_metrics["model"]) == {
    "Dummy",
    "Logistic",
    "Pre+Post MLP",
}

print("[PASS] MLP结构与输出硬验收通过")

print("\n===== Train pooled OOF metrics =====")
print(train_metrics.to_string(index=False))

print("\n===== Independent Valid metrics =====")
print(valid_metrics.to_string(index=False))

print("\n===== Fold summary =====")
print(folds.to_string(index=False))
PY

echo
echo "============================================================"
echo "[FINAL PASS] 任务构建、五折OOF训练、独立Valid验收全部完成"
echo "Completed: $(date -Is)"
echo "TASK_DIR=$TASK_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "============================================================"
