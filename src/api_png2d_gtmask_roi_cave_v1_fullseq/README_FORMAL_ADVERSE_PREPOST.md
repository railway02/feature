# Formal strict Pre+Post Local-CAVE adverse-outcome training

## Frozen scientific policy

- Primary task: record-level adverse outcome (`不良转归：1是；0否`).
- Prediction unit: `record_uid`.
- Grouping unit: `patient_id`.
- A record is included only when its mapped series has **both Local-CAVE Pre
  and Post embeddings**.
- Pre-only, Post-only and no-phase records are ignored and listed in the
  inclusion audit.
- Multiple lesion records from one patient remain separate records.
- All records from one patient stay in the same Train fold.
- Official Valid is evaluation-only.
- Thresholds come from pooled Train OOF predictions.
- Valid bootstrap resamples patients, not individual rows.

## Models

1. Dummy prior.
2. Logistic Deep: nested Train-only selection of PCA dimension and C.
3. Logistic Fusion: deep PCA plus scalar preprocessing/PCA.
4. MLP Deep: grouped inner early stopping and 3-seed ensemble per outer fold.
5. MLP Fusion: deep plus scalar representation, same grouped protocol.

## Install on AutoDL

Copy the four files into:

```text
/root/autodl-tmp/aneurysm/code/
api_gtmask_roi_cave_v5_fullmask_fullseq/
```

Then set the finalized record mapping directory explicitly:

```bash
export MAPPING_DIR=/root/autodl-tmp/aneurysm/manifests/<FINAL_MAPPING_DIRECTORY>
```

That directory must contain:

```text
train_record_series_map.csv
valid_record_series_map.csv
```

## Preflight and cohort build

```bash
CODE=/root/autodl-tmp/aneurysm/code/api_gtmask_roi_cave_v5_fullmask_fullseq

bash "$CODE/run_formal_adverse_prepost.sh" preflight

bash "$CODE/run_formal_adverse_prepost.sh" build --overwrite
```

Inspect before training:

```bash
TASK=/root/autodl-tmp/aneurysm/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_record_task

cat "$TASK/task_summary.json"

python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/root/autodl-tmp/aneurysm/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_record_task")
for split in ("train", "valid"):
    audit = pd.read_csv(root / f"{split}_record_inclusion_audit.csv")
    print("\n", split.upper())
    print(audit["included"].value_counts(dropna=False))
    print(audit.loc[~audit["included"].astype(bool), "exclusion_reason"].value_counts())
PY
```

The exclusion audit must show that every one-phase sample is excluded with a
`strict_prepost_exclusion_*` reason.

## Background formal training

```bash
CODE=/root/autodl-tmp/aneurysm/code/api_gtmask_roi_cave_v5_fullmask_fullseq
REPORT=/root/autodl-tmp/aneurysm/reports/api_gtmask_roi_cave_v5_fullmask_fullseq
mkdir -p "$REPORT"

LOG="$REPORT/adverse_prepost_formal_$(date +%Y%m%d_%H%M%S).log"

nohup setsid bash -lc '
set -euo pipefail
export MAPPING_DIR="'"$MAPPING_DIR"'"
export PROJECT=/root/autodl-tmp/aneurysm
export PYTHON=/root/autodl-tmp/envs/aneurysm-ml/bin/python
export DEVICE=cuda:0
export MLP_SEEDS=3
export BOOTSTRAP_REPEATS=2000

CODE=/root/autodl-tmp/aneurysm/code/api_gtmask_roi_cave_v5_fullmask_fullseq
bash "$CODE/run_formal_adverse_prepost.sh" train
' > "$LOG" 2>&1 < /dev/null &

echo "PID=$!"
echo "LOG=$LOG"
```

Resume after interruption by running the same `train` command without
`--overwrite`. Completed model/fold predictions are reused.

## Outputs

```text
adverse_prepost_record_task/
  task_summary.json
  train_features.npz
  valid_features.npz
  train_records.csv
  valid_records.csv
  train_grouped_folds.csv
  train_record_inclusion_audit.csv
  valid_record_inclusion_audit.csv

adverse_prepost_formal_models/
  metrics.csv
  train_oof_predictions.csv
  valid_predictions.csv
  fold_audit.csv
  valid_patient_cluster_bootstrap_ci.csv
  valid_paired_model_differences.csv
  report.md
  summary.json
  folds/<model>/fold_<k>/...
```

Do not interpret a high Train in-fold score. The primary internal estimate is
pooled `Train_OOF`; the final external estimate is `Valid`.
