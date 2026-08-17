#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/aneurysm}
PYTHON=${PYTHON:-/root/autodl-tmp/envs/aneurysm-ml/bin/python}
CODE="$PROJECT/code/api_gtmask_roi_cave_v5_fullmask_fullseq/matched_ablation_v1"
TASK="$PROJECT/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_matched_ablation_task_v1"
MODELS="$PROJECT/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_matched_ablation_models_v1"

case "${1:-}" in
  freeze)
    "$PYTHON" "$CODE/00_freeze_baseline.py"
    ;;
  build)
    "$PYTHON" "$CODE/01_build_matched_tasks.py"
    ;;
  test)
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$CODE/test_matched_ablation_v1.py"
    ;;
  validate)
    "$PYTHON" "$CODE/04_validate_matched_tasks.py"
    ;;
  smoke-m0|smoke-w0|smoke-wl)
    experiment="${1#smoke-}"
    experiment="${experiment^^}"
    "$PYTHON" "$CODE/02_train_matched_experiment.py" \
      --task-root "$TASK" --output-root "$MODELS" \
      --experiment "$experiment" --mode smoke --fold 1 \
      --device cuda:0 --mlp-seeds 1 --mlp-search-seeds 1
    ;;
  formal-m0|formal-w0|formal-wl)
    experiment="${1#formal-}"
    experiment="${experiment^^}"
    "$PYTHON" "$CODE/02_train_matched_experiment.py" \
      --task-root "$TASK" --output-root "$MODELS" \
      --experiment "$experiment" --mode formal \
      --device cuda:0 --cpu-threads 16 \
      --mlp-seeds 3 --mlp-search-seeds 2 \
      --bootstrap-repeats 2000
    ;;
  *)
    echo "Usage: $0 {freeze|build|test|validate|smoke-m0|smoke-w0|smoke-wl|formal-m0|formal-w0|formal-wl}" >&2
    exit 2
    ;;
esac
