#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/aneurysm}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/segresnet-cave-teacher-v5/bin/python}"
CODE_DIR="${CODE_DIR:-$PROJECT/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready}"
CONFIG="${CONFIG:-$PROJECT/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_pilot.json}"
DEVICE="${DEVICE:-cuda:0}"

case "${1:-}" in
  preflight)
    "$PYTHON" "$CODE_DIR/00_preflight.py" --config "$CONFIG"
    ;;

  train-spatial)
    "$PYTHON" "$CODE_DIR/01_train_spatial_encoder.py" \
      --config "$CONFIG" \
      --fold "${2:-all}" \
      --device "$DEVICE"
    ;;

  extract)
    "$PYTHON" "$CODE_DIR/02_extract_spatial_features.py" \
      --config "$CONFIG" \
      --fold "${2:-all}" \
      --device "$DEVICE"
    ;;

  fusion)
    "$PYTHON" "$CODE_DIR/03_train_fusion.py" \
      --config "$CONFIG" \
      --mode "${2:-all}" \
      --device "$DEVICE"
    ;;

  summarize)
    "$PYTHON" "$CODE_DIR/04_summarize.py" --config "$CONFIG"
    ;;

  all)
    "$PYTHON" "$CODE_DIR/00_preflight.py" --config "$CONFIG"
    "$PYTHON" "$CODE_DIR/01_train_spatial_encoder.py" --config "$CONFIG" --fold all --device "$DEVICE"
    "$PYTHON" "$CODE_DIR/02_extract_spatial_features.py" --config "$CONFIG" --fold all --device "$DEVICE"
    "$PYTHON" "$CODE_DIR/03_train_fusion.py" --config "$CONFIG" --mode all --device "$DEVICE"
    "$PYTHON" "$CODE_DIR/04_summarize.py" --config "$CONFIG"
    ;;

  *)
    cat <<EOF
Usage:
  bash $0 preflight
  bash $0 train-spatial [1..5|all]
  bash $0 extract [1..5|all]
  bash $0 fusion [cave_only|spatial_only|concat|interaction|gated_interaction|all]
  bash $0 summarize
  bash $0 all

Strategies (set in config):
  pilot_single        = all 2D inventory pairs with patient-grouped inner epoch selection; fastest teacher pilot
  strict_crossfit     = fold-safe SegResNet representation; formal OOF
  external_checkpoint = do not train SegResNet; load a supplied checkpoint

Spatial representation:
  global_only     = [G_pre,G_post], teacher baseline
  global_gt_roi   = global + GT ROI ablation
  global_pred_roi = global + predicted ROI ablation

Segmentation population:
  pilot_single    = all 2D inventory pairs
  strict_crossfit = other Train outcome folds only
EOF
    exit 2
    ;;
esac
