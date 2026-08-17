#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_png2d_spatial_backbones_v6_strict
CONFIG=/root/autodl-tmp/aneurysm/configs/api_png2d_spatial_backbones_v6_strict.json
OUT=/root/autodl-tmp/aneurysm/outputs/api_png2d_spatial_backbones_v6_strict
export TORCH_HOME=/root/autodl-tmp/envs/png2d-spatial-v6/torch_cache

stage=${1:-}
case "$stage" in
  preflight) "$PYTHON" "$CODE/00_preflight_expanded_strict.py" --config "$CONFIG" ;;
  smoke) "$PYTHON" "$CODE/01_smoke_expanded_strict_backbones.py" --config "$CONFIG" --device cuda:0 ;;
  train)
    for family in segresnet deeplabv3plus_resnet50_imagenet; do
      for fold in 1 2 3 4 5; do
        "$PYTHON" "$CODE/03_train_strict_segmentation.py" --config "$CONFIG" --family "$family" --fold "$fold" --device cuda:0
      done
    done
    ;;
  audit)
    for family in segresnet deeplabv3plus_resnet50_imagenet; do "$PYTHON" "$CODE/04_audit_strict_segmentation.py" --config "$CONFIG" --family "$family" --device cuda:0; done ;;
  extract)
    for family in segresnet deeplabv3plus_resnet50_imagenet; do "$PYTHON" "$CODE/05_extract_spatial_features.py" --config "$CONFIG" --family "$family" --device cuda:0; done ;;
  featurebanks)
    for family in segresnet deeplabv3plus_resnet50_imagenet; do "$PYTHON" "$CODE/06_build_unified_featurebanks.py" --config "$CONFIG" --family "$family"; "$PYTHON" "$CODE/07_verify_featurebanks.py" --config "$CONFIG" --family "$family"; done ;;
  fusion)
    for family in segresnet deeplabv3plus_resnet50_imagenet; do "$PYTHON" "$CODE/08_run_frozen_teacher_fusion.py" --config "$CONFIG" --family "$family" --device cuda:0; done ;;
  report) "$PYTHON" "$CODE/09_build_final_report.py" --config "$CONFIG" ;;
  deployment)
    "$PYTHON" "$CODE/10_preflight_all2d_segmentation.py" --config "$CONFIG" --device cuda:0
    "$PYTHON" "$CODE/11_train_all2d_segmentation.py" --config "$CONFIG" --family all --device cuda:0
    ;;
  start-full-strict)
    mkdir -p "$OUT/logs"
    nohup setsid bash "$0" train > "$OUT/logs/expanded_strict_full_train.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$OUT/logs/expanded_strict_full_train.pid"
    printf 'started pid=%s log=%s\n' "$pid" "$OUT/logs/expanded_strict_full_train.log"
    ;;
  *) echo "Usage: $0 {preflight|smoke|train|audit|extract|featurebanks|fusion|report|deployment|start-full-strict}"; exit 2 ;;
esac
