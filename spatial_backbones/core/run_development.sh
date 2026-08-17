#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_png2d_spatial_backbones_v6_strict
CONFIG=/root/autodl-tmp/aneurysm/configs/api_png2d_spatial_backbones_v6_strict.json
export TORCH_HOME=/root/autodl-tmp/envs/png2d-spatial-v6/torch_cache

"$PYTHON" "$CODE/00_preflight.py" --config "$CONFIG" --device cuda:0
"$PYTHON" "$CODE/01_run_development_selection.py" --config "$CONFIG" --family segresnet --outer-fold 1 --device cuda:0
"$PYTHON" "$CODE/01_run_development_selection.py" --config "$CONFIG" --family deeplabv3plus_resnet50_imagenet --outer-fold 1 --device cuda:0
"$PYTHON" "$CODE/01_run_development_selection.py" --config "$CONFIG" --family segresnet --outer-fold 4 --device cuda:0
"$PYTHON" "$CODE/01_run_development_selection.py" --config "$CONFIG" --family deeplabv3plus_resnet50_imagenet --outer-fold 4 --device cuda:0
"$PYTHON" "$CODE/02_compare_development_models.py" --config "$CONFIG" --device cuda:0
