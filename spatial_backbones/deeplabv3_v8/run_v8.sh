#!/usr/bin/env bash
set -euo pipefail
PYTHON=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_png2d_spatial_only_v8_deeplab_fused/run_v8.py
CONFIG=/root/autodl-tmp/aneurysm/configs/api_png2d_spatial_only_v8_deeplab_fused.json
case "${1:-}" in
 preflight|extract-train|verify|outcome-oof|compare) "$PYTHON" "$CODE" "$1" --config "$CONFIG" --device cuda:0 ;;
 all-train) for s in preflight extract-train verify outcome-oof compare; do "$0" "$s"; done ;;
 *) echo "Usage: $0 {preflight|extract-train|verify|outcome-oof|compare|all-train}"; exit 2 ;;
esac
