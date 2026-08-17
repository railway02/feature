#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/autodl-tmp/envs/png2d-spatial-v6/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_png2d_spatial_only_v7_masked/run.py
CONFIG=/root/autodl-tmp/aneurysm/configs/api_png2d_spatial_only_v7_masked.json

case "${1:-}" in
  preflight|extract-train|verify|outcome-oof|compare)
    "$PYTHON" "$CODE" "$1" --config "$CONFIG" --device cuda:0 ;;
  all-train)
    for stage in preflight extract-train verify outcome-oof compare; do
      "$0" "$stage"
    done ;;
  *)
    echo "Usage: $0 {preflight|extract-train|verify|outcome-oof|compare|all-train}"
    exit 2 ;;
esac
