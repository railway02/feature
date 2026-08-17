#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/root/autodl-tmp/aneurysm}
SOURCE=$(cd "$(dirname "$0")" && pwd)
TARGET=$PROJECT/code/api_gtmask_roi_cave_v5_fullmask_fullseq
mkdir -p "$PROJECT/code" "$PROJECT/configs" "$PROJECT/manifests/api_gtmask_roi_cave_v5_fullmask_fullseq"
rm -rf "$TARGET"
cp -a "$SOURCE" "$TARGET"
if [[ ! -f "$PROJECT/configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json" ]]; then
  cp "$TARGET/config.example.json" "$PROJECT/configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json"
fi
if [[ ! -f "$PROJECT/manifests/api_gtmask_roi_cave_v5_fullmask_fullseq/manual_mask_mapping.csv" ]]; then
  cp "$TARGET/manual_mask_mapping_template.csv" "$PROJECT/manifests/api_gtmask_roi_cave_v5_fullmask_fullseq/manual_mask_mapping.csv"
fi
echo "Installed: $TARGET"
echo "Config: $PROJECT/configs/api_gtmask_roi_cave_v5_fullmask_fullseq.json"
