#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/root/autodl-tmp/aneurysm}
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=${TARGET:-$ROOT/code/api_fullseq_cave_v3}
CONFIG=${CONFIG:-$ROOT/configs/api_fullseq_cave_v3_frozen.json}

mkdir -p "$TARGET" "$ROOT/configs" "$ROOT/logs" "$ROOT/reports/api_fullseq_cave_v3"
cp -a "$SOURCE"/*.py "$SOURCE"/*.sh "$SOURCE"/*.md "$TARGET/"
if [[ ! -f "$CONFIG" ]]; then
  cp "$SOURCE/frozen_config.example.json" "$CONFIG"
fi
chmod +x "$TARGET"/*.py "$TARGET"/*.sh
python -m py_compile "$TARGET"/*.py

echo "[PASS] deployed code to $TARGET"
echo "[PASS] frozen config at $CONFIG"
