#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/aneurysm"
CODEX="/root/autodl-tmp/tools/codex/bin/codex"
PROMPT="$ROOT/prompts/api_fullseq_v2_full_train_valid_unattended.md"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/logs/codex_api_fullseq_v2_full_train_valid_${STAMP}"

cd "$ROOT"
mkdir -p "$RUN_DIR" "$ROOT/logs"

if pgrep -af '[c]odex.*api_fullseq_v2_full_train_valid' >/dev/null; then
  echo "[STOP] 已存在相关Codex任务"
  exit 1
fi

if pgrep -af '[2]0_extract_api_fullseq_v2_train_valid_pairdata.py|[2]1_build_api_fullseq_v2_train_valid_features.py|run_api_fullseq_v2_full_train_valid.sh' >/dev/null; then
  echo "[STOP] 已存在Full Train/Valid后台任务"
  exit 1
fi

if [[ ! -f "$PROMPT" ]]; then
  echo "[STOP] Prompt不存在：$PROMPT"
  exit 1
fi

echo "$RUN_DIR" > "$ROOT/logs/latest_codex_api_fullseq_v2_full_train_valid_run.txt"

nohup "$CODEX" \
  --ask-for-approval never \
  exec \
  --skip-git-repo-check \
  -C "$ROOT" \
  -m gpt-5.6-sol \
  --sandbox danger-full-access \
  --json \
  --output-last-message "$RUN_DIR/final_message.txt" \
  - \
  < "$PROMPT" \
  > "$RUN_DIR/events.jsonl" \
  2> "$RUN_DIR/codex_stderr.log" \
  < /dev/null &

PID=$!
echo "$PID" > "$RUN_DIR/pid"
echo "$PID" > "$ROOT/logs/api_fullseq_v2_full_train_valid_codex.pid"

echo "CODEX_PID=$PID"
echo "RUN_DIR=$RUN_DIR"
echo "EVENTS=$RUN_DIR/events.jsonl"
echo "STDERR=$RUN_DIR/codex_stderr.log"