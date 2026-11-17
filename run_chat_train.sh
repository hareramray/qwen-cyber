#!/usr/bin/env bash
# Build the scan+chat mix and warm-start-train the 1.5B adapter on it, in one
# resumable step. Every step skips work that is already done, so if this stops
# for any reason, just run it again and it continues where it left off.
#   bash run_chat_train.sh
#
# What it produces:
#   data_chat/     synthetic chat Q&A (xssft.gen.chat)
#   data_mixed/    scan data (data/) + chat data, shuffled, for train & val
#   runs/xss-qlora-1.5b-chat/   a NEW adapter, warm-started from the existing
#                               runs/xss-qlora-1.5b so it keeps the scan skill
#                               and adds prose answers. The original adapter is
#                               left untouched.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python
[ -x "$PY" ] || PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
BASE_ADAPTER=runs/xss-qlora-1.5b          # the scan-tuned adapter to warm-start from
OUT=runs/xss-qlora-1.5b-chat              # the new mixed adapter
N_CHAT=2000
EPOCHS=1                                  # 1 pass over the mix; raise if chat answers stay weak
export PYTHONUNBUFFERED=1
mkdir -p runs

stamp() { echo "=== $(date '+%F %T') $*"; }

# 0. Preconditions: the scan data and the adapter we warm-start from must exist.
if [ ! -f data/train.jsonl ]; then
  echo "missing data/train.jsonl -- run run_pipeline.sh first (it generates the scan data)" >&2
  exit 1
fi
if [ ! -f "$BASE_ADAPTER/adapter_model.safetensors" ]; then
  echo "missing $BASE_ADAPTER/adapter_model.safetensors -- run run_pipeline.sh first (it trains it)" >&2
  exit 1
fi

# 1. Chat data (skip if already built).
if [ -f data_chat/train.jsonl ]; then
  stamp "1 chat data already in data_chat, skipping"
else
  stamp "1 generate chat data"
  "$PY" -m xssft.gen.chat --n "$N_CHAT" --out data_chat
fi

# 2. Mixed data = scan + chat, shuffled (skip if already built). Keeping the scan
#    rows in the mix is what stops the model from forgetting how to scan.
if [ -f data_mixed/train.jsonl ] && [ -f data_mixed/val.jsonl ]; then
  stamp "2 mixed data already in data_mixed, skipping"
else
  stamp "2 build mixed data"
  mkdir -p data_mixed
  cat data/train.jsonl data_chat/train.jsonl | shuf > data_mixed/train.jsonl
  cat data/val.jsonl   data_chat/val.jsonl   | shuf > data_mixed/val.jsonl
  echo "   train=$(wc -l < data_mixed/train.jsonl)  val=$(wc -l < data_mixed/val.jsonl)"
fi

# 3. Train. First launch warm-starts from $BASE_ADAPTER; if it is interrupted,
#    re-running resumes from the latest checkpoint in $OUT (which carries the
#    optimizer state), so --init-adapter is only used for the very first start.
if [ -f "$OUT/adapter_model.safetensors" ]; then
  stamp "3 train: adapter already in $OUT, skipping"
else
  stamp "3 warm-start QLoRA train on the mix"
  "$PY" -m xssft.train --model "$MODEL" --data data_mixed --out "$OUT" \
    --epochs "$EPOCHS" --init-adapter "$BASE_ADAPTER" --resume 2>&1 \
    | tee -a runs/train_1.5b_chat.log
fi

stamp "done -- chat with it:  $PY -m xssft.chat --adapter $OUT"
