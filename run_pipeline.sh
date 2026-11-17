#!/usr/bin/env bash
# Run steps 4a -> 3 -> 4b end to end. Every step resumes, so if this stops for any
# reason, just run it again and it continues where it left off.
#   bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python
[ -x "$PY" ] || PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
TAG=1.5b   # keeps logs/dumps/adapters separate from earlier 7B runs
OUT=runs/xss-qlora-$TAG
export PYTHONUNBUFFERED=1
mkdir -p runs

stamp() { echo "=== $(date '+%F %T') $*"; }

if [ ! -f data/train.jsonl ]; then
  stamp "generate data"
  "$PY" -m xssft.gen.generate --n 6000 --out data
fi

stamp "4a baseline eval"
"$PY" -m xssft.eval --model "$MODEL" --data data \
  --dump runs/baseline_${TAG}_preds.jsonl --resume 2>&1 | tee -a runs/baseline_${TAG}_eval.log

if [ -f "$OUT/adapter_model.safetensors" ]; then
  stamp "3 train: adapter already in $OUT, skipping"
else
  stamp "3 QLoRA train"
  "$PY" -m xssft.train --model "$MODEL" --data data --out "$OUT" --epochs 2 \
    --resume 2>&1 | tee -a runs/train_${TAG}.log
fi

stamp "4b fine-tuned eval"
"$PY" -m xssft.eval --model "$MODEL" --adapter "$OUT" --data data \
  --dump runs/finetuned_${TAG}_preds.jsonl --resume 2>&1 | tee -a runs/finetuned_${TAG}_eval.log

stamp "done"
