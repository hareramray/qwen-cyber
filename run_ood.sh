#!/usr/bin/env bash
# Out-of-distribution check: score the base model and a fine-tuned adapter on
# hand-written examples whose patterns never appear in the generator templates.
# Resumable like run_pipeline.sh: just run it again if it stops.
#   bash run_ood.sh                               # scores runs/xss-qlora-1.5b
#   bash run_ood.sh --adapter runs/xss-qlora-1.5b-chat
# Each adapter writes to its own dump (named after the adapter folder), so runs
# don't overwrite each other and you can compare them side by side.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python
[ -x "$PY" ] || PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
ADAPTER=runs/xss-qlora-1.5b
while [ $# -gt 0 ]; do
  case "$1" in
    --adapter) ADAPTER="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
LABEL=$(basename "$ADAPTER")      # e.g. xss-qlora-1.5b-chat -> dump suffix
export PYTHONUNBUFFERED=1

stamp() { echo "=== $(date '+%F %T') $*"; }

stamp "build OOD set"
"$PY" -m xssft.gen.ood --out data_ood

# Baseline is adapter-independent and shared across runs (resumes instantly once done).
stamp "OOD baseline eval"
"$PY" -m xssft.eval --model "$MODEL" --data data_ood \
  --dump runs/ood_baseline_1.5b_preds.jsonl --resume 2>&1 | tee -a runs/ood_baseline_1.5b_eval.log

stamp "OOD eval for adapter $ADAPTER"
"$PY" -m xssft.eval --model "$MODEL" --adapter "$ADAPTER" --data data_ood \
  --dump runs/ood_${LABEL}_preds.jsonl --resume 2>&1 | tee -a runs/ood_${LABEL}_eval.log

stamp "done -- dump: runs/ood_${LABEL}_preds.jsonl"
