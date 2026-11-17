# xssft — fine-tuning Qwen2.5-Coder-1.5B into an XSS analyst

Trains Qwen2.5-Coder-1.5B-Instruct to find cross-site scripting in **Vue**, **Django**
and **Spring Boot (Java)** code and emit a grounded JSON report (sources, sinks,
data-flow path, output context, severity, fix, PoC).

The report format and the grounding checker live in [`xssft/schema.py`](xssft/schema.py):
every line of input is numbered, and every cited `(file, line, code)` in a report
must exist in the input, so answers can be verified mechanically.

## Pipeline

| Step | Command |
|------|---------|
| 1–2. Generate synthetic train/val/test | `python -m xssft.gen.generate --n 6000 --out data` |
| 3. QLoRA fine-tune (8 GB GPU) | `python -m xssft.train --data data --out runs/xss-qlora-1.5b --epochs 2` |
| 4a. Baseline (base model) | `python -m xssft.eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --data data` |
| 4b. After fine-tune | `python -m xssft.eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter runs/xss-qlora-1.5b --data data` |


.venv\Scripts\python -m xssft.chat --adapter runs\xss-qlora-1.5b-chat


Install training deps: `pip install -r requirements-train.txt`.

### Resuming interrupted runs

- **Training** saves `runs/xss-qlora-1.5b/checkpoint-<step>` every `--save-steps` (default 50)
  optimizer steps, keeping the last two. Re-run the same command with `--resume` to
  continue from the latest one (weights, optimizer, LR schedule and data position), or
  `--resume runs/xss-qlora-1.5b/checkpoint-<step>` for a specific one.
- **Eval** writes each prediction to `--dump` as it finishes. Re-run the same command
  with `--resume` to skip examples already in the dump; metrics cover the full set.
  `--resume` refuses a dump from a different model/adapter/split.

```
python -m xssft.train --data data --out runs/xss-qlora-1.5b --epochs 2 --resume
python -m xssft.eval --data data --dump runs/baseline_1.5b_preds.jsonl --resume
```

## Synthetic data (steps 1–2)

[`xssft/gen/`](xssft/gen/) builds code whose vulnerability label is known by
construction, so no teacher model is needed and every example passes
`validate_report`. Coverage per stack:

- **Vue** ([`vue.py`](xssft/gen/vue.py)) — `v-html`, `innerHTML`, `:href="javascript:"`,
  render-function `domProps.innerHTML`; safe: `{{ }}`, `textContent`, DOMPurify.
- **Django** ([`django.py`](xssft/gen/django.py)) — `HttpResponse` concat, `mark_safe`,
  `|safe`, `{% autoescape off %}`, and the autoescape-on-but-`javascript:`-URL trap;
  safe: autoescaped `{{ }}`, `django.utils.html.escape`.
- **Spring Boot** ([`spring.py`](xssft/gen/spring.py)) — Thymeleaf `th:utext`,
  `@ResponseBody`/`ResponseEntity` HTML concat, JSP `${param.x}`; safe: `th:text`, JSON.

Each family randomises identifiers, routes and multi-file (view→template) flow.
About a third of examples are **safe** (hard negatives) so the model learns not to
over-report, including context traps where escaping is present but wrong for the
output context. Splits are stratified by template, so every pattern appears in train,
val and test.

## Evaluation (step 4)

`xssft/eval.py` reports JSON validity, grounding, detection accuracy, **false-positive
rate on safe code**, recall, sink-line localization, and type/context accuracy. Run the
base model first to establish the baseline the fine-tune must beat.

## Notes for the 8 GB card

`train.py` defaults to 4-bit NF4 weights, LoRA on all attention+MLP projections,
gradient checkpointing, paged 8-bit AdamW, batch 1 × grad-accum 16, seq len 2048.
Lower `--max-len` if you hit OOM; raise `--accum` to keep the effective batch size.

The 1.5B model uses ~1–2 GB of weights in 4-bit (~3 GB in bf16), so on 8 GB you can
also train with `--no-4bit` (plain LoRA, no quantization error) or raise `--batch`.
To go back to the 7B model, pass `--model Qwen/Qwen2.5-Coder-7B-Instruct`.
