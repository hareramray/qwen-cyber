# xssft — fine-tuning Qwen2.5-Coder-1.5B into an XSS analyst

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Base model](https://img.shields.io/badge/base-Qwen2.5--Coder--1.5B-orange.svg)
![Method](https://img.shields.io/badge/method-QLoRA-green.svg)

> **⚠️ For education and authorized security testing only.** This project generates
> proof-of-concept XSS payloads to train a vulnerability *detector*. Use it only on
> systems you own or are explicitly authorized to test. The authors accept no liability
> for misuse; see [LICENSE](LICENSE).

Trains Qwen2.5-Coder-1.5B-Instruct to find cross-site scripting in **Vue**, **Django**
and **Spring Boot (Java)** code and emit a grounded JSON report (sources, sinks,
data-flow path, output context, severity, fix, PoC). An optional second stage mixes in
synthetic Q&A so the same adapter can also answer questions in prose.

The report format and the grounding checker live in [`xssft/schema.py`](xssft/schema.py):
every line of input is numbered, and every cited `(file, line, code)` in a report must
exist in the input, so answers can be verified mechanically.

> **Model weights are not in this repo.** The `.safetensors` adapters and checkpoints
> (~800 MB) are gitignored; rebuild them with the scripts below. All results are
> reproducible on a single 8 GB GPU.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-train.txt   # Windows
# source .venv/bin/activate && pip install -r requirements-train.txt          # Linux/Mac

bash run_pipeline.sh        # generate data -> baseline eval -> QLoRA train -> eval
```

Every script is **resumable** — if it stops (power, OOM, a closed terminal), run the
same command again and it continues from the last checkpoint / finished example.

## Scripts

| Script | What it does | Output |
|--------|--------------|--------|
| [`run_pipeline.sh`](run_pipeline.sh) | Generate data, eval the base model, QLoRA fine-tune, eval the adapter | `runs/xss-qlora-1.5b/` |
| [`run_chat_train.sh`](run_chat_train.sh) | Build scan+chat mix, **warm-start** from the scan adapter, train a combined model | `runs/xss-qlora-1.5b-chat/` |
| [`run_ood.sh`](run_ood.sh) | Score base + an adapter on a hand-written out-of-distribution test set | `runs/ood_*_preds.jsonl` |

```bash
bash run_chat_train.sh                               # adds prose Q&A to the scan adapter
bash run_ood.sh --adapter runs/xss-qlora-1.5b-chat   # generalization check on unseen patterns
```

## Using a trained model

[`xssft/chat.py`](xssft/chat.py) scans files or chats with an adapter:

```bash
# Scan files for XSS (prints the JSON report + a summary, incl. a PoC for authorized testing)
.venv/Scripts/python -m xssft.chat --adapter runs/xss-qlora-1.5b-chat demo/views.py demo/templates/feedback.html

# Interactive: `scan <files>`, `paste` (end with END), `reset`, `quit`, or free-form chat
.venv/Scripts/python -m xssft.chat --adapter runs/xss-qlora-1.5b-chat

# Compare against the un-tuned base model
.venv/Scripts/python -m xssft.chat --base
```

## Results

Metrics: `json_valid` (parses to a report), `grounded` (all cited lines exist),
`detect_acc` (vuln/safe decision), `recall` (vuln found), `fp_rate` (safe code wrongly
flagged — lower is better), `sink_line`/`type_acc`/`context_acc` (localization quality
on true positives).

**Held-out test set (593 examples).** The base model flags almost everything as
vulnerable; the fine-tune fixes both format and discrimination:

| Metric | base | fine-tuned |
|--------|------|-----------|
| json_valid | 65.3% | **100%** |
| grounded | 13.7% | **99.3%** |
| detect_acc | 43.0% | **100%** |
| recall | 97.5% | **100%** |
| fp_rate | 87.5% | **0%** |

⚠️ The test set shares its 20 templates with training, so 100% here reflects **learning
those templates**, not general XSS reasoning. The honest measure is the OOD set below.

**Out-of-distribution set (36 hand-written examples, patterns absent from the
generator** — see [`xssft/gen/ood.py`](xssft/gen/ood.py)**).** `scan-only` is
`xss-qlora-1.5b`; `chat/mix` is the warm-started `xss-qlora-1.5b-chat`:

| Metric | base | scan-only | chat/mix |
|--------|------|-----------|----------|
| json_valid | 66.7% | 97.2% | 97.2% |
| grounded | 22.2% | 66.7% | **83.3%** |
| detect_acc | 41.7% | 55.6% | **58.3%** |
| recall | 100% | 64.7% | 70.6% |
| fp_rate | 90.0% | 50.0% | 50.0% |
| type_acc | 42.9% | 54.5% | **75.0%** |

Fine-tuning generalizes only partially: detection and grounding improve markedly, but a
~50% false-positive rate and missed vuln patterns remain — the ceiling of a 1.5B model
trained on templated data. Adding the chat mix did **not** regress scanning (it slightly
helped). With only 36 examples, treat small shifts as noise.

## Synthetic data

[`xssft/gen/`](xssft/gen/) builds code whose vulnerability label is known by
construction, so no teacher model is needed and every example passes `validate_report`.
Coverage per stack:

- **Vue** ([`vue.py`](xssft/gen/vue.py)) — `v-html`, `innerHTML`, `:href="javascript:"`,
  render-function `domProps.innerHTML`; safe: `{{ }}`, `textContent`, DOMPurify.
- **Django** ([`django.py`](xssft/gen/django.py)) — `HttpResponse` concat, `mark_safe`,
  `|safe`, `{% autoescape off %}`, and the autoescape-on-but-`javascript:`-URL trap;
  safe: autoescaped `{{ }}`, `django.utils.html.escape`.
- **Spring Boot** ([`spring.py`](xssft/gen/spring.py)) — Thymeleaf `th:utext`,
  `@ResponseBody`/`ResponseEntity` HTML concat, JSP `${param.x}`; safe: `th:text`, JSON.

Each family randomises identifiers, routes and multi-file (view→template) flow. About a
third of examples are **safe** (hard negatives) so the model learns not to over-report,
including context traps where escaping is present but wrong for the output context.
Splits are stratified by template, so every pattern appears in train, val and test.

Two generators support the extra stages:

- [`ood.py`](xssft/gen/ood.py) — a **test-only** set using sinks the templates never
  produce (`document.write`, `insertAdjacentHTML`, `setTimeout` strings, FreeMarker
  `?no_esc`, unquoted attributes, sanitiser-then-raw, …). Never train on it.
- [`chat.py`](xssft/gen/chat.py) — free-text Q&A (concepts, payloads for a given
  context, fix critiques, calibrated refusals) that shares the training record shape, to
  mix with the scan data so the model can also chat.

## Resuming interrupted runs

- **Training** saves `runs/<out>/checkpoint-<step>` every `--save-steps` (default 50)
  optimizer steps, keeping the last two. Re-run with `--resume` to continue from the
  latest (weights, optimizer, LR schedule, data position), or `--resume <checkpoint>`
  for a specific one. `--init-adapter <adapter>` instead **warm-starts** from an existing
  adapter's weights (used by `run_chat_train.sh`).
- **Eval** writes each prediction to `--dump` as it finishes. Re-run with `--resume` to
  skip examples already in the dump; metrics cover the full set. `--resume` refuses a
  dump from a different model/adapter/split.

## Notes for the 8 GB card

`train.py` defaults to 4-bit NF4 weights, LoRA on all attention+MLP projections,
gradient checkpointing, 8-bit AdamW, batch 1 × grad-accum 16, seq len 2048. Lower
`--max-len` if you hit OOM; raise `--accum` to keep the effective batch size. The 1.5B
model uses ~1–2 GB in 4-bit (~3 GB in bf16), so on 8 GB you can also train with
`--no-4bit` (plain LoRA, no quantization error). To go back to the 7B model, pass
`--model Qwen/Qwen2.5-Coder-7B-Instruct`.

## Disclaimer

This is a research and educational project. The synthetic data and the model it
produces include proof-of-concept XSS payloads whose sole purpose is to train and
evaluate a vulnerability detector. **Only run payloads against applications you own or
have written authorization to test.** The model is a small (1.5B) assistant: it misses
vulnerabilities and raises false positives (see the out-of-distribution results above),
so treat its output as a first pass to be verified by a human, not as an authoritative
security audit.

## License

[MIT](LICENSE) © 2026 Hareram Ray
