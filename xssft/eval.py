"""Evaluate a model (base or fine-tuned) on the held-out XSS test set.

Metrics:
  json_valid     fraction of answers that parse into a report object
  grounded       fraction whose cited (file,line,code) all exist in the input
  detect_acc     accuracy of the vulnerable / safe decision
  fp_rate        of truly-safe files, fraction wrongly flagged vulnerable
  recall         of truly-vulnerable files, fraction correctly flagged
  sink_line      on true positives, fraction citing the gold sink line
  type_acc       on true positives, XSS-type match
  context_acc    on true positives, output-context match

Run the base model first to get a baseline, then the adapter:
    python -m xssft.eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --data data
    python -m xssft.eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter runs/xss-qlora-1.5b --data data

With --dump, each prediction is written (and flushed) as soon as it finishes. If
the run stops, repeat the same command with --resume: examples already in the
dump are skipped and metrics are reported over the full set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import schema

DUMP_VERSION = 2
RUN_KEYS = ("model", "adapter", "split")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gold_sink_lines(report: dict) -> set[tuple]:
    return {(s["file"], s["line"]) for s in report.get("sinks", [])
            if not s.get("neutralized", False)}


def pred_sink_lines(report: dict) -> set[tuple]:
    out = set()
    for f in report.get("findings", []):
        for step in f.get("path", []):
            out.add((step.get("file"), step.get("line")))
    for s in report.get("sinks", []):
        if not s.get("neutralized", False):
            out.add((s.get("file"), s.get("line")))
    return out


@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")
    inputs = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


def score(rec: dict, answer: str) -> dict:
    """Score one answer. The row holds every per-example outcome the metrics
    need, so a dump file alone is enough to rebuild them on --resume."""
    gold = json.loads(rec["target"])
    gold_vuln = bool(gold["vulnerable"])
    pred = schema.parse_report(answer)
    row = {"template": rec["template"], "label": rec["label"],
           "parsed": pred is not None, "gold_vuln": gold_vuln}
    if pred is None:
        row["raw"] = answer[:800]
        return row

    gold_files = _files_from_prompt(rec["messages"][1]["content"])
    pred_vuln = bool(pred.get("vulnerable"))
    row.update(pred_vuln=pred_vuln,
               grounded=not schema.validate_report(pred, gold_files))
    if gold_vuln and pred_vuln:
        pf = (pred.get("findings") or [{}])[0]
        row.update(
            sink_hit=bool(gold_sink_lines(gold) & pred_sink_lines(pred)),
            type_ok=pf.get("type") == gold["findings"][0]["type"],
            ctx_ok=pf.get("context") == gold["findings"][0]["context"])
    return row


def aggregate(rows: list[dict]) -> dict:
    m = dict(n=0, json_valid=0, grounded=0, detect_ok=0,
             safe_n=0, fp=0, vuln_n=0, recall_hit=0,
             tp=0, sink_hit=0, type_ok=0, ctx_ok=0)
    for r in rows:
        m["n"] += 1
        if not r["parsed"]:
            continue  # unparsed answers count only toward n
        m["json_valid"] += 1
        m["grounded"] += r["grounded"]
        gold_vuln, pred_vuln = r["gold_vuln"], r["pred_vuln"]
        m["detect_ok"] += gold_vuln == pred_vuln
        if gold_vuln:
            m["vuln_n"] += 1
            m["recall_hit"] += pred_vuln
        else:
            m["safe_n"] += 1
            m["fp"] += pred_vuln
        if gold_vuln and pred_vuln:
            m["tp"] += 1
            m["sink_hit"] += r["sink_hit"]
            m["type_ok"] += r["type_ok"]
            m["ctx_ok"] += r["ctx_ok"]
    return m


def load_done(dump: Path, run: dict) -> dict[int, dict]:
    """Read finished rows from an earlier --dump of the same run, keyed by
    example index. Rewrites the file without a half-written trailing line (from a
    hard kill) so appending to it stays valid JSONL."""
    if not dump.exists():
        return {}
    lines = dump.read_text(encoding="utf-8").splitlines()
    done: dict[int, dict] = {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if all(not rest.strip() for rest in lines[n:]):
                print(f"--resume: dropping truncated last line of {dump}")
                continue
            raise SystemExit(f"--resume: {dump}:{n} is not valid JSON")
        if row.get("v") != DUMP_VERSION:
            raise SystemExit(
                f"--resume: {dump} was written by an older eval.py and lacks the "
                "fields needed to rebuild metrics; rerun without --resume "
                "(or point --dump at a new file)")
        other = {k: row.get(k) for k in RUN_KEYS}
        if other != run:
            raise SystemExit(f"--resume: {dump} is from a different run {other}; "
                             f"this run is {run}")
        done[row["idx"]] = row
    with dump.open("w", encoding="utf-8") as fh:
        for idx in sorted(done):
            fh.write(json.dumps(done[idx]) + "\n")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to load")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dump", default=None, help="write per-example predictions to this JSONL")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --dump, run only the missing "
                         "examples, and report metrics over all of them")
    args = ap.parse_args()
    if args.resume and not args.dump:
        ap.error("--resume needs --dump (the file progress is kept in)")

    records = load_jsonl(Path(args.data) / args.split)
    if args.limit:
        records = records[:args.limit]

    run = {"model": args.model, "adapter": args.adapter, "split": args.split}
    done = load_done(Path(args.dump), run) if args.resume else {}
    todo = [i for i in range(len(records)) if i not in done]
    if args.resume:
        print(f"--resume: {len(records) - len(todo)} of {len(records)} already done, "
              f"{len(todo)} to go")

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant = None if args.no_4bit else BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    rows = dict(done)
    dump_fh = None
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        dump_fh = open(args.dump, "a" if args.resume else "w", encoding="utf-8")
    try:
        for i in todo:
            rec = records[i]
            answer = generate(model, tokenizer, rec["messages"], args.max_new_tokens)
            row = {"v": DUMP_VERSION, "idx": i, **run, **score(rec, answer)}
            rows[i] = row
            if dump_fh:
                dump_fh.write(json.dumps(row) + "\n")
                dump_fh.flush()  # an interrupted run loses at most one example
    finally:
        if dump_fh:
            dump_fh.close()

    m = aggregate([rows[i] for i in range(len(records))])

    def pct(a, b):
        return f"{(100.0 * a / b):5.1f}%" if b else "   n/a"

    print(f"\nmodel={args.model} adapter={args.adapter} n={m['n']}")
    print(f"  json_valid   {pct(m['json_valid'], m['n'])}")
    print(f"  grounded     {pct(m['grounded'], m['n'])}   (cited lines all exist)")
    print(f"  detect_acc   {pct(m['detect_ok'], m['n'])}")
    print(f"  recall       {pct(m['recall_hit'], m['vuln_n'])}   (vuln found / vuln total {m['vuln_n']})")
    print(f"  fp_rate      {pct(m['fp'], m['safe_n'])}   (safe flagged / safe total {m['safe_n']})")
    print(f"  sink_line    {pct(m['sink_hit'], m['tp'])}   (on {m['tp']} true positives)")
    print(f"  type_acc     {pct(m['type_ok'], m['tp'])}")
    print(f"  context_acc  {pct(m['ctx_ok'], m['tp'])}")


def _files_from_prompt(user_prompt: str) -> list[dict]:
    """Rebuild [{path, content}] from the numbered listing in the user prompt,
    so grounding checks run against exactly what the model saw."""
    files = []
    lines = user_prompt.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("File: "):
            path = lines[i][len("File: "):].strip()
            # next line is ```; collect until closing ```
            i += 2
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                # strip the "NNN | " prefix added by schema.number_lines
                raw = lines[i]
                bar = raw.find(" | ")
                body.append(raw[bar + 3:] if bar != -1 else raw)
                i += 1
            files.append({"path": path, "content": "\n".join(body)})
        i += 1
    return files


if __name__ == "__main__":
    main()
