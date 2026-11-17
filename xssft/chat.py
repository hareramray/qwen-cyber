"""Talk to the fine-tuned model.

Scan files (the task the model was trained on, prints the JSON report):
    python -m xssft.chat path/to/views.py path/to/template.html

Interactive session:
    python -m xssft.chat
      scan <file> [file ...]   scan files from disk
      paste                    paste code, finish with a line containing only END
      <anything else>          free-form chat (not what it was trained for)
      reset                    clear chat history
      quit                     exit

Options: --adapter PATH (default runs/xss-qlora-1.5b), --base (no adapter, for
comparison), --max-new-tokens N.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextStreamer)

from . import schema

STACKS = {".vue": ("JavaScript", "Vue 3"), ".js": ("JavaScript", ""), ".ts": ("TypeScript", ""),
          ".jsx": ("JavaScript", "React"), ".tsx": ("TypeScript", "React"),
          ".py": ("Python", "Django"), ".java": ("Java", "Spring Boot"),
          ".jsp": ("Java", "Spring Boot"), ".ftlh": ("Java", "Spring Boot"),
          ".php": ("PHP", ""), ".rb": ("Ruby", ""), ".go": ("Go", "")}


def load(model_id: str, adapter: str | None):
    print(f"loading {model_id}" + (f" + {adapter}" if adapter else " (base, no adapter)"),
          file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def reply(model, tok, messages, max_new_tokens: int) -> str:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.pad_token_id,
                         streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True))
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def scan(model, tok, files: list[dict], max_new_tokens: int) -> None:
    lang, fw = "", ""
    for f in files:
        lang, fw = STACKS.get(Path(f["path"]).suffix.lower(), (lang, fw))
        if fw:
            break
    answer = reply(model, tok, schema.build_messages(files, lang, fw), max_new_tokens)
    report = schema.parse_report(answer)
    print()
    if report is None:
        print("!! answer is not a valid JSON report")
        return
    problems = schema.validate_report(report, files)
    verdict = "VULNERABLE" if report.get("vulnerable") else "no XSS found"
    print(f"== {verdict}")
    for fd in report.get("findings", []):
        last = (fd.get("path") or [{}])[-1]
        print(f"   - {fd.get('type')} XSS, {fd.get('context')} context, severity "
              f"{fd.get('severity')} at {last.get('file')}:{last.get('line')}")
        if (fd.get("fix") or {}).get("summary"):
            print(f"     fix: {fd['fix']['summary']}")
    if problems:
        print(f"   (warning: {len(problems)} citation(s) don't match the code: {problems[0]})")


def read_files(paths: list[str]) -> list[dict] | None:
    files = []
    for p in paths:
        try:
            files.append({"path": p.replace("\\", "/"),
                          "content": Path(p).read_text(encoding="utf-8").rstrip("\n")})
        except OSError as e:
            print(f"cannot read {p}: {e}")
            return None
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="files to scan (omit for interactive mode)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default="runs/xss-qlora-1.5b")
    ap.add_argument("--base", action="store_true", help="run without the adapter")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    args = ap.parse_args()

    model, tok = load(args.model, None if args.base else args.adapter)

    if args.files:
        files = read_files(args.files)
        if files:
            scan(model, tok, files, args.max_new_tokens)
        return

    print("ready. commands: scan <file...> | paste | reset | quit | or type a message")
    history: list[dict] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd in ("quit", "exit"):
            break
        if cmd == "reset":
            history = []
            print("history cleared")
        elif cmd == "scan":
            files = read_files(rest.split()) if rest else None
            if files:
                scan(model, tok, files, args.max_new_tokens)
            elif not rest:
                print("usage: scan <file> [file ...]")
        elif cmd == "paste":
            name = input("file name (sets the language, e.g. views.py or App.vue): ").strip() or "snippet.txt"
            print("paste code, then a line with only END:")
            body = []
            while (ln := input()) != "END":
                body.append(ln)
            scan(model, tok, [{"path": name, "content": "\n".join(body)}], args.max_new_tokens)
        else:
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant",
                            "content": reply(model, tok, history, args.max_new_tokens)})


if __name__ == "__main__":
    main()
