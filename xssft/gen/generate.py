"""Generate the synthetic dataset and write train/val/test JSONL.

Every example is re-checked with schema.validate_report before it is kept, so a
bug in a template fails loudly here instead of poisoning training. Splits are
stratified by template id (no template appears only in test) and the three
stacks stay balanced.

Usage:
    python -m xssft.gen.generate --n 6000 --out data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .. import schema
from . import django, spring, vue

STACKS = {"vue": vue.ALL, "django": django.ALL, "spring": spring.ALL}


def _one(template, rng: random.Random) -> dict:
    ex = template(rng)
    problems = schema.validate_report(ex["report"], ex["files"])
    if problems:
        raise AssertionError(f"{ex['template']} produced an invalid report: {problems}")
    return ex


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    templates = [t for stack in STACKS.values() for t in stack]
    out, seen = [], set()
    attempts = 0
    # Round-robin over templates so every family is represented roughly equally.
    while len(out) < n and attempts < n * 40:
        attempts += 1
        template = templates[attempts % len(templates)]
        ex = _one(template, rng)
        key = json.dumps([f["content"] for f in ex["files"]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def to_record(ex: dict) -> dict:
    """Flatten to a training record: messages + target string + metadata."""
    messages = schema.build_messages(ex["files"], ex["language"], ex["framework"])
    target = schema.dump_report(ex["report"])
    return {"messages": messages, "target": target, "template": ex["template"],
            "label": ex["label"], "language": ex["language"], "framework": ex["framework"]}


def split(records: list[dict], val_frac: float, test_frac: float, seed: int):
    """Stratify by template so train/val/test each see every template."""
    by_t: dict[str, list[dict]] = {}
    for r in records:
        by_t.setdefault(r["template"], []).append(r)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for recs in by_t.values():
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test += recs[:n_test]
        val += recs[n_test:n_test + n_val]
        train += recs[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="total examples to generate")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    examples = generate(args.n, args.seed)
    records = [to_record(e) for e in examples]
    train, val, test = split(records, args.val_frac, args.test_frac, args.seed)

    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    def stats(recs):
        return dict(label=Counter(r["label"] for r in recs),
                    framework=Counter(r["framework"] for r in recs))

    print(f"generated {len(records)} unique examples")
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        s = stats(recs)
        print(f"  {name:5} {len(recs):5}  labels={dict(s['label'])}  stacks={dict(s['framework'])}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
