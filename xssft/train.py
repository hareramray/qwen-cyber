"""QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct for XSS analysis.

Tuned for a single 8 GB GPU: 4-bit base weights, gradient checkpointing, paged
8-bit AdamW, LoRA on the attention+MLP projections. Loss is computed only on the
assistant answer (the JSON report), never on the prompt.

    python -m xssft.train --data data --out runs/xss-qlora-1.5b --epochs 2

Checkpoints land in <out>/checkpoint-<step> every --save-steps optimizer steps.
To continue an interrupted run, repeat the same command with --resume (latest
checkpoint in --out) or --resume <out>/checkpoint-<step> (a specific one).

Dependencies (see requirements-train.txt):
    pip install "transformers>=4.44" peft trl datasets accelerate bitsandbytes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE = -100


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ids(result) -> list[int]:
    """apply_chat_template returns a list in older transformers and a BatchEncoding
    in newer ones; normalise to a flat list of token ids."""
    if hasattr(result, "keys"):
        return list(result["input_ids"])
    return list(result)


def build_tokenize_fn(tokenizer, max_len: int):
    """Return input_ids + labels, with prompt tokens masked out of the loss."""

    def fn(rec: dict) -> dict:
        messages = rec["messages"]
        answer = rec["target"]
        # Prompt = system + user with the generation prefix the model sees at inference.
        prompt_ids = _ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True))
        # Full = prompt + the assistant answer + EOS.
        full_messages = messages + [{"role": "assistant", "content": answer}]
        full_ids = _ids(tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False))
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = IGNORE
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}

    return fn


def answer_only_loss(model, inputs, num_items_in_batch=None, use_item_count=True,
                     return_outputs=False):
    """Causal-LM loss that runs lm_head only over the answer span.

    Prompt tokens are masked out of the loss anyway, but the stock forward still
    projects every position onto the 152k-token vocab: ~1.2k positions x 152k fp32
    logits plus their grads is ~2 GB at peak. The answer is a suffix of each
    sequence, so keep logits only from the first labelled position onwards.

    Normalisation matches the Trainer: divide by num_items_in_batch (tokens across
    the whole accumulation window) when the Trainer supplies it and skips its own
    division by the accumulation steps, else take the mean over this batch.
    """
    import torch
    import torch.nn.functional as F

    labels = inputs["labels"]
    seq_len = labels.shape[1]
    labelled_cols = (labels != IGNORE).any(dim=0)
    first = int(labelled_cols.float().argmax()) if labelled_cols.any() else seq_len - 1
    first = max(first, 1)  # position `first` is predicted from logits at first-1
    outputs = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    logits_to_keep=seq_len - first + 1)
    logits = outputs.logits[:, :-1, :].float()  # positions first-1 .. seq_len-2
    targets = labels[:, first:]                  # tokens   first   .. seq_len-1
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="sum")
    if num_items_in_batch is not None and use_item_count:
        denom = num_items_in_batch
        if torch.is_tensor(denom):
            denom = denom.to(loss.device)
    else:
        denom = (targets != IGNORE).sum().clamp(min=1)
    loss = loss / denom
    return (loss, outputs) if return_outputs else loss


def resolve_resume(resume: str | None, out_dir: str) -> str | None:
    """Turn the --resume flag into a checkpoint path for Trainer.train (or None)."""
    if resume is None:
        return None
    if resume != "latest":
        ckpt = Path(resume)
        if not (ckpt / "trainer_state.json").is_file():
            raise SystemExit(f"--resume: {ckpt} is not a Trainer checkpoint "
                             "(no trainer_state.json)")
        print(f"resuming from {ckpt}")
        return str(ckpt)
    out = Path(out_dir)
    ckpts = sorted(
        (d for d in out.glob("checkpoint-*")
         if d.is_dir() and d.name.split("-")[-1].isdigit()
         and (d / "trainer_state.json").is_file()),
        key=lambda d: int(d.name.split("-")[-1]))
    if not ckpts:
        print(f"--resume: no checkpoint found in {out}, starting from scratch")
        return None
    print(f"resuming from {ckpts[-1]}")
    return str(ckpts[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/xss-qlora-1.5b")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-4bit", action="store_true", help="load in bf16 instead of 4-bit")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="stop after N optimizer steps (smoke tests); -1 = use --epochs")
    ap.add_argument("--save-steps", type=int, default=50,
                    help="save a checkpoint every N optimizer steps")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    metavar="CHECKPOINT",
                    help="resume training: bare flag = latest checkpoint in --out, "
                         "or a path to a specific checkpoint-<step> directory")
    ap.add_argument("--init-adapter", default=None, metavar="ADAPTER",
                    help="warm-start: load this existing LoRA adapter's weights and "
                         "keep training them (its adapter_config.json sets the LoRA "
                         "shape, so --lora-r/--lora-alpha are ignored). Unlike "
                         "--resume this carries no optimizer state and trains on the "
                         "current --data. Ignored when --resume loads a checkpoint.")
    args = ap.parse_args()

    resume_from = resolve_resume(args.resume, args.out)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    # Not peft.prepare_model_for_kbit_training: it upcasts every non-4-bit weight to
    # fp32, and for Qwen's 152k vocab that doubles embed_tokens + lm_head to ~4.4 GB,
    # which overflows an 8 GB card into shared system RAM and stalls training.
    # bf16 for those frozen weights is fine; LoRA adapters are still trained in fp32.
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.init_adapter:
        from peft import PeftModel
        # is_trainable=True keeps the loaded LoRA weights as trainable parameters
        # (default would freeze them for inference). The LoRA shape comes from the
        # adapter's own adapter_config.json.
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"warm-start: loaded adapter weights from {args.init_adapter}")
    else:
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data_dir = Path(args.data)
    tok_fn = build_tokenize_fn(tokenizer, args.max_len)
    train_ds = Dataset.from_list(load_jsonl(data_dir / "train.jsonl")).map(
        tok_fn, remove_columns=["messages", "target", "template", "label",
                                "language", "framework"])
    val_ds = Dataset.from_list(load_jsonl(data_dir / "val.jsonl")).map(
        tok_fn, remove_columns=["messages", "target", "template", "label",
                                "language", "framework"])

    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE,
                                      padding=True)

    # transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio)
    # and group_by_length into train_sampling_strategy; pick whichever exists.
    import inspect
    targ_params = inspect.signature(TrainingArguments.__init__).parameters
    compat = {}
    if "warmup_ratio" in targ_params:
        compat["warmup_ratio"] = 0.03
    else:
        compat["warmup_steps"] = 0.03
    if "train_sampling_strategy" in targ_params:
        compat["train_sampling_strategy"] = "group_by_length"
    else:
        compat["group_by_length"] = True

    targs = TrainingArguments(
        **compat,
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        # Not paged_adamw_8bit: its state lives in CUDA managed memory, and on Windows
        # (WDDM) copying that back to save optimizer.pt hangs the checkpoint save.
        # LoRA optimizer state is only ~80 MB, so paging buys nothing here anyway.
        optim="adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
        # eval only needs the loss; otherwise the Trainer keeps every example's
        # full-vocab logits around for compute_metrics, which we don't use
        prediction_loss_only=True,
    )

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            return answer_only_loss(model, inputs, num_items_in_batch,
                                    self.model_accepts_loss_kwargs, return_outputs)

    trainer = AnswerOnlyTrainer(model=model, args=targs, train_dataset=train_ds,
                                eval_dataset=val_ds, data_collator=collator)
    trainer.train(resume_from_checkpoint=resume_from)
    if torch.cuda.is_available():
        print(f"peak GPU memory allocated: "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
