"""LoRA fine-tuning for LocateAnything-3B (GPU box).

NVIDIA did not publish training code, so this is a standard PEFT/LoRA supervised
fine-tune over the model's own chat format, built on transformers + peft +
datasets. It consumes the JSONL from :mod:`motion.data.dataset_la`.

⚠️ VERIFY-ON-GPU: LocateAnything ships as `trust_remote_code` custom modeling, so
the processor call and how image tokens are spliced into the sequence are
model-specific. The two adapter points are marked `# ADAPT:` — everything else is
boilerplate. Confirm against the model card's `LocateAnythingWorker` example on the
GPU box before a long run; do a 10-step smoke run first.

⚠️ LICENSE: LocateAnything is NVIDIA non-commercial/research only (docs/LICENSING.md).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from motion.config import load_yaml, resolve_path
from motion.logging import get_logger

log = get_logger(__name__)


class LAFinetuneConfig(BaseModel):
    model_id: str = "nvidia/LocateAnything-3B"
    dataset_jsonl: str = "data/datasets/la_train.jsonl"
    output_dir: str = "checkpoints/locateanything-ft"
    epochs: float = 3.0
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    max_seq_len: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    save_steps: int = 200
    logging_steps: int = 10
    report_to: str = "tensorboard"  # or "wandb" / "none"
    smoke_steps: int = 0  # >0 → stop after N steps (quick validation run)


def run(config_path: str | Path) -> None:
    cfg = LAFinetuneConfig.model_validate(load_yaml(config_path))
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Install training deps on the GPU box: pip install -e '.[train]'") from e

    log.info("Loading %s", cfg.model_id)
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            task_type="CAUSAL_LM",
        ),
    )
    if cfg.gradient_checkpointing:
        # With PEFT + gradient checkpointing the frozen input embeddings produce no
        # grad unless we force it, or the backward pass silently breaks. (Trainer
        # re-enables checkpointing itself via TrainingArguments below, with
        # use_reentrant=False — so we don't call gradient_checkpointing_enable here.)
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(resolve_path(cfg.dataset_jsonl)), split="train")

    tokenizer = getattr(processor, "tokenizer", processor)

    def collate(batch: list[dict]):
        images, texts, prompt_lens = [], [], []
        for ex in batch:
            images.append(Image.open(ex["image"]).convert("RGB"))
            # ADAPT #1: build the model's chat string. apply_chat_template is the
            # standard path; if the custom processor differs, format here instead.
            full = processor.apply_chat_template(ex["conversations"], tokenize=False)
            texts.append(full)
            # Token length of everything BEFORE the assistant answer, so we can
            # supervise ONLY the answer (boxes) — not the prompt or image tokens.
            prompt_only = processor.apply_chat_template(
                ex["conversations"][:-1], tokenize=False, add_generation_prompt=True
            )
            prompt_lens.append(len(tokenizer(prompt_only, add_special_tokens=False).input_ids))
        # ADAPT #2: some VLM processors take images=..., text=...; others expect a
        # combined "messages" arg. Match the model card's processor signature.
        enc = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
        )
        labels = enc["input_ids"].clone()
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Mask the prompt + image-placeholder span in each row; loss falls only on
        # the target answer tokens. (Note: with left-padding the offset shifts —
        # verify padding_side on the GPU box; this assumes right-padding.)
        for row, plen in enumerate(prompt_lens):
            labels[row, :plen] = -100
        enc["labels"] = labels
        return enc

    args = TrainingArguments(
        output_dir=str(resolve_path(cfg.output_dir)),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_steps=cfg.smoke_steps if cfg.smoke_steps else -1,
        report_to=[cfg.report_to] if cfg.report_to != "none" else [],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate)
    trainer.train()
    out = resolve_path(cfg.output_dir)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    log.info("saved fine-tuned LocateAnything (LoRA) → %s", out)
    log.info("next: `motion convert-gguf --ckpt %s` to run it on the Mac", out)
