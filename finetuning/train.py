"""
Fine-tuning script for Gemma 4 E4B — text-only QLoRA via Unsloth.

Run on Google Colab T4 (free tier) or any GPU with 10GB+ VRAM.

Usage:
    python finetuning/train.py

Required on Colab:
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install datasets huggingface-hub
"""

import os
from pathlib import Path

# ── Load config ────────────────────────────────────────────────────────────────
from finetuning.config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHAT_TEMPLATE,
    DATASET_REPO,
    EPOCHS,
    GRADIENT_ACCUMULATION,
    HUB_MODEL_ID,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LR_SCHEDULER,
    MAX_SEQ_LENGTH,
    OPTIMIZER,
    OUTPUT_DIR,
    PUSH_TO_HUB,
    TARGET_MODULES,
    USE_GRADIENT_CHECKPOINTING,
    WARMUP_RATIO,
)


def format_sample(sample: dict, tokenizer) -> dict:
    """Convert Alpaca-style sample to chat-formatted text."""
    messages = [
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def train() -> None:
    """Run full QLoRA fine-tuning pipeline."""
    # ── Imports (heavy — only available on GPU machine) ────────────────────────
    from unsloth import FastModel  # NOT FastVisionModel — text-only
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # ── Load model + tokenizer ─────────────────────────────────────────────────
    print(f"Loading model: {BASE_MODEL}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        # use_cache handled by Unsloth internally — do NOT set use_cache=False
    )

    # ── Apply LoRA adapters ────────────────────────────────────────────────────
    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=42,
    )

    # ── Set chat template ──────────────────────────────────────────────────────
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    # ── Load + format dataset ──────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET_REPO}")
    raw_dataset = load_dataset(DATASET_REPO, split="train")

    dataset = raw_dataset.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=raw_dataset.column_names,
    )
    print(f"Training samples: {len(dataset)}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=LR_SCHEDULER,
            optim=OPTIMIZER,
            fp16=not __import__("torch").cuda.is_bf16_supported(),
            bf16=__import__("torch").cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

    # ── Push to Hub ────────────────────────────────────────────────────────────
    if PUSH_TO_HUB:
        hf_token = os.getenv("HF_TOKEN", "")
        if not hf_token:
            print("WARNING: HF_TOKEN not set — skipping push to Hub")
        else:
            print(f"Pushing adapter to: {HUB_MODEL_ID}")
            model.push_to_hub(HUB_MODEL_ID, token=hf_token)
            tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token)
            print(f"Pushed to https://huggingface.co/{HUB_MODEL_ID}")
    else:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved locally to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
