"""Load the fine-tuned Gemma 4 adapter from HuggingFace Hub."""

import os
from pathlib import Path

from loguru import logger

HUB_MODEL_ID = os.getenv("HF_MODEL_REPO", "AditHash/gemma4-backend-ai-expert")
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 1024

# Module-level singletons — loaded once at startup
_model = None
_tokenizer = None


def load_model():
    """Load model + tokenizer. Safe to call multiple times (noop after first load)."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    logger.info(f"Loading model from: {HUB_MODEL_ID}")

    # Heavy imports only on GPU machine
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    hf_token = os.getenv("HF_TOKEN", "") or None

    model, tokenizer = FastModel.from_pretrained(
        model_name=HUB_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        token=hf_token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastModel.for_inference(model)

    _model = model
    _tokenizer = tokenizer
    logger.info("Model loaded and ready.")
    return _model, _tokenizer


def generate(question: str) -> str:
    """Generate an answer for a given question. Loads model on first call."""
    import torch

    model, tokenizer = load_model()

    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    if inputs.shape[1] > MAX_SEQ_LENGTH - MAX_NEW_TOKENS:
        raise ValueError(f"Input too long: {inputs.shape[1]} tokens")

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def is_loaded() -> bool:
    return _model is not None
