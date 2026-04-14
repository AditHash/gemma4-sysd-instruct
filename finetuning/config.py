"""Fine-tuning configuration — all hyperparameters in one place."""

# ── Model ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/gemma-4-E4B-it"
# CRITICAL: Text-only fine-tuning → FastModel, NOT FastVisionModel
# from unsloth import FastModel  ← correct
# from unsloth import FastVisionModel  ← WRONG for this project

DATASET_REPO = "AditHash/backend-ai-instruct"

# ── Quantization ───────────────────────────────────────────────────────────────
LOAD_IN_4BIT = True          # QLoRA — E4B QLoRA fits on Colab T4 (16GB)

# ── LoRA ───────────────────────────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_DROPOUT = 0.0           # 0 is optimal for Unsloth

# ── Training ───────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4   # Effective batch size = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"

# ── Unsloth-specific (CRITICAL) ────────────────────────────────────────────────
# "unsloth" not True — reduces VRAM + extends context 4x
# Never set use_cache=False manually — Unsloth handles KV cache bugs automatically
USE_GRADIENT_CHECKPOINTING = "unsloth"

# ── Chat template ──────────────────────────────────────────────────────────────
# Use "gemma-4" for E2B/E4B (non-thinking)
# Use "gemma-4-thinking" ONLY for 26B and 31B
CHAT_TEMPLATE = "gemma-4"

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"
PUSH_TO_HUB = True
HUB_MODEL_ID = "AditHash/gemma4-backend-ai-expert"
