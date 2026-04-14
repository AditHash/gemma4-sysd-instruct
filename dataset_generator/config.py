import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
FINAL_DIR = DATA_DIR / "final"
TOPICS_FILE = Path(__file__).parent / "topics" / "topics.json"

# ── Groq ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Agent temperatures ─────────────────────────────────────────────────────────
GENERATOR_TEMPERATURE: float = 0.8
CRITIC_TEMPERATURE: float = 0.1
REWRITER_TEMPERATURE: float = 0.4

# ── Pipeline thresholds ────────────────────────────────────────────────────────
MAX_SAMPLES_PER_TOPIC: int = int(os.getenv("MAX_SAMPLES_PER_TOPIC", "20"))
CRITIC_THRESHOLD: int = int(os.getenv("CRITIC_THRESHOLD", "10"))
DEDUP_SIMILARITY_THRESHOLD: float = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# ── Retry ──────────────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_WAIT_MIN: float = 1.0
RETRY_WAIT_MAX: float = 8.0

# ── Deduplication ──────────────────────────────────────────────────────────────
DEDUP_MODEL: str = "all-MiniLM-L6-v2"

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO: str = os.getenv("HF_DATASET_REPO", "AditHash/backend-ai-instruct")
