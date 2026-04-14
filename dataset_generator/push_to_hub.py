"""Push final JSONL dataset to HuggingFace Hub."""

import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import login
from loguru import logger

from dataset_generator.config import FINAL_DIR, HF_DATASET_REPO, HF_TOKEN

MIN_SAMPLES = 400


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def push_to_hub(dataset_path: Path | None = None, repo_id: str | None = None) -> None:
    """Push final dataset to HuggingFace Hub.

    Args:
        dataset_path: Path to JSONL file. Defaults to data/final/dataset.jsonl.
        repo_id: HuggingFace dataset repo. Defaults to config value.
    """
    dataset_path = dataset_path or FINAL_DIR / "dataset.jsonl"
    repo_id = repo_id or HF_DATASET_REPO

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set. Add it to .env")

    samples = load_jsonl(dataset_path)
    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")

    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"Dataset has only {len(samples)} samples — minimum is {MIN_SAMPLES}. "
            f"Run the pipeline to generate more data first."
        )

    login(token=HF_TOKEN)

    dataset = Dataset.from_list(samples)
    logger.info(f"Pushing {len(dataset)} samples to {repo_id}...")

    dataset.push_to_hub(
        repo_id,
        private=False,
        commit_message=f"Add {len(dataset)} instruction samples",
    )

    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    push_to_hub()
