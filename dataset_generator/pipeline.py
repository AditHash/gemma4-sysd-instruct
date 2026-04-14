"""Main pipeline orchestrator — Generator → Critic → Rewriter → Deduplicator."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from dataset_generator.agents.critic_agent import CriticAgent
from dataset_generator.agents.deduplicator import Deduplicator
from dataset_generator.agents.generator_agent import GeneratorAgent
from dataset_generator.agents.rewriter_agent import RewriterAgent
from dataset_generator.config import (
    FINAL_DIR,
    FILTERED_DIR,
    RAW_DIR,
    TOPICS_FILE,
    MAX_SAMPLES_PER_TOPIC,
)
from dataset_generator.schemas.sample import Sample


def _load_topics() -> list[str]:
    with TOPICS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _save_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples → {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _process_sample(
    sample: dict[str, Any],
    critic: CriticAgent,
    rewriter: RewriterAgent,
) -> dict[str, Any] | None:
    """Run critic → optional rewrite → second critic pass. Returns kept sample or None."""
    try:
        evaluation = await critic.evaluate(sample)
    except Exception as exc:
        logger.error(f"Critic failed: {exc}")
        return None

    if evaluation["verdict"] == "keep":
        return sample

    # Rewrite pass
    try:
        rewritten = await rewriter.rewrite(sample, evaluation.get("reason", ""))
    except Exception as exc:
        logger.error(f"Rewriter failed: {exc}")
        return None

    # Second critic pass on rewritten sample
    try:
        second_eval = await critic.evaluate(rewritten)
    except Exception as exc:
        logger.error(f"Second critic pass failed: {exc}")
        return None

    if second_eval["verdict"] == "keep":
        return rewritten

    logger.debug(f"Discarding after rewrite (still below threshold): {sample.get('instruction', '')[:60]}...")
    return None


async def _process_topic(
    topic: str,
    generator: GeneratorAgent,
    critic: CriticAgent,
    rewriter: RewriterAgent,
    dry_run: bool = False,
    n: int = MAX_SAMPLES_PER_TOPIC,
) -> list[dict[str, Any]]:
    """Process one topic end-to-end. Returns validated, kept samples."""
    logger.info(f"Processing topic: {topic!r}")

    try:
        raw_samples = await generator.generate(topic, n=n)
    except Exception as exc:
        logger.error(f"Generator failed for {topic!r}: {exc}")
        return []

    if dry_run:
        raw_samples = raw_samples[:5]

    tasks = [_process_sample(s, critic, rewriter) for s in raw_samples]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    kept: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            continue
        try:
            validated = Sample(**result)
            kept.append(validated.model_dump())
        except Exception as exc:
            logger.warning(f"Schema validation failed: {exc}")

    logger.info(f"Topic {topic!r}: {len(raw_samples)} raw → {len(kept)} kept")
    return kept


async def run_pipeline(dry_run: bool = False, topics: list[str] | None = None) -> None:
    """Run the full data generation pipeline.

    Args:
        dry_run: If True, process only 1 topic with 5 samples (for testing).
        topics: Override topic list. If None, loads from topics.json.
    """
    topic_list = topics or _load_topics()
    if dry_run:
        topic_list = topic_list[:1]
        logger.info("DRY RUN: processing 1 topic, 5 samples")

    generator = GeneratorAgent()
    critic = CriticAgent()
    rewriter = RewriterAgent()
    deduplicator = Deduplicator()

    all_samples: list[dict[str, Any]] = []

    for topic in topic_list:
        kept = await _process_topic(
            topic,
            generator=generator,
            critic=critic,
            rewriter=rewriter,
            dry_run=dry_run,
            n=5 if dry_run else MAX_SAMPLES_PER_TOPIC,
        )
        all_samples.extend(kept)

        # Save filtered checkpoint after each topic (resume-friendly)
        checkpoint_path = FILTERED_DIR / "checkpoint.jsonl"
        _save_jsonl(all_samples, checkpoint_path)

    logger.info(f"Total kept before dedup: {len(all_samples)}")

    # Deduplicate
    unique_samples = deduplicator.deduplicate(all_samples)

    # Save final
    _save_jsonl(unique_samples, FINAL_DIR / "dataset.jsonl")
    logger.info(f"Pipeline complete. Final dataset: {len(unique_samples)} samples")


if __name__ == "__main__":
    import argparse
    import sys
    from loguru import logger as _logger

    # Ensure logs are visible in all terminals (Windows CMD, PowerShell, Git Bash)
    _logger.remove()
    _logger.add(sys.stderr, level="INFO", colorize=True,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    parser = argparse.ArgumentParser(description="Run dataset generation pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Test run: 1 topic, 5 samples")
    args = parser.parse_args()

    asyncio.run(run_pipeline(dry_run=args.dry_run))
