"""Deduplicator — removes near-duplicate samples using cosine similarity."""

from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from dataset_generator.config import DEDUP_MODEL, DEDUP_SIMILARITY_THRESHOLD


class Deduplicator:
    """Removes near-duplicate samples based on instruction embedding similarity."""

    def __init__(self) -> None:
        logger.info(f"Loading dedup model: {DEDUP_MODEL}")
        self._model = SentenceTransformer(DEDUP_MODEL)

    def deduplicate(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return samples with near-duplicates removed.

        Keeps the first occurrence when similarity > threshold.
        Runs entirely locally — no API calls.
        """
        if not samples:
            return []

        instructions = [s.get("instruction", "") for s in samples]
        logger.info(f"Encoding {len(instructions)} instructions for deduplication...")

        embeddings = self._model.encode(instructions, normalize_embeddings=True, show_progress_bar=True)
        # embeddings shape: (n, dim) — already L2-normalized, so dot product = cosine sim
        sim_matrix: np.ndarray = np.dot(embeddings, embeddings.T)

        keep_indices: list[int] = []
        dropped = 0

        for i in range(len(samples)):
            is_duplicate = False
            for j in keep_indices:
                if sim_matrix[i, j] > DEDUP_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    dropped += 1
                    logger.debug(
                        f"Dropping duplicate (sim={sim_matrix[i, j]:.3f}): "
                        f"{instructions[i][:60]}..."
                    )
                    break
            if not is_duplicate:
                keep_indices.append(i)

        unique = [samples[i] for i in keep_indices]
        logger.info(f"Deduplication complete: {len(samples)} → {len(unique)} samples ({dropped} dropped)")
        return unique
