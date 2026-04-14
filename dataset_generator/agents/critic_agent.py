"""Critic Agent — scores each sample and decides keep/rewrite."""

import json
from typing import Any

from groq import AsyncGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from dataset_generator.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    CRITIC_TEMPERATURE,
    CRITIC_THRESHOLD,
    MAX_RETRIES,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)

SYSTEM_PROMPT = """\
You are a strict technical quality evaluator for a backend/AI engineering instruction dataset.

Evaluate each sample on three dimensions (1–5 each):
- technical_accuracy: Is the information correct? No hallucinated libraries or APIs?
- clarity: Is the question clear and the answer easy to follow?
- usefulness: Would a senior Python/backend developer find this genuinely useful?

Rules:
- Be harsh. A score of 5 means exceptional, not just adequate.
- A sample with any factual errors must score ≤ 2 on technical_accuracy.
- Output ONLY valid JSON. No markdown, no explanation.
"""

USER_TEMPLATE = """\
Evaluate this instruction-following sample:

Instruction: {instruction}
Input: {input}
Output: {output}

Return ONLY this JSON:
{{
  "technical_accuracy": <1-5>,
  "clarity": <1-5>,
  "usefulness": <1-5>,
  "total": <sum of above>,
  "verdict": "keep" or "rewrite",
  "reason": "<one sentence>"
}}

verdict must be exactly "keep" if total >= {threshold}, else "rewrite".
"""


class CriticAgent:
    """Scores a single sample and returns structured evaluation."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=GROQ_API_KEY)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    async def evaluate(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one sample. Returns dict with scores + verdict."""
        response = await self._client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=CRITIC_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        instruction=sample.get("instruction", ""),
                        input=sample.get("input", ""),
                        output=sample.get("output", ""),
                        threshold=CRITIC_THRESHOLD,
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"Critic JSON parse failed: {exc}. Retrying...")
            raise

        # Enforce verdict consistency with threshold
        total = result.get("total", 0)
        result["verdict"] = "keep" if total >= CRITIC_THRESHOLD else "rewrite"

        logger.debug(
            f"Critic: total={total} verdict={result['verdict']} | "
            f"{sample.get('instruction', '')[:60]}..."
        )
        return result
