"""Generator Agent — produces raw Q&A pairs for a given topic."""

import json
from typing import Any

from groq import AsyncGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from dataset_generator.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GENERATOR_TEMPERATURE,
    MAX_SAMPLES_PER_TOPIC,
    MAX_RETRIES,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)

SYSTEM_PROMPT = """\
You are an expert technical content creator specializing in backend engineering and AI/ML development.

Your task is to generate high-quality instruction-following samples for fine-tuning a language model.

Domain: Backend engineering + AI/ML development
Target audience: Senior Python/backend developers
Focus areas: Python, FastAPI, AWS, Docker, LLM integration, RAG, vector databases, async patterns

Rules:
- Do NOT generate Java, PHP, Ruby, or JavaScript content
- Every instruction must be a clear, specific technical question or task
- Every output must be technically accurate with no hallucinated libraries or APIs
- Code examples must be runnable Python 3.11+ — use type hints, async/await where appropriate
- Output must be a valid JSON array — no markdown fences, no preamble, no trailing text
- Each output field must be at least 3 sentences long and genuinely useful
"""

USER_TEMPLATE = """\
Generate exactly {n} diverse instruction-following samples about: {topic}

Return a JSON array of objects with this exact shape:
[
  {{
    "instruction": "Clear, specific technical question or task",
    "input": "",
    "output": "Detailed, technically accurate answer (3+ sentences, code where relevant)"
  }}
]

Output ONLY the JSON array. No markdown. No explanation. No preamble.
"""


class GeneratorAgent:
    """Generates raw instruction samples for a single topic via Groq."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=GROQ_API_KEY)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    async def generate(self, topic: str, n: int = MAX_SAMPLES_PER_TOPIC) -> list[dict[str, Any]]:
        """Generate `n` raw samples for `topic`. Returns list of dicts."""
        logger.debug(f"Generating {n} samples for topic: {topic!r}")

        response = await self._client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GENERATOR_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(n=n, topic=topic)},
            ],
        )

        raw = response.choices[0].message.content.strip()

        try:
            samples = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON parse failed for topic {topic!r}: {exc}. Retrying...")
            raise

        if not isinstance(samples, list):
            raise ValueError(f"Expected list, got {type(samples)} for topic {topic!r}")

        logger.info(f"Generated {len(samples)} raw samples for: {topic!r}")
        return samples
