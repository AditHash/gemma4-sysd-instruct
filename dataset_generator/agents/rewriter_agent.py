"""Rewriter Agent — rewrites rejected samples based on critic feedback."""

import json
from typing import Any

from groq import AsyncGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from dataset_generator.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    REWRITER_TEMPERATURE,
    MAX_RETRIES,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)

SYSTEM_PROMPT = """\
You are a technical content editor. You will receive a low-quality instruction-following sample
and a critic's feedback explaining what is wrong with it.

Your job: rewrite the sample to fix all identified issues while keeping the same topic.

Rules:
- Domain: backend engineering + AI/ML development (Python, FastAPI, AWS, Docker, LLM, RAG)
- Output must be technically accurate — no hallucinated libraries or APIs
- Keep the same general topic/instruction, but improve the content substantially
- Output must be a valid JSON object — no markdown fences, no preamble
"""

USER_TEMPLATE = """\
Original sample (low quality):
Instruction: {instruction}
Input: {input}
Output: {output}

Critic feedback: {reason}

Rewrite this sample to fix the issues. Return ONLY this JSON:
{{
  "instruction": "improved instruction",
  "input": "",
  "output": "substantially improved output (3+ sentences, correct code if applicable)"
}}
"""


class RewriterAgent:
    """Rewrites a rejected sample using critic feedback."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=GROQ_API_KEY)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    async def rewrite(self, sample: dict[str, Any], critic_reason: str) -> dict[str, Any]:
        """Rewrite `sample` given `critic_reason`. Returns new sample dict."""
        logger.debug(f"Rewriting sample: {sample.get('instruction', '')[:60]}...")

        response = await self._client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=REWRITER_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        instruction=sample.get("instruction", ""),
                        input=sample.get("input", ""),
                        output=sample.get("output", ""),
                        reason=critic_reason,
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        try:
            rewritten = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"Rewriter JSON parse failed: {exc}. Retrying...")
            raise

        rewritten.setdefault("input", "")
        logger.debug(f"Rewritten: {rewritten.get('instruction', '')[:60]}...")
        return rewritten
