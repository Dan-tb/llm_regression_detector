"""LLM-as-judge for summary relevance (1-5)."""
from __future__ import annotations

import json

from .providers import Provider

JUDGE_SYSTEM = (
    "You grade a one-sentence summary of a customer email against a reference summary. "
    "Rate 1 (unrelated/wrong) to 5 (captures the same key point). "
    'Reply with JSON only: {"score": <1-5>}.'
)


async def judge_summary(expected: str, actual: str, provider: Provider, model: str) -> tuple[int, int, int]:
    """Returns (score, prompt_tokens, completion_tokens)."""
    user = f"JUDGE::{expected}::{actual}"
    if not actual.strip():
        return 1, 0, 0
    resp = await provider.complete(JUDGE_SYSTEM, user, model, None)
    try:
        score = int(json.loads(resp.text)["score"])
        return min(5, max(1, score)), resp.prompt_tokens, resp.completion_tokens
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 1, resp.prompt_tokens, resp.completion_tokens
