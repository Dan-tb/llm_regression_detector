"""LLM providers. Gemini (default) and OpenAI for real runs; a deterministic mock for offline demos and CI tests."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .models import PromptConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Provider(Protocol):
    async def complete(self, system: str, user: str, model: str, config: PromptConfig | None = None) -> LLMResponse: ...


class GeminiProvider:
    """Google Gemini via the google-genai SDK. Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment."""

    def __init__(self, max_retries: int = 5) -> None:
        from google import genai  # imported lazily so mock mode needs no key or SDK

        self.client = genai.Client()
        self.max_retries = max_retries

    async def complete(self, system, user, model, config=None) -> LLMResponse:
        import asyncio

        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=0, response_mime_type="application/json"
        )
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.aio.models.generate_content(model=model, contents=user, config=cfg)
                break
            except Exception as e:  # retry rate limits / transient server errors only
                code = getattr(e, "code", None)
                if code not in (429, 500, 502, 503, 504) or attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        u = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            prompt_tokens=(u.prompt_token_count or 0) if u else 0,
            completion_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
        )


class OpenAIProvider:  # optional: `pip install openai` to use
    def __init__(self) -> None:
        from openai import AsyncOpenAI  # imported lazily so mock mode needs no key

        self.client = AsyncOpenAI()

    async def complete(self, system, user, model, config=None) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


_KEYWORDS = {
    "billing": ["charge", "charged", "invoice", "refund", "payment", "billing", "subscription fee", "receipt", "price", "card"],
    "technical": ["error", "crash", "bug", "not working", "doesn't work", "won't load", "timeout", "500", "broken", "sync", "app keeps"],
    "account": ["password", "login", "log in", "sign in", "email address", "username", "account", "2fa", "locked", "delete my"],
}


def _mock_classify(text: str, profile: str) -> str:
    t = text.lower()
    scores = {c: sum(k in t for k in kws) for c, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    category = best if scores[best] > 0 else "general"
    if profile == "degraded" and category == "account":
        return "general"  # simulated regression: prompt change broke account detection
    return category


class MockProvider:
    """Deterministic keyword classifier + word-overlap judge. No network, no cost."""

    async def complete(self, system, user, model, config=None) -> LLMResponse:
        if user.startswith("JUDGE::"):
            _, expected, actual = user.split("::", 2)
            a, b = set(re.findall(r"\w+", expected.lower())), set(re.findall(r"\w+", actual.lower()))
            overlap = len(a & b) / max(len(a | b), 1)
            score = 1 + round(overlap * 8)
            return LLMResponse(json.dumps({"score": min(5, max(1, score))}), 40, 5)
        profile = config.mock_profile if config else "good"
        first = re.split(r"(?<=[.!?])\s", user.strip())[0][:140]
        out = {"category": _mock_classify(user, profile), "summary": first}
        return LLMResponse(json.dumps(out), len(user.split()) + 60, 30)


def get_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY is not set (use --provider mock for an offline run)")
        return GeminiProvider()
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (use --provider mock for an offline run)")
        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name}")
