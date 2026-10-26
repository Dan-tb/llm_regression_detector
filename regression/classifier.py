"""The LLM feature under test: a support-email classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from .models import ClassifierOutput, PromptConfig
from .providers import Provider


@dataclass
class ClassifyResult:
    output: ClassifierOutput | None
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


def build_system_prompt(config: PromptConfig) -> str:
    parts = [config.system_prompt.strip()]
    if config.few_shot:
        parts.append("\nExamples:")
        for ex in config.few_shot:
            parts.append(f"Email: {ex.email}\nOutput: " + json.dumps({"category": ex.category, "summary": ex.summary}))
    return "\n".join(parts)


async def classify_email(email: str, config: PromptConfig, provider: Provider) -> ClassifyResult:
    """Single entry point. The prompt is a parameter, so CI can swap versions freely."""
    resp = await provider.complete(build_system_prompt(config), email, config.model, config)
    try:
        out = ClassifierOutput(**json.loads(resp.text))
        return ClassifyResult(out, resp.prompt_tokens, resp.completion_tokens)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return ClassifyResult(None, resp.prompt_tokens, resp.completion_tokens, error=f"invalid output: {e}"[:200])
