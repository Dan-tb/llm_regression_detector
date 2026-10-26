"""Async test runner with bounded concurrency (keeps rate limits and cost under control)."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from .classifier import classify_email
from .judge import judge_summary
from .models import CaseResult, GoldenDataset, PromptConfig, RunResult, TestCase
from .providers import Provider


async def _run_case(case: TestCase, config: PromptConfig, provider: Provider, judge_model: str, sem: asyncio.Semaphore) -> CaseResult:
    async with sem:
        start = time.perf_counter()
        res = await classify_email(case.input, config, provider)
        latency = (time.perf_counter() - start) * 1000
        if res.output is None:
            return CaseResult(
                case_id=case.id, expected_category=case.expected_category, predicted_category=None,
                predicted_summary="", category_match=False, summary_score=0, latency_ms=latency,
                prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, error=res.error,
            )
        score, jp, jc = await judge_summary(case.expected_summary, res.output.summary, provider, judge_model)
        return CaseResult(
            case_id=case.id, expected_category=case.expected_category,
            predicted_category=res.output.category, predicted_summary=res.output.summary,
            category_match=res.output.category == case.expected_category, summary_score=score,
            latency_ms=latency, prompt_tokens=res.prompt_tokens + jp, completion_tokens=res.completion_tokens + jc,
        )


async def run_eval(config: PromptConfig, dataset: GoldenDataset, provider: Provider,
                   concurrency: int = 8, judge_model: str = "gemini-3.8-flash") -> RunResult:
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(_run_case(c, config, provider, judge_model, sem) for c in dataset.cases))
    return RunResult(
        run_id=uuid.uuid4().hex[:8], prompt_version=config.version, model=config.model,
        dataset_version=dataset.dataset_version, timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cases=list(results),
    )
