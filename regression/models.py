"""Typed interface contracts shared by every part of the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]
CATEGORIES: list[str] = ["billing", "technical", "account", "general"]


class FewShot(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """A versioned prompt. This is the 'code' CI runs against."""

    version: str
    timestamp: str
    model: str = "gemini-3.8-flash"
    system_prompt: str
    few_shot: list[FewShot] = Field(default_factory=list)
    # Only read by the offline mock provider so the demo can simulate a bad prompt.
    mock_profile: Literal["good", "degraded"] = "good"


class ClassifierOutput(BaseModel):
    """Structured output the LLM feature must return."""

    category: Category
    summary: str


class TestCase(BaseModel):
    __test__ = False  # stop pytest trying to collect this as a test class
    id: str
    input: str
    expected_category: Category
    expected_summary: str
    expected_difficulty: Literal["easy", "medium", "hard"] = "easy"
    notes: str = ""


class GoldenDataset(BaseModel):
    dataset_version: str
    cases: list[TestCase]


class CaseResult(BaseModel):
    case_id: str
    expected_category: str
    predicted_category: Optional[str]
    predicted_summary: str
    category_match: bool
    summary_score: int = Field(ge=0, le=5)  # 0 = no/invalid output, 1-5 judge rating
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.category_match


class RunResult(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    dataset_version: str
    timestamp: str
    cases: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def mean_summary_score(self) -> float:
        return sum(c.summary_score for c in self.cases) / len(self.cases) if self.cases else 0.0
