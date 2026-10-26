"""Diffing, thresholds and slow-drift detection: the core value of the system."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .models import CATEGORIES, CaseResult, RunResult

OK, WARN, CRITICAL = "pass", "warn", "critical"


@dataclass
class Thresholds:
    warn: float = 0.03       # absolute pass-rate drop that triggers a warning
    critical: float = 0.08   # absolute pass-rate drop that triggers a critical failure
    drift_window: int = 7
    drift_floor: float = 0.90  # moving average below this => slow-drift warning

    @classmethod
    def from_env(cls, env) -> "Thresholds":
        return cls(
            warn=float(env.get("WARN_THRESHOLD", cls.warn)),
            critical=float(env.get("CRITICAL_THRESHOLD", cls.critical)),
            drift_window=int(env.get("DRIFT_WINDOW", cls.drift_window)),
            drift_floor=float(env.get("DRIFT_FLOOR", cls.drift_floor)),
        )


@dataclass
class Flip:
    case_id: str
    old: CaseResult
    new: CaseResult


@dataclass
class Comparison:
    current: RunResult
    baseline: RunResult | None
    pass_rate_delta: float = 0.0
    summary_delta: float = 0.0
    category_deltas: dict[str, float] = field(default_factory=dict)
    regressions: list[Flip] = field(default_factory=list)
    improvements: list[Flip] = field(default_factory=list)
    latency_delta_ms: float = 0.0
    token_delta: int = 0
    status: str = OK
    drift_warning: bool = False
    notes: list[str] = field(default_factory=list)


def _category_accuracy(run: RunResult) -> dict[str, float]:
    out = {}
    for cat in CATEGORIES:
        cs = [c for c in run.cases if c.expected_category == cat]
        out[cat] = sum(c.passed for c in cs) / len(cs) if cs else 0.0
    return out


def _mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def compare_runs(current: RunResult, baseline: RunResult | None, th: Thresholds) -> Comparison:
    cmp = Comparison(current=current, baseline=baseline)
    if baseline is None:
        cmp.notes.append("No baseline run found; this run becomes the baseline.")
        return cmp

    if baseline.dataset_version != current.dataset_version:
        cmp.notes.append(
            f"Dataset version changed ({baseline.dataset_version} -> {current.dataset_version}); "
            "deltas compare different eval bars."
        )

    cmp.pass_rate_delta = current.pass_rate - baseline.pass_rate
    cmp.summary_delta = current.mean_summary_score - baseline.mean_summary_score
    cur_cat, base_cat = _category_accuracy(current), _category_accuracy(baseline)
    cmp.category_deltas = {c: cur_cat[c] - base_cat[c] for c in CATEGORIES}
    cmp.latency_delta_ms = _mean(c.latency_ms for c in current.cases) - _mean(c.latency_ms for c in baseline.cases)
    cmp.token_delta = sum(c.prompt_tokens + c.completion_tokens for c in current.cases) - sum(
        c.prompt_tokens + c.completion_tokens for c in baseline.cases
    )

    old = {c.case_id: c for c in baseline.cases}
    for new in current.cases:
        o = old.get(new.case_id)
        if o is None:
            continue
        if o.passed and not new.passed:
            cmp.regressions.append(Flip(new.case_id, o, new))
        elif not o.passed and new.passed:
            cmp.improvements.append(Flip(new.case_id, o, new))

    drop = -cmp.pass_rate_delta
    if drop > th.critical:
        cmp.status = CRITICAL
    elif drop > th.warn:
        cmp.status = WARN
    return cmp


def detect_drift(history_pass_rates: list[float], th: Thresholds) -> bool:
    """True when the moving average of the last N runs is below the floor, even if
    no individual run tripped the per-run threshold. Needs a full window."""
    if len(history_pass_rates) < th.drift_window:
        return False
    return statistics.fmean(history_pass_rates[-th.drift_window:]) < th.drift_floor
