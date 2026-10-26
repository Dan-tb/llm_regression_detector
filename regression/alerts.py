"""Slack incoming-webhook alerts."""
from __future__ import annotations

import json
import urllib.request

from .compare import CRITICAL, OK, WARN, Comparison

_EMOJI = {OK: ":white_check_mark:", WARN: ":warning:", CRITICAL: ":rotating_light:"}


def headline(cmp: Comparison) -> str:
    if cmp.baseline is None:
        return f"Baseline recorded: {cmp.current.pass_rate:.0%} accuracy on prompt {cmp.current.prompt_version}"
    n = len(cmp.regressions)
    return (
        f"{n} regression{'s' if n != 1 else ''} detected, accuracy "
        f"{'dropped' if cmp.pass_rate_delta < 0 else 'moved'} from {cmp.baseline.pass_rate:.0%} to {cmp.current.pass_rate:.0%}"
    )


def build_slack_payload(cmp: Comparison, report_url: str | None = None) -> dict:
    status = CRITICAL if cmp.status == CRITICAL else WARN if (cmp.status == WARN or cmp.drift_warning) else OK
    text = f"{_EMOJI[status]} *LLM eval: {status.upper()}* - prompt `{cmp.current.prompt_version}` on `{cmp.current.model}`"
    fields = [f"*{headline(cmp)}*"]
    if cmp.drift_warning:
        fields.append(":chart_with_downwards_trend: Slow drift: moving average is below the configured floor.")
    if cmp.regressions:
        fields.append("Regressed cases: " + ", ".join(f"`{f.case_id}`" for f in cmp.regressions[:10]))
    if report_url:
        fields.append(f"<{report_url}|Full diff report>")
    return {"text": text, "blocks": [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(fields)}},
    ]}


def send_slack(webhook_url: str, payload: dict, timeout: int = 10) -> int:
    req = urllib.request.Request(
        webhook_url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL comes from operator config
        return resp.status
