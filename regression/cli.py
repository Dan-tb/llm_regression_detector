"""CLI entry point: python -m regression.cli --prompt prompts/v2.yaml"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .alerts import build_slack_payload, headline, send_slack
from .compare import CRITICAL, Thresholds, compare_runs, detect_drift
from .dataset import load_dataset
from .prompts import load_prompt
from .providers import get_provider
from .report import render_report
from .runner import run_eval
from .store import RunStore


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Run the golden-dataset regression eval for a prompt version.")
    p.add_argument("--prompt", required=True, help="path to a prompt YAML file")
    p.add_argument("--dataset", default="data/golden_v1.json")
    p.add_argument("--provider", choices=["gemini", "openai", "mock"], default=os.environ.get("EVAL_PROVIDER", "gemini"))
    p.add_argument("--model", default=os.environ.get("EVAL_MODEL"), help="override the model named in the prompt YAML")
    p.add_argument("--db", default="results/runs.db")
    p.add_argument("--report", default="results/report.html")
    p.add_argument("--summary-md", default="results/summary.md", help="PR-comment markdown output")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--report-url", default=os.environ.get("REPORT_URL"))
    p.add_argument("--no-save", action="store_true", help="do not record this run as history")
    args = p.parse_args(argv)

    th = Thresholds.from_env(os.environ)
    config, dataset = load_prompt(args.prompt), load_dataset(args.dataset)
    if args.model:
        config.model = args.model
    store = RunStore(args.db)

    run = asyncio.run(run_eval(config, dataset, get_provider(args.provider), args.concurrency, config.model))
    baseline_runs = store.latest(1, dataset_version=dataset.dataset_version)
    cmp = compare_runs(run, baseline_runs[0] if baseline_runs else None, th)

    if not args.no_save:
        store.save(run)
    history = store.history(20)
    cmp.drift_warning = detect_drift([h[2] for h in history], th)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(cmp, history), encoding="utf-8")
    md = (
        f"### LLM eval: {'CRITICAL' if cmp.status == CRITICAL else cmp.status.upper()}\n"
        f"{headline(cmp)}\n\nPrompt `{run.prompt_version}` / model `{run.model}` / dataset `{run.dataset_version}`\n"
        + (f"\nRegressed: {', '.join(f.case_id for f in cmp.regressions)}\n" if cmp.regressions else "")
        + ("\nSlow drift warning: moving average below floor.\n" if cmp.drift_warning else "")
    )
    Path(args.summary_md).write_text(md, encoding="utf-8")
    print(md)

    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            send_slack(webhook, build_slack_payload(cmp, args.report_url))
        except Exception as e:  # alert failure must not mask the eval result
            print(f"warning: Slack alert failed: {e}", file=sys.stderr)

    return 1 if cmp.status == CRITICAL else 0


if __name__ == "__main__":
    sys.exit(main())
