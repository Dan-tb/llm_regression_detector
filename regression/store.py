"""SQLite persistence: one row per run, full per-case JSON alongside it."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import RunResult


class RunStore:
    def __init__(self, path: str | Path = "results/runs.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE, prompt_version TEXT, model TEXT,
                dataset_version TEXT, timestamp TEXT, pass_rate REAL, mean_summary REAL, payload TEXT)"""
        )
        self.db.commit()

    def save(self, run: RunResult) -> None:
        self.db.execute(
            "INSERT INTO runs (run_id, prompt_version, model, dataset_version, timestamp, pass_rate, mean_summary, payload)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run.run_id, run.prompt_version, run.model, run.dataset_version, run.timestamp,
             run.pass_rate, run.mean_summary_score, run.model_dump_json()),
        )
        self.db.commit()

    def latest(self, n: int = 1, dataset_version: str | None = None) -> list[RunResult]:
        """Most recent n runs, newest first. Optionally restricted to one dataset version
        so a changed eval bar is never compared against an old one."""
        q, args = "SELECT payload FROM runs", []
        if dataset_version:
            q += " WHERE dataset_version = ?"
            args.append(dataset_version)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(n)
        return [RunResult.model_validate_json(r[0]) for r in self.db.execute(q, args)]

    def history(self, n: int = 20) -> list[tuple[str, str, float, float]]:
        """(run_id, prompt_version, pass_rate, mean_summary) oldest -> newest."""
        rows = self.db.execute(
            "SELECT run_id, prompt_version, pass_rate, mean_summary FROM runs ORDER BY seq DESC LIMIT ?", (n,)
        ).fetchall()
        return list(reversed(rows))
