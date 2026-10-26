"""Self-contained HTML diff report (no JS, inline SVG trend chart)."""
from __future__ import annotations

from html import escape

from .compare import CRITICAL, OK, WARN, Comparison
from .models import CATEGORIES

_COLORS = {OK: "#1a7f37", WARN: "#9a6700", CRITICAL: "#cf222e"}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _delta(x: float, pct: bool = True) -> str:
    cls = "neg" if x < 0 else "pos" if x > 0 else ""
    s = f"{x:+.1%}" if pct else f"{x:+.2f}"
    return f'<span class="{cls}">{s}</span>'


def _trend_svg(history: list[tuple[str, str, float, float]], width: int = 560, height: int = 140) -> str:
    if len(history) < 2:
        return "<p class='muted'>Not enough runs for a trend chart yet.</p>"
    pad = 28
    xs = [pad + i * (width - 2 * pad) / (len(history) - 1) for i in range(len(history))]
    lo = min(0.5, min(h[2] for h in history) - 0.05)
    ys = [height - pad - (h[2] - lo) / (1 - lo) * (height - 2 * pad) for h in history]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#0969da"><title>{escape(h[1])}: {h[2]:.1%}</title></circle>'
        f'<text x="{x:.1f}" y="{height - 8}" font-size="10" text-anchor="middle" fill="#57606a">{escape(h[1])}</text>'
        for (x, y, h) in zip(xs, ys, history)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Pass rate over recent runs">'
        f'<text x="4" y="{pad}" font-size="10" fill="#57606a">100%</text>'
        f'<text x="4" y="{height - pad}" font-size="10" fill="#57606a">{lo:.0%}</text>'
        f'<polyline fill="none" stroke="#0969da" stroke-width="2" points="{pts}"/>{dots}</svg>'
    )


def render_report(cmp: Comparison, history: list[tuple[str, str, float, float]]) -> str:
    cur, base = cmp.current, cmp.baseline
    status = CRITICAL if cmp.status == CRITICAL else WARN if (cmp.status == WARN or cmp.drift_warning) else OK
    rows = ""
    if base:
        rows = (
            f"<tr><td>Pass rate</td><td>{_pct(base.pass_rate)}</td><td>{_pct(cur.pass_rate)}</td><td>{_delta(cmp.pass_rate_delta)}</td></tr>"
            f"<tr><td>Mean summary score (1-5)</td><td>{base.mean_summary_score:.2f}</td><td>{cur.mean_summary_score:.2f}</td><td>{_delta(cmp.summary_delta, False)}</td></tr>"
            f"<tr><td>Avg latency</td><td colspan=2></td><td>{cmp.latency_delta_ms:+.1f} ms</td></tr>"
            f"<tr><td>Token usage</td><td colspan=2></td><td>{cmp.token_delta:+d}</td></tr>"
        )
        for c in CATEGORIES:
            rows += f"<tr><td>&nbsp;&nbsp;{c} accuracy</td><td colspan=2></td><td>{_delta(cmp.category_deltas[c])}</td></tr>"
    else:
        rows = f"<tr><td>Pass rate</td><td>n/a</td><td>{_pct(cur.pass_rate)}</td><td>baseline</td></tr>"

    reg_rows = "".join(
        f"<tr><td>{escape(f.case_id)}</td><td>{escape(f.old.expected_category)}</td>"
        f"<td>{escape(str(f.old.predicted_category))}<br><small>{escape(f.old.predicted_summary)}</small></td>"
        f"<td class='bad'>{escape(str(f.new.predicted_category))}<br><small>{escape(f.new.predicted_summary)}</small></td></tr>"
        for f in cmp.regressions
    ) or "<tr><td colspan=4 class='muted'>No regressions.</td></tr>"
    imp_rows = "".join(
        f"<tr><td>{escape(f.case_id)}</td><td>{escape(str(f.old.predicted_category))}</td><td>{escape(str(f.new.predicted_category))}</td></tr>"
        for f in cmp.improvements
    ) or "<tr><td colspan=3 class='muted'>No improvements.</td></tr>"
    notes = "".join(f"<li>{escape(n)}</li>" for n in cmp.notes)
    drift = "<p class='warn'>Slow drift: moving average is below the configured floor.</p>" if cmp.drift_warning else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Eval report {escape(cur.run_id)}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}} th,td{{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}} .badge{{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;background:{_COLORS[status]};font-weight:600}}
.pos{{color:#1a7f37}} .neg,.bad{{color:#cf222e}} .muted{{color:#57606a}} .warn{{color:#9a6700;font-weight:600}} small{{color:#57606a}}
</style></head><body>
<h1>LLM regression report <span class="badge">{status.upper()}</span></h1>
<p class="muted">Run <b>{escape(cur.run_id)}</b> &middot; prompt <b>{escape(cur.prompt_version)}</b> &middot; model <b>{escape(cur.model)}</b>
&middot; dataset <b>{escape(cur.dataset_version)}</b> &middot; {escape(cur.timestamp)}
{f"<br>Baseline: run {escape(base.run_id)} (prompt {escape(base.prompt_version)})" if base else ""}</p>
{drift}<ul>{notes}</ul>
<h2>Scorecard</h2><table><tr><th>Metric</th><th>Baseline</th><th>This run</th><th>Delta</th></tr>{rows}</table>
<h2>Regressions ({len(cmp.regressions)})</h2>
<table><tr><th>Case</th><th>Expected</th><th>Baseline output</th><th>New output</th></tr>{reg_rows}</table>
<h2>Improvements ({len(cmp.improvements)})</h2>
<table><tr><th>Case</th><th>Baseline</th><th>This run</th></tr>{imp_rows}</table>
<h2>Pass-rate trend</h2>{_trend_svg(history)}
</body></html>"""
