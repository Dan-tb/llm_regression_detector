from regression.compare import CRITICAL, OK, WARN, Thresholds, compare_runs, detect_drift
from regression.models import CaseResult, RunResult


def _run(rid, passes, version="v1", ds="d1"):
    cases = [
        CaseResult(case_id=f"c{i}", expected_category="billing", predicted_category="billing" if p else "general",
                   predicted_summary="s", category_match=p, summary_score=4, latency_ms=10, prompt_tokens=5, completion_tokens=5)
        for i, p in enumerate(passes)
    ]
    return RunResult(run_id=rid, prompt_version=version, model="m", dataset_version=ds, timestamp="t", cases=cases)


def test_no_baseline_is_ok():
    assert compare_runs(_run("a", [True] * 10), None, Thresholds()).status == OK


def test_flips_are_detected():
    base, cur = _run("a", [True, True, False, True]), _run("b", [True, False, True, True])
    cmp = compare_runs(cur, base, Thresholds())
    assert [f.case_id for f in cmp.regressions] == ["c1"]
    assert [f.case_id for f in cmp.improvements] == ["c2"]


def test_thresholds_warn_and_critical():
    base = _run("a", [True] * 100)
    assert compare_runs(_run("b", [True] * 96 + [False] * 4), base, Thresholds()).status == WARN
    assert compare_runs(_run("c", [True] * 90 + [False] * 10), base, Thresholds()).status == CRITICAL
    assert compare_runs(_run("d", [True] * 99 + [False]), base, Thresholds()).status == OK


def test_thresholds_configurable():
    base = _run("a", [True] * 100)
    assert compare_runs(_run("b", [True] * 99 + [False]), base, Thresholds(warn=0.005)).status == WARN


def test_drift_needs_full_window_and_low_average():
    th = Thresholds(drift_window=3, drift_floor=0.9)
    assert not detect_drift([0.85, 0.85], th)
    assert detect_drift([0.95, 0.88, 0.86, 0.85], th)
    assert not detect_drift([0.95, 0.95, 0.94], th)


def test_dataset_version_change_is_noted():
    cmp = compare_runs(_run("b", [True], ds="d2"), _run("a", [True], ds="d1"), Thresholds())
    assert any("Dataset version changed" in n for n in cmp.notes)
