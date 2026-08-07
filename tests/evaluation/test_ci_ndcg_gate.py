"""Unit tests for evaluation/ci_ndcg_gate.py.

Per AGENTS.md ("Deterministic functions: strict TDD"), `check_ndcg_gate` is
a pure function tested directly with fabricated ConfigResult lists -- no
file I/O, no live benchmark run. `load_baseline` is exercised separately
against a tmp_path fixture file, never the real committed
evaluation/benchmark_baseline.json (so a future baseline refresh can't
silently break this test).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.ci_ndcg_gate import ConfigResult, check_ndcg_gate, load_baseline, main


def _result(config: str, ndcg: float, run_id: str = "run1", num_queries: int = 15) -> ConfigResult:
    return ConfigResult(config=config, run_id=run_id, mean_ndcg_at_10=ndcg, num_queries=num_queries)


# ---------------------------------------------------------------------------
# check_ndcg_gate() -- pure, deterministic
# ---------------------------------------------------------------------------


def test_passes_when_all_configs_meet_threshold() -> None:
    results = [_result("config_A", 0.5333), _result("config_C", 0.5280)]

    gate = check_ndcg_gate(results, threshold=0.50)

    assert gate.passed is True
    assert gate.failed_configs == []


def test_fails_when_any_config_below_threshold() -> None:
    results = [_result("config_A", 0.5333), _result("config_C", 0.42)]

    gate = check_ndcg_gate(results, threshold=0.50)

    assert gate.passed is False
    assert gate.failed_configs == ["config_C"]


def test_exact_threshold_value_passes() -> None:
    results = [_result("config_A", 0.50)]

    gate = check_ndcg_gate(results, threshold=0.50)

    assert gate.passed is True


def test_raises_on_empty_results() -> None:
    with pytest.raises(ValueError):
        check_ndcg_gate([], threshold=0.50)


# ---------------------------------------------------------------------------
# load_baseline() -- file I/O, isolated to a tmp fixture
# ---------------------------------------------------------------------------


def test_load_baseline_parses_committed_shape(tmp_path: Path) -> None:
    fixture = tmp_path / "baseline.json"
    fixture.write_text(
        json.dumps(
            {
                "_note": "fixture",
                "configs": [
                    {"config": "config_A", "run_id": "abc", "mean_ndcg_at_10": 0.55, "num_queries": 15},
                ],
            }
        ),
        encoding="utf-8",
    )

    results = load_baseline(fixture)

    assert results == [ConfigResult(config="config_A", run_id="abc", mean_ndcg_at_10=0.55, num_queries=15)]


# ---------------------------------------------------------------------------
# main() -- against the real committed baseline
# ---------------------------------------------------------------------------


def test_main_passes_against_the_real_committed_baseline() -> None:
    """Real evaluation/benchmark_baseline.json (Day 9's live Config A/C
    numbers) must clear the >0.50 gate -- this is the actual CI check."""
    assert main() == 0
