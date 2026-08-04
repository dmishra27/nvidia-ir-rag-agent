"""Unit tests for monitoring/term_shift_monitor.py.

Per AGENTS.md ("Deterministic functions: strict TDD"), `term_frequencies`
and `term_shift` are pure functions tested directly with fabricated text —
no I/O. `run()` is tested with injected fake fetch callables, mirroring
tests/monitoring/test_drift_detector.py's `run()` tests.
"""

from __future__ import annotations

from monitoring.term_shift_monitor import (
    TermShiftResult,
    term_frequencies,
    term_shift,
    run,
)


# ---------------------------------------------------------------------------
# term_frequencies() — pure, deterministic
# ---------------------------------------------------------------------------


def test_term_frequencies_counts_relative_to_total_tokens() -> None:
    freqs = term_frequencies(["cuda cuda malloc"])

    assert freqs["cuda"] == 2 / 3
    assert freqs["malloc"] == 1 / 3


def test_term_frequencies_aggregates_across_documents() -> None:
    freqs = term_frequencies(["cuda malloc", "cuda memcpy"])

    assert freqs["cuda"] == 2 / 4


def test_term_frequencies_empty_corpus_returns_empty_dict() -> None:
    assert term_frequencies([]) == {}


def test_term_frequencies_lowercases_and_strips_punctuation() -> None:
    freqs = term_frequencies(["CUDA, cuda!"])

    assert set(freqs.keys()) == {"cuda"}
    assert freqs["cuda"] == 1.0


def test_term_frequencies_is_deterministic() -> None:
    texts = ["cuda malloc memcpy", "cuda kernel launch"]

    assert term_frequencies(texts) == term_frequencies(texts)


# ---------------------------------------------------------------------------
# term_shift() — baseline vs current diff
# ---------------------------------------------------------------------------


def test_term_shift_no_shift_for_identical_corpora() -> None:
    texts = ["cuda malloc memcpy kernel"] * 10

    result = term_shift(texts, texts)

    assert result.shifted_terms == []
    assert result.new_terms == []
    assert result.dropped_terms == []


def test_term_shift_detects_new_terms() -> None:
    result = term_shift(
        baseline_texts=["cuda malloc"] * 10,
        current_texts=["cuda malloc tensorrt"] * 10,
    )

    assert "tensorrt" in result.new_terms


def test_term_shift_detects_dropped_terms() -> None:
    result = term_shift(
        baseline_texts=["cuda malloc legacy_api"] * 10,
        current_texts=["cuda malloc"] * 10,
    )

    assert "legacy_api" in result.dropped_terms


def test_term_shift_flags_terms_above_threshold_only() -> None:
    result = term_shift(
        baseline_texts=["alpha beta"] * 100,
        current_texts=["alpha alpha alpha beta"] * 100,
        shift_threshold=0.02,
    )

    shifted_names = {s.term for s in result.shifted_terms}
    assert "alpha" in shifted_names


def test_term_shift_respects_top_n() -> None:
    baseline = ["term1 term2 term3 term4 term5"] * 10
    current = ["shift1 shift2 shift3 shift4 shift5"] * 10

    result = term_shift(baseline, current, top_n=2, shift_threshold=0.0)

    assert len(result.shifted_terms) == 2


def test_term_shift_records_doc_counts() -> None:
    result = term_shift(baseline_texts=["a"] * 3, current_texts=["b"] * 7)

    assert result.num_baseline_docs == 3
    assert result.num_current_docs == 7


# ---------------------------------------------------------------------------
# run() — Airflow-task-shaped entry point, injected fakes
# ---------------------------------------------------------------------------


def test_run_calls_both_fetch_fns_and_returns_result() -> None:
    calls: list = []

    def fetch_baseline() -> list[str]:
        calls.append("baseline")
        return ["cuda malloc"] * 5

    def fetch_current() -> list[str]:
        calls.append("current")
        return ["cuda tensorrt"] * 5

    result = run(fetch_baseline, fetch_current)

    assert isinstance(result, TermShiftResult)
    assert calls == ["baseline", "current"]
    assert "tensorrt" in result.new_terms


def test_run_empty_baseline_and_current() -> None:
    result = run(lambda: [], lambda: [])

    assert result.num_baseline_docs == 0
    assert result.num_current_docs == 0
    assert result.shifted_terms == []
