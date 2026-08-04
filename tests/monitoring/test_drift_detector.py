"""Unit tests for monitoring/drift_detector.py.

Per AGENTS.md ("Deterministic functions: strict TDD"), `population_stability_index`
and `compute_drift` are pure numpy functions tested directly with fabricated
arrays — no embedding model, no I/O. `run()` is tested with injected fake
fetch/embed callables, mirroring tests/agents/test_retrieval_agent.py's
constructor-injection convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from monitoring.drift_detector import (
    DriftResult,
    compute_drift,
    population_stability_index,
    run,
)


# ---------------------------------------------------------------------------
# population_stability_index() — pure, deterministic
# ---------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    baseline = rng.normal(size=1000)

    psi = population_stability_index(baseline, baseline.copy())

    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_is_small_for_similar_distributions() -> None:
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=2000)
    current = rng.normal(loc=0.0, scale=1.0, size=2000)

    psi = population_stability_index(baseline, current)

    assert 0.0 <= psi < 0.10


def test_psi_is_large_for_shifted_distributions() -> None:
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=2000)
    current = rng.normal(loc=5.0, scale=1.0, size=2000)

    psi = population_stability_index(baseline, current)

    assert psi >= 0.25


def test_psi_is_deterministic_given_same_inputs() -> None:
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    current = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    psi_1 = population_stability_index(baseline, current, buckets=5)
    psi_2 = population_stability_index(baseline, current, buckets=5)

    assert psi_1 == psi_2


def test_psi_raises_on_empty_baseline() -> None:
    with pytest.raises(ValueError):
        population_stability_index(np.array([]), np.array([1.0, 2.0]))


def test_psi_raises_on_empty_current() -> None:
    with pytest.raises(ValueError):
        population_stability_index(np.array([1.0, 2.0]), np.array([]))


def test_psi_is_zero_for_constant_baseline() -> None:
    baseline = np.ones(100)
    current = np.ones(100) * 5

    psi = population_stability_index(baseline, current)

    assert psi == 0.0


# ---------------------------------------------------------------------------
# compute_drift() — embedding-shaped wrapper
# ---------------------------------------------------------------------------


def test_compute_drift_no_drift_for_identical_embeddings() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 16))

    result = compute_drift(embeddings, embeddings.copy())

    assert isinstance(result, DriftResult)
    assert result.severity == "none"
    assert result.is_drifted is False
    assert result.baseline_size == 200
    assert result.current_size == 200


def test_compute_drift_flags_significant_drift() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0.0, size=(200, 16))
    current = rng.normal(loc=10.0, size=(200, 16))

    result = compute_drift(baseline, current)

    assert result.severity == "significant"
    assert result.is_drifted is True
    assert result.psi >= 0.25


# ---------------------------------------------------------------------------
# run() — Airflow-task-shaped entry point, injected fakes
# ---------------------------------------------------------------------------


def test_run_returns_no_drift_result_when_no_queries_fetched() -> None:
    baseline = np.ones((10, 4))

    result = run(baseline, fetch_queries_fn=lambda: [], embed_fn=lambda qs: np.ones((len(qs), 4)))

    assert result.current_size == 0
    assert result.is_drifted is False
    assert result.severity == "none"


def test_run_calls_embed_fn_with_fetched_queries() -> None:
    baseline = np.ones((10, 4))
    calls: list = []

    def embed_fn(queries: list[str]) -> np.ndarray:
        calls.append(queries)
        return np.ones((len(queries), 4))

    run(baseline, fetch_queries_fn=lambda: ["q1", "q2"], embed_fn=embed_fn)

    assert calls == [["q1", "q2"]]


def test_run_returns_drift_result_from_embed_fn_output() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=(50, 8))

    result = run(
        baseline,
        fetch_queries_fn=lambda: ["q1"] * 50,
        embed_fn=lambda qs: rng.normal(loc=10.0, size=(len(qs), 8)),
    )

    assert isinstance(result, DriftResult)
    assert result.current_size == 50
