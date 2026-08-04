"""Layer 8 drift detector: Population Stability Index (PSI) on sampled query
embeddings vs a stored baseline distribution.

Per AGENTS.md's mock-embedding-calls-in-tests rule, `population_stability_index`
is a pure numpy function (no I/O, fully deterministic) and `compute_drift`
takes already-embedded baseline/current arrays — nothing here loads a real
embedding model. `run()` is the Airflow-task-shaped entry point: it takes
injected `fetch_queries_fn` / `embed_fn` callables (e.g. a Postgres query_log
read and DenseIndex's embedding call) so the daily DAG task wires real
dependencies while unit tests supply fakes, mirroring
agents/retrieval_agent.py's constructor-injection convention.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import structlog
from pydantic import BaseModel

log = structlog.get_logger()

DEFAULT_BUCKETS = 10
SIGNIFICANT_DRIFT_THRESHOLD = 0.25  # PSI >= 0.25 -> significant distribution shift
MODERATE_DRIFT_THRESHOLD = 0.10  # 0.10-0.25 -> moderate shift, worth watching


class DriftResult(BaseModel):
    psi: float
    baseline_size: int
    current_size: int
    is_drifted: bool
    severity: str  # "none" | "moderate" | "significant"


def _severity(psi: float) -> str:
    if psi >= SIGNIFICANT_DRIFT_THRESHOLD:
        return "significant"
    if psi >= MODERATE_DRIFT_THRESHOLD:
        return "moderate"
    return "none"


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, buckets: int = DEFAULT_BUCKETS
) -> float:
    """PSI between two 1-D scalar samples, bucketed on baseline quantiles.

    PSI = sum_i (cur_pct_i - base_pct_i) * ln(cur_pct_i / base_pct_i)
    Bucket edges come from baseline quantiles so each baseline bucket holds
    ~1/buckets of the baseline mass by construction; current is binned
    against those same fixed edges. Epsilon-smoothed to avoid log(0)/div-by-0
    when a bucket is empty in one sample.
    """
    if baseline.size == 0 or current.size == 0:
        raise ValueError("baseline and current must be non-empty")

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, buckets + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-4
    base_pct = base_counts / baseline.size + eps
    cur_pct = cur_counts / current.size + eps

    return float(np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct)))


def _project_to_scalar(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine similarity of each embedding to the baseline centroid — a
    single scalar per query capturing how 'on-topic' it is relative to the
    baseline distribution, suitable for PSI bucketing."""
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    norms = np.where(norms == 0, 1e-8, norms)
    return (embeddings @ centroid) / norms


def compute_drift(
    baseline_embeddings: np.ndarray, current_embeddings: np.ndarray, buckets: int = DEFAULT_BUCKETS
) -> DriftResult:
    centroid = baseline_embeddings.mean(axis=0)
    baseline_scores = _project_to_scalar(baseline_embeddings, centroid)
    current_scores = _project_to_scalar(current_embeddings, centroid)

    psi = population_stability_index(baseline_scores, current_scores, buckets=buckets)
    severity = _severity(psi)
    return DriftResult(
        psi=psi,
        baseline_size=len(baseline_embeddings),
        current_size=len(current_embeddings),
        is_drifted=severity != "none",
        severity=severity,
    )


def run(
    baseline_embeddings: np.ndarray,
    fetch_queries_fn: Callable[[], list[str]],
    embed_fn: Callable[[list[str]], np.ndarray],
    buckets: int = DEFAULT_BUCKETS,
) -> DriftResult:
    """Airflow-task-shaped entry point: fetch today's sampled queries, embed
    them, and compare against a pre-computed baseline embedding matrix."""
    queries = fetch_queries_fn()
    log.info("drift_detector_run", stage="drift_detector", num_queries=len(queries))
    if not queries:
        log.warning("drift_detector_no_queries", stage="drift_detector")
        return DriftResult(
            psi=0.0, baseline_size=len(baseline_embeddings), current_size=0, is_drifted=False, severity="none"
        )

    current_embeddings = embed_fn(queries)
    result = compute_drift(baseline_embeddings, current_embeddings, buckets=buckets)
    log.info(
        "drift_detector_complete",
        stage="drift_detector",
        psi=result.psi,
        severity=result.severity,
        is_drifted=result.is_drifted,
    )
    return result
