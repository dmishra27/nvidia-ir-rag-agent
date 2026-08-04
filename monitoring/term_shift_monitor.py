"""Layer 8 term-shift monitor: BM25 vocabulary term-frequency drift vs a
30-day baseline.

`_tokenize` mirrors retrieval/bm25_index.py's word-boundary regex exactly
(same local-duplication convention as retrieval/chunk_quality.py) so term
stats here bucket identically to what BM25Index actually indexes.
`term_frequencies` and `term_shift` are pure, deterministic functions over
already-fetched text — no I/O, no model — per AGENTS.md's mock-everything-
in-tests rule; `run()` is the Airflow-task-shaped entry point that wires a
real corpus fetch (30-day-old baseline snapshot vs current chunks/queries)
via injected callables, mirroring monitoring/drift_detector.py's `run()`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import structlog
from pydantic import BaseModel

log = structlog.get_logger()

DEFAULT_TOP_N = 20
SHIFT_THRESHOLD = 0.02  # min |delta relative frequency| to flag a term


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class TermShift(BaseModel):
    term: str
    baseline_freq: float
    current_freq: float
    delta: float


class TermShiftResult(BaseModel):
    shifted_terms: list[TermShift]
    num_baseline_docs: int
    num_current_docs: int
    new_terms: list[str]  # in current, absent from baseline
    dropped_terms: list[str]  # in baseline, absent from current


def term_frequencies(texts: list[str]) -> dict[str, float]:
    """Relative term frequency across a corpus: count(term) / total_tokens."""
    counts: Counter[str] = Counter()
    total = 0
    for text in texts:
        tokens = _tokenize(text)
        counts.update(tokens)
        total += len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def term_shift(
    baseline_texts: list[str],
    current_texts: list[str],
    top_n: int = DEFAULT_TOP_N,
    shift_threshold: float = SHIFT_THRESHOLD,
) -> TermShiftResult:
    baseline_freq = term_frequencies(baseline_texts)
    current_freq = term_frequencies(current_texts)

    all_terms = set(baseline_freq) | set(current_freq)
    shifts = [
        TermShift(
            term=term,
            baseline_freq=baseline_freq.get(term, 0.0),
            current_freq=current_freq.get(term, 0.0),
            delta=current_freq.get(term, 0.0) - baseline_freq.get(term, 0.0),
        )
        for term in all_terms
    ]
    shifted = sorted(
        (s for s in shifts if abs(s.delta) >= shift_threshold),
        key=lambda s: abs(s.delta),
        reverse=True,
    )[:top_n]

    return TermShiftResult(
        shifted_terms=shifted,
        num_baseline_docs=len(baseline_texts),
        num_current_docs=len(current_texts),
        new_terms=sorted(t for t in current_freq if t not in baseline_freq),
        dropped_terms=sorted(t for t in baseline_freq if t not in current_freq),
    )


def run(
    fetch_baseline_texts_fn: Callable[[], list[str]],
    fetch_current_texts_fn: Callable[[], list[str]],
    top_n: int = DEFAULT_TOP_N,
    shift_threshold: float = SHIFT_THRESHOLD,
) -> TermShiftResult:
    """Airflow-task-shaped entry point: fetch the 30-day baseline snapshot
    and today's corpus, and diff their term-frequency distributions."""
    baseline_texts = fetch_baseline_texts_fn()
    current_texts = fetch_current_texts_fn()
    log.info(
        "term_shift_monitor_run",
        stage="term_shift_monitor",
        num_baseline=len(baseline_texts),
        num_current=len(current_texts),
    )
    result = term_shift(baseline_texts, current_texts, top_n=top_n, shift_threshold=shift_threshold)
    log.info(
        "term_shift_monitor_complete",
        stage="term_shift_monitor",
        num_shifted=len(result.shifted_terms),
        num_new=len(result.new_terms),
        num_dropped=len(result.dropped_terms),
    )
    return result
