"""Unit tests for streamlit_app/live_data.py's pure/file-based helpers.

MLflow/Postgres-backed fetchers (fetch_live_benchmark_summaries,
fetch_per_query_ndcg, fetch_ragas_scores, fetch_citation_accuracy) are
exercised indirectly by tests/streamlit_app/test_tabs.py's AppTest runs
(which hit real services when available, and the tab modules' own fallback
path otherwise) rather than mocked here individually -- this file covers
the two helpers that need no live service at all: the real committed JSON
artifact and the pure aggregation over it.
"""

from __future__ import annotations

import pandas as pd

from streamlit_app.live_data import citation_accuracy_by_query, load_citation_judgments


def test_load_citation_judgments_reads_the_real_committed_file() -> None:
    df = load_citation_judgments()
    assert len(df) == 27  # Day 9's real judgment count, per mock_data.py's DAY9_CITATION_ACCURACY docstring
    assert set(df.columns) >= {"query_id", "claim", "chunk_id", "supported", "rationale"}


def test_citation_accuracy_by_query_computes_per_query_fraction() -> None:
    judgments = pd.DataFrame(
        [
            {"query_id": "Q1", "supported": True},
            {"query_id": "Q1", "supported": True},
            {"query_id": "Q1", "supported": False},
            {"query_id": "Q2", "supported": True},
        ]
    )
    result = citation_accuracy_by_query(judgments)
    q1 = result[result["query_id"] == "Q1"].iloc[0]
    q2 = result[result["query_id"] == "Q2"].iloc[0]
    assert (q1["supported"], q1["total"]) == (2, 3)
    assert q1["accuracy"] == 2 / 3
    assert (q2["supported"], q2["total"]) == (1, 1)
    assert q2["accuracy"] == 1.0


def test_citation_accuracy_by_query_matches_known_day9_aggregate() -> None:
    # Sanity-check the real file end to end: per-query fractions should
    # aggregate back to mock_data.py's committed DAY9_CITATION_ACCURACY.
    judgments = load_citation_judgments()
    per_query = citation_accuracy_by_query(judgments)
    total_supported = int(per_query["supported"].sum())
    total_claims = int(per_query["total"].sum())
    assert total_claims == 27
    assert round(total_supported / total_claims, 4) == 0.7037
