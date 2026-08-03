"""Unit tests for evaluation/benchmark_runner.py.

Per AGENTS.md, no real BM25/dense index, reranker model, MLflow tracking
server, or Postgres connection is touched — rerankers are injected fakes
(mirroring tests/retrieval/test_reranker_router.py), MLflow is patched
module-level, and Postgres writes use the same fake-session-factory pattern
as tests/api/test_middleware.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from evaluation.benchmark_runner import (
    CONFIG_B_DEFERRED_REASON,
    BenchmarkRow,
    aggregate,
    build_candidate_pools,
    load_benchmark_queries,
    load_relevance_labels,
    log_to_mlflow,
    run_config,
    write_to_postgres,
)
from evaluation.relevance_labeller import BenchmarkQuery, RelevanceLabel, write_jsonl
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


class _FakeBM25:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def search(self, query: str, top_k: int) -> list[Candidate]:
        return self._results


class _FakeDense:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def search(self, query: str, top_k: int) -> list[Candidate]:
        return self._results


# ---------------------------------------------------------------------------
# load_benchmark_queries() / load_relevance_labels() — jsonl round trip
# ---------------------------------------------------------------------------


def test_load_benchmark_queries_round_trips(tmp_path) -> None:
    path = tmp_path / "benchmark_queries.jsonl"
    queries = [BenchmarkQuery(query_id="bq01", query="q1"), BenchmarkQuery(query_id="bq02", query="q2")]
    write_jsonl(path, queries)

    loaded = load_benchmark_queries(path)

    assert loaded == queries


def test_load_relevance_labels_groups_by_query_id(tmp_path) -> None:
    path = tmp_path / "relevance_labels.jsonl"
    labels = [
        RelevanceLabel(query_id="bq01", chunk_id="c1", label=1, rationale="r"),
        RelevanceLabel(query_id="bq01", chunk_id="c2", label=0, rationale="r"),
        RelevanceLabel(query_id="bq02", chunk_id="c3", label=1, rationale="r"),
    ]
    write_jsonl(path, labels)

    relevance = load_relevance_labels(path)

    assert relevance == {"bq01": {"c1": 1, "c2": 0}, "bq02": {"c3": 1}}


# ---------------------------------------------------------------------------
# build_candidate_pools() — shared RRF top-100 pool per query
# ---------------------------------------------------------------------------


def test_build_candidate_pools_fuses_bm25_and_dense_per_query() -> None:
    bm25 = _FakeBM25([_c("b1", rank=1)])
    dense = _FakeDense([_c("b1", rank=1), _c("d1", rank=2)])
    queries = [BenchmarkQuery(query_id="bq01", query="q1")]

    pools = build_candidate_pools(queries, bm25, dense, pool_size=10)

    assert list(pools.keys()) == ["bq01"]
    assert [c.chunk_id for c in pools["bq01"]] == ["b1", "d1"]


def test_build_candidate_pools_one_entry_per_query() -> None:
    bm25 = _FakeBM25([_c("b1", rank=1)])
    dense = _FakeDense([_c("b1", rank=1)])
    queries = [BenchmarkQuery(query_id="bq01", query="q1"), BenchmarkQuery(query_id="bq02", query="q2")]

    pools = build_candidate_pools(queries, bm25, dense, pool_size=10)

    assert set(pools.keys()) == {"bq01", "bq02"}


# ---------------------------------------------------------------------------
# run_config() — reranks each query's pool and scores against relevance
# ---------------------------------------------------------------------------


def _identity_rerank(query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
    return candidates[:top_k]


def test_run_config_produces_one_row_per_query_with_a_pool() -> None:
    queries = [BenchmarkQuery(query_id="bq01", query="q1"), BenchmarkQuery(query_id="bq02", query="q2")]
    pools = {"bq01": [_c("c1", rank=1)], "bq02": [_c("c2", rank=1)]}
    relevance = {"bq01": {"c1": 1}, "bq02": {}}

    rows = run_config("config_A_ms_marco", _identity_rerank, queries, pools, relevance)

    assert [r.query_id for r in rows] == ["bq01", "bq02"]
    assert all(r.config == "config_A_ms_marco" for r in rows)


def test_run_config_ndcg_and_mrr_reflect_relevance_judgment() -> None:
    queries = [BenchmarkQuery(query_id="bq01", query="q1")]
    pools = {"bq01": [_c("c1", rank=1), _c("c2", rank=2)]}
    relevance = {"bq01": {"c1": 1}}

    rows = run_config("config_A_ms_marco", _identity_rerank, queries, pools, relevance)

    assert rows[0].ndcg_at_10 == 1.0
    assert rows[0].mrr == 1.0
    assert rows[0].prec_at_3 == 0.5


def test_run_config_skips_queries_with_no_pool() -> None:
    queries = [BenchmarkQuery(query_id="bq01", query="q1")]
    rows = run_config("config_A_ms_marco", _identity_rerank, queries, pools={}, relevance={})

    assert rows == []


def test_run_config_applies_cost_per_query() -> None:
    queries = [BenchmarkQuery(query_id="bq01", query="q1")]
    pools = {"bq01": [_c("c1", rank=1)]}

    rows = run_config("config_C_cohere_rerank", _identity_rerank, queries, pools, {}, cost_per_query=0.002)

    assert rows[0].cost_usd == 0.002


def test_run_config_latency_ms_is_nonnegative() -> None:
    queries = [BenchmarkQuery(query_id="bq01", query="q1")]
    pools = {"bq01": [_c("c1", rank=1)]}

    rows = run_config("config_A_ms_marco", _identity_rerank, queries, pools, {})

    assert rows[0].latency_ms >= 0


# ---------------------------------------------------------------------------
# aggregate() — mean/total across rows
# ---------------------------------------------------------------------------


def test_aggregate_computes_means_and_total_cost() -> None:
    rows = [
        BenchmarkRow(config="c", query_id="q1", ndcg_at_10=1.0, mrr=1.0, prec_at_3=1.0, prec_at_5=1.0, prec_at_10=1.0, latency_ms=10.0, cost_usd=0.01),
        BenchmarkRow(config="c", query_id="q2", ndcg_at_10=0.0, mrr=0.0, prec_at_3=0.0, prec_at_5=0.0, prec_at_10=0.0, latency_ms=20.0, cost_usd=0.01),
    ]

    result = aggregate(rows)

    assert result["ndcg_at_10"] == 0.5
    assert result["latency_ms"] == 15.0
    assert result["cost_usd"] == 0.02


def test_aggregate_empty_rows_is_all_zero() -> None:
    result = aggregate([])

    assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# log_to_mlflow() — MLflow calls, module patched
# ---------------------------------------------------------------------------


def test_log_to_mlflow_sets_experiment_and_logs_metrics() -> None:
    rows = [
        BenchmarkRow(config="config_A_ms_marco", query_id="q1", ndcg_at_10=1.0, mrr=1.0, prec_at_3=1.0, prec_at_5=1.0, prec_at_10=1.0, latency_ms=10.0, cost_usd=0.0),
    ]

    with patch("evaluation.benchmark_runner.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        log_to_mlflow("config_A_ms_marco", rows)

    mock_mlflow.set_experiment.assert_called_once_with("reranker_benchmark")
    mock_mlflow.start_run.assert_called_once_with(run_name="config_A_ms_marco")
    assert mock_mlflow.log_metric.called
    assert mock_mlflow.log_metrics.called


# ---------------------------------------------------------------------------
# write_to_postgres() — SQLAlchemy ORM writes, fake session factory
# ---------------------------------------------------------------------------


def _configure_session(mock_session_factory: MagicMock) -> MagicMock:
    mock_session = MagicMock()
    mock_session_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_session_factory.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session


def test_write_to_postgres_writes_one_row_per_benchmark_row() -> None:
    rows = [
        BenchmarkRow(config="config_A_ms_marco", query_id="q1", ndcg_at_10=1.0, mrr=1.0, prec_at_3=1.0, prec_at_5=1.0, prec_at_10=1.0, latency_ms=10.0, cost_usd=0.0),
        BenchmarkRow(config="config_A_ms_marco", query_id="q2", ndcg_at_10=0.5, mrr=0.5, prec_at_3=0.5, prec_at_5=0.5, prec_at_10=0.5, latency_ms=12.0, cost_usd=0.0),
    ]
    mock_sf = MagicMock()
    mock_session = _configure_session(mock_sf)

    written = write_to_postgres(rows, run_id="run1", session_factory=mock_sf)

    assert written == 2
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()
    row = mock_session.add.call_args_list[0][0][0]
    assert row.run_id == "run1"
    assert row.config == "config_A_ms_marco"
    assert row.query_id == "q1"


def test_write_to_postgres_returns_zero_and_logs_on_failure() -> None:
    rows = [BenchmarkRow(config="c", query_id="q1", ndcg_at_10=1.0, mrr=1.0, prec_at_3=1.0, prec_at_5=1.0, prec_at_10=1.0, latency_ms=10.0, cost_usd=0.0)]
    mock_sf = MagicMock(side_effect=RuntimeError("db down"))

    written = write_to_postgres(rows, run_id="run1", session_factory=mock_sf)

    assert written == 0


# ---------------------------------------------------------------------------
# Config B — explicitly deferred, not run
# ---------------------------------------------------------------------------


def test_config_b_deferred_reason_documents_oom() -> None:
    assert "bge" in CONFIG_B_DEFERRED_REASON.lower()
    assert "OOM" in CONFIG_B_DEFERRED_REASON or "oom" in CONFIG_B_DEFERRED_REASON.lower()
