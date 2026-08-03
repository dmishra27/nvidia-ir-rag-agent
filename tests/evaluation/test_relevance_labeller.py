"""Unit tests for evaluation/relevance_labeller.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these
tests never load BM25/dense indices or call a real Anthropic client. Fake
BM25/dense objects stand in for retrieval, and a MagicMock response (mirroring
tests/agents/test_qa_agent.py's convention) stands in for Claude.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from evaluation.relevance_labeller import (
    BENCHMARK_QUERIES,
    BenchmarkQuery,
    RelevanceLabel,
    RelevancePair,
    build_pairs,
    label_pairs,
    write_jsonl,
)
from retrieval.candidates import Candidate


def _c(chunk_id: str, text: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=text, score=1.0, rank=rank)


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


def _judge_response(relevant: bool, rationale: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "judge_relevance"
    block.input = {"relevant": relevant, "rationale": rationale}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _text_only_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# BENCHMARK_QUERIES — the fixed 50-query set
# ---------------------------------------------------------------------------


def test_benchmark_queries_has_fifty_entries() -> None:
    assert len(BENCHMARK_QUERIES) == 50


def test_benchmark_query_ids_are_unique() -> None:
    ids = [q.query_id for q in BENCHMARK_QUERIES]
    assert len(set(ids)) == len(ids)


def test_benchmark_query_texts_are_nonempty() -> None:
    assert all(q.query.strip() for q in BENCHMARK_QUERIES)


# ---------------------------------------------------------------------------
# build_pairs() — retrieval, RRF fusion, top-1 candidate per query
# ---------------------------------------------------------------------------


def test_build_pairs_takes_top_ranked_fused_candidate_per_query() -> None:
    bm25 = _FakeBM25([_c("b1", "bm25 text", rank=1)])
    dense = _FakeDense([_c("b1", "bm25 text", rank=1), _c("d2", "dense text", rank=2)])
    queries = [BenchmarkQuery(query_id="q1", query="cudaMalloc parameters")]

    pairs = build_pairs(queries, bm25, dense)

    assert len(pairs) == 1
    assert pairs[0].query_id == "q1"
    assert pairs[0].query == "cudaMalloc parameters"
    assert pairs[0].chunk_id == "b1"
    assert pairs[0].passage_text == "bm25 text"


def test_build_pairs_produces_one_pair_per_query() -> None:
    bm25 = _FakeBM25([_c("b1", "text", rank=1)])
    dense = _FakeDense([_c("b1", "text", rank=1)])
    queries = [
        BenchmarkQuery(query_id="q1", query="query one"),
        BenchmarkQuery(query_id="q2", query="query two"),
    ]

    pairs = build_pairs(queries, bm25, dense)

    assert [p.query_id for p in pairs] == ["q1", "q2"]


def test_build_pairs_skips_query_with_no_fused_results() -> None:
    bm25 = _FakeBM25([])
    dense = _FakeDense([])
    queries = [BenchmarkQuery(query_id="q1", query="no results query")]

    pairs = build_pairs(queries, bm25, dense)

    assert pairs == []


# ---------------------------------------------------------------------------
# label_pairs() — Claude Sonnet forced tool call, mocked
# ---------------------------------------------------------------------------


def test_label_pairs_relevant_maps_to_label_one() -> None:
    pair = RelevancePair(query_id="q1", query="cudaMalloc parameters", chunk_id="c1", passage_text="cudaMalloc(void**, size_t)")
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "directly answers the query")

    labels = label_pairs([pair], client)

    assert labels == [RelevanceLabel(query_id="q1", chunk_id="c1", label=1, rationale="directly answers the query")]


def test_label_pairs_not_relevant_maps_to_label_zero() -> None:
    pair = RelevancePair(query_id="q1", query="cudaMalloc parameters", chunk_id="c1", passage_text="unrelated text about NVLink")
    client = MagicMock()
    client.messages.create.return_value = _judge_response(False, "does not mention cudaMalloc")

    labels = label_pairs([pair], client)

    assert labels[0].label == 0


def test_label_pairs_forces_judge_relevance_tool_choice() -> None:
    pair = RelevancePair(query_id="q1", query="q", chunk_id="c1", passage_text="p")
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "r")

    label_pairs([pair], client)

    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "judge_relevance"}
    assert kwargs["tools"][0]["name"] == "judge_relevance"


def test_label_pairs_multiple_pairs_each_get_their_own_call() -> None:
    pairs = [
        RelevancePair(query_id="q1", query="q1", chunk_id="c1", passage_text="p1"),
        RelevancePair(query_id="q2", query="q2", chunk_id="c2", passage_text="p2"),
    ]
    client = MagicMock()
    client.messages.create.side_effect = [
        _judge_response(True, "r1"),
        _judge_response(False, "r2"),
    ]

    labels = label_pairs(pairs, client)

    assert client.messages.create.call_count == 2
    assert [label.label for label in labels] == [1, 0]


def test_label_pairs_skips_pair_on_missing_tool_use_block() -> None:
    pair = RelevancePair(query_id="q1", query="q", chunk_id="c1", passage_text="p")
    client = MagicMock()
    client.messages.create.return_value = _text_only_response("no structured output")

    labels = label_pairs([pair], client)

    assert labels == []


def test_label_pairs_skips_pair_on_client_error() -> None:
    pair = RelevancePair(query_id="q1", query="q", chunk_id="c1", passage_text="p")
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("api error")

    labels = label_pairs([pair], client)

    assert labels == []


# ---------------------------------------------------------------------------
# write_jsonl() — output format
# ---------------------------------------------------------------------------


def test_write_jsonl_writes_one_json_object_per_line(tmp_path) -> None:
    out_path = tmp_path / "out.jsonl"
    records = [
        RelevanceLabel(query_id="q1", chunk_id="c1", label=1, rationale="r1"),
        RelevanceLabel(query_id="q2", chunk_id="c2", label=0, rationale="r2"),
    ]

    write_jsonl(out_path, records)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json

    assert json.loads(lines[0])["query_id"] == "q1"
    assert json.loads(lines[1])["label"] == 0
