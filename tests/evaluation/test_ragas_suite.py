"""Unit tests for evaluation/ragas_suite.py.

Per AGENTS.md ("LLM quality: RAGAS threshold gates in CI (not unit tests)"
and "Mock all embedding and LLM calls in unit tests"), `ragas.evaluate` is
always patched — these tests never run a real RAGAS metric (which would
itself call an LLM) or connect to MLflow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.qa_agent import QAState
from evaluation.ragas_suite import build_ragas_dataset, log_to_mlflow, run_ragas_eval, select_metrics
from retrieval.candidates import Candidate


def _state(query_id: str, query: str, answer: str, chunks: list[Candidate]) -> QAState:
    return QAState(query_id=query_id, query=query, answer=answer, reranked_results=chunks)


_STATES = [
    _state("q1", "cudaMalloc parameters", "cudaMalloc takes a pointer and a size.", [Candidate(chunk_id="c1", text="cudaMalloc(void**, size_t)", score=1.0, rank=1)]),
    _state("q2", "NVLink bandwidth", "NVLink 4.0 provides 900 GB/s.", [Candidate(chunk_id="c2", text="NVLink 4.0: 900 GB/s bidirectional", score=1.0, rank=1)]),
]


# ---------------------------------------------------------------------------
# build_ragas_dataset()
# ---------------------------------------------------------------------------


def test_build_ragas_dataset_maps_query_answer_contexts() -> None:
    dataset = build_ragas_dataset(_STATES)

    assert dataset["question"] == ["cudaMalloc parameters", "NVLink bandwidth"]
    assert dataset["answer"] == ["cudaMalloc takes a pointer and a size.", "NVLink 4.0 provides 900 GB/s."]
    assert dataset["contexts"] == [["cudaMalloc(void**, size_t)"], ["NVLink 4.0: 900 GB/s bidirectional"]]


def test_build_ragas_dataset_omits_ground_truth_when_not_supplied() -> None:
    dataset = build_ragas_dataset(_STATES)

    assert "ground_truth" not in dataset.column_names


def test_build_ragas_dataset_includes_ground_truth_when_supplied() -> None:
    dataset = build_ragas_dataset(_STATES, ground_truths={"q1": "reference answer 1"})

    assert dataset["ground_truth"] == ["reference answer 1", ""]


def test_build_ragas_dataset_falls_back_to_fused_results_when_reranked_empty() -> None:
    state = QAState(
        query_id="q3", query="q", answer="a",
        fused_results=[Candidate(chunk_id="c3", text="fused context", score=1.0, rank=1)],
        reranked_results=[],
    )

    dataset = build_ragas_dataset([state])

    assert dataset["contexts"] == [["fused context"]]


def test_build_ragas_dataset_handles_none_answer() -> None:
    state = QAState(query_id="q4", query="q", answer=None, reranked_results=[])

    dataset = build_ragas_dataset([state])

    assert dataset["answer"] == [""]


# ---------------------------------------------------------------------------
# select_metrics()
# ---------------------------------------------------------------------------


def test_select_metrics_excludes_context_precision_without_ground_truth() -> None:
    metrics = select_metrics(None)

    names = {m.name for m in metrics}
    assert names == {"faithfulness", "answer_relevancy"}


def test_select_metrics_excludes_context_precision_with_empty_ground_truth_dict() -> None:
    metrics = select_metrics({})

    names = {m.name for m in metrics}
    assert names == {"faithfulness", "answer_relevancy"}


def test_select_metrics_includes_context_precision_with_ground_truth() -> None:
    metrics = select_metrics({"q1": "reference"})

    names = {m.name for m in metrics}
    assert names == {"faithfulness", "answer_relevancy", "context_precision"}


# ---------------------------------------------------------------------------
# run_ragas_eval() — evaluate() patched
# ---------------------------------------------------------------------------


def test_run_ragas_eval_empty_states_returns_empty_without_calling_evaluate() -> None:
    with patch("evaluation.ragas_suite.evaluate") as mock_evaluate:
        result = run_ragas_eval([])

    assert result == {}
    mock_evaluate.assert_not_called()


def test_run_ragas_eval_averages_per_row_scores() -> None:
    mock_result = MagicMock()
    mock_result.scores = [
        {"faithfulness": 1.0, "answer_relevancy": 0.8},
        {"faithfulness": 0.5, "answer_relevancy": 0.6},
    ]

    with patch("evaluation.ragas_suite.evaluate", return_value=mock_result) as mock_evaluate:
        result = run_ragas_eval(_STATES)

    assert result == {"faithfulness": 0.75, "answer_relevancy": 0.7}
    mock_evaluate.assert_called_once()


def test_run_ragas_eval_passes_dataset_metrics_llm_embeddings() -> None:
    mock_result = MagicMock()
    mock_result.scores = [{"faithfulness": 1.0, "answer_relevancy": 1.0}]
    fake_llm = MagicMock()
    fake_embeddings = MagicMock()

    with patch("evaluation.ragas_suite.evaluate", return_value=mock_result) as mock_evaluate:
        run_ragas_eval(_STATES[:1], llm=fake_llm, embeddings=fake_embeddings)

    _, kwargs = mock_evaluate.call_args
    assert kwargs["llm"] is fake_llm
    assert kwargs["embeddings"] is fake_embeddings
    assert len(kwargs["metrics"]) == 2


# ---------------------------------------------------------------------------
# log_to_mlflow()
# ---------------------------------------------------------------------------


def test_log_to_mlflow_sets_experiment_and_logs_metrics() -> None:
    scores = {"faithfulness": 0.9, "answer_relevancy": 0.8}

    with patch("evaluation.ragas_suite.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        log_to_mlflow(scores)

    mock_mlflow.set_experiment.assert_called_once_with("ragas_eval")
    mock_mlflow.log_metrics.assert_called_once_with(scores)


def test_log_to_mlflow_skips_run_when_scores_empty() -> None:
    with patch("evaluation.ragas_suite.mlflow") as mock_mlflow:
        log_to_mlflow({})

    mock_mlflow.start_run.assert_not_called()
