"""RAGAS evaluation suite: faithfulness, answer relevancy, context
precision over agents/qa_agent.py's QAState outputs, per SKILLS.md's
write-ragas-eval pattern.

`build_ragas_dataset` follows that pattern's exact column names
(question/answer/contexts/ground_truth) — this project's patched ragas
0.4.3 accepts the legacy schema, per AGENTS.md's version note.
`context_precision` requires a `ground_truth` reference per query;
`faithfulness` and `answer_relevancy` do not. Since this project has no
hand-curated reference answers, `ground_truth` is an optional dict
(query_id -> reference answer) rather than always derived from the QA
agent's own answer — grading an answer's faithfulness against itself would
be circular. `select_metrics` drops `context_precision` whenever no
ground truth is supplied, rather than silently no-op the metric on a
missing column.

Neither OpenAI (ragas's default LLM/embeddings backend) nor an OpenAI key
exists in this project (AGENTS.md is Anthropic-only), so `run_ragas_eval`
takes `llm`/`embeddings` as optional injected ragas wrappers —
`build_default_llm`/`build_default_embeddings` construct the real
Claude Sonnet + e5-base-v2 backends lazily (mirrors
retrieval/reranker_msmarco.py's `load()` classmethod split), so importing
this module never requires langchain_anthropic or sentence-transformers,
and unit tests never call a real LLM or load a real embedding model, per
AGENTS.md's mock-everything-in-tests rule.
"""

from __future__ import annotations

import sys
from typing import Any

import mlflow
import structlog
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Metric, answer_relevancy, context_precision, faithfulness

from agents import qa_agent
from agents.qa_agent import QAState
from evaluation.relevance_labeller import BENCHMARK_QUERIES

load_dotenv()
log = structlog.get_logger()

MODEL = "claude-sonnet-5"
DEFAULT_EMBED_MODEL = "intfloat/e5-base-v2"
MLFLOW_EXPERIMENT = "ragas_eval"
SAMPLE_SIZE = 10


def build_ragas_dataset(qa_states: list[QAState], ground_truths: dict[str, str] | None = None) -> Dataset:
    data: dict[str, list[Any]] = {
        "question": [s.query for s in qa_states],
        "answer": [s.answer or "" for s in qa_states],
        "contexts": [[c.text for c in (s.reranked_results or s.fused_results)] for s in qa_states],
    }
    if ground_truths is not None:
        data["ground_truth"] = [ground_truths.get(s.query_id, "") for s in qa_states]
    return Dataset.from_dict(data)


def select_metrics(ground_truths: dict[str, str] | None) -> list[Metric]:
    metrics: list[Metric] = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.append(context_precision)
    else:
        log.warning("ragas_context_precision_skipped", stage="ragas_suite", reason="no ground_truth supplied")
    return metrics


def build_default_llm() -> Any:
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatAnthropic(model=MODEL))


def build_default_embeddings() -> Any:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL))


def _average_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: sum(row[k] for row in scores) / len(scores) for k in keys}


def run_ragas_eval(
    qa_states: list[QAState],
    ground_truths: dict[str, str] | None = None,
    llm: Any = None,
    embeddings: Any = None,
) -> dict[str, float]:
    if not qa_states:
        return {}

    log.info("run_ragas_eval", stage="ragas_suite", num_queries=len(qa_states))
    dataset = build_ragas_dataset(qa_states, ground_truths)
    metrics = select_metrics(ground_truths)
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    scores = _average_scores(result.scores)
    log.info("run_ragas_eval_complete", stage="ragas_suite", **scores)
    return scores


def log_to_mlflow(scores: dict[str, float], experiment: str = MLFLOW_EXPERIMENT) -> None:
    if not scores:
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="ragas_eval"):
        mlflow.log_metrics(scores)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sample_queries = BENCHMARK_QUERIES[:SAMPLE_SIZE]

    qa_states = [qa_agent.run(bq.query, query_id=bq.query_id) for bq in sample_queries]

    llm = build_default_llm()
    embeddings = build_default_embeddings()
    scores = run_ragas_eval(qa_states, llm=llm, embeddings=embeddings)
    log_to_mlflow(scores)
    print(f"RAGAS scores over {len(qa_states)} queries: {scores}")


if __name__ == "__main__":
    main()
