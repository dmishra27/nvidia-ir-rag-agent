"""Prompt variant comparison for the QA agent's `generate` node (criterion 2).

Design, hypothesis, confirm/falsify conditions: docs/uat/prompt_variant_comparison.md
(written before this script ran). This is the protocol section's implementation.

Mirrors run_day9_ragas.py's and run_day9_citation_judge.py's "no model loading"
approach: reuses the already-computed, live-reranked Config A contexts
(evaluation/day9_config_a_contexts.json) for the same 10 queries those two
scripts scored, so no BM25/dense/Qdrant/cross-encoder load is needed here --
only two live Claude Sonnet generate() calls per query (one per prompt
variant) plus the existing RAGAS suite and citation judge, both API-only.

Per-query results (both variants' QA states, per-query RAGAS scores, and every
citation judgment) are written to evaluation/prompt_variant_comparison.json
before any comparison is drawn -- committed alongside this script, per the
project's persist-before-analyse standard (0149ca4).
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import io
import json
import sys
from pathlib import Path

import anthropic
import mlflow
import structlog
from ragas import evaluate

from agents.qa_agent import PROMPT_VARIANTS, QAState, make_generate_node
from evaluation.citation_judge import citation_accuracy, judge_qa_output
from evaluation.ragas_suite import build_default_llm, build_ragas_dataset, select_metrics
from retrieval.candidates import Candidate
from run_day9_ragas import CohereRagasEmbeddings

log = structlog.get_logger()

CONTEXTS_PATH = Path("evaluation/day9_config_a_contexts.json")
OUT_PATH = Path("evaluation/prompt_variant_comparison.json")
SAMPLE_SIZE = 10
MLFLOW_EXPERIMENT = "prompt_variant_comparison"


def build_qa_states(variant: str) -> list[QAState]:
    contexts = json.loads(CONTEXTS_PATH.read_text(encoding="utf-8"))
    query_ids = list(contexts.keys())[:SAMPLE_SIZE]

    generate = make_generate_node(prompt_variant=variant)
    states: list[QAState] = []
    for query_id in query_ids:
        entry = contexts[query_id]
        candidates = [Candidate(**c) for c in entry["reranked"]]
        state = QAState(query_id=query_id, query=entry["query"], reranked_results=candidates)
        result = generate(state)
        print(f"[{variant}] {query_id}: {result.answer[:80] if result.answer else '(no answer)'}...")
        states.append(result)
    return states


def score_ragas_per_query(qa_states: list[QAState], llm: object, embeddings: object) -> list[dict[str, float]]:
    """Like evaluation/ragas_suite.run_ragas_eval, but returns the per-row scores
    (result.scores) instead of only the mean -- this comparison reports per-query,
    not just aggregate, per the project's reporting standard."""
    dataset = build_ragas_dataset(qa_states, ground_truths=None)
    metrics = select_metrics(ground_truths=None)
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    return list(result.scores)


def score_citations_per_query(qa_states: list[QAState], client: anthropic.Anthropic) -> dict[str, list[dict]]:
    judgments_by_query: dict[str, list[dict]] = {}
    for state in qa_states:
        judgments = judge_qa_output(state, client)
        judgments_by_query[state.query_id] = [j.model_dump() for j in judgments]
        acc = citation_accuracy(judgments)
        print(f"{state.query_id}: citation_accuracy={acc:.2f} ({len(judgments)} judged)")
    return judgments_by_query


def _mean(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r[key] is not None]
    return sum(values) / len(values) if values else 0.0


def _log_to_mlflow_or_warn(variant: str, aggregate: dict[str, float]) -> None:
    """Best-effort MLflow logging. This comparison's real output is
    evaluation/prompt_variant_comparison.json (written unconditionally below,
    per this project's persist-everything standard) -- MLflow is a secondary
    sink, and docker-compose.yml's mlflow service has a documented history of
    being down (DEF-18, and Day 8's "mlflow container stopped" note). A
    tracking-server outage should not cost the real result, so any
    connection failure here is caught and logged, not raised."""
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"prompt_variant_{variant}"):
            mlflow.log_param("prompt_variant", variant)
            mlflow.log_metrics(aggregate)
    except Exception as exc:
        log.warning("mlflow_logging_skipped", variant=variant, stage="prompt_variant_comparison", exc=str(exc))
        print(f"[{variant}] MLflow logging skipped (tracking server unreachable): {exc}")


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    llm = build_default_llm()
    embeddings = CohereRagasEmbeddings()
    anthropic_client = anthropic.Anthropic()

    report: dict[str, dict] = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    for variant in PROMPT_VARIANTS:
        print(f"\n=== variant: {variant} ===")
        qa_states = build_qa_states(variant)
        ragas_scores = score_ragas_per_query(qa_states, llm, embeddings)
        citation_judgments = score_citations_per_query(qa_states, anthropic_client)

        per_query = []
        for state, ragas_row in zip(qa_states, ragas_scores):
            judgments = citation_judgments[state.query_id]
            acc = sum(1 for j in judgments if j["supported"]) / len(judgments) if judgments else None
            per_query.append(
                {
                    "query_id": state.query_id,
                    "query": state.query,
                    "answer": state.answer,
                    "answer_length_chars": len(state.answer or ""),
                    "num_citations": len(state.citations),
                    "faithfulness": ragas_row.get("faithfulness"),
                    "answer_relevancy": ragas_row.get("answer_relevancy"),
                    "citation_accuracy": acc,
                    "num_judgments": len(judgments),
                }
            )

        aggregate = {
            "faithfulness": _mean(per_query, "faithfulness"),
            "answer_relevancy": _mean(per_query, "answer_relevancy"),
            "citation_accuracy": _mean(per_query, "citation_accuracy"),
            "mean_answer_length_chars": _mean(per_query, "answer_length_chars"),
            "mean_num_citations": _mean(per_query, "num_citations"),
        }
        print(f"[{variant}] aggregate: {aggregate}")

        _log_to_mlflow_or_warn(variant, aggregate)

        # Written after each variant, not just at the end -- a later variant's
        # crash (e.g. the credit exhaustion this script hit on its first live
        # run) must not cost an earlier variant's already-real result.
        report[variant] = {"per_query": per_query, "aggregate": aggregate}
        OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[{variant}] saved -> {OUT_PATH}")

    print(f"\nSaved per-query + aggregate results for {len(PROMPT_VARIANTS)} variants -> {OUT_PATH}")


if __name__ == "__main__":
    main()
