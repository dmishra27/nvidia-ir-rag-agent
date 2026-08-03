"""Day 9 Task 5: live citation judge over 10 real QA agent outputs.

Reuses run_day9_ragas.py's approach for getting real QA agent output
without a torch/dense-model reload: Config A's live-reranked contexts
(evaluation/day9_config_a_contexts.json) become each QAState's
`reranked_results`, and only `generate` (agents/qa_agent.py's forced
`answer_with_citations` Claude call) actually runs — retrieve/rerank are
not re-executed since their live output already exists from Task 2.

The full RAGAS `evaluate()` run (Task 4) was deferred per user instruction
(see day_09_storyline.md) after `generate` surfaced a real bug: Claude
occasionally returns a citations list item as a bare string instead of
{claim, chunk_ids}, which crashed agents/qa_agent.py's unhandled list
comprehension. That's now fixed (skip malformed entries, log a warning —
see tests/agents/test_qa_agent.py's
test_skips_malformed_citation_entries_instead_of_crashing) and is exercised
for real by this script's 10 live generate() calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic
import mlflow
import structlog

from agents.qa_agent import QAState, make_generate_node
from evaluation.citation_judge import citation_accuracy, judge_qa_output
from retrieval.candidates import Candidate

log = structlog.get_logger()

CONTEXTS_PATH = Path("evaluation/day9_config_a_contexts.json")
QA_STATES_OUT_PATH = Path("evaluation/day9_qa_states.json")
JUDGMENTS_OUT_PATH = Path("evaluation/day9_citation_judgments.json")
SAMPLE_SIZE = 10


def build_qa_states() -> list[QAState]:
    contexts = json.loads(CONTEXTS_PATH.read_text(encoding="utf-8"))
    query_ids = list(contexts.keys())[:SAMPLE_SIZE]

    generate = make_generate_node()
    states: list[QAState] = []
    for query_id in query_ids:
        entry = contexts[query_id]
        candidates = [Candidate(**c) for c in entry["reranked"]]
        state = QAState(query_id=query_id, query=entry["query"], reranked_results=candidates)
        result = generate(state)
        num_citations = sum(len(c.chunk_ids) for c in result.citations)
        print(f"{query_id}: {len(result.citations)} claims, {num_citations} cited chunk_ids")
        states.append(result)
    return states


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    qa_states = build_qa_states()
    QA_STATES_OUT_PATH.write_text(
        json.dumps([s.model_dump() for s in qa_states], indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved {len(qa_states)} QA states -> {QA_STATES_OUT_PATH}")

    client = anthropic.Anthropic()
    all_judgments = []
    for state in qa_states:
        judgments = judge_qa_output(state, client)
        all_judgments.extend(judgments)
        acc = citation_accuracy(judgments)
        print(f"{state.query_id}: citation_accuracy={acc:.2f} ({len(judgments)} judged)")

    JUDGMENTS_OUT_PATH.write_text(
        json.dumps([j.model_dump() for j in all_judgments], indent=2), encoding="utf-8"
    )
    overall = citation_accuracy(all_judgments)
    print(f"\nOverall citation_accuracy: {overall:.4f} ({len(all_judgments)} (claim, chunk) pairs judged)")
    print(f"Saved judgments -> {JUDGMENTS_OUT_PATH}")

    mlflow.set_experiment("citation_judge")
    with mlflow.start_run(run_name="day9_citation_judge"):
        mlflow.log_metric("citation_accuracy", overall)
        mlflow.log_metric("num_judgments", len(all_judgments))
        mlflow.log_metric("num_queries", len(qa_states))


if __name__ == "__main__":
    main()
