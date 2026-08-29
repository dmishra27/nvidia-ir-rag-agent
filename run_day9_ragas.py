"""Day 9 Task 4: live RAGAS evaluation over 10 real QA agent outputs.

Two adaptations keep this Claude/Cohere-API-only, per Day 9's "no model
loading" instruction:

1. Retrieval is not re-run. agents/qa_agent.py's `retrieve`/`rerank` nodes
   need BM25Index + DenseIndex (the latter loads e5-base-v2/torch) — instead
   this script reuses run_day9_benchmark_ac.py's already-computed,
   live-reranked Config A contexts (evaluation/day9_config_a_contexts.json)
   for 10 queries, builds a QAState per query with `reranked_results`
   pre-populated, and calls `make_generate_node()(state)` directly. This is
   the QA agent's real `generate` node (forced `answer_with_citations` tool
   call to Claude Sonnet) — only the retrieve/rerank nodes are bypassed
   because their outputs already exist from a live run today.
2. RAGAS's `answer_relevancy` metric needs an embeddings backend.
   evaluation/ragas_suite.py's `build_default_embeddings()` loads
   e5-base-v2/torch (deferred, same reason as above) and
   langchain_community's `CohereEmbeddings` wrapper is broken in the
   installed version (raises `KeyError: 'user_agent'` at construction —
   verified directly). `CohereRagasEmbeddings` below is a ~15-line adapter
   over `cohere.Client().embed()` implementing ragas's `BaseRagasEmbeddings`
   interface directly, avoiding both.

`context_precision` needs a `ground_truth` reference this project doesn't
have (see evaluation/ragas_suite.py's `select_metrics`), so this run scores
faithfulness + answer_relevancy only, consistent with Day 8's design.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import json
import sys
import warnings
from pathlib import Path

import structlog
from ragas.embeddings.base import BaseRagasEmbeddings

from agents.qa_agent import QAState, make_generate_node
from evaluation.ragas_suite import build_default_llm, log_to_mlflow, run_ragas_eval
from retrieval.candidates import Candidate

warnings.filterwarnings("ignore", category=DeprecationWarning)
log = structlog.get_logger()

CONTEXTS_PATH = Path("evaluation/day9_config_a_contexts.json")
QA_STATES_OUT_PATH = Path("evaluation/day9_qa_states.json")
SAMPLE_SIZE = 10
COHERE_EMBED_MODEL = "embed-english-v3.0"


class CohereRagasEmbeddings(BaseRagasEmbeddings):
    """Minimal ragas BaseRagasEmbeddings adapter over cohere.Client().embed()
    — API-based, no local model, unlike ragas_suite.py's e5-base-v2 default."""

    def __init__(self, model: str = COHERE_EMBED_MODEL) -> None:
        super().__init__()
        import cohere

        self._client = cohere.Client()
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(texts=texts, model=self._model, input_type="search_document")
        return [list(e) for e in response.embeddings]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


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
        print(f"{query_id}: {result.answer[:80] if result.answer else '(no answer)'}...")
        states.append(result)
    return states


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    qa_states = build_qa_states()
    QA_STATES_OUT_PATH.write_text(
        json.dumps([s.model_dump() for s in qa_states], indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved {len(qa_states)} QA states -> {QA_STATES_OUT_PATH}")

    llm = build_default_llm()
    embeddings = CohereRagasEmbeddings()
    scores = run_ragas_eval(qa_states, llm=llm, embeddings=embeddings)
    log_to_mlflow(scores)
    print(f"RAGAS scores over {len(qa_states)} queries: {scores}")


if __name__ == "__main__":
    main()
