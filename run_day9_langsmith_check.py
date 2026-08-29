"""Day 9 Task 6: verify LangSmith tracing is active.

LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY/LANGCHAIN_PROJECT/LANGCHAIN_ENDPOINT
(EU) are already set in .env — nothing to change there. This script proves
a trace actually lands by invoking agents/qa_agent.py's compiled LangGraph
graph (a langchain-core Runnable, auto-traced once those env vars are set)
with fake BM25/dense/router objects — memory-safe, no model load — so
`generate`'s one real, unmocked Claude call is the only live component,
and its containing LangGraph run is what should appear in the LangSmith
EU dashboard under project "nvidia-ir-rag-agent".
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import sys

from agents.qa_agent import QAState, build_graph
from retrieval.candidates import Candidate

CANDIDATES = [
    Candidate(
        chunk_id="langsmith-demo-chunk",
        text="cudaMalloc allocates linear memory on the device and returns a pointer via its first argument.",
        score=1.0,
        rank=1,
    )
]


class _FakeIndex:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        return CANDIDATES


class _FakeRouter:
    def rerank(self, query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        return candidates[:top_k]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    graph = build_graph(_FakeIndex(), _FakeIndex(), _FakeRouter())
    initial = QAState(query="What does cudaMalloc do?", query_id="langsmith-check-01")

    result = graph.invoke(initial)

    print("query_id:", result["query_id"] if isinstance(result, dict) else result.query_id)
    answer = result["answer"] if isinstance(result, dict) else result.answer
    print("answer:", (answer or "")[:200])
    print("\nIf LANGCHAIN_TRACING_V2=true and the API key is valid, a trace for this")
    print("run should now appear in the LangSmith EU dashboard under project")
    print("'nvidia-ir-rag-agent': https://eu.smith.langchain.com/")


if __name__ == "__main__":
    main()
