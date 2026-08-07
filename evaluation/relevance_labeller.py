"""Semi-automated relevance labelling: Claude Sonnet first-pass binary
(0/1) judgments for the fixed 50-query benchmark set.

Per SKILLS.md's run-reranker-benchmark ("Always use the same 50-query set
... and same RRF top-100 candidates as input to all three configs"),
`BENCHMARK_QUERIES` is the single fixed 50-query set every other Day 8
module (benchmark_runner, ragas_suite) draws its sample from.

`build_pairs()` retrieves one (query, passage) pair per query — the
top-ranked RRF-fused candidate, mirroring agents/retrieval_agent.py's
retrieve step (BM25 top-100 + dense top-100 -> RRF) but keeping only rank 1
per query. This gives a sparse, MS-MARCO-style binary judgment set (one
labelled passage per query) rather than exhaustive per-query pooling —
appropriate for a first-pass label set sized for a single evaluator
(Claude) rather than a TREC-scale annotation effort. Unjudged candidates
are treated as non-relevant by evaluation/retrieval_metrics.py.

`label_pairs()` forces a `judge_relevance` tool call per pair (mirrors
agents/qa_agent.py's `answer_with_citations` pattern) so the label comes
back as a structured {relevant, rationale} pair rather than parsed from
free text. Per AGENTS.md, the Anthropic client is instantiated directly in
`main()` (not constructor-injected) and unit tests patch it out entirely,
mirroring agents/qa_agent.py's convention for LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anthropic
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel

from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse

load_dotenv()
log = structlog.get_logger()

MODEL = "claude-sonnet-5"
CANDIDATE_POOL_SIZE = 100
BENCHMARK_QUERIES_PATH = Path("evaluation/benchmark_queries.jsonl")
RELEVANCE_LABELS_PATH = Path("evaluation/relevance_labels.jsonl")

JUDGE_TOOL: dict[str, Any] = {
    "name": "judge_relevance",
    "description": (
        "Judge whether a passage is relevant to a query: does the passage "
        "contain information that directly answers or substantively addresses "
        "the query?"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "True if the passage is relevant to the query, false otherwise.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence explaining the judgment.",
            },
        },
        "required": ["relevant", "rationale"],
    },
}


class BenchmarkQuery(BaseModel):
    query_id: str
    query: str


class RelevancePair(BaseModel):
    query_id: str
    query: str
    chunk_id: str
    passage_text: str


class RelevanceLabel(BaseModel):
    query_id: str
    chunk_id: str
    label: int
    rationale: str


# ── The fixed 50-query benchmark set ────────────────────────────────────────
# Categories: exact CUDA API (bq01-10), memory/performance internals
# (bq11-20), hardware specs (bq21-30), semantic/conceptual (bq31-40),
# mixed/legacy terminology (bq41-50) — mirrors run_uat_superiority.py's
# category structure at 50-query scale.

BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(query_id=f"bq{i:02d}", query=q)
    for i, q in enumerate(
        [
            "CUDA cudaMalloc function parameters",
            "cudaMemcpyAsync stream parameter",
            "CUDA error cudaErrorInvalidValue description",
            "cudaDeviceSynchronize return value",
            "dim3 struct constructor syntax",
            "cudaMallocHost pinned memory benefits",
            "cudaStreamCreate flags parameter",
            "cudaEventRecord usage for timing kernels",
            "cudaGetLastError versus cudaPeekAtLastError",
            "cudaMemGetInfo free and total memory",
            "shared memory bank conflicts and how to avoid them",
            "memory coalescing rules for global memory access patterns",
            "warp divergence performance impact",
            "occupancy versus performance tradeoffs",
            "register pressure and its effect on occupancy",
            "latency hiding through instruction level parallelism",
            "L2 cache persistence for CUDA kernels",
            "unified memory page migration overhead",
            "asynchronous memory prefetching with cudaMemPrefetchAsync",
            "stream priorities and concurrent kernel execution",
            "NVLink 4.0 bandwidth specifications",
            "NVSwitch topology for multi-GPU communication",
            "H100 HBM3 memory capacity",
            "streaming multiprocessor count per GPU architecture",
            "Tensor Core precision modes supported",
            "PCIe Gen5 versus NVLink bandwidth comparison",
            "GPU Direct RDMA for network transfers",
            "multi-instance GPU partitioning",
            "thermal design power limits for datacenter GPUs",
            "NVDEC hardware video decode throughput",
            "how does GPU memory work for parallel processing",
            "best practices for optimising neural network training",
            "what causes memory errors in GPU applications",
            "how to make GPU programs run faster",
            "problems with threads executing different code paths",
            "techniques for reducing kernel launch overhead",
            "strategies for balancing compute and memory bound kernels",
            "why does my CUDA kernel run slower than expected",
            "how to profile a CUDA application for bottlenecks",
            "approaches to overlapping data transfer and computation",
            "shader processor count per streaming multiprocessor",
            "global memory coalescing techniques",
            "texture memory caching behavior",
            "constant memory broadcast access pattern",
            "warp scheduler instruction issue rate",
            "CUDA thread synchronization performance overhead",
            "cooperative groups grid synchronization",
            "dynamic parallelism kernel launch from device code",
            "cuDNN convolution algorithm selection",
            "NCCL all-reduce collective communication pattern",
        ],
        start=1,
    )
]


# ── Retrieval: one top-ranked RRF-fused candidate per query ────────────────


def build_pairs(
    queries: list[BenchmarkQuery], bm25_index: BM25Index, dense_index: DenseIndex
) -> list[RelevancePair]:
    pairs: list[RelevancePair] = []
    for bq in queries:
        log.info("build_pairs_retrieve", query_id=bq.query_id, stage="build_pairs")
        bm25_results: list[Candidate] = bm25_index.search(bq.query, top_k=CANDIDATE_POOL_SIZE)
        dense_results: list[Candidate] = dense_index.search(bq.query, top_k=CANDIDATE_POOL_SIZE)
        fused = fuse(bm25_results, dense_results, top_k=1)
        if not fused:
            log.warning("build_pairs_no_results", query_id=bq.query_id, stage="build_pairs")
            continue
        top = fused[0]
        pairs.append(
            RelevancePair(query_id=bq.query_id, query=bq.query, chunk_id=top.chunk_id, passage_text=top.text)
        )
    return pairs


# ── Labelling: Claude Sonnet forced tool call per pair ──────────────────────


def label_pairs(pairs: list[RelevancePair], client: anthropic.Anthropic, model: str = MODEL) -> list[RelevanceLabel]:
    labels: list[RelevanceLabel] = []
    for pair in pairs:
        log.info("label_pair", query_id=pair.query_id, stage="label_pairs")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "judge_relevance"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Judge whether this passage is relevant to the query.\n\n"
                            f"Query: {pair.query}\n\nPassage: {pair.passage_text}"
                        ),
                    }
                ],
            )
        except Exception as exc:
            log.error("label_pair_failed", query_id=pair.query_id, stage="label_pairs", exc=str(exc))
            continue

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            log.error("label_pair_no_tool_use", query_id=pair.query_id, stage="label_pairs")
            continue

        labels.append(
            RelevanceLabel(
                query_id=pair.query_id,
                chunk_id=pair.chunk_id,
                label=1 if tool_block.input.get("relevant") else 0,
                rationale=tool_block.input.get("rationale", ""),
            )
        )
    return labels


# ── Output ───────────────────────────────────────────────────────────────


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (record.model_dump_json() for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bm25_index = BM25Index.load()
    dense_index = DenseIndex.connect()

    pairs = build_pairs(BENCHMARK_QUERIES, bm25_index, dense_index)
    write_jsonl(BENCHMARK_QUERIES_PATH, BENCHMARK_QUERIES)
    print(f"Wrote {len(BENCHMARK_QUERIES)} queries to {BENCHMARK_QUERIES_PATH}")

    client = anthropic.Anthropic()
    labels = label_pairs(pairs, client)
    write_jsonl(RELEVANCE_LABELS_PATH, labels)
    print(f"Wrote {len(labels)} relevance labels to {RELEVANCE_LABELS_PATH}")


if __name__ == "__main__":
    main()
