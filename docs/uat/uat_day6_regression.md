# Day 6 UAT — ms-marco Re-ranker Regression (Q1 and Q12)

## Methodology

Memory-safe regression pass, run via `run_uat_day6_regression.py` on the
8GB dev box (see host memory constraint notes): no BM25/dense/RRF indices
were rebuilt and no Qdrant connection was opened. Instead, the cached
top-3 **RRF hybrid candidate pool** (`method_c_rrf`) for Q1 and Q12 was
loaded directly from
[`docs/uat/uat_superiority_cases_raw.json`](uat_superiority_cases_raw.json)
(produced during the Day 5 superiority UAT) and passed straight into
`retrieval/reranker_msmarco.py::MSMarcoReranker`, which loads only
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB, CPU) — the `live_fast`
`RERANKER_MODE` config from AGENTS.md.

Scope is intentionally narrow: 2 queries, 3 candidates each, one reranker.
This validates that the Day 6 `MSMarcoReranker` class re-scores real
cached RRF output correctly (no regressions from the `reranker_router`
work), not a full benchmark — that's the three-way `run-reranker-benchmark`
skill, deferred until bge/Cohere configs and more memory headroom are
available.

Raw output: [`docs/uat/uat_day6_regression_raw.json`](uat_day6_regression_raw.json).

---

## Q1 — case1_bm25_lexical_superiority

**Query**: "CUDA cudaMalloc function parameters"

| Stage | Rank | chunk_id | Score | Text (first ~100 chars) |
|---|---|---|---|---|
| RRF (cached, pre-rerank) | 1 | `81b9c458ed8d5bbf219819b8` | 0.0317 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| **ms-marco (rank 1)** | 1 | `81b9c458ed8d5bbf219819b8` | -1.3748 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to,` |

**Read**: ms-marco confirms the RRF top-1 unchanged — the cross-encoder
agrees with the fused BM25+dense signal on this candidate set, even though
its own score is negative (ms-marco logits are unbounded and centered near
zero/negative for weak matches; this is still the highest of the 3
candidates). No regression: this is the same chunk case1 flagged for Day 5
as the RRF top-1.

## Q12 — case5_dense_failure_bm25_advantage

**Query**: "cudaDeviceSynchronize return value"

| Stage | Rank | chunk_id | Score | Text (first ~100 chars) |
|---|---|---|---|---|
| RRF (cached, pre-rerank) | 1 | `7168ba67e35c613f13986864` | 0.0297 | *(not the reranked top-1 — see below)* |
| RRF (cached) | 2 | `02bb6a205ba73aa9763b937c` | 0.0284 | `. cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDevi` |
| **ms-marco (rank 1)** | 1 | `02bb6a205ba73aa9763b937c` | 6.2774 | `. cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDevi` |

**Read**: ms-marco **promotes** the RRF rank-2 chunk to rank 1 — a chunk
whose text directly answers "return value" ("...returns an error if one of
the preceding tasks has failed") over the cached RRF rank-1. This is the
expected behavior for the case5 scenario (dense underperforms, BM25/RRF
carries it partway, cross-encoder re-ranking should sharpen it further) and
is a positive signal for `reranker_router` wiring into the retrieval graph.

---

## Verdict

Both queries ran end-to-end through `MSMarcoReranker.rerank()` against real
cached candidates with no crashes, no OOM (host had ~450MB free at run
time; peak usage stayed within the model's ~90MB CPU footprint), and no
`sentence-transformers`/`torch` errors beyond an expected unauthenticated
HF Hub rate-limit warning. Q1 shows rank stability, Q12 shows a correct
promotion — no regression in the Day 6 reranker path. Full 15-query /
3-way (ms-marco + bge + Cohere) regression remains deferred to a
higher-memory session, consistent with the Day 5 UAT deferral.
