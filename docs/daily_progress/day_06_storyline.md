# Day 6 — ms-marco Re-ranker, Router, LangGraph Retrieval Agent

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `retrieval/reranker_msmarco.py` | 78 | 10 (`tests/retrieval/test_reranker_msmarco.py`, 160 lines) |
| `retrieval/reranker_router.py` | 131 | 22 (`tests/retrieval/test_reranker_router.py`, 276 lines) |
| `agents/retrieval_agent.py` | 141 | 25 (`tests/agents/test_retrieval_agent.py`, 328 lines) |

New tests today: 55. Suite total: **208/208 passing**.

## 2. Why it matters

Day 5 closed candidate generation (BM25 + dense + RRF); Day 6 closes
Layer 3b — turning a fused candidate pool into a final ranked answer.
`retrieval/reranker_msmarco.py` wraps `cross-encoder/ms-marco-MiniLM-L-6-v2`:
unlike BM25/dense/SPLADE it scores `(query, candidate_text)` pairs directly
rather than searching an index, so it can correct ranking mistakes the
first-stage retrievers make — specifically the Day 5 UAT's RRF
"corroboration bias," where a chunk two mediocre signals both rank
moderately outranks a chunk one signal ranks highly, because RRF sums
`1/(k+rank)` across signals without ever looking at the query text again.
A cross-encoder re-reads the actual query against the actual candidate text,
so it isn't fooled by that kind of double-counting.

`retrieval/reranker_router.py` turns AGENTS.md's five `RERANKER_MODE`
values into a concrete, testable design: `live_frontier` → `live_quality` →
`live_fast` → `fallback` form an ordered serving chain (highest quality
first), and the router degrades to the next tier down whenever the
configured tier is unavailable (not wired up) or raises, terminating at
`fallback` (BM25 rank order), which needs no model and never fails.
`benchmark` mode is deliberately excluded from that degradation chain — it's
an explicit, non-degrading choice that requires its own injected runner and
raises `NotImplementedError` rather than silently falling back if one isn't
provided. Only `live_fast` (ms-marco) is wired to a real model as of Day 6;
`live_quality` (bge-reranker-v2-m3) and `live_frontier` (Cohere Rerank v3)
remain `None` tiers that degrade cleanly until built.

`agents/retrieval_agent.py` is the first LangGraph graph in Layer 3: a
three-node `retrieve → rerank → return_results` pipeline over a pydantic
`AgentState`, mirroring the Day 2 Text-to-SQL agent's node-factory /
`model_copy(update=...)` pattern. `retrieve` runs BM25 + dense in parallel
and RRF-fuses them (Day 5's pipeline, reused unchanged); `rerank` hands the
fused pool to `RerankerRouter`; `return_results` selects the reranked list,
falling back to the fused pool if reranking produced nothing (e.g. an
unrecoverable router error). The BM25 index, dense index, and router are all
constructor-injected, so the 25 agent tests are pure schema/contract tests —
they assert which `AgentState` fields get set, preserved, or cleared on
error, never actual retrieval content — and never load a real index, connect
to Qdrant, or load a cross-encoder, per AGENTS.md's rule to mock all
embedding/LLM calls in unit tests.

## 3. Day 1 → 6 narrative

- **Day 1** (`1bc7f0c`): project scaffolding — schema, docker-compose,
  `AGENTS.md`/`SKILLS.md` contracts.
- **Day 2** (`42431ea`): Airflow 3 ingestion DAG (650 lines) — PyMuPDF parse,
  LangChain chunk, quality score, SQLAlchemy ORM write. 72 tests. Text-to-SQL
  LangGraph agent (`aebd277`, 21 tests) and direct ingest runner (`daa4ae9`)
  landed the corpus: **5,389 chunks in Postgres**, all 4 Text-to-SQL queries
  verified live.
- **Day 3** (`3fee692`): BM25 sparse index + shared `Candidate` dataclass —
  the first of the four Layer-3 retrieval signals, 108/108 tests passing,
  live search verified.
- **Day 4** (`14bbcfa`, `0568c5b`): SPLADE sparse index (second signal,
  18 tests); bi-encoder evaluation picks e5-base-v2 (NDCG@10 0.5088) as the
  dense encoder and populates Qdrant with 5,389 points; 3 of 4 MCP servers
  (postgres, qdrant, airflow) written, structlog-to-stderr fix resolves MCP
  stdout corruption. 126/126 tests passing.
- **Day 5** (`373c9f5`, `4ba23cf`, `3c9e415`): dense search wrapper over the
  populated Qdrant collection (third signal, 12 tests) and RRF fusion
  (15 tests) combine BM25 + dense into a single hybrid ranking, live-verified
  end to end. 153/153 tests passing. A 15-query UAT across 6 superiority
  cases followed, surfacing two regressions: RRF corroboration bias burying
  the correct `cudaMalloc` chunk behind two moderately-ranked competitors
  (Q1), and a need to confirm re-ranking preserves dense's already-correct
  rank-1 result rather than disturbing it (Q12).
- **Day 6** (today): ms-marco cross-encoder re-ranker (10 tests), 5-tier
  `RERANKER_MODE` router with graceful degradation (22 tests), and a
  three-node LangGraph retrieval agent (`retrieve → rerank → return_results`,
  25 contract tests) close Layer 3b. 208/208 tests passing. Live UAT
  regression verification for Q1/Q12 **deferred** — see §4.

## 4. UAT regression verification — deferred

The plan for Day 6 was to re-run Q1 ("CUDA cudaMalloc function parameters")
and Q12 ("cudaDeviceSynchronize return value") end to end through the new
`retrieval_agent.run()` pipeline and confirm the ms-marco re-ranker (a) pulls
the `cudaMalloc` definition chunk (`cc6c8e53936d04e9b192a7d5`) to rank 1,
fixing Q1's RRF corroboration-bias regression, and (b) keeps the
`cudaDeviceSynchronize` error-return chunk (`02bb6a205ba73aa9763b937c`) at
rank 1, confirming Q12.

This session's host had **critically low free memory** (0.33–0.49GB free of
7.65GB total, Windows actively compressing memory pages) and a Docker/WSL2
backend that was already failing (`docker ps` returned a 500 Internal Server
Error from `dockerDesktopLinuxEngine`, twice). Loading two more ML models —
the e5-base-v2 dense encoder (~440MB) and the ms-marco cross-encoder
(~90MB), plus PyTorch runtime overhead — on top of that risked destabilizing
the host, consistent with this project's standing rule to check free memory
and avoid concurrent heavy ML/infra work on this 8GB CPU-only machine. Per
the user's decision, the live check is deferred rather than forced through.

All code that will be exercised by the live check is unit-tested against
fakes today (55 new tests across the reranker, router, and agent), so the
mechanism is verified; only the empirical "does ms-marco actually fix Q1 and
confirm Q12 against the real corpus" question remains open.

**Next session**: once free memory allows (or by reusing the pre-computed
BM25/dense candidate pools already captured in
`docs/uat/uat_superiority_cases_raw.json` for Q1/Q12, avoiding a fresh
dense-index/BM25-index load), run `retrieval_agent.run()` — or
`MSMarcoReranker.load().rerank()` directly over the cached pools — for both
queries and record the before/after rank of the target chunk.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | 3/4 servers written | postgres (live), qdrant (live-verified), airflow (written, not yet live), mlflow deferred to Week 4 |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 3/4 signals + RRF done | BM25 (Day 3), SPLADE (Day 4), dense (Day 4/5), RRF fusion (Day 5); ColBERT outstanding |
| Layer 3b (re-ranking) | ms-marco + router + agent done, live UAT regression check deferred | `retrieval/reranker_msmarco.py`, `retrieval/reranker_router.py`, `agents/retrieval_agent.py` |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 208/208 passing (72 chunk-quality + 21 Text-to-SQL + 15 BM25
+ 18 SPLADE + 12 dense + 15 RRF + 10 ms-marco reranker + 22 reranker router
+ 25 retrieval agent, plus contract/fixture tests).

**Host memory note**: this session hit the same danger zone flagged on
Day 4 and Day 5, but worse — free RAM dropped to under 0.5GB out of 7.65GB
total with Windows actively compressing memory, and the Docker/WSL2 backend
itself became unresponsive (`docker ps` 500 errors) before any new model was
loaded. Unlike Day 5, where stopping one container bought enough headroom to
proceed, this time the decision was to stop rather than push through:
building and unit-testing the Day 6 code did not require loading any real
model, so all of it shipped safely; only the live empirical verification
step was deferred to a session with more free memory.
