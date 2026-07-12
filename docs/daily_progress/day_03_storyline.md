# Day 3 — BM25 Sparse Retrieval

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `retrieval/candidates.py` | 13 | (shared dataclass, exercised by every retrieval test) |
| `retrieval/bm25_index.py` | 94 | 15 (`tests/retrieval/test_bm25_index.py`, 141 lines) |
| `run_bm25_search.py` | 41 | — (live search runner, not unit-tested) |

New tests today: 15. Suite total: **108/108 passing**.

## 2. Why it matters

Day 3 is the first day of Layer 3 (retrieval) from `AGENTS.md`: BM25 +
SPLADE + dense + ColBERT, fused via RRF. It starts with the frozen
`Candidate(chunk_id, text, score, rank)` dataclass in
`retrieval/candidates.py` — every later retrieval signal (SPLADE and dense
on Day 4, RRF fusion on Day 5) returns this exact shape, which is what makes
fusing heterogeneous rankers possible without per-signal glue code. `BM25Index`
wraps `rank_bm25.BM25Okapi` over `chunk_text` sourced from Postgres via the
Day 2 ORM models, with disk persistence (`pickle`) so the index doesn't have
to be rebuilt from Postgres on every query-time run.

## 3. Day 1 → 3 narrative

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

## 4. Live search verification

`run_bm25_search.py` loads the persisted index (`data/indexes/bm25_index.pkl`,
built from all 5,389 chunks) and runs two queries used as the standing
comparison set for every retrieval day since:

**Query: "NVLink bandwidth"** — top result `97724de868d508c6796761ea`
(score 15.95), a DRAM/VRAM write-bandwidth metrics chunk — an exact lexical
match on "bandwidth", as expected from BM25.

**Query: "CUDA memory allocation"** — top result `7cb10cb8b14c66c9987417cb`
(score 11.08), the `cudaMemAllocNodeParams::dptr` API reference chunk —
again a direct term match, since BM25 has no notion of semantic similarity
beyond shared tokens.

This BM25-only ranking becomes the baseline that Day 4's dense/SPLADE
signals and Day 5's RRF fusion are compared against.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | Not started | — |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 1/4 signals done | BM25 (Day 3); SPLADE, dense, ColBERT, RRF outstanding |
| Layer 3b (re-ranking) | Not started | — |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 108/108 passing (72 chunk-quality + 21 Text-to-SQL + 15 BM25).
