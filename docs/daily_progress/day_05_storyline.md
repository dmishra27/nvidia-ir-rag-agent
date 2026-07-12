# Day 5 — Dense Index, RRF Fusion, Hybrid Search Complete

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `retrieval/dense_index.py` | 123 | 12 (`tests/retrieval/test_dense_index.py`, 208 lines) |
| `retrieval/rrf_fusion.py` | 43 | 15 (`tests/retrieval/test_rrf_fusion.py`, 170 lines) |
| `run_hybrid_search.py` | 75 | — (live BM25+dense+RRF verification runner) |
| `docs/daily_progress/day_01_storyline.md` | — | — (backfilled) |
| `docs/daily_progress/day_02_storyline.md` | — | — (backfilled) |
| `docs/daily_progress/day_03_storyline.md` | — | — (backfilled) |
| `docs/daily_progress/README.md` | — | — (new — daily progress table of contents) |

New tests today: 27. Suite total: **153/153 passing**.

## 2. Why it matters

Day 5 closes the loop Day 3 and Day 4 opened. `retrieval/dense_index.py`
does *not* re-embed the corpus — Day 4's `populate_qdrant.py` already put
5,389 e5-base-v2 embeddings into the `nvidia_ir_chunks` Qdrant collection —
it only has to embed the query and hand it to Qdrant's own ranking, returning
the same `Candidate` shape BM25 (Day 3) and SPLADE (Day 4) already use.
`retrieval/rrf_fusion.py` is the piece that makes "hybrid" real: it combines
any two ranked `Candidate` lists by summing `1 / (k + rank)` per list a
chunk appears in (`k=60`), so a chunk that ranks well on *both* lexical and
semantic search outranks one that only wins on a single signal. Both the
Qdrant client and the query encoder in `DenseIndex` are constructor-injected
specifically so the 12 dense-index unit tests never load `torch` or
`sentence-transformers` or open a network connection, matching the project
rule (`AGENTS.md`) to mock all embedding/LLM calls in unit tests — the real
model is only loaded once, by `run_hybrid_search.py`, for live verification.

With this, Layer 3's candidate-generation side is 3 of 4 signals plus fusion:
BM25 (Day 3), SPLADE (Day 4), dense (Day 4 index + Day 5 search wrapper),
and RRF fusion (Day 5). Only ColBERT and Layer 3b re-ranking remain on the
retrieval critical path.

## 3. Day 1 → 5 narrative

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
- **Day 5** (today): dense search wrapper over the populated Qdrant
  collection (third signal, 12 tests) and RRF fusion (15 tests) combine
  BM25 + dense into a single hybrid ranking, live-verified end to end.
  153/153 tests passing.

## 4. Live hybrid search verification

`run_hybrid_search.py` loads the persisted Day 3 BM25 index and connects to
the live Day 4 Qdrant collection, retrieves a top-100 candidate pool from
each signal, and fuses them with RRF (`k=60`) down to the top 5.

**Query: "NVLink bandwidth"**

| Rank | RRF chunk_id | RRF score | Also in BM25 top-5? | Also in dense top-5? |
|---|---|---|---|---|
| 1 | `97724de868d508c6796761ea` | 0.0328 | Yes (#1) | Yes (#1) |
| 2 | `aae9e31dae08f840680449cd` | 0.0304 | Yes (#2) | No |
| 3 | `0fc9e36da0ab43104dee747a` | 0.0300 | No | Yes (#2) |
| 4 | `fa64dc43a5422c77d04b1f27` | 0.0295 | Yes (#5) | No |
| 5 | `6a54459b6095940046063514` | 0.0287 | Yes (#3) | No |

BM25-only vs. RRF top-5 overlap: **4/5** — RRF kept BM25's exact-match
strength but pulled in one dense-only chunk (`0fc9e36d...`, a CUDA C++ Best
Practices Guide "Theoretical Bandwidth Calculation" section) that BM25
missed because the query terms don't appear verbatim in it.

**Query: "CUDA memory allocation"**

| Rank | RRF chunk_id | RRF score | Also in BM25 top-5? | Also in dense top-5? |
|---|---|---|---|---|
| 1 | `3ddd571f8cec6d1fc5ad1de4` | 0.0301 | No | Yes (#4) |
| 2 | `1e0a96307aaa32b6dd123d4b` | 0.0296 | Yes (#2) | No |
| 3 | `aa69983083fbf22181cc774f` | 0.0291 | No | Yes (#2) |
| 4 | `5a4c576d74df682e3144960b` | 0.0287 | Yes (#3) | No |
| 5 | `82991021558df26bc1eb76ed` | 0.0282 | No | No (ranked outside both top-5s, favored by combined signal) |

BM25-only vs. RRF top-5 overlap: **2/5** — this query is more generic
("CUDA memory allocation" matches many API-reference chunks lexically), so
dense semantic similarity disagrees with BM25 much more than on the narrow
"NVLink bandwidth" query, and RRF's fused ranking looks meaningfully
different from BM25-only.

**Does RRF improve over BM25-only?** On both queries RRF keeps every
double-ranked chunk (present in both signals) at or near the top — exactly
the intended effect — while surfacing dense-only chunks that are topically
relevant but lexically dissimilar (e.g. the "Theoretical Bandwidth
Calculation" chunk for query 1). The generic query (query 2) shows the
larger shift, which matches expectations: BM25 and dense search agree most
when the query terms are rare and specific, and diverge more on common
phrasing where semantic similarity has more chunks to choose from.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | 3/4 servers written | postgres (live), qdrant (live-verified), airflow (written, not yet live), mlflow deferred to Week 4 |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 3/4 signals + RRF done | BM25 (Day 3), SPLADE (Day 4), dense (Day 4/5), RRF fusion (Day 5); ColBERT outstanding |
| Layer 3b (re-ranking) | Not started | — |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 153/153 passing (72 chunk-quality + 21 Text-to-SQL + 15 BM25
+ 18 SPLADE + 12 dense + 15 RRF, plus contract/fixture tests).

**Host memory note**: this session started with ~515MB free RAM against an
8GB CPU-only host with 5 Docker containers running (`postgres`, `qdrant`,
`streamlit`, `api`, `mlflow`) — the same danger zone that deferred Airflow
on Day 4. The `mlflow` container (not needed for this session's work) was
stopped first, bringing free memory to ~800MB before loading `e5-base-v2`
for live verification; the run completed without incident.
