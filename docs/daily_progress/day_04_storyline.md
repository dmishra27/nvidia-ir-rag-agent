# Day 4 — Dense Retrieval, SPLADE, and MCP Tool-Calling

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `retrieval/biencoder_eval.py` | 433 | (eval harness, not unit-tested — produces `biencoder_eval_results.json`) |
| `retrieval/splade_index.py` | 168 | 18 (`tests/retrieval/test_splade_index.py`, 193 lines) |
| `retrieval/populate_qdrant.py` | 116 | — (one-shot corpus embedding + upsert runner) |
| `mcp/mcp_qdrant/server.py` | 117 | live-verified (direct call, see §3) |
| `mcp/mcp_airflow/server.py` | 85 | written, not live-verified (deferred, see §3) |
| `docker-compose.yml` | +qdrant, +airflow services | — |
| `.mcp.json` | registers both new servers as stdio tools | — |
| `biencoder_eval_results.json` | eval output artifact | — |

New tests today: 18. Suite total: **126/126 passing**.

## 2. Why it matters

Day 3 gave the project exact-match retrieval (BM25). Day 4 adds the two legs
a hybrid retriever needs to beat pure lexical search: a **learned sparse**
signal (SPLADE, term-expansion aware) and a **dense semantic** signal
(bi-encoder cosine search over Qdrant). Layer 3 of the architecture
(`AGENTS.md`) calls for BM25 + SPLADE + dense + ColBERT fused via RRF —
today closes the SPLADE and dense gaps, leaving ColBERT as the only
unimplemented candidate-generation signal. The bi-encoder eval also
produces the artifact (`biencoder_eval_results.json`) that
`mcp_qdrant/server.py` reads at runtime to pick which embedding model to
use for query-time search, so the eval and the serving path are wired
together rather than duplicated.

## 3. Day 1 → 4 narrative

- **Day 1**: project scaffolding — schema, docker-compose, `AGENTS.md`/`SKILLS.md` contracts.
- **Day 2** (`42431ea`): Airflow 3 ingestion DAG (650 lines) — PyMuPDF parse, LangChain chunk, quality score, SQLAlchemy ORM write. 72 tests.
- Text-to-SQL LangGraph agent (`aebd277`, 21 tests) and direct ingest runner (`daa4ae9`) landed the corpus: **5,389 chunks in Postgres**, all 4 Text-to-SQL queries verified live.
- **Day 3** (`3fee692`): BM25 sparse index + shared `Candidate` dataclass — the first of the four Layer-3 retrieval signals, 108/108 tests passing, live search verified.
- **Day 4** (today): SPLADE sparse index (second signal) reuses the Day 3 `Candidate` shape so the eventual RRF fusion step can treat BM25 and SPLADE interchangeably. Bi-encoder evaluation picks a dense model (third signal) against the same 5,389-chunk corpus and 10-query judged set used to reason about Day 3's BM25 quality, so results are comparable across days. `mcp_qdrant`/`mcp_airflow` extend Layer 1 (MCP tool-calling) from the single `nvidia-ir-postgres` server built earlier to 3 of the planned 4 servers — `nvidia-ir-mlflow` remains for Week 4.

This keeps the project on the Layer 3 critical path: BM25 (Day 3) → SPLADE + dense (Day 4) → RRF fusion + ColBERT (next) → Layer 3b re-ranking.

## 4. Bi-encoder evaluation results

5,389 chunks, 10 hand-judged queries (graded relevance 1–2), NDCG@10, dense-only retrieval.

| Model | Params | Corpus embed time | Mean NDCG@10 | Mean query latency | p95 query latency | Outcome |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 265s | 0.4469 | 42.7ms | 109.3ms | baseline |
| **e5-base-v2** | 109M | 3,383s (56min, batch=16) | **0.5088** | 72.1ms | 117.8ms | **winner** |
| bge-m3 | 570M | — | — | — | — | **deferred** — OOM during model *load* (not batching) on this 8GB CPU-only host, confirmed even at batch_size=8 over a 1,000-chunk sample. Revisit with GPU/cloud compute. |

e5-base-v2 was selected as the production dense encoder: highest NDCG@10
among models that could actually run on this hardware. `mcp_qdrant/server.py`
reads this file at startup and reruns `_winning_model()` so a re-run of the
eval automatically repoints query-time search — no hardcoded model name in
the serving path.

Qdrant collection `nvidia_ir_chunks`: 5,389 points, 768-dim (e5-base-v2),
cosine distance, status green — populated and confirmed live via the
Qdrant REST API and by direct invocation of `search_vectors`/
`collection_stats` (top result score 0.838 for a held-out query, matching
point count against `GET /collections/nvidia_ir_chunks`).

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | 3/4 servers written | postgres (live), qdrant (live-verified via direct call), airflow (written, not yet live — see below), mlflow deferred to Week 4 |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 2/4 signals done | BM25 (Day 3), SPLADE (Day 4), dense/Qdrant (Day 4); ColBERT + RRF fusion outstanding |
| Layer 3b (re-ranking) | Not started | — |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 126/126 passing (72 ingestion/quality + 21 Text-to-SQL + 15 BM25 + 18 SPLADE, plus contract/fixture tests).

**Deferred out of Day 4** (host-memory constraint, revisit next session):
- Airflow container did not start this session. Live host free memory
  measured at ~686MB (not the ~6.3GB expected after the Docker Desktop
  v4.81.0 update) once other running containers were accounted for —
  the same danger zone that caused a near-freeze earlier in the week.
  Starting Airflow standalone (DB migration + webserver + scheduler +
  triggerer) was judged too risky at that memory level, so
  `mcp_airflow/server.py` remains written but not live-verified against a
  running Airflow instance. Resume by starting Airflow alone with
  `docker stats` open once free memory is confirmed stable and headroom is
  actually available.
- `nvidia-ir-qdrant` MCP server is registered in `.mcp.json` but did not
  register as a callable tool within this Claude Code session (likely a
  slow `sentence-transformers`/torch import exceeding the MCP handshake
  window) — needs a Claude Code restart to confirm the actual MCP-protocol
  round trip, though the underlying `search_vectors`/`collection_stats`
  logic was verified live by direct invocation.
- `pgadmin` container hit `exec /entrypoint.sh: exec format error` after
  the Docker Desktop update; re-pulling the image and recreating the
  container did not fix it. Stopped rather than left crash-looping. Not on
  the Day 4 critical path — revisit if DB admin UI access is needed.
