# Day 7 — QA Agent, FastAPI Endpoints, structlog Instrumentation

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `agents/qa_agent.py` | 226 | 22 (`tests/agents/test_qa_agent.py`, 391 lines) |
| `api/main.py` | 31 | covered by endpoint tests below |
| `api/middleware.py` | 91 | 4 (`tests/api/test_middleware.py`, 87 lines) |
| `api/dependencies.py` | 31 | covered by endpoint tests below |
| `api/schemas/__init__.py` | 61 | covered by endpoint tests below |
| `api/routers/search.py` | 59 | 4 (`tests/api/test_main.py::TestSearch`) |
| `api/routers/ask.py` | 55 | 4 (`tests/api/test_main.py::TestAsk`) |
| `api/routers/health.py` | 21 | 1 (`tests/api/test_main.py::TestHealth`) |
| `schema/models.py` (+`RequestLog`) | 85 | exercised via middleware tests |

New tests today: 35 (22 QA agent + 13 API/middleware). Suite total:
**243/243 passing**.

## 2. Why it matters

Day 6 closed Layer 3b (candidate pool → final ranked list via ms-marco +
`RerankerRouter`). Day 7 turns that ranked list into an answer and puts the
whole pipeline behind an HTTP API — the first code in this project a
non-Claude-Code client (a browser, curl, the eventual Streamlit app) can
actually call.

`agents/qa_agent.py` adds a third LangGraph node, `generate`, on top of
Day 6's `retrieve`/`rerank` (reused as a matching `QAState` pair, not the
literal `AgentState` node functions, since the schemas diverge). `generate`
grounds Claude Sonnet (`claude-sonnet-5`) in the top-`top_k` re-ranked
passages and forces a `answer_with_citations` tool call rather than parsing
citations out of free text — the model returns `{answer, citations: [{claim,
chunk_ids}]}` directly, so "citation extraction per claim" is a schema
constraint enforced by the tool's `input_schema`, not a regex over prose.
Per `agents/text_to_sql_agent.py`'s established convention for LLM nodes,
`anthropic.Anthropic()` is instantiated directly inside `generate` (not
constructor-injected, unlike the BM25/dense/router components), and all 22
tests patch `agents.qa_agent.anthropic.Anthropic` — never a live API call,
per AGENTS.md.

`api/` turns `retrieval_agent.run()` and `qa_agent.run()` into three FastAPI
routes: `POST /search` (re-ranked candidates, no generation), `POST /ask`
(grounded answer + citations), and `GET /health` (pure liveness probe — no
DB/Qdrant/model calls, so its own latency never tracks a downstream
dependency's). Both `/search` and `/ask` accept `RERANKER_MODE` as a query
parameter (`?RERANKER_MODE=live_quality`), letting a caller flip re-ranking
tiers per-request without changing the JSON body shape. BM25 index, dense
index, and the ms-marco reranker are wired as FastAPI dependencies
(`api/dependencies.py`, `lru_cache`-backed so the real objects load once per
process) rather than imported globally, so `app.dependency_overrides` lets
all 9 endpoint tests swap in fakes — no real index load, no Qdrant
connection, no cross-encoder load in the test suite, matching the same
constructor-injection discipline `retrieval_agent.py` established on Day 6.

`api/middleware.py`'s `RequestLatencyMiddleware` measures per-request
latency and writes one row to `request_log` (Postgres) per call, via the new
`RequestLog` SQLAlchemy model added to `schema/models.py` — never a raw SQL
string, per AGENTS.md. The `stage` column groups these rows by endpoint
(`search`/`ask`/`health`) so they can be queried by request-log dashboards
later; this middleware does **not** capture per-LangGraph-node timing
(retrieve/rerank/generate individually) — that would require threading
timers through `AgentState`/`QAState`, which is Layer 6 monitoring/ work,
not something observable at the ASGI layer. A failed DB write is logged and
swallowed rather than raised, so monitoring can never break an actual
response — the same graceful-degradation posture `RerankerRouter` uses for
re-ranking tiers. The session factory is constructor-injected
(`create_app(session_factory=...)`), so `tests/api/test_middleware.py`
verifies the exact row written using a fake session, with no live Postgres
connection.

structlog instrumentation: every new node (`retrieve`, `rerank`, `generate`)
and every new HTTP handler (`search_request`, `ask_request`,
`request_latency`) logs with `query_id` and `stage` fields, consistent with
every prior day's modules.

## 3. Day 1 → 7 narrative

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
  cases followed, surfacing two regressions to re-check: Q1 (RRF
  corroboration bias) and Q12 (dense-fails/BM25-wins).
- **Day 6** (`d7a998f`): ms-marco cross-encoder re-ranker (10 tests), 5-tier
  `RERANKER_MODE` router with graceful degradation (22 tests), and a
  three-node LangGraph retrieval agent (25 contract tests) close Layer 3b.
  208/208 tests passing. Live UAT regression check deferred to low free
  memory (0.33–0.49GB of 7.65GB).
- **Day 7 pt. 1** (`139a9cc`): the deferred UAT regression, run
  memory-safely by reranking cached RRF candidate pools instead of rebuilding
  BM25/dense/RRF — Q1's rank-1 confirmed stable, Q12's rank-1 correctly
  promoted from RRF's rank-2 by ms-marco.
- **Day 7 pt. 2** (today): QA agent (`retrieve → rerank → generate`, Claude
  Sonnet grounded in top-10 passages, per-claim citations via forced tool
  call, 22 tests) and a 3-endpoint FastAPI service (`/search`, `/ask`,
  `/health`) with `RERANKER_MODE`-aware dependency injection and a
  request-latency-to-Postgres middleware (13 tests). 243/243 tests passing.

## 4. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | 3/4 servers written | postgres (live), qdrant (live-verified), airflow (written, not yet live), mlflow deferred to Week 4 |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 3/4 signals + RRF done | BM25 (Day 3), SPLADE (Day 4), dense (Day 4/5), RRF fusion (Day 5); ColBERT outstanding |
| Layer 3b (re-ranking) | Done, live-verified | `retrieval/reranker_msmarco.py`, `retrieval/reranker_router.py`, `agents/retrieval_agent.py`, Day 7 UAT regression (`139a9cc`) |
| QA agent | Done (unit-tested; live Claude call not yet exercised) | `agents/qa_agent.py` — retrieve → rerank → generate, per-claim citations |
| API | Done (unit-tested; not yet run live with `uvicorn`) | `api/main.py`, `api/routers/{search,ask,health}.py`, `api/middleware.py` |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 243/243 passing (72 chunk-quality + 21 Text-to-SQL + 15 BM25
+ 18 SPLADE + 12 dense + 15 RRF + 10 ms-marco reranker + 22 reranker router +
25 retrieval agent + 22 QA agent + 13 API/middleware, plus contract/fixture
tests).

**Open items for next session**: run `uvicorn api.main:app` and hit
`/search`/`/ask`/`/health` live against the real corpus and a real
Anthropic call (everything shipped today is unit-tested against fakes, per
AGENTS.md's mock-everything-in-tests rule, so the live path is unverified);
confirm `request_log` rows actually land in Postgres outside the mocked
middleware tests; wire `RequestLog` into a monitoring dashboard query
(Layer 6).
