# nvidia-ir-rag-agent — Daily Progress

Storyline docs recording what was built, why, and what was live-verified,
one file per working day. Each doc is self-contained (what/why/narrative/
cumulative status) but the "Day 1 → N narrative" section in every doc
recaps everything before it, so any single file can be read on its own.

| Day | Focus | Tests added | Suite total | Doc |
|---|---|---|---|---|
| 1 | Project scaffolding — schema, docker-compose, `AGENTS.md`/`SKILLS.md` | 0 | 0/0 | [day_01_storyline.md](day_01_storyline.md) |
| 2 | Ingestion pipeline (Airflow DAG) + Text-to-SQL semantic layer, 5,389 chunks landed in Postgres | 93 | 93/93 | [day_02_storyline.md](day_02_storyline.md) |
| 3 | BM25 sparse retrieval + shared `Candidate` dataclass | 15 | 108/108 | [day_03_storyline.md](day_03_storyline.md) |
| 4 | Bi-encoder eval (e5-base-v2 wins), SPLADE sparse index, Qdrant populated, 3 MCP servers | 18 | 126/126 | [day_04_storyline.md](day_04_storyline.md) |
| 5 | Dense index (Qdrant), RRF fusion, hybrid search live-verified | 27 | 153/153 | [day_05_storyline.md](day_05_storyline.md) |
| 6 | ms-marco re-ranker, reranker router, LangGraph retrieval agent | 55 | 208/208 | [day_06_storyline.md](day_06_storyline.md) |
| 7 | QA agent, FastAPI endpoints, structlog instrumentation | 35 | 243/243 | [day_07_storyline.md](day_07_storyline.md) |
| 8 | Evaluation stack — relevance labeller, Cohere reranker, RAGAS suite, citation judge, benchmark runner | 80 | 323/323 | [day_08_storyline.md](day_08_storyline.md) |
| 9 | Live Config A/C benchmark, citation accuracy (0.7037), LangSmith tracing | 1 | 324/324 | [day_09_storyline.md](day_09_storyline.md) |
| 10 | Q10 citation fix, mcp-mlflow server | 2 | 326/326 | [day_10_storyline.md](day_10_storyline.md) |
| 11 | RAGAS live (real bug fixed), multi-agent orchestrator, drift/term-shift monitors, OTel + Jaeger live | 42 | 368/368 | [day_11_storyline.md](day_11_storyline.md) |
| 12 | EDA agent, Phoenix config, quality regression monitor, Streamlit UI, drift DAG, Docker/Render deploy, CI/CD gate | 61 | 429/429 | [day_12_storyline.md](day_12_storyline.md) |
| 13 | CI fully green (torch wheel + 4 more stacked pip conflicts + ragas patch + mypy strict debt), Slackbot + HITL feedback, feedback aggregator DAG, Streamlit live data, README, Render deploy prepared | 40 | 469/469 | [day_13_storyline.md](day_13_storyline.md) |
| 14 | Render deployment confirmed live, A2A protocol (retrieval→QA agent handoff), final evaluation report, CHANGELOG.md, **v1.0.0 tagged release** | 11 | 480/480 | [day_14_storyline.md](day_14_storyline.md) |

## Architecture reference

Full 8-layer architecture, MCP server list, coding standards, and folder
structure live in [`AGENTS.md`](../../AGENTS.md). Reusable code patterns
live in [`SKILLS.md`](../../SKILLS.md).

## Layer 3 (retrieval) progress at a glance

| Signal | Day | Status |
|---|---|---|
| BM25 (lexical) | 3 | Done |
| SPLADE (learned sparse) | 4 | Done |
| Dense (bi-encoder / Qdrant) | 4 (index), 5 (search wrapper) | Done |
| RRF fusion | 5 | Done |
| ColBERT | — | Outstanding |
| Layer 3b re-ranking (3-way benchmark) | 8 (msmarco), 9 (live A/C benchmark) | Config A/C done; B (bge-reranker-v2-m3) deferred, needs GPU |
