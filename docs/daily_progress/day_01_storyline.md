# Day 1 — Project Scaffolding

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `AGENTS.md` | 82 | — (contract doc, not code) |
| `SKILLS.md` | 51 | — (contract doc, not code) |
| `schema/schema.sql` | 134 | — (DDL, exercised indirectly by later ORM tests) |
| `docker-compose.yml` | 33 | — (postgres, pgadmin, qdrant, airflow services) |
| `requirements.txt` | — | — (pinned dependency set) |
| `.gitignore` | 36 | — |

No unit tests today — this is the scaffolding commit (`1bc7f0c`) that every
later day builds on.

## 2. Why it matters

Everything downstream depends on Day 1's two contract files and one schema.
`AGENTS.md` fixes the 8-layer architecture (Layer 0 pre-retrieval quality
through Layer 8 drift/HITL), the MCP server list, coding standards
(structlog-only logging, Pydantic v2, SQLAlchemy ORM, ruff/mypy strict), and
the re-ranker mode contract — Claude Code reads this file at the start of
every session in this repo. `SKILLS.md` captures reusable code patterns
(`retrieve-hybrid`, `build-langgraph-node`, `write-ragas-eval`,
`build-mcp-tool`, `add-otel-span`, `run-reranker-benchmark`) so later days
don't have to re-derive them. `schema/schema.sql` defines all 11 Postgres
tables up front — `doc_metadata` and `chunks` get populated on Day 2,
`chunk_quality` on Day 2, and the remaining eight (`coverage_log`,
`duplicate_flags`, `query_log`, `eval_results`, `request_log`, `error_log`,
`feedback_log`, `benchmark_results`) stay empty until the layers that write
to them (ingestion coverage, evaluation, monitoring, HITL) are built in
later weeks.

## 3. Day 1 narrative

- **Day 1** (`1bc7f0c`): project scaffolding — schema, docker-compose,
  `AGENTS.md`/`SKILLS.md` contracts. No application code yet; this commit
  exists purely to fix the architecture and interfaces before any layer is
  implemented, so later days have a stable contract to build against instead
  of discovering structure ad hoc.

## 4. What's in the schema and compose file

`schema/schema.sql` — 11 tables, applied automatically via
`docker-entrypoint-initdb.d` on first postgres start:

| Table | Purpose | First populated |
|---|---|---|
| `doc_metadata` | Per-document metadata (title, source URL, GPU family, page count) | Day 2 |
| `chunks` | Chunk text + position + token count, FK to `doc_metadata` | Day 2 |
| `chunk_quality` | Heuristic quality score per chunk (Layer 0) | Day 2 |
| `coverage_log` | Ingestion coverage checks (manifest vs. indexed) | Layer 2 coverage check (later) |
| `duplicate_flags` | Near-duplicate chunk pairs | Layer 0 dedup (later) |
| `query_log` | Every retrieval query issued | Layer 3+ serving (later) |
| `eval_results` | Per-query RAGAS/DeepEval scores | Layer 4-5 (later) |
| `request_log` | Per-stage latency/status for the API | Layer 6 (later) |
| `error_log` | API errors | Layer 6 (later) |
| `feedback_log` | Slackbot HITL reactions | Layer 8 (later) |
| `benchmark_results` | 3-way re-ranker benchmark rows | Layer 3b (later) |

`docker-compose.yml` — four services from day one: `postgres:16` (schema
auto-applied, healthcheck-gated), `pgadmin` (DB admin UI, depends on
postgres health), `qdrant/qdrant:latest` (vector DB, not populated until
Day 4), and `apache/airflow:3.0.2` in `standalone` mode (DAG folder mounted,
not run live until later — Day 2's ingestion actually runs via a direct
Python runner instead, see `day_02_storyline.md`).

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Not started | Schema table exists, no scorer yet |
| Layer 1 (MCP tool-calling) | Not started | — |
| Layer 2 (ingestion) | Not started | Compose service defined, no DAG yet |
| Layer 3 (retrieval) | Not started | — |
| Layer 3b (re-ranking) | Not started | — |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | Schema tables exist, no writers yet |

**Test suite**: 0/0 (no code yet).
