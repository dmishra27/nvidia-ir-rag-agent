# Changelog

All notable changes to this project are documented here, grouped by the
day-by-day build log this project shipped as (see
[`docs/daily_progress/`](docs/daily_progress/) for the full narrative
behind each entry). This project ships its first tagged release at
`v1.0.0`, capturing all 14 days of build history rather than a series of
incremental prior tags — format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-08

### Added

**Day 1 — Project scaffolding**
Postgres schema (11 tables), `docker-compose.yml`, `AGENTS.md`/`SKILLS.md`
project conventions.

**Day 2 — Ingestion + semantic layer**
Airflow 3 DAG (`airflow/dags/ingest_nvidia_docs.py`): PyMuPDF parse →
LangChain chunk → quality score → SQLAlchemy ORM → Postgres. 5,389 chunks
landed. `agents/text_to_sql_agent.py` — natural-language-to-SQL semantic
layer over the ingested corpus.

**Day 3 — BM25 retrieval**
`retrieval/bm25_index.py`, shared `retrieval/candidates.py::Candidate`
dataclass used by every retrieval/re-ranking module since.

**Day 4 — Bi-encoder eval, SPLADE, Qdrant, MCP servers**
`retrieval/biencoder_eval.py` (e5-base-v2 wins), `retrieval/splade_index.py`,
Qdrant collection populated (`retrieval/populate_qdrant.py`), first 3 MCP
servers (`mcp/mcp_postgres/`, `mcp/mcp_qdrant/`).

**Day 5 — Dense index + RRF fusion**
`retrieval/dense_index.py`, `retrieval/rrf_fusion.py` — hybrid search
(BM25 + dense + RRF) live-verified end to end.

**Day 6 — ms-marco re-ranker + LangGraph retrieval agent**
`retrieval/reranker_msmarco.py`, `retrieval/reranker_router.py`,
`agents/retrieval_agent.py` (LangGraph: retrieve → rerank → return_results).

**Day 7 — QA agent + FastAPI + structlog**
`agents/qa_agent.py` (LangGraph: retrieve → rerank → generate, forced
tool-call citations), `api/main.py`/`api/routers/` (`/search`, `/ask`,
`/health`), structlog instrumentation project-wide.

**Day 8 — Evaluation stack**
`evaluation/relevance_labeller.py`, `retrieval/reranker_cohere.py`,
`evaluation/retrieval_metrics.py` (NDCG/MRR/Precision@K),
`evaluation/benchmark_runner.py` (3-way A/B/C re-ranker benchmark),
`evaluation/citation_judge.py`, `evaluation/ragas_suite.py`.

**Day 9 — Live benchmark + citation results**
Config A/C benchmark run for real (NDCG@10 0.5333 / 0.5280, 15 queries),
citation judge run for real (0.7037 accuracy, 27 claims), LangSmith
tracing verified, real `qa_agent.py` bug found and fixed live.

**Day 10 — mcp-mlflow server**
`mcp/mcp_mlflow/server.py`, a real production bug fix (Q10's stringified
citation output).

**Day 11 — Orchestrator, RAGAS live, drift monitors, OTel**
`agents/orchestrator.py` (multi-agent: retrieve → rerank → generate →
evaluate), `agents/eval_agent.py`, RAGAS live-verified with a real
bypass-temperature bug found and fixed, `monitoring/drift_detector.py`
(PSI) + `monitoring/term_shift_monitor.py`, OpenTelemetry + Jaeger live
end to end, `mcp/mcp_mlflow/server.py` live-verified.

**Day 12 — EDA agent, Phoenix, quality regression, Streamlit, CI/CD**
`agents/eda_agent.py` (PandasAI conversational EDA),
`monitoring/phoenix_config.py` (Arize Phoenix instrumentation),
`monitoring/quality_regression.py` (3-day-decline alerting),
`streamlit_app/` (5-tab UI shell), `airflow/dags/drift_monitor.py`,
`evaluation/ci_ndcg_gate.py`, `Dockerfile`, `render.yaml`,
`.github/workflows/ci.yml` (first version — not yet green).

**Day 13 — CI green, Slackbot + HITL, live Streamlit, README, Render**
- Fixed CI end to end: torch CPU wheel index, 4 more pip resolver
  conflicts it unmasked (`pyarrow`/`pillow`/`pdfplumber`/
  `openinference-semantic-conventions`/`datasets`, `langchain-google-vertexai`
  removed), a ragas post-install patch (`scripts/patch_ragas.py` +
  `requirements_notes.txt`), and 44 real `mypy --strict` errors across
  production code (`tests/`/root scripts scoped out of mypy, matching the
  existing `run_*.py` ruff convention).
- `slackbot/app.py`/`handlers.py`/`feedback_handler.py` — Slack Bolt app
  (Socket Mode), `/nvidia-search` slash command, 👍/👎 HITL feedback →
  `feedback_log` (new `FeedbackLog` ORM model).
- `monitoring/feedback_aggregator.py` + `airflow/dags/feedback_aggregator.py`
  — weekly HITL feedback aggregation.
- `streamlit_app/live_data.py` — `benchmark_tab.py`/`eval_dashboard.py`
  wired to live MLflow/Postgres/JSON data (5 real-data charts, up from 2),
  with graceful fallback and a capped MLflow HTTP timeout.
- `README.md` (repo had none before), `.env.example`.
- Render deploy prepared (`render.yaml`, pre-existing from Day 12); deploy
  action itself is manual/dashboard-only.

**Day 14 — A2A protocol, final eval report, v1.0.0 release**
- `agents/a2a_protocol.py` — typed `AgentMessage` handoff envelope between
  `retrieval_agent.py` and `qa_agent.py`; `run_handoff()` runs retrieval
  once and hands its ranked results to `qa_agent`'s `generate` node
  directly, instead of each agent independently re-retrieving.
- `reports/final_eval_report.md` — consolidated NDCG/RAGAS/citation-accuracy
  results, re-ranker comparison, and research context, sourced entirely
  from this project's real committed Day 9/11 numbers.
- Render deployment confirmed live (`/health` 200, Swagger UI at `/docs`);
  `/search`/`/ask` confirmed 500ing for the documented reason (BM25 index
  not shipped in the deployed image).
- First tagged release: `v1.0.0`.

### Fixed

- Day 9: `agents/qa_agent.py` crash on a specific query shape, found via
  the live benchmark run.
- Day 10: Q10's citation output was a stringified list instead of a real
  list — found and fixed live.
- Day 11: RAGAS's `bypass_temperature` handling — a real bug, not a mock
  artifact, found running RAGAS live for the first time.
- Day 13: CI's entire install→lint→typecheck→test→gate pipeline — six
  stacked pre-existing defects, none related to each other, uncovered
  one at a time by watching real GitHub Actions runs (see that day's
  storyline §2-3 for the full chain).

### Known limitations (carried into v1.0.0, not resolved)

- Config B (bge-reranker-v2-m3) — GPU-hardware-blocked, never benchmarked
  at any scope.
- ColBERT retrieval — not started.
- Full 50-query × top-100-candidate benchmark — not run (15-query smoke
  scope only).
- Render deployment's `/search`/`/ask` — 500, BM25 index not shipped in
  the deployed image (documented, not silently broken).
- `drift_monitor.py`/`feedback_aggregator.py` DAGs — unit-tested, never
  run against a live Airflow scheduler.
- Slack alert wiring for the two monitoring DAGs' `alert_fn` — not
  connected, despite `slackbot/` existing since Day 13.
