# nvidia-ir-rag-agent — AGENTS.md

## Project identity
AI-powered hybrid information retrieval system over NVIDIA public technical
documentation. Portfolio project by Debabrata Mishra (dmishra27).
SIGIR 2024 co-author — directly extends neural passage quality estimation research.
GitHub: https://github.com/dmishra27/nvidia-ir-rag-agent

## Architecture overview — 8 layers
Layer 0: Pre-retrieval quality (chunk scorer, freshness, dedup, coverage)
Layer 1: MCP tool-calling (4 servers: postgres, qdrant, mlflow, airflow)
Layer 2: Ingestion (Airflow 3 DAG, PyMuPDF, LangChain splitters, spaCy)
Layer 3: Retrieval (BM25 + SPLADE + dense + ColBERT + RRF)
Layer 3b: Re-ranking (ms-marco / bge-reranker-v2-m3 / Cohere — 3-way benchmark)
Layer 4-5: Evaluation (RAGAS, DeepEval, citation judge, NDCG, MRR)
Layer 6: Monitoring (per-stage latency, error rate, throughput)
Layer 7: Observability (LangSmith, OpenTelemetry+Jaeger, structlog, Arize Phoenix)
Layer 8: Drift + HITL (PSI, term shift, quality regression, Slackbot feedback)

## MCP servers
nvidia-ir-postgres: query all 11 tables
  (doc_metadata, chunks, chunk_quality, coverage_log, duplicate_flags,
   query_log, eval_results, request_log, error_log, feedback_log, benchmark_results)
nvidia-ir-qdrant: vector search, collection stats (Week 2+)
nvidia-ir-mlflow: experiment metrics, benchmark results (Week 4+)
nvidia-ir-airflow: DAG status, task logs (Week 2+)

## Coding standards
- Python 3.11+. Type hints on all function signatures.
- structlog for all logging. Never use print() or logging.info().
  Every log call must include query_id and stage fields.
- Pydantic v2 models for all data schemas.
- pytest for all tests. TDD for deterministic functions.
  Mock all embedding and LLM calls in unit tests.
- ruff for formatting. mypy strict mode.
- SQLAlchemy ORM for all database writes. Never raw SQL strings.
- Environment variables from .env via python-dotenv. Never hardcode keys.

## Re-ranker configuration
Controlled by RERANKER_MODE env var:
  live_fast     -> ms-marco-MiniLM-L-6-v2 (default, <1s CPU)
  live_quality  -> bge-reranker-v2-m3 via sentence-transformers
  live_frontier -> Cohere Rerank v3 API
  benchmark     -> all three run in parallel, results to MLflow
  fallback      -> BM25 rank order, no re-ranking

## Test patterns
- Deterministic functions: strict TDD, test first
- Embedding/LLM calls: mock with numpy fixtures
- LangGraph nodes: contract tests on state schema only
- End-to-end: VCR cassette fixtures, no live API calls in CI
- LLM quality: RAGAS threshold gates in CI (not unit tests)

## Folder structure
retrieval/    bm25_index, splade_index, dense_index, rrf_fusion,
              reranker_msmarco, reranker_bge, reranker_cohere, reranker_router,
              chunk_quality, dedup
agents/       retrieval_agent, cleaning_agent, qa_agent, eval_agent,
              eda_agent, orchestrator, a2a_protocol, text_to_sql_agent
mcp/          mcp_postgres/, mcp_qdrant/, mcp_mlflow/, mcp_airflow/
api/          main.py, routers/, schemas/, middleware.py, telemetry.py
monitoring/   freshness_monitor, drift_detector, term_shift_monitor,
              quality_regression, phoenix_config
evaluation/   ragas_suite, deepeval_suite, citation_judge,
              retrieval_metrics, benchmark_runner, plot_benchmark,
              relevance_labeller
schema/       schema.sql, SQLAlchemy models, Alembic migrations
airflow/dags/ ingest_nvidia_docs, coverage_check, drift_monitor,
              quality_regression, feedback_aggregator

## Key versions installed
torch: 2.13.0+cpu
transformers: 5.13.0
sentence-transformers: 5.6.0
langchain: 1.3.12
langgraph: 1.2.8
langsmith: 0.10.0
ragas: 0.4.3 (patched — see requirements_notes.txt)
mlflow: 3.14.0
qdrant-client: 1.18.0
fastapi: 0.139.0
python: 3.11.9
