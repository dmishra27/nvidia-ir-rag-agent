# nvidia-ir-rag-agent

A hybrid-retrieval RAG agent over NVIDIA's public CUDA/GPU technical
documentation, with a full retrieval-evaluation, LLM-evaluation, and
human-feedback stack behind it — not just a search box.

Built by Debabrata Mishra ([dmishra27](https://github.com/dmishra27)) as an
**LLM Zoomcamp 2026 capstone submission**. Portfolio context: a 14-day build
log lives in [`docs/daily_progress/`](docs/daily_progress/), tagged
`v1.0.0` (see [`CHANGELOG.md`](CHANGELOG.md)); this project extends the
author's SIGIR 2024 co-authored work on neural passage-quality estimation
into a production-shaped RAG system.

**→ [Criteria-to-evidence mapping](#criteria-to-evidence-mapping)** — the
fastest way for a reviewer to check any single rubric line against the code.

## Problem statement

NVIDIA's CUDA documentation is fragmented across several separate,
differently-structured references — a CUDA C++ Programming Guide, a Best
Practices Guide, a Math API reference, a Runtime API reference, and a
profiling tool guide — each covering the same underlying concepts from a
different angle. A developer asking "how does `cudaMemcpyAsync`'s stream
parameter work?" has to know which document covers it, search each
separately, and cross-reference API signatures (which live in the Runtime
API reference) against the prose explanation of *why* it behaves that way
(which lives in the Programming Guide).

This project builds one retrieval + re-ranking + LLM-grounded answer
pipeline over that corpus, so a question gets a cited answer regardless of
which document actually contains it — with the retrieval and generation
quality actually measured, not just demoed:

1. **Real retrieval-quality measurement** — BM25 vs dense vs RRF fusion
   compared head-to-head over 15 designed queries, and two re-ranker
   configs benchmarked on NDCG@10/MRR/latency/cost (see
   [`/evaluation`](#live-deployment)).
2. **Per-claim citation grounding**, checked by a separate LLM-as-judge
   pass — not "the answer mentions a source," but "does chunk X actually
   support claim Y."
3. **A closed monitoring + human-feedback loop** — a Slack bot where real
   👍/👎 reactions write to `feedback_log`, aggregated weekly, not a
   one-shot eval run.

**Corpus** (5,389 chunks, 5 documents — see `api/static/index.html`'s
corpus panel, served at `/`):
CUDA C++ Programming Guide · CUDA C++ Best Practices Guide · CUDA Math API
Reference · CUDA Runtime API · Nsight Systems User Guide. Topics outside
these five documents (NVLink, H100, TensorRT, etc.) are out of scope by
construction — queries about them return low-relevance results, not
nothing, and the UI says so.

## Live deployment

| Endpoint | Status as of 2026-08-16 |
|---|---|
| [`nvidia-ir-rag-agent.onrender.com/health`](https://nvidia-ir-rag-agent.onrender.com/health) | ✅ `200 OK` — `{"status": "ok", "service": "nvidia-ir-rag-agent"}`, confirmed live. Note: earlier checks the same day returned `503 Service Unavailable` on every route for ~7 minutes straight before recovering — consistent with a Render free/starter-tier spin-down rather than a code defect, but it means the service isn't guaranteed to be warm on first click. |
| [`nvidia-ir-rag-agent.onrender.com/evaluation`](https://nvidia-ir-rag-agent.onrender.com/evaluation) | ✅ Confirmed live — loads `api/static/evaluation.html`, "Retrieval Evaluation" heading, NDCG@10 0.5333/0.5280 figures present. |
| `POST /search` | ✅ Confirmed live — `{"query": "cudaMalloc function parameters", "top_k": 3}` returned three ranked results, rank 1 the `cudaMalloc(void**, size_t)` signature chunk, with scores `1/61`, `1/62`, `1/63` — the single-ranker RRF signature confirming BM25-only fallback per `render.yaml`'s `RERANKER_MODE=fallback`. |
| `POST /ask` | Not verified — requires `ANTHROPIC_API_KEY`, which is not configured on the deployed instance. |

**For working `/search` and `/ask` with real results verified locally**,
run it yourself — see [Setup & run](#setup--run). `docker compose up -d`
runs the same code Render runs, against your own Postgres/Qdrant instead
of Render's.

## Architecture

```
 User (Streamlit / Slack /nvidia-search / curl)
        │
        ▼
 FastAPI  POST /ask  ──►  agents/qa_agent.py (LangGraph: retrieve → rerank → generate)
        │                        │
        │              ┌─────────┴─────────┐
        │              ▼                   ▼
        │        BM25 + Dense          RerankerRouter
        │      (retrieval/rrf_fusion)  (RERANKER_MODE: live_fast
        │              │                default = ms-marco cross-
        │              └────────┬───────  encoder; degrades to
        │                       ▼          fallback = BM25 order)
        │             Claude Sonnet 5, forced tool call
        │             (answer + per-claim chunk_id citations)
        ▼
 AskResponse{answer, citations[], query_id}
        │
        ├──► Slack: handlers.py posts top-3 citations;
        │           feedback_handler.py records 👍/👎 → feedback_log
        │
        └──► Streamlit: 5 tabs (search / benchmark / eval / monitoring / drift)
```

**Surrounding infrastructure** (not on the request path above, but what
produces and evaluates the data the request path uses):

- **Ingestion** — `airflow/dags/ingest_nvidia_docs.py`: PyMuPDF parse →
  LangChain chunk → quality score → SQLAlchemy → Postgres.
- **Evaluation** — `evaluation/benchmark_runner.py` (NDCG/MRR/latency
  across re-ranker configs), `evaluation/ragas_suite.py` (faithfulness /
  answer relevancy), `evaluation/citation_judge.py` (per-claim LLM judge).
- **Monitoring** — `monitoring/drift_detector.py` (PSI),
  `monitoring/term_shift_monitor.py`, `monitoring/quality_regression.py`,
  `monitoring/feedback_aggregator.py`, all surfaced in the Streamlit
  dashboard and traced via OpenTelemetry/Jaeger + LangSmith.

Full 8-layer detail and folder structure: [`AGENTS.md`](AGENTS.md).

## Criteria-to-evidence mapping

Self-assessed against the published rubric. Where a criterion is only
partly met, that's stated here rather than left implicit — verify against
the linked file/endpoint directly.

| Rubric criterion | Assessment | Evidence |
|---|---|---|
| **Problem description** (2) | Met | This README's [Problem statement](#problem-statement); page description in `api/static/index.html` |
| **Retrieval flow** (2) | Met | Knowledge base (BM25 + dense over Postgres/Qdrant) + LLM generation, both real: `agents/qa_agent.py`, `POST /ask` |
| **Retrieval evaluation** (2) | Met, scope-caveated | `docs/uat/uat_superiority_cases_executed.md` — BM25 vs dense vs RRF, 15 queries / 6 designed cases, live-run. `evaluation/benchmark_runner.py` + `/evaluation` — re-ranker Config A (ms-marco) beats Config C (Cohere) on NDCG@10 (0.5333 vs 0.5280) and **A is the default `RERANKER_MODE`** actually served. Caveat: `benchmark_baseline.json`'s own `_note` field says this is a 15-query smoke-scope run at top-3, not the full 50-query × top-100 run `benchmark_runner.py` is designed for. |
| **LLM evaluation** (2) | ⚠️ Partial | `evaluation/ragas_suite.py` (faithfulness 0.7616, answer relevancy 0.2497, real RAGAS run) and `evaluation/citation_judge.py` (LLM-as-judge citation accuracy, 0.7037 over 27 claims) are both real, live-measured evaluations of generation quality. **What's missing**: only one generation model/prompt (`claude-sonnet-5`, hardcoded in `agents/qa_agent.py`) was ever used — no comparison across multiple LLMs or prompts with a "best one" selected. If the rubric requires that comparison specifically, treat this as not met. |
| **Interface** (2) | Met | FastAPI (`/search`, `/ask`, `/health`, `/docs`), guided search UI (`/`), evaluation report (`/evaluation`), Streamlit (`streamlit_app/app.py`, 5 tabs), Slack bot (`slackbot/app.py`) |
| **Ingestion pipeline** (2) | Met, caveated | `airflow/dags/ingest_nvidia_docs.py` — real Airflow 3 TaskFlow DAG (fetch → parse → chunk → score → write → log coverage), runs inside `docker-compose.yml`'s `airflow` service. Caveat: the 5,389-chunk corpus actually in this repo was loaded via `run_ingest_direct.py`, a direct-Python mirror of the same pipeline — the DAG itself has not been run against a live Airflow scheduler in this project. |
| **Monitoring** (2) | Met | Feedback, two input paths into the same `feedback_log` table: (1) Slack 👍/👎 via `slackbot/feedback_handler.py` (`source="slackbot"`), and (2) a thumbs up/down control on each result card at `/` via `POST /feedback` (`api/routers/feedback.py`, `source="web"`) — both write the real `FeedbackLog` ORM model, no schema change; `monitoring/feedback_aggregator.py` aggregates weekly. Dashboard: `streamlit_app/benchmark_tab.py`, `eval_dashboard.py`, `monitoring_tab.py`, `drift_tab.py` — 5+ charts (ranking-quality bar, latency bar, per-query NDCG box-plot, NDCG-vs-gate, per-query citation accuracy, quality-regression trend, per-stage latency, request/error rate, PSI drift, term shift) |
| **Containerization** (2) | Met, not literally everything | `docker-compose.yml` — 9 services: `api`, `streamlit`, `postgres`, `qdrant`, `mlflow`, `pgadmin`, `airflow`, `jaeger`, `phoenix`. Caveat: the Slack bot and the 4 MCP servers run on the host, not in `docker-compose.yml` — see [Setup & run](#setup--run). |
| **Reproducibility** (2) | Met | All 384 pinned dependencies in `requirements.txt` (`==`, verified directly — the file is UTF-16, which breaks naive `grep` but not `pip`); `.env.example` covers every variable with working local defaults; `data/indexes/bm25_index.pkl` committed to git so `/search` works with zero setup. Full dense/hybrid reproduction needs a documented ~86-minute local embedding pass — see [Setup & run](#setup--run). |
| **Hybrid search** (1, best practice) | Met | `retrieval/rrf_fusion.py` fuses BM25 (`retrieval/bm25_index.py`) + dense (`retrieval/dense_index.py`) |
| **Document re-ranking** (1, best practice) | Met | `retrieval/reranker_msmarco.py` (cross-encoder), wired as the default tier in `retrieval/reranker_router.py` |
| **Query rewriting** (1, best practice) | ❌ Not evidenced | No query rewriting, expansion, or HyDE-style reformulation exists anywhere in the codebase (verified by grep — no matches) |
| **Bonus: cloud deployment** (2) | Met, with a warm-up caveat | `render.yaml` deploys `POST /search`/`/ask`/`/health` to Render, correctly configured for the 512 MB limit. `/health` and `/evaluation` both confirmed live (`200 OK`) — see [Live deployment](#live-deployment). Caveat: the same URL returned `503` on every route for ~7 minutes earlier the same day before recovering, so it isn't guaranteed warm on a reviewer's first click. |

## Setup & run

Quick version below; **full step-by-step instructions, the complete
environment-variable table, the port map, and host-run components are in
[`setup.md`](setup.md).**

```bash
git clone https://github.com/dmishra27/nvidia-ir-rag-agent.git
cd nvidia-ir-rag-agent
cp .env.example .env        # every value already has a working local default
docker compose up -d        # api, streamlit, postgres, qdrant, mlflow, pgadmin, airflow, jaeger, phoenix
curl http://localhost:8001/health
curl -X POST http://localhost:8001/search -H 'content-type: application/json' -d '{"query": "cudaMemcpyAsync"}'
```

Windows PowerShell mangles that inline-JSON `-d` payload (you'll get a
`json_invalid` error, not a broken API) — write the body to a file first:

```powershell
'{"query":"cudaMalloc function parameters","top_k":3}' | Out-File -Encoding ascii body.json
curl.exe -s -X POST http://localhost:8001/search -H "content-type: application/json" -d "@body.json"
```

`data/indexes/bm25_index.pkl` is committed to git, so `/search` returns
real BM25 results after `docker compose up -d` — no data download or
embedding step required. The default `RERANKER_MODE=live_fast` also queries
the dense index, but on a fresh clone Qdrant's `nvidia_ir_chunks`
collection doesn't exist yet (it's created by the one-time ~86-minute
`populate_qdrant.py` step — see
[`setup.md`](setup.md#4-populate-the-corpus)). When the dense query fails,
retrieval **degrades to BM25-only** and logs a `dense_retrieval_degraded`
warning; you still get real ranked results, just without the dense/RRF
half until you populate Qdrant. If retrieval fails outright (BM25 index
missing too), `/search` returns HTTP 503 with an `error` field rather than
an empty `200`. An `ANTHROPIC_API_KEY` is needed for `/ask` to generate
answers (not for `/search`).

**Tests:**

```bash
pytest          # 520 tests, all mock embedding/LLM/DB/MLflow calls — no live services needed
ruff check .
mypy .           # strict on agents/, api/, retrieval/, monitoring/, evaluation/, schema/, mcp/, slackbot/, streamlit_app/
```

## Example query

A real, live-run example — `agents/qa_agent.py`'s actual output, from Day
9's benchmark set:

> **Q:** cudaMemcpyAsync stream parameter
>
> **A:** Based on the available passages, `cudaMemcpyAsync()` is described
> as a non-blocking variant of `cudaMemcpy()` in which control returns to
> the calling thread immediately rather than waiting for the copy to
> complete — this asynchronous behavior is tied to the stream on which the
> operation is enqueued. However, the passages provided do not explicitly
> detail the specific "stream" parameter (its type, default value, or
> usage) for `cudaMemcpyAsync()`; they only confirm its non-blocking nature
> and list it alongside related memory copy functions.
>
> **Citations:**
> 1. *"cudaMemcpyAsync() is a non-blocking variant of cudaMemcpy() in which
>    control is returned to the calling thread before the copy completes."*
>    → `242e353090d3f493c8ef64dc`
> 2. *"cudaMemcpyAsync is listed as a related function alongside cudaMemcpy,
>    cudaMemcpy2D, ..."* → `f9f67e7b94c193a8782f538e`

The model declines to invent the stream parameter's type/default when the
retrieved passages don't cover it — this is what `evaluation/citation_judge.py`
and RAGAS faithfulness are there to catch when it *doesn't* happen.

## Known limitations

Stated plainly, matching the [criteria mapping](#criteria-to-evidence-mapping)
above rather than repeating it:

- **Render runs BM25-only.** `RERANKER_MODE=fallback` on the deployed
  service is load-bearing, not a quality choice: `render.yaml`'s own
  comment documents that Render's 512 MB plan can't fit
  `DenseIndex.connect()` + `MSMarcoReranker.load()` together — confirmed
  via Render's Events log reporting "Ran out of memory (used over 512MB)"
  on every `/search` call before this was set. **The full BM25 + dense +
  RRF + cross-encoder re-ranking pipeline only runs locally** via
  `docker compose up -d`.
- **Render's free/starter tier isn't guaranteed warm.** `/health` and
  `/evaluation` are confirmed live as of this writing, but the same URL
  returned `503` on every route for ~7 minutes earlier the same session —
  see [Live deployment](#live-deployment). A reviewer's first click could
  land during a spin-down/recovery window.
- **No query rewriting** — not implemented anywhere in the codebase.
- **LLM evaluation compares metrics, not models** — RAGAS and the citation
  judge evaluate one fixed generation model; no multi-model/multi-prompt
  comparison was run.
- **Config B (bge-reranker-v2-m3)** never ran at any scope — hardware-
  blocked (OOMs at model load on this project's 8 GB CPU-only dev machine).
- **15-query smoke-scope benchmark**, not the full 50-query × top-100 run
  `evaluation/benchmark_runner.py` is designed for.
- **Ingestion DAG not live-scheduler-run** — the committed corpus was
  loaded via `run_ingest_direct.py`, a direct-Python mirror of
  `airflow/dags/ingest_nvidia_docs.py`, not by triggering the DAG under a
  running Airflow scheduler.
- **Slack bot and MCP servers are not containerized** — they run on the
  host against `docker compose`'s published ports; see
  [`setup.md`](setup.md#7-things-that-still-run-on-the-host).
- **No human evaluation** of answer quality — RAGAS faithfulness is an
  automated LLM-judged proxy, not a human review.
- ColBERT retrieval — not started.
- Quality-regression and drift DAGs are unit-tested but not run against a
  live Airflow scheduler.
- **Dense retrieval on a fresh clone is unavailable until Qdrant is
  populated** (`populate_qdrant.py`, ~86 min). `agents/retrieval_agent.py`
  now degrades to BM25-only when the dense query fails — logging
  `dense_retrieval_degraded` — instead of discarding the BM25 half, so
  `/search` returns real results on the default `RERANKER_MODE=live_fast`
  config out of the box. Earlier versions of this section warned that a
  dense failure took the BM25 results down with it (clean-clone test
  CCT-NVIR-2026-001, F-14); that was correct against the then-current code
  and is fixed as of this commit.

## Project docs

- [`AGENTS.md`](AGENTS.md) — full 8-layer architecture, coding standards,
  MCP server list, folder structure.
- [`SKILLS.md`](SKILLS.md) — reusable code patterns.
- [`reports/final_eval_report.md`](reports/final_eval_report.md) — the
  single most current, consolidated evaluation writeup (NDCG, RAGAS,
  citation judge, explicit known-gaps section, research context).
- [`docs/uat/uat_superiority_cases_executed.md`](docs/uat/uat_superiority_cases_executed.md) —
  full per-query BM25 vs dense vs RRF comparison behind the retrieval-
  evaluation criterion above.
- [`docs/daily_progress/`](docs/daily_progress/) — day-by-day build log,
  one storyline per working day; start at
  [`docs/daily_progress/README.md`](docs/daily_progress/README.md).
- [`setup.md`](setup.md) — full setup/run instructions, environment
  variables, port map.
