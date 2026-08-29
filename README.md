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

## Quickstart

The rest of this README and [`setup.md`](setup.md) are thorough by design;
this block is the whole happy path in one place. **BM25 search only — no
API keys, no data download, no embedding step.**

```bash
git clone https://github.com/dmishra27/nvidia-ir-rag-agent.git
cd nvidia-ir-rag-agent
cp .env.example .env                 # every value already has a working local default
docker compose up -d                 # ~10 min first build; pulls + builds 9 services
docker compose ps                    # wait for `api` to read `healthy` (not just `up`)

curl -X POST http://localhost:8001/search \
  -H 'content-type: application/json' \
  -d '{"query": "cudaMalloc function parameters", "top_k": 3}'
```

**Expect:** three ranked results, rank 1 the `cudaMalloc(void**, size_t)`
signature chunk (`cc6c8e53936d04e9b192a7d5`), RRF scores `1/61`, `1/62`,
`1/63`. **The first call takes a few minutes** (cold model/index load under
memory pressure); subsequent calls are fast. Then open
`http://localhost:8001/` for the guided UI, `http://localhost:8001/evaluation`
for the retrieval-quality report, `http://localhost:8501` for the 5-tab
Streamlit dashboard.

`/ask` (LLM-generated cited answers) additionally needs `ANTHROPIC_API_KEY`
in `.env`. Dense/hybrid retrieval additionally needs the one-time
`populate_qdrant.py` embedding pass (§4 of [`setup.md`](setup.md)) — until
then retrieval degrades to BM25-only and logs `dense_retrieval_degraded`.

## Requirements

Measured on the 8 GB CPU-only dev machine during a clean-clone
reproducibility test (`docs/uat/clean_clone_test_findings.md`), not
estimated.

| | Requirement | Notes |
|---|---|---|
| **Python** | 3.11 **exactly** | Host-run scripts (`run_*.py`, `retrieval/populate_qdrant.py`) hard-fail on any other version; `.python-version` pins it. The install breaks on 3.14 (`pandasai==3.0.0` has no wheel) with an error that never names the version. Not needed for the containers — only for host-run tooling, tests, and ingestion. |
| **Free RAM** | 4 GB min; 8 GB total is marginal | The full 9-service stack leaves ~0.5 GB free on an 8 GB host. Don't run the ingestion/embedding pass with the whole stack up, and don't `docker restart` a single service with the stack up — either OOM-kills the `api` container. |
| **Disk** | ~6 GB | ~2.5 GB venv + ~3 GB Docker images. |
| **Time to first BM25 query** | ~50 min | 34 min `pip install` + ~10 min image build + ~5 min `run_ingest_direct.py`. |
| **Time to full hybrid pipeline** | ~2 h 20 min | above + ~86 min `populate_qdrant.py` embedding pass. |
| **Ports** | 8001, 8501, 5432, 5050, 6333, 6334, 5001, 8080, 16686, 4317, 4318, 6006 | All must be free. Only **one instance of this project can run at a time** — see [`setup.md` §3](setup.md#3-bring-up-the-stack). |
| **External accounts** | None for BM25 `/search` | `ANTHROPIC_API_KEY` for `/ask`; `COHERE_API_KEY` for the Config C benchmark only. `dvc pull` needs **no** credentials. |

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
Reference · CUDA Runtime API · Nsight Systems User Guide. Queries about
topics these five documents don't cover (NVLink, H100, TensorRT, etc.)
return low-relevance results, not nothing, and the UI says so.

These gaps are a mix of deliberate scoping and known ingestion debt, not
one clean design line:

- **NVLink / H100** — the Ampere and Hopper architecture whitepapers and
  the A100 datasheet *are* DVC-tracked (`data/raw/*.dvc`) but were never
  wired into the ingestion manifest (`run_ingest_direct.py`), so nothing
  from them is in the index. Tracked-but-not-ingested (DEF-19).
- **TensorRT / cuDNN / Thrust** — dropped from the manifest because their
  NVIDIA PDF URLs now 404 (NVIDIA serves an identical HTML error page for
  all three); ingesting them would need re-sourced PDFs (DEF-20).

See `run_ingest_direct.py`'s `MANIFEST` comment and
`docs/uat/correction_notice_a1.md` §6.

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

## Beyond the core rubric

Work that sits outside the required criteria and is easy to miss on a
first pass. Each is real and runnable, not a stub.

### ☁️ Cloud deployment — `render.yaml`

`POST /search` · `/ask` · `/health` deployed to Render, configured for the
free-tier 512 MB cap (`RERANKER_MODE=fallback`, BM25-only — the memory
math is in `render.yaml`'s own comments). Live endpoints and the
spin-down caveat are in [Live deployment](#live-deployment).
[`render.yaml`](render.yaml)

### ✅ CI/CD quality gate — `.github/workflows/ci.yml`

Every push runs: checkout → CPU-only-torch install → `patch_ragas.py` →
`ruff check` → `mypy` (strict on all production packages) → `pytest` →
an **NDCG@10 regression gate** that fails the build if retrieval quality
drops below `evaluation/benchmark_baseline.json`'s threshold.
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)

### 🔭 Observability stack

- **OpenTelemetry → Jaeger** — `api/telemetry.py`, `traced_stage` spans
  through `agents/retrieval_agent.py` + `agents/qa_agent.py`; UI at
  `:16686` (`ENABLE_TRACING=true`).
- **Arize Phoenix** — `monitoring/phoenix_config.py`, LLM-call tracing at
  `:6006` (`ENABLE_PHOENIX=true`).
- **MLflow** — `reranker_benchmark`, `ragas_eval`, `citation_judge`
  experiments at `:5001`; the Streamlit benchmark tab reads from it live.
- **structlog** — single-line JSON logs, one `query_id` correlated across
  every stage and both endpoints.
- **LangSmith** — optional, `LANGCHAIN_TRACING_V2` (verified in Day 9).

### 🤖 Slack bot + 4 MCP servers

- **Slack bot** — `slackbot/app.py`, Socket Mode: `/nvidia-search` slash
  command, `/ask` answers with top-3 citations, 👍/👎 reactions write to
  `feedback_log` (`source="slackbot"`).
- **MCP servers** — `mcp/mcp_postgres`, `mcp/mcp_qdrant`, `mcp/mcp_mlflow`,
  `mcp/mcp_airflow`: query the corpus DB, the dense index, experiment
  history, and DAG status from any MCP-compatible client (`.mcp.json`).

Both run on the host, not in `docker compose` — see
[`setup.md` §7](setup.md#7-things-that-still-run-on-the-host).

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
| **Interface** (2) | Met | FastAPI (`/search`, `/ask`, `/health` liveness + `/health/ready` dependency-assertion, `/docs`), guided search UI (`/`), evaluation report (`/evaluation`), Streamlit (`streamlit_app/app.py`, 5 tabs), Slack bot (`slackbot/app.py`) |
| **Ingestion pipeline** (2) | Met, caveated | `airflow/dags/ingest_nvidia_docs.py` — real Airflow 3 TaskFlow DAG (fetch → parse → chunk → score → write → log coverage), runs inside `docker-compose.yml`'s `airflow` service. Caveat: the 5,389-chunk corpus actually in this repo was loaded via `run_ingest_direct.py`, a direct-Python mirror of the same pipeline — the DAG itself has not been run against a live Airflow scheduler in this project. |
| **Monitoring** (2) | Met | Feedback, two input paths into the same `feedback_log` table: (1) Slack 👍/👎 via `slackbot/feedback_handler.py` (`source="slackbot"`), and (2) a thumbs up/down control on each result card at `/` via `POST /feedback` (`api/routers/feedback.py`, `source="web"`) — both write the real `FeedbackLog` ORM model, no schema change; `monitoring/feedback_aggregator.py` aggregates weekly. Dashboard: `streamlit_app/benchmark_tab.py`, `eval_dashboard.py`, `monitoring_tab.py`, `drift_tab.py` — 5+ charts (ranking-quality bar, latency bar, per-query NDCG box-plot, NDCG-vs-gate, per-query citation accuracy, quality-regression trend, per-stage latency, request/error rate, PSI drift, term shift) |
| **Containerization** (2) | Met, not literally everything | `docker-compose.yml` — 9 services: `api`, `streamlit`, `postgres`, `qdrant`, `mlflow`, `pgadmin`, `airflow`, `jaeger`, `phoenix`. Caveat: the Slack bot and the 4 MCP servers run on the host, not in `docker-compose.yml` — see [Setup & run](#setup--run). |
| **Reproducibility** (2) | Met | All 384 pinned dependencies in `requirements.txt` (`==`, verified directly — the file is UTF-16, which breaks naive `grep` but not `pip`); `.env.example` covers every variable with working local defaults; `data/indexes/bm25_index.pkl` committed to git so `/search` works with zero setup; the 8-PDF source corpus is DVC-pinned and `dvc pull`s with no credentials — see [Data versioning](#data-versioning). Full dense/hybrid reproduction needs a documented ~86-minute local embedding pass — see [Setup & run](#setup--run). |
| **Hybrid search** (1, best practice) | Met | `retrieval/rrf_fusion.py` fuses BM25 (`retrieval/bm25_index.py`) + dense (`retrieval/dense_index.py`) |
| **Document re-ranking** (1, best practice) | Met | `retrieval/reranker_msmarco.py` (cross-encoder), wired as the default tier in `retrieval/reranker_router.py` |
| **Query rewriting** (1, best practice) | ❌ Not evidenced | No query rewriting, expansion, or HyDE-style reformulation exists anywhere in the codebase (verified by grep — no matches) |
| **Bonus: cloud deployment** (2) | Met, with a warm-up caveat | `render.yaml` deploys `POST /search`/`/ask`/`/health` to Render, correctly configured for the 512 MB limit. `/health` and `/evaluation` both confirmed live (`200 OK`) — see [Live deployment](#live-deployment). Caveat: the same URL returned `503` on every route for ~7 minutes earlier the same day before recovering, so it isn't guaranteed warm on a reviewer's first click. |

## Setup & run

The happy path is in [Quickstart](#quickstart) above; **full step-by-step
instructions, the environment-variable table, the port map, data
versioning, and host-run components are in [`setup.md`](setup.md).** A few
things worth knowing before you start:

- **First `/search` after `up` takes a few minutes**, not seconds — cold
  model and index loads under memory pressure (measured at 101–281 s on
  the 8 GB dev host). It is not hung.
- **Windows PowerShell mangles an inline-JSON `-d` payload** (you get a
  `json_invalid` error, not a broken API) — write the body to a file:
  ```powershell
  '{"query":"cudaMalloc function parameters","top_k":3}' | Out-File -Encoding ascii body.json
  curl.exe -s -X POST http://localhost:8001/search -H "content-type: application/json" -d "@body.json"
  ```
- **`data/indexes/bm25_index.pkl` is committed and baked into the image**,
  so BM25 `/search` works with zero data setup. The default
  `RERANKER_MODE=live_fast` also queries the dense index; on a fresh clone
  Qdrant's `nvidia_ir_chunks` collection doesn't exist until the one-time
  ~86-min `populate_qdrant.py` pass ([`setup.md` §4](setup.md#4-populate-the-corpus)),
  so retrieval **degrades to BM25-only** and logs `dense_retrieval_degraded`.
  If retrieval fails outright (BM25 missing too), `/search` returns HTTP
  503 with an `error` field, never an empty `200`.
- **`/health` is liveness only; `/health/ready` asserts dependencies**
  (Postgres reachable, BM25 loadable) and returns 503 with a per-check
  body when one is down.

**Tests:**

```bash
pytest          # 561 tests, all mock embedding/LLM/DB/MLflow calls — no live services needed
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

## Data versioning

The source corpus is DVC-tracked, so a stranger with only the clone URL
rebuilds the exact 5,389-chunk corpus every evaluation figure is anchored
to (clean-clone test: `dvc pull` fetched all 8 blobs, sha256-verified
identical).

- **8 files** under `data/raw/*.dvc` — the 5 ingested CUDA PDFs plus 3
  tracked-but-not-yet-ingested (`ampere_architecture_whitepaper`,
  `hopper_architecture_whitepaper`, `a100_datasheet` — see
  [Problem statement](#problem-statement)).
- **No credentials.** The DVC remote (`.dvc/config`) is
  `https://raw.githubusercontent.com/.../dvc-storage/` — the blobs live on
  the **`dvc-storage` orphan branch** of this same public repo, so
  `dvc pull` is an anonymous HTTPS GET.
- **A shallow / single-branch clone still needs the remote reachable** —
  the blobs are on that orphan branch, not on `main`, and are not in the
  working tree until `dvc pull` runs.

```bash
pip install dvc            # or: pipx install dvc
dvc pull                   # populates data/raw/*.pdf from the dvc-storage branch
```

Full history of what was broken and how it was fixed:
[`docs/explainers/data_versioning_explained.md`](docs/explainers/data_versioning_explained.md),
[`docs/explainers/dvc_chronology.md`](docs/explainers/dvc_chronology.md).

## Project docs

- [`AGENTS.md`](AGENTS.md) — full 8-layer architecture, coding standards,
  MCP server list, folder structure.
- [`SKILLS.md`](SKILLS.md) — reusable code patterns.
- [`docs/explainers/data_versioning_explained.md`](docs/explainers/data_versioning_explained.md)
  — how the DVC corpus pinning works and why.
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
