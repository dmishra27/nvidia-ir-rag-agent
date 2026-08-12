# nvidia-ir-rag-agent

A hybrid-retrieval RAG agent over NVIDIA's public technical documentation —
CUDA Runtime API, best-practices guides, architecture whitepapers — with a
full evaluation/monitoring/HITL stack behind it, not just a search box.

Portfolio project by Debabrata Mishra ([dmishra27](https://github.com/dmishra27)),
built as a 14-day build log (see [Project docs](#project-docs) below),
tagged `v1.0.0` (see [`CHANGELOG.md`](CHANGELOG.md)).
SIGIR 2024 co-author — this project directly extends prior neural
passage-quality-estimation research into a production-shaped RAG system.

## Live demo

| Endpoint | Status |
|---|---|
| [Live API health check](https://nvidia-ir-rag-agent.onrender.com/health) | ✅ `200 OK` — `{"status": "ok", "service": "nvidia-ir-rag-agent"}` |
| [Swagger UI](https://nvidia-ir-rag-agent.onrender.com/docs) | ✅ accessible — `POST /search`, `POST /ask`, `GET /health` all listed |
| `POST /search` / `POST /ask` | ❌ still `500` as of the last check — see below |

The service itself is genuinely live (`POSTGRES_URL`/`RERANKER_MODE`
configured in the Render dashboard). `/search` and `/ask` previously 500'd
for exactly the reason this README and `render.yaml` documented before
deploying: `retrieval/bm25_index.py`'s `BM25Index.load()` reads
`data/indexes/bm25_index.pkl`, and the whole `data/` tree is gitignored, so
the container had no index file to load — an unhandled exception during
FastAPI's dependency resolution, before either route's body ever ran.

**Partially fixed, not yet resolved live**: `data/indexes/bm25_index.pkl`
is now committed directly to git (`.dockerignore` negates it back through
the otherwise-blanket `data/` exclusion: `!data/indexes` /
`!data/indexes/**`), and `Dockerfile` copies it into the image — verified
locally with an isolated test build that the file actually lands there.
But repeated live checks against the deployed URL after that fix was
pushed still return `500`, not the expected `200`/empty-results (`/health`
did briefly stop responding mid-check, consistent with a redeploy
happening, then came back — but `/search` was still `500` afterward).
Possible causes not distinguishable from outside the Render dashboard:
the new image may not have finished building/deploying yet, Render's
`autoDeploy` may not have triggered, or something else in the request
path is failing before the fix even gets exercised. **Unresolved as of
this writing** — checking Render's own build/deploy logs directly is the
next step, not something observable from a client-side `curl`.

**Still open**: dense retrieval isn't wired up — `DenseIndex.connect()`
needs a populated Qdrant collection via `QDRANT_CLOUD_URL`/`QDRANT_CLOUD_API_KEY`,
which nothing has pointed the deployed service at yet. And because
`agents/retrieval_agent.py`'s retrieve node wraps *both* the BM25 and dense
calls in one `try`/`except`, a dense-search failure discards the
already-succeeded BM25 results too (`return_results()` comes back
empty on any retrieval error, not BM25-only) — so once redeployed, expect
`/search`/`/ask` to return `200` with `results: []` rather than 500, not
real BM25-only hybrid results. Splitting that error handling so a dense
failure degrades to BM25-only instead of empty is a natural next step, not
done here.

**For working `/search` and `/ask` with real results today**, run it
locally — see [How to run](#how-to-run) below (`docker compose up -d`
brings up the `api` container itself now, loading the same
`BM25Index`/`DenseIndex` from your local `data/`/Qdrant instead of
Render's still-Qdrant-less one).

## Problem statement

NVIDIA's own technical documentation is large, fragmented across dozens of
separate PDFs/HTML guides (CUDA C++ Programming Guide, CUDA C++ Best
Practices Guide, CUDA Runtime API reference, architecture whitepapers for
each GPU generation), inconsistently structured, and duplicated across
versions. A developer asking "how does `cudaMemcpyAsync`'s stream parameter
work?" has to know *which* document covers it, search each one separately,
and cross-reference API signatures against prose explanations that live in
a different document from the reference itself.

This project builds a single hybrid-retrieval + re-ranking + LLM-grounded
answer pipeline over that fragmented corpus, with three things most
"RAG demo" projects skip:

1. **Real retrieval quality measurement** — BM25 vs SPLADE vs dense vs RRF
   fusion, three re-ranker configs A/B/C benchmarked head-to-head on
   NDCG@10/MRR/latency/cost, not just "it returns something."
2. **Per-claim citation grounding**, judged by a separate LLM-as-judge
   pass — not just "the answer mentions the source," but "does chunk X
   actually support claim Y."
3. **A closed monitoring + human-feedback loop** — drift detection, quality
   regression alerts, and a Slack bot where real users' 👍/👎 reactions feed
   back into `feedback_log` for later analysis, not a one-shot eval run.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 8  Drift + HITL          PSI drift · term shift · quality      │
│                                 regression · Slack 👍/👎 → feedback_log │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 7  Observability         LangSmith · OpenTelemetry+Jaeger ·    │
│                                 structlog · Arize Phoenix             │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6  Monitoring            per-stage latency · error rate ·      │
│                                 throughput (api/middleware.py)        │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4-5 Evaluation           RAGAS · DeepEval · citation judge ·   │
│                                 NDCG/MRR · benchmark_runner (A/B/C)   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3b Re-ranking            ms-marco │ bge-reranker-v2-m3 │       │
│                                 Cohere Rerank v3  (RERANKER_MODE)     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3  Retrieval             BM25 + SPLADE + dense(Qdrant) + RRF   │
│                                 fusion  (ColBERT: not started)        │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2  Ingestion              Airflow 3 DAG: PyMuPDF parse →       │
│                                 LangChain chunk → quality score →     │
│                                 SQLAlchemy → Postgres                 │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 1  MCP tool-calling      4 servers: postgres · qdrant ·        │
│                                 mlflow · airflow                      │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 0  Pre-retrieval quality  chunk scorer · freshness · dedup ·   │
│                                 coverage                               │
└─────────────────────────────────────────────────────────────────────┘

  User (Slack /nvidia-search or Streamlit)
        │
        ▼
  FastAPI  /ask  ──►  agents/qa_agent.py (LangGraph: retrieve → rerank → generate)
        │                     │
        │            ┌────────┴────────┐
        │            ▼                 ▼
        │      BM25 + Dense        RerankerRouter
        │      (RRF fusion)     (ms-marco / bge / Cohere)
        │            │                 │
        │            └────────┬────────┘
        │                     ▼
        │           Claude Sonnet 5, forced tool call
        │           (answer + per-claim chunk_id citations)
        ▼
  AskResponse{answer, citations[], query_id}
        │
        ├──► Slack: handlers.py formats blocks, posts top-3 citations
        │           feedback_handler.py records 👍/👎 → feedback_log
        │
        └──► Streamlit: 5 tabs (search / benchmark / eval / monitoring / drift)
```

Full 8-layer detail, MCP server list, coding standards, and folder
structure live in [`AGENTS.md`](AGENTS.md); reusable code patterns live in
[`SKILLS.md`](SKILLS.md).

## How to run

Everything below works from a clean clone at any tagged commit — no step
depends on state left over from a previous run.

### 0. Prerequisites

- Docker + Docker Compose v2 (`docker compose version` — bundled with
  current Docker Desktop; the old standalone `docker-compose` binary works
  too, just substitute the hyphenated form below).
- Python 3.11 + a virtualenv, **only** for the corpus-ingestion step below
  and for anything else that runs on the host rather than in a container
  (tests, the MCP servers, the Slack bot, one-off `run_*.py` scripts) —
  the FastAPI app and Streamlit UI themselves no longer need this.
- An `ANTHROPIC_API_KEY` if you want `/ask` or the Slack bot to actually
  generate answers — see the feature/key matrix in step 2.

### 1. Clone

```bash
git clone https://github.com/dmishra27/nvidia-ir-rag-agent.git
cd nvidia-ir-rag-agent
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Every value in `.env.example` already has a working default for a local
`docker compose up` — you do **not** need to fill in anything to get the
stack running and `/search` returning real BM25 results. Fill in keys only
for the features that need them:

| Variable | Required for | Works without it? |
|---|---|---|
| `ANTHROPIC_API_KEY` | `/ask` (Claude Sonnet 5 generation + citations), Slack bot, RAGAS eval | No — `/ask` returns a 500 without it. `/search` and `/health` are unaffected. |
| `COHERE_API_KEY` | `RERANKER_MODE=live_frontier` / `benchmark` | Only needed for those two modes — default `RERANKER_MODE=live_fast` uses a local cross-encoder, no key. |
| `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY` | Render/prod deploy only | Yes — local dev uses `QDRANT_URL` against the `qdrant` container instead. |
| `LANGCHAIN_API_KEY` | LangSmith tracing | Yes — optional, off without it. |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | Slack bot (`python -m slackbot.app`, not part of `docker compose up`) | Yes — only needed if you run the Slack bot. |
| everything else (`POSTGRES_*`, `QDRANT_URL`, `MLFLOW_TRACKING_URI`, `RERANKER_MODE`, `ENABLE_TRACING`, `ENABLE_PHOENIX`, `OTLP_ENDPOINT`, `PHOENIX_ENDPOINT`) | wiring between services | Yes — already defaulted to match `docker-compose.yml`. |

`.env`'s `POSTGRES_URL`/`QDRANT_URL`/`MLFLOW_TRACKING_URI` point at
`localhost` — that's correct for host-run tooling (MCP servers, `run_*.py`
scripts, notebooks, `pytest`). The `api`/`streamlit` containers don't read
these from `.env` at all; `docker-compose.yml` wires them to the
in-network service names (`postgres`, `qdrant`, `mlflow`) directly, since
`localhost` inside a container means the container itself, not the host.

### 3. Bring up the stack

```bash
docker compose up -d
```

This builds and starts **every** service this project runs, including the
app itself:

| Service | What it is |
|---|---|
| `api` | FastAPI (`/search`, `/ask`, `/health`) — builds from the root `Dockerfile`'s `api` target |
| `streamlit` | The 5-tab UI — builds from the same `Dockerfile`'s `streamlit` target |
| `postgres` | Chunks/quality/coverage/benchmark tables |
| `qdrant` | Dense vector index (`nvidia_ir_chunks` collection) |
| `mlflow` | Experiment tracking (`reranker_benchmark`, `ragas_eval`, `citation_judge` runs) |
| `pgadmin` | Postgres admin UI |
| `airflow` | Standalone scheduler for the ingestion/monitoring DAGs |
| `jaeger` | OpenTelemetry trace UI (`ENABLE_TRACING=true`) |
| `phoenix` | Arize Phoenix trace UI (`ENABLE_PHOENIX=true`) |

The `api`/`streamlit` images share one `pip install` layer (`Dockerfile`'s
`deps` stage — see its comment for why) but that layer is still a ~3GB
torch/transformers/langchain/mlflow/streamlit install; expect the first
`docker compose up -d` to take **5–15 minutes** depending on bandwidth.
Rebuilds after that are fast — Docker reuses the cached layer as long as
`requirements.txt` hasn't changed.

`api` won't report healthy until `postgres`/`qdrant` do (it `depends_on`
them with `condition: service_healthy`); `streamlit` waits on `api` the
same way. `docker compose ps` shows you where things are.

### 4. Populate the corpus

The `api`/`streamlit` containers ship with the BM25 index already built
(`data/indexes/bm25_index.pkl` is committed to git and baked into the
image — see the `Dockerfile`'s comment), so `/search`/`/ask` work
immediately with BM25-only results. **Dense/hybrid retrieval needs
Qdrant's `nvidia_ir_chunks` collection populated**, which is a host-run,
one-time step (not part of `docker compose up` — it's a data load, not a
service):

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/patch_ragas.py                     # see requirements_notes.txt

python run_ingest_direct.py       # PDFs -> parse -> chunk -> score -> Postgres (5,389 chunks)
python retrieval/populate_qdrant.py   # embeds those chunks (e5-base-v2) -> upserts into Qdrant
```

Both read `.env`'s `localhost`-pointed `POSTGRES_URL`/`QDRANT_URL`, so they
talk to the same `docker compose`-published ports from the host. The
embedding step is the slow part — `populate_qdrant.log` shows it took
~86 minutes over 5,389 chunks on a CPU-only machine; that's expected, not
hung. Re-running `populate_qdrant.py` is safe (it caches embeddings to
`data/indexes/qdrant_corpus_embeddings.npy` and upserts by deterministic
point ID, so a second run reuses the cache and overwrites rather than
duplicating points); neither script needs to be redone unless you drop the
`qdrant_data`/`postgres_data` volumes.

### 5. Verify

```bash
curl http://localhost:8001/health          # {"status": "ok", "service": "nvidia-ir-rag-agent"}
curl -X POST http://localhost:8001/search -H 'content-type: application/json' -d '{"query": "cudaMemcpyAsync"}'
```

Then open `http://localhost:8501` for the Streamlit UI, or
`http://localhost:8001/docs` for Swagger.

### 6. Port map

| Port | Service | What's there |
|---|---|---|
| `8001` | `api` | FastAPI — `/search`, `/ask`, `/health`, `/docs` (container listens on 8000 internally; 8001 on the host since 8000 is frequently already taken) |
| `8501` | `streamlit` | 5-tab UI |
| `5432` | `postgres` | `nvidia_ir_db` |
| `5050` | `pgadmin` | `admin@nvidia-ir.local` / `admin` |
| `6333` / `6334` | `qdrant` | REST / gRPC |
| `5001` | `mlflow` | Tracking UI + REST |
| `8080` | `airflow` | Standalone webserver |
| `16686` / `4317` / `4318` | `jaeger` | UI / OTLP gRPC / OTLP HTTP |
| `6006` | `phoenix` | UI + OTLP/HTTP trace receiver (`/v1/traces`) |

### 7. Things that still run on the host

Not containerized — each needs the `.venv` from step 4 and reads `.env`'s
`localhost`-pointed URLs:

```bash
# Slack bot (Socket Mode — needs SLACK_BOT_TOKEN + SLACK_APP_TOKEN). Its
# FASTAPI_BASE_URL default (localhost:8000) targets a locally-run uvicorn,
# not the containerized `api` service -- point it at localhost:8001 instead
# if you want the bot talking to the Dockerized API.
python -m slackbot.app

# MCP servers (mcp/mcp_postgres, mcp/mcp_qdrant, mcp/mcp_mlflow,
# mcp/mcp_airflow) -- launched by Claude Code per .mcp.json, not run
# directly.
```

### 8. Tests

```bash
pytest                 # 501 tests, all mocked (no live API/DB/MLflow calls — see AGENTS.md)
ruff check .
mypy .                  # strict on agents/, api/, retrieval/, monitoring/, evaluation/, schema/, mcp/, slackbot/, streamlit_app/
```

No backing services need to be running for `pytest` — including
`tests/streamlit_app/test_tabs.py`'s benchmark/eval-dashboard tests, which
mock `streamlit_app/live_data.py`'s MLflow/Postgres client constructors
(DEF-04) rather than depending on a live MLflow, so the suite stays fast
and deterministic whether or not `docker compose up` has been run.

## Example query

A real, live-run example from Day 9's benchmark set (not fabricated) —
`agents/qa_agent.py`'s actual output, forced-tool-call citations included:

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

Note the model correctly declines to invent the stream parameter's type/
default when the retrieved passages don't cover it — this is what
`evaluation/citation_judge.py` and RAGAS faithfulness are there to catch
when it *doesn't* happen.

## Benchmark results (Day 9, live-measured, 15 queries × top-3 RRF pool)

| Config | Re-ranker | NDCG@10 | MRR | Prec@3 | Latency (ms/query) | Cost/query |
|---|---|---|---|---|---|---|
| A | ms-marco-MiniLM-L-6-v2 (local CPU cross-encoder) | 0.5333 | 0.5333 | 0.2444 | 48.9 | $0.00 |
| C | Cohere Rerank v3 (hosted API) | 0.5280 | 0.5333 | 0.2444 | 291.2 | $0.002 |
| B | bge-reranker-v2-m3 | — | — | — | — | — |

Config B is hardware-blocked (OOMs at model load on this project's 8GB-RAM/
CPU-only dev machine) and deferred to a GPU environment — shown here as
"not run," not silently omitted; see `evaluation/benchmark_runner.py`'s
`CONFIG_B_DEFERRED_REASON`.

CI's NDCG regression gate (`evaluation/ci_ndcg_gate.py`) fails any push that
drops Config A below **0.50** against this committed baseline.

**Evaluation (Day 11 live RAGAS + Day 9 citation judge):**

| Metric | Score | Source |
|---|---|---|
| Faithfulness | 0.7616 | RAGAS, 10 queries, MLflow run `7d8f1005` |
| Answer relevancy | 0.2497 | RAGAS, 10 queries, MLflow run `7d8f1005` |
| Citation accuracy | 0.7037 | LLM-judge, 27 claims across 10 queries |

`streamlit_app/benchmark_tab.py` and `eval_dashboard.py` show these same
numbers **live** — queried from MLflow/Postgres on every render, falling
back to this table's committed values only if MLflow/Postgres aren't
reachable (see `streamlit_app/live_data.py`).

## Streamlit UI

Five tabs (`streamlit_app/app.py`), Plotly charts throughout, all served
from one `streamlit run streamlit_app/app.py`:

1. **Search** — enter a query, see retrieved passages + the grounded
   answer with citations (fixed demo query/answer, no live model call by
   design — this tab is a UI shell, not a benchmark surface).
2. **Benchmark** — Config A vs C, live from MLflow: ranking-quality bar
   chart, latency bar chart, **and** a per-query NDCG@10 box-plot pulled
   live from Postgres' `benchmark_results` table (45 real rows) — not just
   the two aggregate charts.
3. **Eval Dashboard** — faithfulness/answer_relevancy/citation_accuracy/
   NDCG headline metrics (live MLflow), NDCG-vs-CI-gate chart, **a
   per-query citation-accuracy chart** built from the real 27-judgment Day
   9 file, and a quality-regression trend (this last chart's *logic* is
   real — `monitoring/quality_regression.py`'s actual `evaluate_regression()`
   — run over a clearly-labeled synthetic 10-day history, since no live
   daily-quality table exists yet).
4. **Monitoring** — per-stage latency (BM25/dense/RRF/rerank/LLM) and
   request/error-rate time series.
5. **Drift** — PSI drift snapshot + term-frequency shift (Blackwell-era
   terms rising, Hopper-era terms receding).

Every live-data chart degrades gracefully with a visible banner (not a
crash) if MLflow/Postgres aren't reachable — including in CI, which never
runs live services, per `AGENTS.md`'s "no live API calls in CI" rule.

## Slack bot + human-in-the-loop feedback

`slackbot/app.py` (Socket Mode — no public URL needed):

- **`/nvidia-search <question>`** → calls `POST /ask` → posts the answer +
  top-3 citations as Slack blocks, with a `query_id` footer.
- **👍 / 👎 reactions** on the bot's own answer → `slackbot/feedback_handler.py`
  looks the `query_id` back up from the message text and writes one row to
  `feedback_log` (SQLAlchemy ORM — no raw SQL, per `AGENTS.md`).
- **`airflow/dags/feedback_aggregator.py`** — weekly TaskFlow DAG,
  aggregates the week's 👍/👎 into a total/up/down/down-rate summary and
  flags queries with the most negative feedback (`monitoring/feedback_aggregator.py`
  is the unit-tested pure core; the DAG just wires a Postgres read + JSON
  history to it, mirroring `airflow/dags/drift_monitor.py`'s shape).

## Testing / CI

- **501 tests**, all mocking embedding/LLM/DB/MLflow calls per `AGENTS.md`'s
  "Mock all embedding and LLM calls in unit tests" rule — `pytest` runs in
  under 2 minutes with no live services (DEF-04: `tests/streamlit_app/test_tabs.py`'s
  benchmark/eval-dashboard tests used to fall through to a live MLflow
  client with a 15s timeout per call when no MLflow was running, pushing
  the suite past 7 minutes non-deterministically — now mocked).
- `.github/workflows/ci.yml`: checkout → install (CPU-only torch wheel
  index) → patch ragas (see `requirements_notes.txt`) → ruff → mypy
  (strict on production code) → pytest → NDCG regression gate. All green
  on `main`.
- `mypy --strict` covers `agents/`, `api/`, `retrieval/`, `monitoring/`,
  `evaluation/`, `schema/`, `mcp/`, `slackbot/`, `streamlit_app/` — `tests/`
  and the root-level `run_*.py`/`verify_phase_c*.py` one-off scripts are
  scoped out (not part of the shipped package), matching the existing
  `run_*.py` ruff per-file-ignore's reasoning.

## Deploy (Render.com)

`render.yaml` deploys the FastAPI service (`/search`, `/ask`, `/health`) as
a Render Blueprint — connect this repo in the Render dashboard, paste the
secrets Render prompts for (`ANTHROPIC_API_KEY`, `COHERE_API_KEY`,
`POSTGRES_URL`, etc. — see `render.yaml`'s `sync: false` list), and it
builds the root `Dockerfile`.

**Known limitation** (documented in `render.yaml` itself; fix pushed, not
yet confirmed live): this gets the *service* deployed and `/health`
responding, not the full retrieval stack live. `BM25Index.load()`'s
missing index file — confirmed as the live 500's actual cause — has a fix
pushed: `data/indexes/bm25_index.pkl` is now committed to git and copied
into the image (see [Live demo](#live-demo) for the `.dockerignore`
mechanism, verified locally with an isolated test build). Repeated live
checks against the deployed URL after that push still return `500`,
though — see [Live demo](#live-demo) for the current unresolved status.
Separately, `DenseIndex.connect()` still needs a populated Qdrant cloud
collection via `QDRANT_CLOUD_URL`/`QDRANT_CLOUD_API_KEY`, which nothing
has pointed the deployed service at yet — once the BM25 500 is actually
resolved, expect `/search`/`/ask` to return `200` with empty results
rather than real hybrid search results (retrieval_agent.py's single
try/except around both BM25 and dense search means a dense failure
discards BM25's results too, not just skips dense).

**Live URL:** see [Live demo](#live-demo) at the top of this file.

## Project docs

Day-by-day build log, one storyline doc per working day, in
[`docs/daily_progress/`](docs/daily_progress/) — each is self-contained but
recaps everything before it, so any single day can be read standalone.
Start at [`docs/daily_progress/README.md`](docs/daily_progress/README.md)
for the full day-by-day index, or jump straight to
[`day_13_storyline.md`](docs/daily_progress/day_13_storyline.md) for the
most recent work (this Slackbot/HITL/live-Streamlit/README/Render session).

## Known limitations / open items

- Config B (bge-reranker-v2-m3) needs a GPU environment — deferred.
- ColBERT retrieval — not started.
- Full 50-query × top-100-candidate benchmark — not run (15-query smoke
  scope only; see [`reports/final_eval_report.md`](reports/final_eval_report.md)).
- Quality-regression and drift DAGs are unit-tested but not yet run
  against a live Airflow scheduler (this dev machine's local venv
  deliberately doesn't install `apache-airflow` — DAGs run inside
  `docker-compose.yml`'s `airflow` service image instead).
- Slack alert wiring for `drift_monitor.py`/`feedback_aggregator.py`'s
  `alert_fn` — both already `structlog.warning()` on their own; posting
  that back into Slack via `slack_sdk.WebClient.chat_postMessage` is a
  natural next step now that `slackbot/` exists.
- Render deploy: `/search`/`/ask` are still `500` live despite the BM25
  index fix being pushed (see [Deploy](#deploy-rendercom) above) — cause
  not yet confirmed, needs the Render dashboard's own logs to diagnose.
  Separately, once that's resolved, dense/Qdrant retrieval is still not
  wired up, and `agents/retrieval_agent.py`'s combined BM25+dense error
  handling means a dense failure will discard BM25's results too rather
  than degrading to BM25-only.
