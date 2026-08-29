# Setup & run — nvidia-ir-rag-agent

Everything below works from a clean clone at any tagged commit — no step
depends on state left over from a previous run. See the main
[`README.md`](README.md) for the problem statement, architecture, and
rubric-evidence mapping; this file is procedural only.

## 0. Prerequisites & requirements

- Docker + Docker Compose v2 (`docker compose version` — bundled with
  current Docker Desktop; the old standalone `docker-compose` binary works
  too, just substitute the hyphenated form below).
- **Python 3.11 exactly** + a virtualenv, **only** for the corpus-ingestion
  step below and for anything else that runs on the host rather than in a
  container (tests, the MCP servers, the Slack bot, one-off `run_*.py`
  scripts) — the FastAPI app and Streamlit UI themselves don't need this.
  Not 3.10, not 3.12+: `requirements.txt` resolves cleanly only on 3.11
  (`pandasai==3.0.0` has no wheel for 3.14, and the resolver error never
  names the version as the cause). `.python-version` pins it, and every
  host-run `run_*.py` / `retrieval/populate_qdrant.py` script now aborts
  immediately with `requires Python 3.11, found X.Y` rather than failing
  deep in an import.
- An `ANTHROPIC_API_KEY` if you want `/ask` or the Slack bot to actually
  generate answers — see the env var table below.

### Measured resource requirements

From a clean-clone reproducibility test on an 8 GB, CPU-only Windows host
(`docs/uat/clean_clone_test_findings.md` §3) — observed, not estimated:

| | Requirement | Evidence / notes |
|---|---|---|
| **Free RAM** | 4 GB minimum; 8 GB total is marginal | Full 9-service stack leaves ~0.5 GB free on 8 GB. Running `run_ingest_direct.py` / `populate_qdrant.py` with the whole stack up, or `docker restart`-ing one service with the stack up, OOM-kills the `api` container (exit 137). Stop `airflow`, `phoenix`, `jaeger`, `streamlit` before the embedding pass. |
| **Disk** | ~6 GB | ~2.5 GB `.venv` + ~3 GB Docker images. |
| **Time to first BM25 query** | ~50 min | 34 min `pip install` + ~10 min first image build + ~5 min `run_ingest_direct.py`. |
| **Time to full hybrid pipeline** | ~2 h 20 min | above + ~86 min `populate_qdrant.py` (5,389 chunks, e5-base-v2, CPU). |
| **First `/search` latency** | minutes, not seconds | 101–281 s cold, under memory pressure — model + index load on first request. Not hung. |
| **External accounts** | none for BM25 `/search` | `ANTHROPIC_API_KEY` → `/ask`; `COHERE_API_KEY` → Config C benchmark only; `dvc pull` → none. |

## 1. Clone

```bash
git clone https://github.com/dmishra27/nvidia-ir-rag-agent.git
cd nvidia-ir-rag-agent
```

### 1b. (Optional) Fetch the source PDFs with DVC

`/search`, `/ask`, and the tests do **not** need this — the BM25 index is
committed and the 5,389-chunk corpus is rebuilt by `run_ingest_direct.py`
in §4 from PDFs it also pins. You only need `dvc pull` if you want the
original source PDFs in `data/raw/` (e.g. to re-parse or extend the
manifest).

```bash
pip install dvc            # standalone, or pipx install dvc
dvc pull                   # 8 blobs, sha256-verified, no credentials
```

The DVC remote (`.dvc/config`) points at
`https://raw.githubusercontent.com/dmishra27/nvidia-ir-rag-agent/dvc-storage/`
— the blobs live on the **`dvc-storage` orphan branch** of this same
public repo, so `dvc pull` is an anonymous HTTPS GET with no cloud account.
Note that a shallow or single-branch clone still reaches that branch over
the network; the blobs are never in the `main` working tree until you pull.
8 files are tracked: the 5 ingested CUDA PDFs plus
`ampere_architecture_whitepaper` / `hopper_architecture_whitepaper` /
`a100_datasheet`, which are versioned but not in the ingestion manifest
(see [README](README.md#problem-statement)). Background:
[`docs/explainers/data_versioning_explained.md`](docs/explainers/data_versioning_explained.md).

## 2. Configure `.env`

```bash
cp .env.example .env
```

Every value in `.env.example` already has a working default for a local
`docker compose up` — you do **not** need to fill in anything to get the
stack running and `/search` returning real BM25 results. Fill in keys only
for the features that need them.

### Every environment variable

| Variable | Purpose | Required for | Works without it? |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Sonnet 5 API key | `/ask` generation + citations, Slack bot, RAGAS eval, citation judge | No — `/ask` returns HTTP 503 with an `error` field without it. `/search` and `/health` are unaffected. |
| `COHERE_API_KEY` | Cohere Rerank v3 API key | `RERANKER_MODE=live_frontier` / `benchmark` | Only needed for those two modes — default `RERANKER_MODE=live_fast` uses a local cross-encoder, no key. |
| `POSTGRES_URL` | Full connection string, host-run tooling | MCP servers, `run_*.py` scripts, notebooks, `pytest` | Yes, already defaulted to match `docker-compose.yml`'s published port. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Postgres container credentials | `docker-compose.yml`'s `postgres` service | Yes, already defaulted. |
| `QDRANT_URL` | Local Qdrant REST endpoint | Host-run tooling (embedding script, MCP server) | Yes, already defaulted. |
| `QDRANT_API_KEY` | Local Qdrant auth | Only if you've enabled auth on the local container | Yes — empty by default (no auth locally). |
| `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY` | Managed Qdrant Cloud endpoint | Render/prod deploy only | Yes — local dev uses `QDRANT_URL` against the `qdrant` container instead. |
| `ENABLE_PHOENIX` | Toggle Arize Phoenix instrumentation | Phoenix tracing | Yes — off by default. |
| `PHOENIX_ENDPOINT` | Phoenix OTLP/HTTP receiver | Phoenix tracing | Yes, already defaulted to the `phoenix` container's port. |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | Benchmark runner, RAGAS suite, citation judge, live Streamlit charts | Yes, already defaulted to the `mlflow` container's published port. |
| `ENABLE_TRACING` | Toggle OpenTelemetry export | Jaeger tracing | Yes — off by default. |
| `OTLP_ENDPOINT` | Jaeger OTLP gRPC receiver | Jaeger tracing | Yes, already defaulted. |
| `LANGCHAIN_API_KEY` | LangSmith API key | LangSmith tracing | Yes — optional, off without it. |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGCHAIN_ENDPOINT` | LangSmith config | LangSmith tracing | Yes, already defaulted. |
| `RERANKER_MODE` | Which re-ranker tier to serve | Every `/search`/`/ask` request | Yes — defaults to `live_fast` (local cross-encoder, no external key). |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | Slack Bolt Socket Mode credentials | Slack bot (`python -m slackbot.app`, not part of `docker compose up`) | Yes — only needed if you run the Slack bot. |
| `FASTAPI_BASE_URL` | Where the Slack bot calls `/ask` | Slack bot | Yes, defaults to `localhost:8000` — point it at `localhost:8001` if you want the bot talking to the Dockerized `api` service instead of a locally-run `uvicorn`. |
| `WANDB_API_KEY`, `WANDB_PROJECT` | Weights & Biases | Optional, unused by the shipped pipeline | Yes — optional. |
| `OLLAMA_BASE_URL` | Local Ollama endpoint | Optional, unused by the shipped pipeline | Yes — optional. |
| `NGROK_AUTHTOKEN` | ngrok tunnel | Optional, dev convenience only | Yes — optional. |

`.env`'s `POSTGRES_URL`/`QDRANT_URL`/`MLFLOW_TRACKING_URI` point at
`localhost` — that's correct for host-run tooling (MCP servers, `run_*.py`
scripts, notebooks, `pytest`). The `api`/`streamlit` containers don't read
these from `.env` at all; `docker-compose.yml` wires them to the
in-network service names (`postgres`, `qdrant`, `mlflow`) directly, since
`localhost` inside a container means the container itself, not the host.

## 3. Bring up the stack

```bash
docker compose up -d
```

This builds and starts **every containerized service** this project runs,
including the app itself:

| Service | What it is |
|---|---|
| `api` | FastAPI (`/search`, `/ask`, `/health`, `/`, `/evaluation`, `/docs`) — builds from the root `Dockerfile`'s `api` target |
| `streamlit` | The 5-tab UI — builds from the same `Dockerfile`'s `streamlit` target |
| `postgres` | Chunks/quality/coverage/benchmark/feedback tables |
| `qdrant` | Dense vector index (`nvidia_ir_chunks` collection) |
| `mlflow` | Experiment tracking (`reranker_benchmark`, `ragas_eval`, `citation_judge` runs) |
| `pgadmin` | Postgres admin UI |
| `airflow` | Standalone scheduler for the ingestion/monitoring DAGs |
| `jaeger` | OpenTelemetry trace UI (`ENABLE_TRACING=true`) |
| `phoenix` | Arize Phoenix trace UI (`ENABLE_PHOENIX=true`) |

Not containerized — see [§7](#7-things-that-still-run-on-the-host).

The `api`/`streamlit` images share one `pip install` layer (`Dockerfile`'s
`deps` stage) but that layer is still a ~3GB torch/transformers/langchain/
mlflow/streamlit install; expect the first `docker compose up -d` to take
**5–15 minutes** depending on bandwidth. Rebuilds after that are fast —
Docker reuses the cached layer as long as `requirements.txt` hasn't
changed.

`api` won't report healthy until `postgres`/`qdrant` do (it `depends_on`
them with `condition: service_healthy`); `streamlit` waits on `api` the
same way.

### After `up`: check, don't assume

`docker compose up -d` prints a **launch** report, not a **health**
assertion — during the clean-clone test it reported all nine services
`Healthy` while two were actually off the network, and `pgadmin` sat dead
(exit 255) for four hours unnoticed. So:

```bash
docker compose ps          # confirm every service is `running (healthy)`, not just `up`
```

If any service is missing, unhealthy, or restarting, **tear the whole
stack down before retrying** — a re-run of `up` after a partial start
reports success without repairing the network, and the API then can't
resolve `postgres`:

```bash
docker compose down        # not just `up` again
docker compose up -d
```

### Only one instance of this project at a time

The compose ports are fixed defaults: `5432`, `6333`/`6334`, `5001`,
`8501`, `5050`, `8080`, `16686`, `4317`/`4318`, `6006` (only `api` was
moved off `8000` to `8001`). If another copy of this repo — or anything
else on those ports — is already running, `docker compose up -d` fails
**loudly** on `qdrant`/`6333` and **silently** on `postgres`/`5432` (the
container just never starts and nothing names it). Stop the other stack
first; a second instance cannot run alongside the first.

## 4. Populate the corpus

The `api`/`streamlit` containers ship with the BM25 index already built
(`data/indexes/bm25_index.pkl` is committed to git and baked into the
image), so `/search`/`/ask` work immediately with BM25-only results.
**Dense/hybrid retrieval needs Qdrant's `nvidia_ir_chunks` collection
populated**, which is a host-run, one-time step (not part of
`docker compose up` — it's a data load, not a service):

```bash
python --version                                  # must be 3.11.x — see §0
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/patch_ragas.py                     # see requirements_notes.txt

python run_ingest_direct.py           # PDFs -> parse -> chunk -> score -> Postgres (5,389 chunks)
python retrieval/populate_qdrant.py   # embeds those chunks (e5-base-v2) -> upserts into Qdrant
```

Both scripts (and every other `run_*.py`) call a Python-version guard
first and exit with `requires Python 3.11, found X.Y` if the shell has
fallen out of the venv — a real failure mode: during the clean-clone test
the shell silently reverted to system Python three times, each producing a
convincing but false "defect". If you see that message, re-activate
`.venv` and re-run.

Both read `.env`'s `localhost`-pointed `POSTGRES_URL`/`QDRANT_URL`, so they
talk to the same `docker compose`-published ports from the host. The
embedding step is the slow part — it took ~86 minutes over 5,389 chunks on
a CPU-only machine; that's expected, not hung. Re-running
`populate_qdrant.py` is safe (it caches embeddings to
`data/indexes/qdrant_corpus_embeddings.npy` and upserts by deterministic
point ID, so a second run reuses the cache and overwrites rather than
duplicating points); neither script needs to be redone unless you drop the
`qdrant_data`/`postgres_data` volumes.

An alternative, not required to get `/search`/`/ask` working: the same
pipeline also exists as an Airflow 3 TaskFlow DAG
(`airflow/dags/ingest_nvidia_docs.py`), runnable from the `airflow`
container's webserver at `http://localhost:8080` — see [Known
limitations](README.md#known-limitations) for its current status.

## 5. Verify

```bash
curl http://localhost:8001/health          # {"status": "ok", "service": "nvidia-ir-rag-agent"} — liveness only
curl http://localhost:8001/health/ready     # asserts Postgres + BM25; 200 {"status":"ready",...} or 503 with per-check detail
curl -X POST http://localhost:8001/search -H 'content-type: application/json' -d '{"query": "cudaMemcpyAsync"}'
```

**The first `/search` after `up` takes minutes, not seconds** (101–281 s
measured on the 8 GB dev host — cold model/index loads under memory
pressure). It is not hung; subsequent calls are fast.

`/health` is a pure liveness probe — it stays `200 ok` even if Postgres or
retrieval is down, on purpose, so it can't wedge container startup
ordering. Use **`/health/ready`** for the dependency assertion: it returns
`503` and names the failing dependency (`{"checks":{"postgres":"error:
...","bm25":"ok"}}`) when the stack isn't actually able to serve.

Then open `http://localhost:8501` for the Streamlit UI, `http://localhost:8001`
for the guided search UI, `http://localhost:8001/evaluation` for the
retrieval-evaluation report, or `http://localhost:8001/docs` for Swagger.

## 6. Port map

| Port | Service | What's there |
|---|---|---|
| `8001` | `api` | FastAPI — `/`, `/evaluation`, `/search`, `/ask`, `/health` (+ `/health/ready`), `/docs` (container listens on 8000 internally; 8001 on the host since 8000 is frequently already taken) |
| `8501` | `streamlit` | 5-tab UI |
| `5432` | `postgres` | `nvidia_ir_db` |
| `5050` | `pgadmin` | `admin@nvidia-ir.local` / `admin` |
| `6333` / `6334` | `qdrant` | REST / gRPC |
| `5001` | `mlflow` | Tracking UI + REST |
| `8080` | `airflow` | Standalone webserver |
| `16686` / `4317` / `4318` | `jaeger` | UI / OTLP gRPC / OTLP HTTP |
| `6006` | `phoenix` | UI + OTLP/HTTP trace receiver (`/v1/traces`) |

## 7. Things that still run on the host

Not containerized — each needs the `.venv` from §4 and reads `.env`'s
`localhost`-pointed URLs:

```bash
# Slack bot (Socket Mode -- needs SLACK_BOT_TOKEN + SLACK_APP_TOKEN). Its
# FASTAPI_BASE_URL default (localhost:8000) targets a locally-run uvicorn,
# not the containerized `api` service -- point it at localhost:8001 instead
# if you want the bot talking to the Dockerized API.
python -m slackbot.app

# MCP servers (mcp/mcp_postgres, mcp/mcp_qdrant, mcp/mcp_mlflow,
# mcp/mcp_airflow) -- launched by an MCP-compatible client per .mcp.json,
# not run directly.
```

## 8. Tests / lint / typecheck

```bash
pytest                 # 561 tests, all mocked (no live API/DB/MLflow calls — see AGENTS.md)
ruff check .
mypy .                  # strict on agents/, api/, retrieval/, monitoring/, evaluation/, schema/, mcp/, slackbot/, streamlit_app/
```

No backing services need to be running for `pytest` — including
`tests/streamlit_app/test_tabs.py`'s benchmark/eval-dashboard tests, which
mock `streamlit_app/live_data.py`'s MLflow/Postgres client constructors
rather than depending on a live MLflow, so the suite stays fast and
deterministic whether or not `docker compose up` has been run.

`.github/workflows/ci.yml` runs: checkout → install (CPU-only torch wheel
index) → patch ragas (see `requirements_notes.txt`) → ruff → mypy (strict
on production code) → pytest → NDCG regression gate.
