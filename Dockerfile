# nvidia-ir-rag-agent — multi-stage image for the two Python services this
# project actually ships: the FastAPI app (api/main.py: /search, /ask,
# /health) and the Streamlit UI (streamlit_app/app.py's 5 tabs). Not used
# for Airflow, MLflow, or the MCP servers -- each of those has its own
# runtime shape and stays on docker-compose.yml's images for local dev.
#
# `deps` installs the ~3GB torch/transformers/langchain/mlflow/streamlit
# stack exactly once; `api` and `streamlit` both build FROM it. docker-
# compose.yml points both services at this one Dockerfile with a different
# `target:`, so the second service's build is a cache hit on the expensive
# pip-install layer instead of paying a second ~5-15min cold install --
# streamlit_app/benchmark_tab.py and eval_dashboard.py import
# evaluation/benchmark_runner.py (torch-backed rerankers), so there's no
# meaningfully lighter dependency set available for the UI image anyway.
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: PyMuPDF (fitz) and tokenizer libs need a C toolchain to build
# wheels that don't ship prebuilt manylinux binaries for every combination;
# libgomp1 is torch's OpenMP runtime at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# See requirements_notes.txt: ragas==0.4.3 unconditionally imports a
# langchain-community submodule this project's pinned langchain-community
# doesn't ship; patch it out of the installed package rather than pull in
# langchain-google-vertexai (which conflicts with pandasai's pyarrow pin).
COPY scripts/patch_ragas.py scripts/patch_ragas.py
RUN python scripts/patch_ragas.py

COPY agents/ agents/
COPY api/ api/
COPY evaluation/ evaluation/
COPY monitoring/ monitoring/
COPY retrieval/ retrieval/
COPY schema/ schema/
COPY AGENTS.md SKILLS.md ./

# BM25Index.load()'s default path (retrieval/bm25_index.py's
# DEFAULT_INDEX_PATH). data/ is gitignored wholesale, but this one built
# artifact was deliberately force-added to git (see .dockerignore's
# !data/indexes negation, mirroring this) so the deployed image has
# something to load -- BM25Index.load() no longer needs a populated
# external store the way DenseIndex.connect() still does.
COPY data/indexes/ data/indexes/

# Reranker weights (ms-marco-MiniLM-L-6-v2) are downloaded from HF Hub at
# first request, not baked into the image -- only matters for RERANKER_MODE
# live_fast/live_quality/live_frontier, since fallback never loads a
# cross-encoder at all. The dense Qdrant collection is also *not* image
# content -- DenseIndex.connect() needs QDRANT_URL (docker-compose.yml's
# `qdrant` service locally) or QDRANT_CLOUD_URL/QDRANT_CLOUD_API_KEY
# (managed Qdrant in prod) pointed at a populated collection -- run
# run_ingest_direct.py against it after `docker compose up -d` (see
# README.md's quickstart).

# `api` is deliberately the *last* stage in this file: render.yaml builds
# this Dockerfile with no `--target` (Render's Blueprint spec has no field
# for one), and a target-less `docker build` on a multi-stage file defaults
# to the last stage -- so this ordering is what keeps the existing Render
# deploy building the API image rather than the UI. docker-compose.yml
# always builds both stages by explicit `target:`, so it's unaffected by
# this ordering either way.
# ---------------------------------------------------------------------------
FROM deps AS streamlit

COPY streamlit_app/ streamlit_app/

# Streamlit puts the *script's* directory on sys.path, not the CWD, so
# /app must be explicit for `from streamlit_app import ...` to resolve.
# docker-compose.yml also sets this; baking it in keeps the image
# self-sufficient when run outside compose.
ENV PYTHONPATH=/app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# `api` is deliberately the *last* stage in this file: render.yaml builds
# this Dockerfile with no `--target` (Render's Blueprint spec has no field
# for one), and a target-less `docker build` on a multi-stage file defaults
# to the last stage -- so this ordering is what keeps the existing Render
# deploy building the API image rather than the UI. docker-compose.yml
# always builds both stages by explicit `target:`, so it's unaffected by
# this ordering either way.
# DO NOT append new stages below this one.
# ---------------------------------------------------------------------------
FROM deps AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

