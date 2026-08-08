# nvidia-ir-rag-agent — production image for the FastAPI service
# (api/main.py: /search, /ask, /health). Not used for Airflow, Streamlit,
# or the MCP servers -- each of those has its own runtime shape and stays
# on docker-compose.yml's images for local dev.
FROM python:3.11-slim AS base

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

# Reranker weights (ms-marco-MiniLM-L-6-v2, RERANKER_MODE=live_fast default)
# are downloaded from HF Hub at first request, not baked into the image --
# keeps the image small; the tradeoff is a cold-start latency spike on the
# first /search or /ask call per container. The dense Qdrant collection is
# still *not* image content -- DenseIndex.connect() needs a populated cloud
# collection wired up via QDRANT_CLOUD_URL/QDRANT_CLOUD_API_KEY, which
# nothing has pointed the deployed service at yet. Unlike the previous
# BM25-file-missing failure (which 500'd via an unhandled exception during
# FastAPI's Depends() resolution, before the route body ever ran),
# agents/retrieval_agent.py's retrieve node wraps *both* the BM25 and dense
# calls in one try/except -- a dense-search failure discards the already-
# succeeded BM25 results too and sets state.error, so return_results()
# comes back empty rather than BM25-only. Net effect once this image
# includes the BM25 index: /search and /ask should return `200` with
# `results: []` (not 500) until Qdrant is wired up for real -- see
# render.yaml and README.md's "Known limitations" for that open item.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
