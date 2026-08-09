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

# Reranker weights (ms-marco-MiniLM-L-6-v2) are downloaded from HF Hub at
# first request, not baked into the image -- only matters for RERANKER_MODE
# live_fast/live_quality/live_frontier, since fallback (render.yaml's
# current default) never loads a cross-encoder at all. The dense Qdrant
# collection is still *not* image content -- DenseIndex.connect() needs a
# populated cloud collection wired up via QDRANT_CLOUD_URL/
# QDRANT_CLOUD_API_KEY, which nothing has pointed the deployed service at
# yet. That used to sink /search and /ask entirely: agents/retrieval_agent.py
# and agents/qa_agent.py's retrieve nodes wrapped *both* the BM25 and dense
# calls in one try/except, so a dense-search failure (or, on this plan's
# 512MB RAM, DenseIndex.connect()/MSMarcoReranker.load() themselves OOMing
# before a request was even served) discarded the already-succeeded BM25
# results too. api/dependencies.py now skips loading DenseIndex/
# MSMarcoReranker entirely under RERANKER_MODE=fallback, and the retrieve
# nodes treat that None as a deliberate skip rather than a failure -- net
# effect: /search and /ask return real BM25-ranked results without Qdrant
# wired up at all. See render.yaml and README.md's "Known limitations" for
# what's still open (real hybrid dense+BM25 search needs Qdrant + a bigger
# plan).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
