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

# Reranker weights (ms-marco-MiniLM-L-6-v2, RERANKER_MODE=live_fast default)
# are downloaded from HF Hub at first request, not baked into the image --
# keeps the image small; the tradeoff is a cold-start latency spike on the
# first /search or /ask call per container. BM25's index.pkl and the dense
# Qdrant collection are *not* image contents: both are gitignored build/data
# artifacts (see .gitignore's trailing "data/") that must be populated
# against the target environment's Postgres/Qdrant before this container
# can serve real results -- see render.yaml and day_12_storyline.md's
# "known limitations" for the open item this leaves.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
