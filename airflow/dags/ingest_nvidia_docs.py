"""Airflow 3 ingestion DAG — nvidia-ir-rag-agent Layer 2.

Pipeline per run:
  fetch_manifest → download_pdfs → parse_pdfs → chunk_docs
      → score_chunks → write_to_postgres → log_coverage

Staging files are written to INGEST_STAGING_DIR/{run_id}/ so that large
text content never travels through XCom.  The directory is local; for a
distributed Airflow deployment, point INGEST_STAGING_DIR at a shared NFS
or S3-backed path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pendulum

# ── project root on sys.path so schema/retrieval imports resolve ──────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import fitz  # PyMuPDF
import requests
import structlog
import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from schema.models import (
    Chunk,
    ChunkQuality,
    CoverageLog,
    DocMetadata,
    get_engine,
    get_session_factory,
)
from retrieval.chunk_quality import score_chunk

load_dotenv(_PROJECT_ROOT / ".env")

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 1500       # characters ≈ 375 tokens
CHUNK_OVERLAP: int = 150     # characters
DOWNLOAD_TIMEOUT: int = 120  # seconds per PDF

STAGING_BASE: Path = Path(
    os.getenv("INGEST_STAGING_DIR", tempfile.gettempdir())
) / "nvidia_ir_ingest"

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# PDF manifest — NVIDIA public technical documentation
# ---------------------------------------------------------------------------

NVIDIA_PDF_MANIFEST: list[dict[str, str]] = [
    {
        "url": "https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf",
        "title": "CUDA C++ Programming Guide",
        "gpu_family": "CUDA",
        "doc_type": "programming_guide",
    },
    {
        "url": "https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf",
        "title": "CUDA C++ Best Practices Guide",
        "gpu_family": "CUDA",
        "doc_type": "best_practices",
    },
    {
        "url": "https://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf",
        "title": "CUDA Math API Reference",
        "gpu_family": "CUDA",
        "doc_type": "api_reference",
    },
    {
        "url": "https://docs.nvidia.com/cuda/pdf/CUDA_Runtime_API.pdf",
        "title": "CUDA Runtime API Reference",
        "gpu_family": "CUDA",
        "doc_type": "api_reference",
    },
    {
        "url": "https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Developer-Guide.pdf",
        "title": "cuDNN Developer Guide",
        "gpu_family": "cuDNN",
        "doc_type": "developer_guide",
    },
    {
        "url": "https://docs.nvidia.com/deeplearning/tensorrt/pdf/TensorRT-Developer-Guide.pdf",
        "title": "TensorRT Developer Guide",
        "gpu_family": "TensorRT",
        "doc_type": "developer_guide",
    },
    {
        "url": "https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf",
        "title": "Thrust Quick Start Guide",
        "gpu_family": "CUDA",
        "doc_type": "quick_start",
    },
    {
        "url": "https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf",
        "title": "Nsight Systems User Guide",
        "gpu_family": "Nsight",
        "doc_type": "user_guide",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_id(url: str) -> str:
    """Stable 16-char hex ID derived from the source URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _nearest_heading(page_offsets: list[tuple[int, str]], pos: int) -> str:
    """Return the most recent section heading at or before character position `pos`."""
    heading = ""
    for offset, h in page_offsets:
        if offset > pos:
            break
        if h:
            heading = h
    return heading


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

@dag(
    dag_id="ingest_nvidia_docs",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
    },
    tags=["nvidia-ir", "ingestion", "layer-2"],
)
def ingest_nvidia_docs() -> None:

    @task()
    def fetch_manifest() -> list[dict[str, str]]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        log.info("fetch_manifest", stage="fetch_manifest", query_id=run_id, count=len(NVIDIA_PDF_MANIFEST))
        return NVIDIA_PDF_MANIFEST

    @task()
    def download_pdfs(manifests: list[dict[str, str]]) -> list[dict[str, str]]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        staging_dir = STAGING_BASE / run_id / "pdfs"
        staging_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, str]] = []
        for entry in manifests:
            url = entry["url"]
            doc_id = _doc_id(url)
            dest = staging_dir / f"{doc_id}.pdf"

            log.info("download_start", stage="download", query_id=run_id, doc_id=doc_id, url=url)
            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65_536):
                        fh.write(chunk)
                log.info(
                    "download_ok",
                    stage="download",
                    query_id=run_id,
                    doc_id=doc_id,
                    bytes=dest.stat().st_size,
                )
                results.append({
                    "doc_id": doc_id,
                    "local_path": str(dest),
                    "title": entry["title"],
                    "source_url": url,
                    "gpu_family": entry["gpu_family"],
                    "doc_type": entry["doc_type"],
                    "status": "downloaded",
                })
            except Exception as exc:
                log.warning(
                    "download_failed",
                    stage="download",
                    query_id=run_id,
                    doc_id=doc_id,
                    error=str(exc),
                )
                results.append({
                    "doc_id": doc_id,
                    "local_path": "",
                    "title": entry["title"],
                    "source_url": url,
                    "gpu_family": entry["gpu_family"],
                    "doc_type": entry["doc_type"],
                    "status": "download_failed",
                })

        return results

    @task()
    def parse_pdfs(downloaded: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Extract text and section headings from each PDF using PyMuPDF."""
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        staging_dir = STAGING_BASE / run_id / "parsed"
        staging_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for doc in downloaded:
            doc_id = doc["doc_id"]
            if doc["status"] != "downloaded":
                results.append({**doc, "page_count": 0, "text_path": "", "status": doc["status"]})
                continue

            log.info("parse_start", stage="parse", query_id=run_id, doc_id=doc_id)
            try:
                pdf = fitz.open(doc["local_path"])
                pages: list[dict[str, Any]] = []

                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    raw = page.get_text("dict")
                    text_parts: list[str] = []
                    headings: list[str] = []

                    for block in raw.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                span_text = span.get("text", "").strip()
                                if not span_text:
                                    continue
                                size: float = span.get("size", 10.0)
                                flags: int = span.get("flags", 0)
                                is_bold = bool(flags & 16)
                                # Large font (≥12pt) or bold short text → heading candidate
                                if (size >= 12.0 or is_bold) and len(span_text) < 120:
                                    headings.append(span_text)
                                text_parts.append(span_text)

                    pages.append({
                        "page_num": page_num,
                        "text": " ".join(text_parts),
                        "headings": headings,
                    })

                pdf.close()

                text_path = staging_dir / f"{doc_id}.json"
                text_path.write_text(
                    json.dumps(pages, ensure_ascii=False), encoding="utf-8"
                )

                log.info(
                    "parse_ok",
                    stage="parse",
                    query_id=run_id,
                    doc_id=doc_id,
                    pages=len(pages),
                )
                results.append({
                    **doc,
                    "page_count": len(pages),
                    "text_path": str(text_path),
                    "status": "parsed",
                })

            except Exception as exc:
                log.warning(
                    "parse_failed",
                    stage="parse",
                    query_id=run_id,
                    doc_id=doc_id,
                    error=str(exc),
                )
                results.append({**doc, "page_count": 0, "text_path": "", "status": "parse_failed"})

        return results

    @task()
    def chunk_docs(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split each document into overlapping chunks via LangChain RecursiveCharacterTextSplitter."""
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        staging_dir = STAGING_BASE / run_id / "chunks"
        staging_dir.mkdir(parents=True, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        results: list[dict[str, Any]] = []
        for doc in parsed:
            doc_id = doc["doc_id"]
            text_path = doc.get("text_path", "")
            if doc["status"] != "parsed" or not text_path:
                results.append({**doc, "chunks_path": "", "chunk_count": 0})
                continue

            log.info("chunk_start", stage="chunk", query_id=run_id, doc_id=doc_id)
            try:
                pages: list[dict[str, Any]] = json.loads(
                    Path(text_path).read_text(encoding="utf-8")
                )

                # Build full document text and track page start offsets for heading attribution.
                full_parts: list[str] = []
                page_offsets: list[tuple[int, str]] = []  # (char_offset, first_heading)
                cursor = 0

                for page in pages:
                    heading = page["headings"][0] if page.get("headings") else ""
                    prefix = f"[{heading}]\n" if heading else ""
                    part = prefix + page["text"]
                    page_offsets.append((cursor, heading))
                    full_parts.append(part)
                    cursor += len(part) + 2  # +2 for "\n\n" separator

                full_text = "\n\n".join(full_parts)
                raw_chunks = splitter.split_text(full_text)

                # Walk through raw_chunks in document order to assign character positions.
                search_from = 0
                chunks: list[dict[str, Any]] = []

                for idx, chunk_text in enumerate(raw_chunks):
                    # Search for the chunk's start position, advancing the cursor.
                    probe = chunk_text[:80] if len(chunk_text) >= 80 else chunk_text
                    pos = full_text.find(probe, search_from)
                    if pos == -1:
                        pos = search_from
                    search_from = pos + max(len(chunk_text) - CHUNK_OVERLAP, 1)

                    heading = _nearest_heading(page_offsets, pos)
                    chunk_id = hashlib.sha256(
                        f"{doc_id}:{idx}:{chunk_text[:64]}".encode()
                    ).hexdigest()[:24]

                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_text": chunk_text,
                        "chunk_index": idx,
                        "token_count": _count_tokens(chunk_text),
                        "section_heading": heading,
                    })

                chunks_path = staging_dir / f"{doc_id}.json"
                chunks_path.write_text(
                    json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
                )

                log.info(
                    "chunk_ok",
                    stage="chunk",
                    query_id=run_id,
                    doc_id=doc_id,
                    chunks=len(chunks),
                )
                results.append({
                    **doc,
                    "chunks_path": str(chunks_path),
                    "chunk_count": len(chunks),
                    "status": "chunked",
                })

            except Exception as exc:
                log.warning(
                    "chunk_failed",
                    stage="chunk",
                    query_id=run_id,
                    doc_id=doc_id,
                    error=str(exc),
                )
                results.append({**doc, "chunks_path": "", "chunk_count": 0, "status": "chunk_failed"})

        return results

    @task()
    def score_chunks(chunked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score each chunk with the heuristic quality scorer and write quality JSON to staging."""
        ctx = get_current_context()
        run_id: str = ctx["run_id"]

        results: list[dict[str, Any]] = []
        for doc in chunked:
            doc_id = doc["doc_id"]
            chunks_path = doc.get("chunks_path", "")
            if doc["status"] != "chunked" or not chunks_path:
                results.append({**doc, "quality_path": ""})
                continue

            log.info("score_start", stage="score_quality", query_id=run_id, doc_id=doc_id)
            try:
                chunks: list[dict[str, Any]] = json.loads(
                    Path(chunks_path).read_text(encoding="utf-8")
                )

                quality_records: list[dict[str, Any]] = []
                for chunk in chunks:
                    result = score_chunk(chunk["chunk_id"], chunk["chunk_text"])
                    quality_records.append({
                        "chunk_id": result.chunk_id,
                        "quality_score": result.quality_score,
                        "mean_sent_len": result.mean_sent_len,
                        "nonascii_ratio": result.nonascii_ratio,
                        "stopword_ratio": result.stopword_ratio,
                        "flagged": result.flagged,
                    })

                quality_dir = STAGING_BASE / run_id / "quality"
                quality_dir.mkdir(parents=True, exist_ok=True)
                quality_path = quality_dir / f"{doc_id}.json"
                quality_path.write_text(json.dumps(quality_records), encoding="utf-8")

                flagged = sum(1 for r in quality_records if r["flagged"])
                log.info(
                    "score_ok",
                    stage="score_quality",
                    query_id=run_id,
                    doc_id=doc_id,
                    total=len(quality_records),
                    flagged=flagged,
                )
                results.append({
                    **doc,
                    "quality_path": str(quality_path),
                    "status": "scored",
                })

            except Exception as exc:
                log.warning(
                    "score_failed",
                    stage="score_quality",
                    query_id=run_id,
                    doc_id=doc_id,
                    error=str(exc),
                )
                results.append({**doc, "quality_path": "", "status": "score_failed"})

        return results

    @task()
    def write_to_postgres(scored: list[dict[str, Any]]) -> dict[str, Any]:
        """Write doc_metadata, chunks, and chunk_quality to Postgres using SQLAlchemy ORM."""
        ctx = get_current_context()
        run_id: str = ctx["run_id"]

        engine = get_engine()
        SessionFactory = get_session_factory(engine)

        docs_written = 0
        chunks_written = 0
        quality_written = 0
        failed_docs: list[str] = []

        log.info(
            "write_start",
            stage="write_postgres",
            query_id=run_id,
            doc_count=len(scored),
        )

        for doc in scored:
            doc_id = doc["doc_id"]
            chunks_path = doc.get("chunks_path", "")
            quality_path = doc.get("quality_path", "")

            if not chunks_path or not Path(chunks_path).exists():
                failed_docs.append(doc_id)
                continue

            try:
                with SessionFactory() as session:
                    # ── doc_metadata: upsert ─────────────────────────────
                    existing_doc: DocMetadata | None = session.get(DocMetadata, doc_id)
                    if existing_doc is None:
                        session.add(DocMetadata(
                            doc_id=doc_id,
                            title=doc["title"],
                            source_url=doc["source_url"],
                            gpu_family=doc.get("gpu_family"),
                            doc_type=doc.get("doc_type"),
                            page_count=doc.get("page_count", 0),
                            last_ingested=datetime.now(timezone.utc),
                            ingestion_run=run_id,
                        ))
                        docs_written += 1
                    else:
                        existing_doc.last_ingested = datetime.now(timezone.utc)
                        existing_doc.ingestion_run = run_id
                        existing_doc.page_count = doc.get("page_count", existing_doc.page_count)

                    # ── chunks: insert new only (content-addressed by chunk_id) ──
                    chunks: list[dict[str, Any]] = json.loads(
                        Path(chunks_path).read_text(encoding="utf-8")
                    )
                    for c in chunks:
                        if session.get(Chunk, c["chunk_id"]) is None:
                            session.add(Chunk(
                                chunk_id=c["chunk_id"],
                                doc_id=doc_id,
                                chunk_text=c["chunk_text"],
                                chunk_index=c["chunk_index"],
                                token_count=c["token_count"],
                                section_heading=c.get("section_heading") or None,
                            ))
                            chunks_written += 1

                    # ── chunk_quality: upsert ────────────────────────────
                    if quality_path and Path(quality_path).exists():
                        quality_records: list[dict[str, Any]] = json.loads(
                            Path(quality_path).read_text(encoding="utf-8")
                        )
                        for qr in quality_records:
                            existing_q: ChunkQuality | None = session.get(ChunkQuality, qr["chunk_id"])
                            if existing_q is None:
                                session.add(ChunkQuality(
                                    chunk_id=qr["chunk_id"],
                                    quality_score=qr["quality_score"],
                                    mean_sent_len=qr["mean_sent_len"],
                                    nonascii_ratio=qr["nonascii_ratio"],
                                    stopword_ratio=qr["stopword_ratio"],
                                    flagged=qr["flagged"],
                                ))
                            else:
                                existing_q.quality_score = qr["quality_score"]
                                existing_q.mean_sent_len = qr["mean_sent_len"]
                                existing_q.nonascii_ratio = qr["nonascii_ratio"]
                                existing_q.stopword_ratio = qr["stopword_ratio"]
                                existing_q.flagged = qr["flagged"]
                                existing_q.scored_at = datetime.now(timezone.utc)
                            quality_written += 1

                    session.commit()

                log.info(
                    "write_doc_ok",
                    stage="write_postgres",
                    query_id=run_id,
                    doc_id=doc_id,
                    chunks=len(chunks),
                )

            except Exception as exc:
                failed_docs.append(doc_id)
                log.error(
                    "write_doc_failed",
                    stage="write_postgres",
                    query_id=run_id,
                    doc_id=doc_id,
                    error=str(exc),
                )

        summary: dict[str, Any] = {
            "run_id": run_id,
            "docs_written": docs_written,
            "chunks_written": chunks_written,
            "quality_written": quality_written,
            "failed_doc_ids": failed_docs,
        }
        log.info("write_done", stage="write_postgres", query_id=run_id, **summary)
        return summary

    @task()
    def log_coverage(
        write_summary: dict[str, Any],
        manifests: list[dict[str, str]],
    ) -> None:
        """Write a CoverageLog entry so downstream monitors can check ingestion completeness."""
        ctx = get_current_context()
        run_id: str = ctx["run_id"]

        engine = get_engine()
        SessionFactory = get_session_factory(engine)

        all_manifest_ids = {_doc_id(m["url"]) for m in manifests}
        failed_ids = set(write_summary.get("failed_doc_ids", []))
        indexed_ids = all_manifest_ids - failed_ids

        total_manifest = len(all_manifest_ids)
        total_indexed = len(indexed_ids)
        missing = sorted(failed_ids)
        coverage_pct = (total_indexed / total_manifest * 100.0) if total_manifest else 0.0

        with SessionFactory() as session:
            session.add(CoverageLog(
                run_id=run_id,
                total_manifest=total_manifest,
                total_indexed=total_indexed,
                coverage_pct=round(coverage_pct, 2),
                missing_docs=missing,
            ))
            session.commit()

        log.info(
            "coverage_logged",
            stage="coverage",
            query_id=run_id,
            total_manifest=total_manifest,
            total_indexed=total_indexed,
            coverage_pct=round(coverage_pct, 2),
            missing=missing,
        )

    # ── Wire the DAG ─────────────────────────────────────────────────────────
    manifests = fetch_manifest()
    downloaded = download_pdfs(manifests)
    parsed = parse_pdfs(downloaded)
    chunked = chunk_docs(parsed)
    scored = score_chunks(chunked)
    write_summary = write_to_postgres(scored)
    log_coverage(write_summary, manifests)


ingest_nvidia_docs()
