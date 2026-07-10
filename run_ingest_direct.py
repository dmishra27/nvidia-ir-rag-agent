"""Direct Python runner for the ingest_nvidia_docs DAG pipeline.

Runs the full pipeline without the Airflow scheduler:
  download_pdfs -> parse_pdfs -> chunk_docs -> score_chunks -> write_to_postgres -> log_coverage

Exit codes: 0 = at least one doc written, 1 = nothing written.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import requests
import structlog
import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from schema.models import (
    Chunk,
    ChunkQuality,
    CoverageLog,
    DocMetadata,
    get_engine,
    get_session_factory,
)
from retrieval.chunk_quality import score_chunk

log = structlog.get_logger()

# ── Constants (mirror the DAG) ────────────────────────────────────────────────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
DOWNLOAD_TIMEOUT = 120
STAGING_BASE = Path(tempfile.gettempdir()) / "nvidia_ir_ingest"
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

MANIFEST: list[dict[str, str]] = [
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _nearest_heading(page_offsets: list[tuple[int, str]], pos: int) -> str:
    heading = ""
    for offset, h in page_offsets:
        if offset > pos:
            break
        if h:
            heading = h
    return heading


# ── Task functions (no Airflow context) ──────────────────────────────────────

def download_pdfs(run_id: str) -> list[dict[str, Any]]:
    staging_dir = STAGING_BASE / run_id / "pdfs"
    staging_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    results: list[dict[str, Any]] = []
    for entry in MANIFEST:
        url = entry["url"]
        doc_id = _doc_id(url)
        dest = staging_dir / f"{doc_id}.pdf"
        log.info("download_start", stage="download", query_id=run_id, doc_id=doc_id, title=entry["title"])
        try:
            resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()
            raw = b"".join(resp.iter_content(chunk_size=65_536))
            if raw[:5] != b"%PDF-":
                raise ValueError(f"Not a PDF (starts with {raw[:8]!r})")
            dest.write_bytes(raw)
            log.info("download_ok", stage="download", query_id=run_id, doc_id=doc_id, bytes=len(raw))
            results.append({
                "doc_id": doc_id, "local_path": str(dest),
                "title": entry["title"], "source_url": url,
                "gpu_family": entry["gpu_family"], "doc_type": entry["doc_type"],
                "status": "downloaded",
            })
        except Exception as exc:
            log.warning("download_failed", stage="download", query_id=run_id, doc_id=doc_id, error=str(exc))
            results.append({
                "doc_id": doc_id, "local_path": "",
                "title": entry["title"], "source_url": url,
                "gpu_family": entry["gpu_family"], "doc_type": entry["doc_type"],
                "status": "download_failed",
            })
    return results


def parse_pdfs(downloaded: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
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
                            if (size >= 12.0 or is_bold) and len(span_text) < 120:
                                headings.append(span_text)
                            text_parts.append(span_text)
                pages.append({"page_num": page_num, "text": " ".join(text_parts), "headings": headings})
            pdf.close()
            text_path = staging_dir / f"{doc_id}.json"
            text_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
            log.info("parse_ok", stage="parse", query_id=run_id, doc_id=doc_id, pages=len(pages))
            results.append({**doc, "page_count": len(pages), "text_path": str(text_path), "status": "parsed"})
        except Exception as exc:
            log.warning("parse_failed", stage="parse", query_id=run_id, doc_id=doc_id, error=str(exc))
            results.append({**doc, "page_count": 0, "text_path": "", "status": "parse_failed"})
    return results


def chunk_docs(parsed: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    staging_dir = STAGING_BASE / run_id / "chunks"
    staging_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
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
            pages: list[dict[str, Any]] = json.loads(Path(text_path).read_text(encoding="utf-8"))
            full_parts: list[str] = []
            page_offsets: list[tuple[int, str]] = []
            cursor = 0
            for page in pages:
                heading = page["headings"][0] if page.get("headings") else ""
                prefix = f"[{heading}]\n" if heading else ""
                part = prefix + page["text"]
                page_offsets.append((cursor, heading))
                full_parts.append(part)
                cursor += len(part) + 2
            full_text = "\n\n".join(full_parts)
            raw_chunks = splitter.split_text(full_text)
            search_from = 0
            chunks: list[dict[str, Any]] = []
            for idx, chunk_text in enumerate(raw_chunks):
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
                    "chunk_id": chunk_id, "doc_id": doc_id,
                    "chunk_text": chunk_text, "chunk_index": idx,
                    "token_count": _count_tokens(chunk_text), "section_heading": heading,
                })
            chunks_path = staging_dir / f"{doc_id}.json"
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            log.info("chunk_ok", stage="chunk", query_id=run_id, doc_id=doc_id, chunks=len(chunks))
            results.append({**doc, "chunks_path": str(chunks_path), "chunk_count": len(chunks), "status": "chunked"})
        except Exception as exc:
            log.warning("chunk_failed", stage="chunk", query_id=run_id, doc_id=doc_id, error=str(exc))
            results.append({**doc, "chunks_path": "", "chunk_count": 0, "status": "chunk_failed"})
    return results


def score_chunks(chunked: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    quality_dir = STAGING_BASE / run_id / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for doc in chunked:
        doc_id = doc["doc_id"]
        chunks_path = doc.get("chunks_path", "")
        if doc["status"] != "chunked" or not chunks_path:
            results.append({**doc, "quality_path": ""})
            continue
        log.info("score_start", stage="score_quality", query_id=run_id, doc_id=doc_id)
        try:
            chunks: list[dict[str, Any]] = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
            quality_records: list[dict[str, Any]] = []
            for chunk in chunks:
                r = score_chunk(chunk["chunk_id"], chunk["chunk_text"])
                quality_records.append({
                    "chunk_id": r.chunk_id, "quality_score": r.quality_score,
                    "mean_sent_len": r.mean_sent_len, "nonascii_ratio": r.nonascii_ratio,
                    "stopword_ratio": r.stopword_ratio, "flagged": r.flagged,
                })
            quality_path = quality_dir / f"{doc_id}.json"
            quality_path.write_text(json.dumps(quality_records), encoding="utf-8")
            flagged = sum(1 for r in quality_records if r["flagged"])
            log.info("score_ok", stage="score_quality", query_id=run_id, doc_id=doc_id,
                     total=len(quality_records), flagged=flagged)
            results.append({**doc, "quality_path": str(quality_path), "status": "scored"})
        except Exception as exc:
            log.warning("score_failed", stage="score_quality", query_id=run_id, doc_id=doc_id, error=str(exc))
            results.append({**doc, "quality_path": "", "status": "score_failed"})
    return results


def write_to_postgres(scored: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    docs_written = chunks_written = quality_written = 0
    failed_docs: list[str] = []
    log.info("write_start", stage="write_postgres", query_id=run_id, doc_count=len(scored))

    for doc in scored:
        doc_id = doc["doc_id"]
        chunks_path = doc.get("chunks_path", "")
        quality_path = doc.get("quality_path", "")
        if not chunks_path or not Path(chunks_path).exists():
            failed_docs.append(doc_id)
            continue
        try:
            with SessionFactory() as session:
                existing = session.get(DocMetadata, doc_id)
                if existing is None:
                    session.add(DocMetadata(
                        doc_id=doc_id, title=doc["title"], source_url=doc["source_url"],
                        gpu_family=doc.get("gpu_family"), doc_type=doc.get("doc_type"),
                        page_count=doc.get("page_count", 0),
                        last_ingested=datetime.now(timezone.utc), ingestion_run=run_id,
                    ))
                    docs_written += 1
                else:
                    existing.last_ingested = datetime.now(timezone.utc)
                    existing.ingestion_run = run_id

                chunks: list[dict[str, Any]] = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
                for c in chunks:
                    if session.get(Chunk, c["chunk_id"]) is None:
                        session.add(Chunk(
                            chunk_id=c["chunk_id"], doc_id=doc_id,
                            chunk_text=c["chunk_text"], chunk_index=c["chunk_index"],
                            token_count=c["token_count"],
                            section_heading=c.get("section_heading") or None,
                        ))
                        chunks_written += 1

                if quality_path and Path(quality_path).exists():
                    quality_records: list[dict[str, Any]] = json.loads(
                        Path(quality_path).read_text(encoding="utf-8")
                    )
                    for qr in quality_records:
                        eq = session.get(ChunkQuality, qr["chunk_id"])
                        if eq is None:
                            session.add(ChunkQuality(
                                chunk_id=qr["chunk_id"], quality_score=qr["quality_score"],
                                mean_sent_len=qr["mean_sent_len"], nonascii_ratio=qr["nonascii_ratio"],
                                stopword_ratio=qr["stopword_ratio"], flagged=qr["flagged"],
                            ))
                        else:
                            eq.quality_score = qr["quality_score"]
                            eq.flagged = qr["flagged"]
                        quality_written += 1

                session.commit()
            log.info("write_doc_ok", stage="write_postgres", query_id=run_id,
                     doc_id=doc_id, chunks=len(chunks))
        except Exception as exc:
            failed_docs.append(doc_id)
            log.error("write_doc_failed", stage="write_postgres", query_id=run_id,
                      doc_id=doc_id, error=str(exc))

    summary = {
        "run_id": run_id, "docs_written": docs_written,
        "chunks_written": chunks_written, "quality_written": quality_written,
        "failed_doc_ids": failed_docs,
    }
    log.info("write_done", stage="write_postgres", query_id=run_id, **summary)
    return summary


def log_coverage(summary: dict[str, Any], run_id: str) -> None:
    all_ids = {_doc_id(m["url"]) for m in MANIFEST}
    failed_ids = set(summary.get("failed_doc_ids", []))
    indexed_ids = all_ids - failed_ids
    total_manifest = len(all_ids)
    total_indexed = len(indexed_ids)
    coverage_pct = (total_indexed / total_manifest * 100.0) if total_manifest else 0.0
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        session.add(CoverageLog(
            run_id=run_id, total_manifest=total_manifest,
            total_indexed=total_indexed, coverage_pct=round(coverage_pct, 2),
            missing_docs=sorted(failed_ids),
        ))
        session.commit()
    log.info("coverage_logged", stage="coverage", query_id=run_id,
             total_manifest=total_manifest, total_indexed=total_indexed,
             coverage_pct=round(coverage_pct, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import uuid
    run_id = f"direct_{uuid.uuid4().hex[:8]}"
    print(f"\n=== Direct ingestion run: {run_id} ===\n")

    print("Step 1/6  fetch_manifest")
    print(f"          {len(MANIFEST)} documents in manifest")

    print("\nStep 2/6  download_pdfs")
    downloaded = download_pdfs(run_id)
    ok = [d for d in downloaded if d["status"] == "downloaded"]
    fail = [d for d in downloaded if d["status"] != "downloaded"]
    print(f"          downloaded={len(ok)}  failed={len(fail)}")
    for d in fail:
        print(f"          SKIP  {d['title']}")

    print("\nStep 3/6  parse_pdfs")
    parsed = parse_pdfs(downloaded, run_id)
    parsed_ok = [d for d in parsed if d["status"] == "parsed"]
    print(f"          parsed={len(parsed_ok)}")

    print("\nStep 4/6  chunk_docs")
    chunked = chunk_docs(parsed, run_id)
    chunked_ok = [d for d in chunked if d["status"] == "chunked"]
    total_chunks = sum(d.get("chunk_count", 0) for d in chunked_ok)
    print(f"          chunked={len(chunked_ok)}  total_chunks={total_chunks}")

    print("\nStep 5/6  score_chunks")
    scored = score_chunks(chunked, run_id)
    scored_ok = [d for d in scored if d["status"] == "scored"]
    print(f"          scored={len(scored_ok)}")

    print("\nStep 6a/6  write_to_postgres")
    summary = write_to_postgres(scored, run_id)
    print(f"          docs_written={summary['docs_written']}")
    print(f"          chunks_written={summary['chunks_written']}")
    print(f"          quality_written={summary['quality_written']}")
    print(f"          failed_docs={summary['failed_doc_ids']}")

    print("\nStep 6b/6  log_coverage")
    log_coverage(summary, run_id)

    print(f"\n=== Ingestion complete: run_id={run_id} ===")
    print(f"    docs={summary['docs_written']}  chunks={summary['chunks_written']}")
    return 0 if summary["docs_written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
