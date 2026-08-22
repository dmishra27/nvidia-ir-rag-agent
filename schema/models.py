from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class DocMetadata(Base):
    __tablename__ = "doc_metadata"

    doc_id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    source_url = Column(Text)
    gpu_family = Column(Text)
    doc_type = Column(Text)
    page_count = Column(Integer)
    last_ingested = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    ingestion_run = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    # SHA-256 of the source PDF's bytes. _doc_id (see ingest scripts) only
    # hashes the URL, so without this a document revision at the same URL
    # is invisible. See DEF-20.
    content_sha256 = Column(String)


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("doc_metadata.doc_id"))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    token_count = Column(Integer)
    section_heading = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class ChunkQuality(Base):
    __tablename__ = "chunk_quality"

    chunk_id = Column(String, ForeignKey("chunks.chunk_id"), primary_key=True)
    quality_score = Column(Float, nullable=False)
    mean_sent_len = Column(Float)
    nonascii_ratio = Column(Float)
    stopword_ratio = Column(Float)
    flagged = Column(Boolean, default=False)
    scored_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class CoverageLog(Base):
    __tablename__ = "coverage_log"

    run_id = Column(String, primary_key=True)
    total_manifest = Column(Integer)
    total_indexed = Column(Integer)
    coverage_pct = Column(Float)
    missing_docs: Column[list[str]] = Column(ARRAY(String))
    checked_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class RequestLog(Base):
    __tablename__ = "request_log"

    request_id = Column(String, primary_key=True)
    query_id = Column(String)
    endpoint = Column(Text)
    reranker_config = Column(Text)
    stage = Column(Text)
    duration_ms = Column(Float)
    status_code = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class QueryLog(Base):
    __tablename__ = "query_log"

    query_id = Column(String, primary_key=True)
    session_id = Column(String)
    query_text = Column(Text, nullable=False)
    reranker_config = Column(Text)
    num_candidates = Column(Integer)
    source = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvalResults(Base):
    __tablename__ = "eval_results"

    eval_id = Column(String, primary_key=True)
    query_id = Column(String, ForeignKey("query_log.query_id"))
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    citation_accuracy = Column(Float)
    hallucination_rate = Column(Float)
    ndcg_at_10 = Column(Float)
    mrr = Column(Float)
    evaluated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class BenchmarkResults(Base):
    __tablename__ = "benchmark_results"

    result_id = Column(String, primary_key=True)
    run_id = Column(String)
    config = Column(Text, nullable=False)
    query_id = Column(String)
    ndcg_at_10 = Column(Float)
    mrr = Column(Float)
    prec_at_3 = Column(Float)
    prec_at_5 = Column(Float)
    prec_at_10 = Column(Float)
    faithfulness = Column(Float)
    citation_accuracy = Column(Float)
    latency_ms = Column(Float)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


class FeedbackLog(Base):
    __tablename__ = "feedback_log"

    feedback_id = Column(String, primary_key=True)
    query_id = Column(String, ForeignKey("query_log.query_id"))
    user_id = Column(String)
    reaction = Column(String)
    source = Column(String, default="slackbot")
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


def get_engine(url: str | None = None) -> Engine:
    db_url = url or os.environ["POSTGRES_URL"]
    return create_engine(db_url, pool_pre_ping=True)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)
