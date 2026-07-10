from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP
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
    missing_docs = Column(ARRAY(String))
    checked_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))


def get_engine(url: str | None = None):
    db_url = url or os.environ["POSTGRES_URL"]
    return create_engine(db_url, pool_pre_ping=True)


def get_session_factory(engine=None) -> sessionmaker[Session]:
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)
