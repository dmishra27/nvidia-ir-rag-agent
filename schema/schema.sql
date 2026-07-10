-- schema.sql — nvidia-ir-rag-agent
-- Run automatically on first postgres container start

-- 1. Document metadata
CREATE TABLE IF NOT EXISTS doc_metadata (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source_url      TEXT,
    gpu_family      TEXT,
    doc_type        TEXT,
    page_count      INTEGER,
    last_ingested   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingestion_run   TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT REFERENCES doc_metadata(doc_id),
    chunk_text      TEXT NOT NULL,
    chunk_index     INTEGER,
    token_count     INTEGER,
    section_heading TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- 3. Chunk quality
CREATE TABLE IF NOT EXISTS chunk_quality (
    chunk_id        TEXT REFERENCES chunks(chunk_id),
    quality_score   FLOAT NOT NULL,
    mean_sent_len   FLOAT,
    nonascii_ratio  FLOAT,
    stopword_ratio  FLOAT,
    flagged         BOOLEAN DEFAULT FALSE,
    scored_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Coverage log
CREATE TABLE IF NOT EXISTS coverage_log (
    run_id          TEXT PRIMARY KEY,
    total_manifest  INTEGER,
    total_indexed   INTEGER,
    coverage_pct    FLOAT,
    missing_docs    TEXT[],
    checked_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Duplicate flags
CREATE TABLE IF NOT EXISTS duplicate_flags (
    chunk_id_a      TEXT REFERENCES chunks(chunk_id),
    chunk_id_b      TEXT REFERENCES chunks(chunk_id),
    similarity      FLOAT,
    flagged_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Query log
CREATE TABLE IF NOT EXISTS query_log (
    query_id        TEXT PRIMARY KEY,
    session_id      TEXT,
    query_text      TEXT NOT NULL,
    reranker_config TEXT,
    num_candidates  INTEGER,
    source          TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Evaluation results
CREATE TABLE IF NOT EXISTS eval_results (
    eval_id             TEXT PRIMARY KEY,
    query_id            TEXT REFERENCES query_log(query_id),
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    citation_accuracy   FLOAT,
    hallucination_rate  FLOAT,
    ndcg_at_10          FLOAT,
    mrr                 FLOAT,
    evaluated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. Request log
CREATE TABLE IF NOT EXISTS request_log (
    request_id      TEXT PRIMARY KEY,
    query_id        TEXT,
    endpoint        TEXT,
    reranker_config TEXT,
    stage           TEXT,
    duration_ms     FLOAT,
    status_code     INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at);

-- 9. Error log
CREATE TABLE IF NOT EXISTS error_log (
    error_id        TEXT PRIMARY KEY,
    endpoint        TEXT,
    error_type      TEXT,
    message         TEXT,
    stack_trace     TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 10. Feedback log
CREATE TABLE IF NOT EXISTS feedback_log (
    feedback_id     TEXT PRIMARY KEY,
    query_id        TEXT REFERENCES query_log(query_id),
    user_id         TEXT,
    reaction        TEXT,
    source          TEXT DEFAULT 'slackbot',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 11. Benchmark results
CREATE TABLE IF NOT EXISTS benchmark_results (
    result_id         TEXT PRIMARY KEY,
    run_id            TEXT,
    config            TEXT NOT NULL,
    query_id          TEXT,
    ndcg_at_10        FLOAT,
    mrr               FLOAT,
    prec_at_3         FLOAT,
    prec_at_5         FLOAT,
    prec_at_10        FLOAT,
    faithfulness      FLOAT,
    citation_accuracy FLOAT,
    latency_ms        FLOAT,
    cost_usd          FLOAT DEFAULT 0.0,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_benchmark_config ON benchmark_results(config);
CREATE INDEX IF NOT EXISTS idx_benchmark_run ON benchmark_results(run_id);
