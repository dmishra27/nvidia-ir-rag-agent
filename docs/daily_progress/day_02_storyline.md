# Day 2 — Ingestion Pipeline and Text-to-SQL Semantic Layer

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `airflow/dags/ingest_nvidia_docs.py` | 650 | (exercised via `test_chunk_quality.py` + live run, not DAG-level unit tests) |
| `schema/models.py` | 72 | (SQLAlchemy ORM models, exercised by every DB-touching test) |
| `retrieval/chunk_quality.py` | 147 | 72 (`tests/retrieval/test_chunk_quality.py`, 373 lines) |
| `agents/text_to_sql_agent.py` | 266 | 21 (`tests/agents/test_text_to_sql_agent.py`, 288 lines) |
| `run_ingest_direct.py` | 439 | — (direct pipeline runner, live-verified not unit-tested) |
| `run_agent_queries.py` | 43 | — (live query runner against the agent) |
| DVC-tracked corpus (`921281c`) | — | 5 PDFs verified clean |

New tests today: 93 (72 chunk-quality + 21 Text-to-SQL). Suite total:
**93/93 passing**.

## 2. Why it matters

Day 1 fixed the schema; Day 2 is the first day data actually flows through
it. `airflow/dags/ingest_nvidia_docs.py` implements Layer 2 (ingestion) per
`AGENTS.md`: PyMuPDF (`fitz`) parses each PDF, `langchain_text_splitters`'
`RecursiveCharacterTextSplitter` chunks the text, `retrieval/chunk_quality.py`
scores every chunk (Layer 0 — heuristic sentence-length / non-ASCII /
stopword-ratio quality, flagging anything below 0.40), and `schema/models.py`
writes everything through the SQLAlchemy ORM (never raw SQL, per coding
standards) into `doc_metadata`, `chunks`, and `chunk_quality`.
`agents/text_to_sql_agent.py` is a separate but related piece of work: a
LangGraph agent backed by Claude (`claude-opus-4-8`) that answers natural-
language questions about the corpus by selecting one of four tools
(`count_docs_by_gpu_family`, `chunks_below_quality_threshold`, etc.) and
running the resulting query — a semantic layer over the same 11-table schema
that Layer 1's MCP servers will later expose more generally.

## 3. Day 1 → 2 narrative

- **Day 1** (`1bc7f0c`): project scaffolding — schema, docker-compose,
  `AGENTS.md`/`SKILLS.md` contracts.
- **Day 2** (`42431ea`): Airflow 3 ingestion DAG (650 lines) — PyMuPDF parse,
  LangChain chunk, quality score, SQLAlchemy ORM write. 72 tests.
- Corpus DVC-tracked (`921281c`, 5 PDFs verified clean), module `__init__.py`
  stubs added (`015d044`).
- Text-to-SQL LangGraph agent (`aebd277`, 21 tests) and direct ingest runner
  (`daa4ae9`) landed the corpus: **5,389 chunks in Postgres**, all 4
  Text-to-SQL queries verified live.

## 4. Ingestion pipeline and live verification

The DAG's task graph — `download_pdfs -> parse_pdfs -> chunk_docs ->
score_chunks -> write_to_postgres -> log_coverage` — was run via
`run_ingest_direct.py` rather than the Airflow scheduler, so the pipeline
could be verified without bringing up the full `standalone` Airflow
container. This landed **5,389 chunks** across the 5 DVC-tracked NVIDIA PDFs
(CUDA C++ Programming Guide, CUDA C++ Best Practices Guide, CUDA Math API
Reference, CUDA Runtime API Reference, Nsight Systems User Guide) into
Postgres.

`run_agent_queries.py` then exercised the Text-to-SQL agent live against
that data with 4 benchmark questions:

| # | Question | Tool selected |
|---|---|---|
| Q1 | How many documents are indexed per GPU family? | `count_docs_by_gpu_family` |
| Q2 | Which chunks scored below the quality threshold of 0.60? | `chunks_below_quality_threshold` |
| Q3 | What is the average chunk quality score per document? | (per-doc aggregate tool) |
| Q4 | How many total chunks were ingested across all documents? | (corpus count tool) |

All 4 returned correct tool selection, valid SQL results, and a grounded
natural-language answer — the first end-to-end proof that the schema, the
ORM models, and an LLM-driven query layer work together against real data.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests |
| Layer 1 (MCP tool-calling) | Not started | Text-to-SQL agent uses direct Anthropic tool-use, not MCP — precursor, not the layer itself |
| Layer 2 (ingestion) | Done | Airflow DAG (written), direct runner (live-verified), 5,389 chunks in Postgres |
| Layer 3 (retrieval) | Not started | — |
| Layer 3b (re-ranking) | Not started | — |
| Layers 4–8 (eval, monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 93/93 passing (72 chunk-quality + 21 Text-to-SQL).
