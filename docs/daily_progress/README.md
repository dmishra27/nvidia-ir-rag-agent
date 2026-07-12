# nvidia-ir-rag-agent — Daily Progress

Storyline docs recording what was built, why, and what was live-verified,
one file per working day. Each doc is self-contained (what/why/narrative/
cumulative status) but the "Day 1 → N narrative" section in every doc
recaps everything before it, so any single file can be read on its own.

| Day | Focus | Tests added | Suite total | Doc |
|---|---|---|---|---|
| 1 | Project scaffolding — schema, docker-compose, `AGENTS.md`/`SKILLS.md` | 0 | 0/0 | [day_01_storyline.md](day_01_storyline.md) |
| 2 | Ingestion pipeline (Airflow DAG) + Text-to-SQL semantic layer, 5,389 chunks landed in Postgres | 93 | 93/93 | [day_02_storyline.md](day_02_storyline.md) |
| 3 | BM25 sparse retrieval + shared `Candidate` dataclass | 15 | 108/108 | [day_03_storyline.md](day_03_storyline.md) |
| 4 | Bi-encoder eval (e5-base-v2 wins), SPLADE sparse index, Qdrant populated, 3 MCP servers | 18 | 126/126 | [day_04_storyline.md](day_04_storyline.md) |
| 5 | Dense index (Qdrant), RRF fusion, hybrid search live-verified | 27 | 153/153 | [day_05_storyline.md](day_05_storyline.md) |

## Architecture reference

Full 8-layer architecture, MCP server list, coding standards, and folder
structure live in [`AGENTS.md`](../../AGENTS.md). Reusable code patterns
live in [`SKILLS.md`](../../SKILLS.md).

## Layer 3 (retrieval) progress at a glance

| Signal | Day | Status |
|---|---|---|
| BM25 (lexical) | 3 | Done |
| SPLADE (learned sparse) | 4 | Done |
| Dense (bi-encoder / Qdrant) | 4 (index), 5 (search wrapper) | Done |
| RRF fusion | 5 | Done |
| ColBERT | — | Outstanding |
| Layer 3b re-ranking (3-way benchmark) | — | Not started |
