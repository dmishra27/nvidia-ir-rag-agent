# Day 8 — Evaluation Stack, Benchmark Runner, RAGAS Suite, Citation Judge

## 1. What was built

| File | Lines | Tests |
|---|---|---|
| `evaluation/retrieval_metrics.py` | 58 | 18 (`tests/evaluation/test_retrieval_metrics.py`, 163 lines) |
| `retrieval/reranker_cohere.py` | 99 | 9 (`tests/retrieval/test_reranker_cohere.py`, 160 lines) |
| `evaluation/relevance_labeller.py` | 250 | 13 (`tests/evaluation/test_relevance_labeller.py`, 217 lines) |
| `schema/models.py` (+`QueryLog`, `EvalResults`, `BenchmarkResults`) | 131 | exercised via benchmark_runner tests |
| `evaluation/benchmark_runner.py` | 266 | 15 (`tests/evaluation/test_benchmark_runner.py`, 254 lines) |
| `evaluation/citation_judge.py` | 152 | 12 (`tests/evaluation/test_citation_judge.py`, 188 lines) |
| `evaluation/ragas_suite.py` | 134 | 13 (`tests/evaluation/test_ragas_suite.py`, 163 lines) |
| `agents/retrieval_agent.py` (`build_default_router` +Cohere tier) | 145 | no new tests — untested real-model factory, same as the pre-existing ms-marco wiring |

New tests today: 80. Suite total: **323/323 passing**.

## 2. Why it matters

Days 1-7 built the retrieval and generation pipeline; nothing yet *measured*
it beyond hand-run UAT scripts. Day 8 opens Layers 4-5: a repeatable
evaluation stack that scores retrieval quality (NDCG@10, MRR, Precision@K),
re-ranker configs against each other (cost and latency included), and the
QA agent's actual output (faithfulness, answer relevancy, context
precision, citation accuracy) — the difference between "it looks right in a
UAT transcript" and "it's measured against a fixed benchmark every time the
pipeline changes."

`evaluation/relevance_labeller.py` defines the project's one fixed
50-query benchmark set (`BENCHMARK_QUERIES`, categorised exact-API /
memory-performance / hardware-spec / semantic / mixed, 10 each) — per
SKILLS.md's run-reranker-benchmark contract, every other Day 8 module reads
from this same set rather than each picking its own sample. `build_pairs()`
retrieves one top-ranked RRF-fused passage per query (mirroring
`agents/retrieval_agent.py`'s retrieve step), and `label_pairs()` forces a
`judge_relevance` Claude Sonnet tool call per pair — the same
forced-tool-call pattern `agents/qa_agent.py` established for citations, now
reused for binary relevance judgment. The result is a sparse, MS-MARCO-style
judgment set (one labelled passage per query, unjudged = non-relevant) sized
for a single LLM evaluator rather than TREC-scale pooling.

`retrieval/reranker_cohere.py` is the third re-ranking tier
(`live_frontier` in `RERANKER_MODE`'s serving chain) — Cohere Rerank v3,
built to the exact shape `retrieval/reranker_msmarco.py` established: an
injectable client Protocol, a `load()` classmethod that does the real
`cohere.Client()` construction, unit tests that never touch the network.
`agents/retrieval_agent.py`'s `build_default_router()` now wires it in
alongside ms-marco, closing part of the gap Day 6/7 left open (only
`live_fast` was wired; `live_quality`/bge is still unbuilt and unwired).

`evaluation/retrieval_metrics.py` is pure math (NDCG@10, MRR, Precision@K)
with no I/O, so it's the one Day 8 module with zero mocking in its tests —
`evaluation/benchmark_runner.py` is the only caller. `benchmark_runner.py`
runs **Config A (ms-marco)** and **Config C (Cohere)** over the identical
RRF top-100 pool per query (never re-fused per config, per SKILLS.md), scores
each against `relevance_labels.jsonl`, and logs per-query + aggregate
metrics twice: to MLflow experiment `reranker_benchmark` (per-config runs,
per-query metric history via `step=`) and to the `benchmark_results` table
via the new `BenchmarkResults` SQLAlchemy model — never a raw SQL string,
per AGENTS.md. **Config B (bge-reranker-v2-m3) is not run** —
`CONFIG_B_DEFERRED_REASON` documents it as hardware-blocked (OOM at model
load under this machine's 8GB RAM / CPU-only constraint) and deferred to a
GPU environment, the same posture Day 6 took toward its live UAT check.

`evaluation/citation_judge.py` closes a gap `agents/qa_agent.py` left open
since Day 7: `generate` forces the model to structure citations as
`{claim, chunk_ids}`, but nothing verified the model actually followed its
own grounding instructions. `judge_qa_output()` walks every `(claim,
chunk_id)` pair in a `QAState`'s citations; if the chunk_id was never
actually retrieved, it's flagged unsupported without spending an LLM call
(a hallucinated citation isn't a judgment call), otherwise a forced
`judge_citation` Claude Sonnet tool call scores it. `citation_accuracy()`
is the fraction supported.

`evaluation/ragas_suite.py` follows SKILLS.md's write-ragas-eval pattern
literally — the legacy `question`/`answer`/`contexts`/`ground_truth` column
names, which this project's patched ragas 0.4.3 still accepts (verified
directly: `evaluate()` reaches the OpenAI-credentials error, not a
column-schema error, when called with the legacy schema). Two deviations
from a naive read of the skill: (1) neither OpenAI nor an `OPENAI_API_KEY`
exists in this Anthropic-only project, so `run_ragas_eval` takes injected
`llm`/`embeddings` ragas wrappers, with `build_default_llm`/
`build_default_embeddings` lazily constructing the real Claude Sonnet
(`langchain_anthropic.ChatAnthropic`) and e5-base-v2
(`langchain_community.embeddings.HuggingFaceEmbeddings`) backends — mirrors
`reranker_msmarco.py`'s `load()` split so importing the module never
requires those packages. (2) `context_precision` needs a `ground_truth`
reference per query, which this project has no hand-curated set for; rather
than fabricate one from the QA agent's own answer (circular — grading an
answer against itself), `select_metrics()` drops `context_precision`
whenever no `ground_truth` dict is supplied and logs why, running only
`faithfulness`/`answer_relevancy` in that case.

## 3. Live artifact generation — deferred

Per the project's standing host-memory rule and Day 6's precedent, free
memory was checked before any live step: **0.22-0.79GB free of 7.65GB**
across repeated checks this session —
the same danger zone that forced Day 6 to defer its UAT regression run.
Loading the dense encoder (e5-base-v2, torch), the ms-marco cross-encoder,
and making live Claude/Cohere calls concurrently with the already-running
Postgres/Qdrant/MLflow/API/Streamlit containers risked thrashing or an OOM
crash. Given the choice, the user chose to defer rather than force it
through or free memory mid-session.

**Deferred, not run this session**:
- `evaluation/relevance_labeller.py main()` — real `benchmark_queries.jsonl`
  and `relevance_labels.jsonl` (50 live BM25+dense+RRF retrievals + 50 live
  Claude judgments)
- `evaluation/benchmark_runner.py main()` — Config A + Config C over the
  live 50-query set, real MLflow run + `benchmark_results` rows
- `evaluation/ragas_suite.py main()` — 10 live QA agent runs + RAGAS scoring
- `evaluation/citation_judge.py` — live citation judgments over those 10 runs

All four are fully unit-tested against fakes/mocks (80 tests, 0 real
network/model calls), so the *code* is verified; only the *live data* is
outstanding. `evaluation/benchmark_queries.jsonl` and
`evaluation/relevance_labels.jsonl` do not yet exist in the repo.

## 4. Day 1 → 8 narrative

- **Day 1** (`1bc7f0c`): project scaffolding — schema, docker-compose,
  `AGENTS.md`/`SKILLS.md` contracts.
- **Day 2** (`42431ea`): Airflow 3 ingestion DAG (650 lines) — PyMuPDF parse,
  LangChain chunk, quality score, SQLAlchemy ORM write. 72 tests. Text-to-SQL
  LangGraph agent (`aebd277`, 21 tests) and direct ingest runner (`daa4ae9`)
  landed the corpus: **5,389 chunks in Postgres**, all 4 Text-to-SQL queries
  verified live.
- **Day 3** (`3fee692`): BM25 sparse index + shared `Candidate` dataclass —
  the first of the four Layer-3 retrieval signals, 108/108 tests passing,
  live search verified.
- **Day 4** (`14bbcfa`, `0568c5b`): SPLADE sparse index (second signal,
  18 tests); bi-encoder evaluation picks e5-base-v2 (NDCG@10 0.5088) as the
  dense encoder and populates Qdrant with 5,389 points; 3 of 4 MCP servers
  (postgres, qdrant, airflow) written, structlog-to-stderr fix resolves MCP
  stdout corruption. 126/126 tests passing.
- **Day 5** (`373c9f5`, `4ba23cf`, `3c9e415`): dense search wrapper over the
  populated Qdrant collection (third signal, 12 tests) and RRF fusion
  (15 tests) combine BM25 + dense into a single hybrid ranking, live-verified
  end to end. 153/153 tests passing. A 15-query UAT across 6 superiority
  cases followed, surfacing two regressions to re-check: Q1 (RRF
  corroboration bias) and Q12 (dense-fails/BM25-wins).
- **Day 6** (`d7a998f`): ms-marco cross-encoder re-ranker (10 tests), 5-tier
  `RERANKER_MODE` router with graceful degradation (22 tests), and a
  three-node LangGraph retrieval agent (25 contract tests) close Layer 3b.
  208/208 tests passing. Live UAT regression check deferred to low free
  memory (0.33–0.49GB of 7.65GB).
- **Day 7** (`139a9cc`, `460c245`): the deferred UAT regression, run
  memory-safely by reranking cached RRF candidate pools (Q1's rank-1
  confirmed stable, Q12's rank-1 correctly promoted from RRF's rank-2 by
  ms-marco); QA agent (`retrieve → rerank → generate`, per-claim citations
  via forced tool call, 22 tests) and a 3-endpoint FastAPI service
  (`/search`, `/ask`, `/health`) with a request-latency-to-Postgres
  middleware (13 tests). 243/243 tests passing.
- **Day 8** (today): evaluation stack opens Layers 4-5 — 50-query benchmark
  set + Claude relevance labelling (`relevance_labeller.py`), a third
  re-ranking tier (`reranker_cohere.py`, now wired into
  `retrieval_agent.build_default_router`), NDCG@10/MRR/Precision@K
  (`retrieval_metrics.py`), a two-config benchmark runner logging to MLflow
  and `benchmark_results` (`benchmark_runner.py`, Config B documented as
  hardware-deferred), a citation-accuracy LLM judge (`citation_judge.py`),
  and a RAGAS suite for faithfulness/answer relevancy/context precision
  (`ragas_suite.py`). 80 new tests, 323/323 passing. Live artifact
  generation (real benchmark data, live benchmark/RAGAS/citation-judge runs)
  deferred to low free memory (0.22–0.79GB of 7.65GB) — see §3.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0 (chunk quality) | Done | `retrieval/chunk_quality.py`, 373 lines of tests (Day 2) |
| Layer 1 (MCP tool-calling) | 3/4 servers written | postgres (live), qdrant (live-verified), airflow (written, not yet live), mlflow deferred to Week 4 |
| Layer 2 (ingestion) | Done | Airflow DAG, 5,389 chunks in Postgres |
| Layer 3 (retrieval) | 3/4 signals + RRF done | BM25 (Day 3), SPLADE (Day 4), dense (Day 4/5), RRF fusion (Day 5); ColBERT outstanding |
| Layer 3b (re-ranking) | 2/3 live tiers wired | ms-marco + Cohere wired into `RerankerRouter`/`build_default_router` (Day 6, Day 8); bge (`live_quality`) still unbuilt |
| QA agent | Done (unit-tested; live Claude call not yet exercised) | `agents/qa_agent.py` — retrieve → rerank → generate, per-claim citations |
| API | Done (unit-tested; not yet run live with `uvicorn`) | `api/main.py`, `api/routers/{search,ask,health}.py`, `api/middleware.py` |
| Layer 4-5 (evaluation) | Code + tests done; live data generation deferred | `evaluation/{relevance_labeller,retrieval_metrics,benchmark_runner,citation_judge,ragas_suite}.py` — see §3 |
| Layers 6-8 (monitoring, observability, drift/HITL) | Not started | — |

**Test suite**: 323/323 passing (243 through Day 7, +18 retrieval metrics +
9 Cohere reranker + 13 relevance labeller + 15 benchmark runner + 12
citation judge + 13 RAGAS suite = 80 new).

**Open items for next session**: run the four deferred live steps (§3) once
free memory allows — this generates the real
`evaluation/benchmark_queries.jsonl` / `relevance_labels.jsonl`, populates
`benchmark_results` and MLflow's `reranker_benchmark` experiment with real
Config A/C numbers, and produces the first live RAGAS + citation-accuracy
scores; run `uvicorn api.main:app` live against the real corpus (still
outstanding from Day 7); build the bge-reranker-v2-m3 tier in a GPU
environment to close Config B and `live_quality`.
