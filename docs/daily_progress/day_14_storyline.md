# Day 14 — A2A Protocol, Final Evaluation Report, v1.0.0 Release

## 1. What was built / run

| File | Purpose |
|---|---|
| `README.md` | Updated (twice, before this day's other work): honest Render deployment status — `/health` confirmed 200 OK, Swagger UI confirmed accessible at `/docs`, `/search`/`/ask` confirmed 500ing for the documented reason (BM25 index not shipped in the deployed image), local quickstart pointed to for working search |
| `agents/a2a_protocol.py` | New: typed `AgentMessage` handoff envelope (`RetrievalHandoffPayload`/`ErrorPayload`) between `agents/retrieval_agent.py` and `agents/qa_agent.py`; `run_handoff()` runs retrieval once and hands its ranked results directly into `qa_agent`'s `generate` node, instead of each agent independently re-retrieving and re-ranking the same query |
| `tests/agents/test_a2a_protocol.py` | New: 11 tests — envelope construction/consumption (pure), `run_handoff()` end to end with fake BM25/dense/router objects and a mocked Anthropic client |
| `reports/final_eval_report.md` | New: consolidated NDCG/RAGAS/citation-accuracy results, Config A/C comparison, per-query citation breakdown, known-gaps section, research context — every number sourced from this project's real committed Day 9/11 artifacts, none re-measured or projected |
| `CHANGELOG.md` | New: Days 1-14 feature history, Keep-a-Changelog-style, first tagged release `v1.0.0` |
| `docs/daily_progress/day_14_storyline.md` | This file |
| `v1.0.0` git tag | New: first tagged release |

## 2. README — Render deployment status confirmed live (two rounds)

Two follow-up updates to Day 13's README work, both verified directly
before writing anything (not taken on faith):

1. First report: `/health` returning 200 with Swagger UI accessible.
   Checked myself first — 5 separate attempts (HTTP/2, HTTP/1.1, root
   path, up to 60s each) all timed out with **0 bytes received**, not a
   4xx/5xx, across roughly 10 minutes. Surfaced that contradiction rather
   than writing the README as if it were confirmed; asked how to proceed;
   was told to add the URL anyway, unverified, with the failure noted —
   done that way, committed as `d153493`.
2. Second report, minutes later: `/health` genuinely live (`200 OK`,
   exact JSON body), Swagger UI accessible. Verified both directly again
   before updating anything — this time confirmed. Updated the README's
   Live Demo section accordingly (`a873435`).
3. Third report: `/search` 500ing, attributed to the BM25 index being
   DVC-tracked and not available on Render's free tier. Verified the 500
   directly (`curl -X POST .../search`) before writing it up. The precise
   mechanism is slightly more specific than "DVC-tracked" — `data/` is
   gitignored wholesale and never enters the Docker image at all; DVC
   itself only tracks `data/raw/`/`data/chunks/` specifically, not the
   built `bm25_index.pkl` — so the README states the gitignore/image-build
   mechanism directly rather than repeating the DVC framing verbatim,
   while crediting the same underlying point. This is exactly the "known
   limitation" `render.yaml` and the README already documented *before*
   deploying (Day 13 §8) — now confirmed true, not a new discovery.
   Committed as `f12f5c7`.

## 3. A2A protocol — real handoff, not a spec implementation

`agents/qa_agent.py`'s own `retrieve`/`rerank` nodes duplicate
`agents/retrieval_agent.py`'s logic almost exactly — both independently
run BM25 + dense + RRF + re-ranking over the same query, because nothing
formalizes one agent handing its already-computed results to the other.
`agents/a2a_protocol.py` fixes that specific gap: `AgentMessage` is a
typed envelope (`sender`/`recipient`/`query_id`/`message_type`/`payload`),
`build_retrieval_handoff()` packages a finished `AgentState` into one
(or an `"error"` message if the retrieval agent itself failed —
callers don't need a separate error check), and `apply_retrieval_handoff()`
consumes it straight into `QAState.reranked_results` — `qa_agent`'s
`generate` node already reads `reranked_results or fused_results`
(`make_generate_node`), so a handed-off state skips its retrieve/rerank
nodes entirely rather than re-running them. `run_handoff()` wires the
whole thing end to end as a real, callable demo, not just an unused
envelope class.

Deliberately *not* an implementation of Google's full A2A spec (HTTP/
JSON-RPC transport, `AgentCard` discovery, etc.) — every agent in this
project runs in-process via LangGraph (`AGENTS.md`'s Layer 1), so there is
no transport to protocol-ize, only the message shape between two Python
function calls. The task's own "Agent handoff messaging between Retrieval
and QA agents" framing is what got built, at the scope this codebase
actually needs.

`_run_retrieval_agent()` (private, in this new module) mirrors
`retrieval_agent.py::run()`'s body rather than calling it directly,
because that function returns only `list[Candidate]` — it discards the
`AgentState.error` field this module's handoff needs to detect a failed
retrieval and produce an `"error"` message instead of a
`"retrieval_handoff"` one. Not a change to `retrieval_agent.py`'s existing
public signature (other callers depend on it returning
`list[Candidate]`), just a second, purpose-built way to run the same
graph.

11 new tests: envelope construction/consumption tested as pure
transformations (no I/O); `run_handoff()` tested end to end with fake
BM25/dense/router objects and `unittest.mock.patch("agents.qa_agent.anthropic.Anthropic")`
— the same mocking target `tests/agents/test_qa_agent.py` already uses,
since `make_generate_node`'s closure references `agents.qa_agent`'s own
`anthropic` import, not this module's.

## 4. Final evaluation report — synthesis, not new measurement

`reports/final_eval_report.md` pulls together every real number this
project has produced across Days 8-13 into one document, with a finding
that only becomes visible by reading them together: Config A/C's NDCG@10
(§1) looks solid (0.53), but RAGAS's answer-relevancy score is low (0.25,
§2) — and the per-query citation breakdown (§3) shows exactly why:
`Q3`/`Q5` each have only 1/3 claims supported, because the retrieved
passages didn't cover those questions, and the model correctly declined
to invent an answer (faithful, but not relevant to what was asked). That's
a retrieval-recall problem, not a generation-quality problem — a
conclusion no single existing dashboard/storyline stated explicitly,
because RAGAS (Day 11) and the citation judge (Day 9) were separate runs
never read side by side before now.

Every number in the report links back to its real source (MLflow run ID,
storyline section, or committed JSON file) — nothing here is a fresh
benchmark run or a projection. §5 states the evaluation's own gaps
plainly: 15-query smoke scope (not the full 50-query design),
Config B never measured at all, RAGAS at 10 queries, no live multi-day
regression history, and citation-judge/RAGAS being two independent runs
rather than one joint measurement over the same queries.

§6 connects this project's Layer 0 chunk-quality scoring
(`retrieval/chunk_quality.py`) to the author's SIGIR 2024 co-authorship
and the sibling `MSc_Dissertation_Document_Quality_Estimation` project in
this same workspace (a T5-based learned quality estimator for PyTerrier
reranking) — stated accurately as a **heuristic** stand-in for that
research direction (sentence-length/non-ASCII/stopword-ratio scoring),
not a claim that a neural model was ported here. A learned Layer 0
estimator is named as a natural next step, not built this session.

## 5. CHANGELOG.md and the v1.0.0 tag

`CHANGELOG.md` documents all 14 days under a single `[1.0.0]` entry —
this project ships its first tag capturing the full build history rather
than a series of incremental prior releases, so a Keep-a-Changelog
`### Added` list grouped by day (with `### Fixed` for the real bugs found
along the way: Day 9's `qa_agent.py` crash, Day 10's Q10 citation
stringification, Day 11's RAGAS `bypass_temperature` bug, Day 13's
six-defect CI chain) reads more honestly than inventing a `0.x` version
history that never actually existed as separate releases.

`git tag -a v1.0.0 -m "nvidia-ir-rag-agent v1.0.0"`, pushed to `origin`.

## 6. Open items carried into v1.0.0 (unchanged from Day 13, restated for the release)

1. Full 50-query × top-100-candidate benchmark — not run.
2. Config B (bge-reranker-v2-m3) — GPU-hardware-blocked, never measured.
3. ColBERT retrieval — not started.
4. Render's `/search`/`/ask` — 500, needs a built index baked into the
   image or `QDRANT_CLOUD_URL` + managed Postgres wired up for real.
5. `drift_monitor.py`/`feedback_aggregator.py` — unit-tested, never run
   against a live Airflow scheduler.
6. Slack alert wiring for those two DAGs' `alert_fn` — not connected.
7. A joint RAGAS + citation-judge run over the same query set — the
   final eval report's §5 gap, not just a §6 aside.
