# Day 11 — RAGAS Live (Real Bug Fixed), Multi-Agent Orchestrator, Drift/Term-Shift Monitors, OTel + Jaeger Live

## 1. What was built / run

| File | Purpose |
|---|---|
| `evaluation/ragas_suite.py` (+1 arg) | `build_default_llm()` now passes `bypass_temperature=True` to `LangchainLLMWrapper` — fixes a real bug, see §3 |
| `agents/eval_agent.py` | New: `EvalState` (QAState superset) + `make_evaluate_node`, wraps `evaluation/citation_judge.py` as a LangGraph node |
| `agents/orchestrator.py` | New: composes `retrieval_agent`'s retrieve/rerank, `qa_agent`'s generate, `eval_agent`'s evaluate into one graph — `retrieve → rerank → generate → evaluate` |
| `tests/agents/test_eval_agent.py`, `test_orchestrator.py` | 11 new contract tests, fully mocked (fake indexes/router, `MagicMock` Anthropic clients) |
| `monitoring/drift_detector.py` | New: `population_stability_index` (pure numpy PSI), `compute_drift` (embedding-centroid projection), Airflow-task-shaped `run()` |
| `monitoring/term_shift_monitor.py` | New: `term_frequencies`, `term_shift` (BM25-tokenizer-matched), Airflow-task-shaped `run()` |
| `tests/monitoring/` | 25 new tests — deterministic PSI/term-frequency math, no I/O |
| `api/telemetry.py` | New: `configure_tracing()` (OTLP → Jaeger), `traced_stage()` context manager with injectable tracer |
| `agents/retrieval_agent.py`, `agents/qa_agent.py` | Instrumented: spans around BM25 search, dense search, RRF fusion, rerank, and the LLM call |
| `tests/api/test_telemetry.py` | 6 new tests against `InMemorySpanExporter`, no Jaeger dependency |
| `docker-compose.yml` | Added `jaeger` service (`jaegertracing/all-in-one:1.60`, OTLP enabled, ports 16686/4317/4318) |

## 2. Memory management

Free memory swung from **~430MB down to ~80-95MB and back** over the
session — never a single runaway process, just the same chronic low-headroom
pattern as Days 4/9/10. Handled it the same way each time:

1. **Before touching anything**, stopped `deploy-streamlit-1` and
   `deploy-api-1` (unrelated containers from other work eating headroom),
   keeping `deploy-mlflow-1` up for logging — per the standing instruction
   and [[user-host-memory-constraint]].
2. **Verified pytest first** (326/326, no segfault this time — better luck
   than Day 10's 0.36GB crash) before starting any new work.
3. When free memory bottomed out at ~80MB right after the first RAGAS run,
   **no new heavy process was started** — waited for the RAGAS subprocess to
   fully exit (which released its memory) before running the next `pytest`
   pass, rather than forcing concurrent work.
4. **Jaeger was gated correctly**: the instruction's specific criterion was
   WSL2 free > 2GB (not Windows host free) after Tasks 1-4 — WSL2 sat at a
   comfortable ~4.7GB free the whole session (Docker's memory pool is
   separate from Windows host RAM), so Jaeger started rather than deferring,
   and its actual footprint after boot was ~10MB.
5. Every `pytest` run and every code edit stayed sequential — never two
   memory/CPU-heavy things at once.

## 3. Results

### RAGAS live run — real bug found and fixed, not just deferred again

Attempt 1 (`run_day9_ragas.py`, `bypass_temperature` not yet applied): all
20 RAGAS judge calls failed with
`invalid_request_error: temperature is deprecated for this model`. Traced
to `ragas.llms.LangchainLLMWrapper`: it overrides `temperature` per-call for
self-consistency sampling by default, but `claude-sonnet-5` rejects any
non-default `temperature` (extended thinking forces it fixed) — the same
class of constraint the library's own docstring calls out for OpenAI's o1
series, with a documented escape hatch (`bypass_temperature=True`) that just
wasn't wired up. Fixed in `evaluation/ragas_suite.py::build_default_llm()`.

Attempt 2 (same script, fix applied): completed in ~30s for the RAGAS
scoring step (vs. Attempt 1's 4m31s of retried failures). Real scores over
10 live QA-agent outputs:

| Metric | Score |
|---|---|
| faithfulness | 0.7616 |
| answer_relevancy | 0.2497 |

Logged to MLflow experiment `ragas_eval`, run `7d8f1005`. The low
`answer_relevancy` is a legitimate signal, not a bug: several of the 10
sampled QA outputs are correct refusals ("the provided passages do not
contain...") because Config A's retrieval didn't surface the right chunk —
RAGAS scores a non-answer as low-relevancy regardless of whether the
refusal itself was the right call, so this number is really flagging
retrieval coverage gaps for those queries, not generation quality.

This closes a gap that had been carried since Day 8 (deferred 3 times
running — Days 8, 9, 10 — each time on memory grounds without ever getting
far enough to hit this bug).

### Multi-agent orchestrator

`agents/eval_agent.py` fills a gap AGENTS.md's folder structure had
described but nothing had built yet: a standalone eval-agent module wrapping
`evaluation/citation_judge.py`'s LLM-as-judge as a LangGraph node (chosen
over wrapping `ragas_suite.py` directly — citation judging operates
per-query already, matching the orchestrator's single-query state flow,
where RAGAS's `Dataset`-batched design does not). `agents/orchestrator.py`
then composes all three agents' nodes — none rewritten, all reused as-is —
into `retrieve → rerank → generate → evaluate` over `EvalState`, a `QAState`
subclass. Verified: 337/337 (326 + 11 new) after this stage, all mocked, no
live index/Qdrant/cross-encoder/API call.

### PSI drift detector + term-shift monitor

Both follow the same shape: a pure, deterministic core function (fully
unit-tested, no model/I-O) plus an Airflow-task-shaped `run()` taking
injected fetch/embed callables, mirroring `agents/retrieval_agent.py`'s
constructor-injection convention. `population_stability_index` buckets on
baseline quantiles with epsilon-smoothing; `compute_drift` projects
embeddings to a scalar (cosine similarity to the baseline centroid) before
bucketing, since PSI is a 1-D-distribution statistic. `term_shift` reuses
`retrieval/bm25_index.py`'s tokenizer regex (duplicated locally, matching
the existing `chunk_quality.py` convention rather than importing a private
cross-module helper) so term stats bucket identically to what BM25 actually
indexes. Neither was run against real Postgres data or a real embedding
model this session — both are unit-verified only (25/25 new tests), with
real wiring (query_log reads, DenseIndex embedding calls) left to whenever
an actual Airflow DAG registration happens.

### OpenTelemetry + Jaeger — live, not deferred

WSL2 had 4.7GB free after Tasks 1-4, clearing the >2GB gate, so this ran the
full path rather than stopping at code-only. `api/telemetry.py`'s
`traced_stage()` context manager takes an injectable `tracer` (OTel's global
`TracerProvider` can only be set once per process, which breaks per-test
isolation — the first version of `tests/api/test_telemetry.py` hit exactly
this and needed a DI fix) and was wired into `agents/retrieval_agent.py`
and `agents/qa_agent.py` around all 5 named stages: `bm25`, `dense`, `rrf`,
`rerank`, `llm`. Added a `jaeger` service to `docker-compose.yml`
(`jaegertracing/all-in-one:1.60`, OTLP receiver enabled) and started it —
booted healthy at ~10MB RSS. Live end-to-end smoke test: called
`configure_tracing()` + `traced_stage("bm25", "smoke-test-q1", top_k=100)`,
force-flushed, and confirmed via Jaeger's HTTP API
(`/api/traces?service=nvidia-ir-rag-agent`) that the span landed with the
correct `query_id`/`stage`/`top_k` tags. Full suite re-verified afterward:
368/368, so instrumenting the two existing agent files broke nothing.

### mcp-mlflow — live-verified

Day 10 left this built but never invoked. This session called both tools
directly against `deploy-mlflow-1`: `list_experiments()` returned all 4 real
experiments (`ragas_eval`, `citation_judge`, `reranker_benchmark`,
`Default`); `get_benchmark_experiment("reranker_benchmark", limit=5)`
returned real Day 9 runs (`config_C_cohere_rerank` ×2, `config_A_ms_marco`)
with their actual logged metrics. Confirms the server built Day 10 works
end-to-end, not just structurally.

## 4. Day 1 → 11 narrative

- **Days 1-7**: see `docs/daily_progress/day_07_storyline.md` §3 — scaffolding,
  ingestion DAG, BM25/SPLADE/dense/RRF retrieval, ms-marco re-ranking +
  router, QA agent, FastAPI service. 243/243 tests passing by Day 7.
- **Day 8** (`3220885`): evaluation stack built — relevance labeller, Cohere
  re-ranking tier, retrieval metrics, benchmark runner, citation judge, RAGAS
  suite. 323/323 tests passing. Live artifact generation deferred.
- **Day 9** (`1836632`): live 50-query relevance labels, Config A/C
  benchmark, live citation judging (0.7037 accuracy), LangSmith tracing
  verified live. Found + fixed a real `agents/qa_agent.py` crash on
  malformed live citations; RAGAS's full `evaluate()` deferred after running
  over an hour. 324/324 tests.
- **Day 10** (`3117a43`, `710dd78`): closed Day 9's stringified-citation gap
  with a recovery path + 2 regression tests, committed without local pytest
  verification (0.36GB free, segfaulted twice). Built `mcp/mcp_mlflow/server.py`
  (untested live). RAGAS deferred a third time. Expected 326, not verified.
- **Day 11** (today): pytest verified first — 326/326, no segfault. RAGAS
  finally run live twice — first attempt surfaced a real `bypass_temperature`
  bug in the `claude-sonnet-5` + ragas integration, second attempt (after the
  one-line fix) produced real scores (faithfulness 0.7616, answer_relevancy
  0.2497). mcp-mlflow live-verified against real Day 9 MLflow data. Built
  and verified the multi-agent orchestrator (`retrieve → rerank → generate →
  evaluate`, 337/337), PSI drift detector + term-shift monitor (362/362), and
  OTel + Jaeger — instrumented 5 pipeline stages and confirmed a real span
  landing in a live Jaeger container (368/368 final).

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0-3b (ingestion → re-ranking) | Done | See Day 7 storyline |
| QA agent | Live-verified Day 9; Q10 stringified-citation gap closed Day 10; now OTel-instrumented | `agents/qa_agent.py` |
| API | Unit-tested; still not run live with `uvicorn` | Carried over from Day 7 |
| Layer 1 (MCP tool-calling) | 3 of 4 servers built and now all live-verified (`qdrant`, `airflow`, `mlflow`); `postgres` still outstanding | `mcp/mcp_mlflow/server.py` |
| Layer 1 (multi-agent) | Orchestrator built and verified (retrieve→rerank→generate→evaluate); `a2a_protocol` not started | `agents/orchestrator.py`, `agents/eval_agent.py` |
| Layer 4-5 (evaluation) | Config A/C + citation judge live-verified (Day 9); RAGAS now live-verified with real scores (Day 11); Config B outstanding | MLflow `ragas_eval` run `7d8f1005` |
| Layer 6 (monitoring) | PSI drift detector + term-shift monitor built, unit-verified; not yet wired to a live Airflow DAG or real Postgres/embedding data | `monitoring/drift_detector.py`, `monitoring/term_shift_monitor.py` |
| Layer 7 (observability) | LangSmith live-verified (Day 9); OTel + Jaeger now live-verified (5 stages instrumented, real span confirmed in Jaeger); Arize Phoenix not started | `api/telemetry.py`, Jaeger UI `localhost:16686` |
| Layer 8 (drift + HITL) | PSI/term-shift built this session (see Layer 6); Slackbot feedback loop not started | — |

**Test suite**: 368/368 (326 baseline + 11 orchestrator/eval-agent + 25
monitoring + 6 telemetry), live-verified this session.

**Open items for next session**:
1. Build the `mcp_postgres` server (only `qdrant`/`airflow`/`mlflow` exist).
2. Wire `monitoring/drift_detector.py` / `term_shift_monitor.py` to a real
   Airflow DAG (`airflow/dags/drift_monitor.py` doesn't exist yet) with real
   Postgres `query_log` reads and a real embedding call.
3. `agents/a2a_protocol.py` — not started.
4. Arize Phoenix instrumentation (Layer 7 remainder).
5. Run the full 50-query × top-100-pool benchmark (carried over from Day 9).
6. Build bge-reranker-v2-m3 (Config B) in a GPU environment (carried over).
7. Run `uvicorn api.main:app` live (carried over from Day 7).
8. Investigate the low `answer_relevancy` (0.2497) — sample more than 10
   queries, or specifically check whether it's dominated by retrieval-miss
   refusals vs. genuinely poor answers on queries with good context.
9. Slackbot HITL feedback loop (Layer 8 remainder) — not started.
