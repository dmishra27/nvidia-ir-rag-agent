# Day 10 — Q10 Citation Fix Committed, mcp-mlflow Server, RAGAS Deferred Again

## 1. What was built / run

| File | Purpose |
|---|---|
| `agents/qa_agent.py` (+31 lines) | `generate` node now recovers citations sent as a stringified JSON array (Q10 case from Day 9 §4), instead of just skipping them |
| `tests/agents/test_qa_agent.py` (+2 tests) | Regression coverage: successful recovery, and unparseable-string fallback to `[]` without crashing |
| `mcp/mcp_mlflow/server.py` | New MCP server: `list_experiments`, `get_benchmark_experiment` — thin `MlflowClient` wrapper, no model loading |
| `mcp/mcp_mlflow/__init__.py` | Package marker (empty, matches `mcp_qdrant`/`mcp_airflow` convention) |
| `.mcp.json` | Registered `nvidia-ir-mlflow` stdio server alongside `nvidia-ir-qdrant`/`nvidia-ir-airflow` |

Commit: `3117a43` — `fix: qa_agent defensive citation parsing for stringified JSON variant + regression test`.

## 2. Memory management

Free memory at session start: **0.4GB of 7.65GB**, dropping to **0.36GB**
after a few minutes — no single runaway process (Docker Desktop, the WSL2
VM, and several Chrome/Edge processes each held 50-300MB; nothing left over
from a prior session). `pytest` segfaulted twice: once on full-suite
`--collect-only`, once on `tests/agents/test_qa_agent.py` alone. Both times
the only import in common was `agents/qa_agent.py` → `retrieval/dense_index.py`,
which pulls in `sentence-transformers`/`torch` at module level even though
this fix doesn't touch retrieval.

Per user decision, three scoping calls were made instead of trying to force
the memory issue:
1. The Q10 fix was **committed on the strength of manual diff review**,
   not a local test run — the commit message says so explicitly rather than
   implying `pytest` was green.
2. **RAGAS was skipped again this session.** `evaluation/ragas_suite.py`
   imports `agents.qa_agent` at module level (same segfaulting chain), and
   Day 9 §3 already showed a full RAGAS `evaluate()` run takes over an hour
   even when memory is healthy. Deferred to a session where memory has
   actual headroom, not attempted at 0.36GB free.
3. **Day 10 scope was cut to memory-safe work only** — a pure-Python MCP
   server (`mlflow` client, no model loading) plus this doc. OTel/Jaeger/Phoenix
   instrumentation and the multi-agent orchestrator (Layers 6-8) need new
   Docker containers or heavier imports and are pushed to Day 11.

`mcp/mcp_mlflow/server.py` was **not live-invoked** this session (no MCP
host round-trip, no `mlflow.tracking.MlflowClient` call against
`http://localhost:5000`) — written to match `mcp/mcp_airflow/server.py`'s
established pattern exactly (stderr-only structlog, lazy client singleton,
`FastMCP` tool decorators) and registered in `.mcp.json`, but functional
verification is carried to next session alongside the rest of the deferred
live work.

## 3. Results

### Q10 citation-parsing fix

Day 9 §4 left the stringified-whole-array case (`citations` arriving as a
JSON *string* rather than a native list) as a known, deliberately-unfixed
gap — safely skipped, not recovered. This session closes that gap:
`make_generate_node`'s `generate` closure now does
`isinstance(raw_citations, str)` → `json.loads` → use the parsed list if it's
actually a list, else log `generate_citations_stringified_unparseable` and
fall back to `[]`. Two new tests
(`test_recovers_citations_sent_as_a_stringified_json_array`,
`test_unparseable_stringified_citations_falls_back_to_empty_without_crashing`)
cover both branches against the existing `_tool_response` mock helper.

**Test count**: not live-verified this session (see §2). Day 9 closed at
324/324; this diff adds 2 tests, so the expectation carried forward is
**326**, pending a real `pytest` run once memory allows.

### mcp-mlflow MCP server

`list_experiments()` calls `MlflowClient.search_experiments()`;
`get_benchmark_experiment(experiment_name="reranker_benchmark", limit=20)`
resolves the experiment by name, then `search_runs()` ordered by
`start_time DESC`, returning each run's `run_id`/`status`/`metrics`/`params`.
Exists to make Day 9's `reranker_benchmark` and `citation_judge` MLflow
experiments queryable from a Claude session without opening the MLflow UI —
same motivation as `mcp_airflow` for DAG status. Untested live this session
(§2); next session should confirm it actually returns Day 9's `a55012a4`/
`c827cd71`/`db983987` runs once `deploy-mlflow-1` is up and memory allows a
client call.

## 4. Day 1 → 10 narrative

- **Days 1-7**: see `docs/daily_progress/day_07_storyline.md` §3 — scaffolding,
  ingestion DAG, BM25/SPLADE/dense/RRF retrieval, ms-marco re-ranking +
  router, QA agent, FastAPI service. 243/243 tests passing by Day 7.
- **Day 8** (`3220885`): evaluation stack built — relevance labeller, Cohere
  re-ranking tier, retrieval metrics, benchmark runner, citation judge, RAGAS
  suite. 323/323 tests passing. Live artifact generation deferred to low
  free memory.
- **Day 9** (`1836632`): live 50-query relevance labels, Config A/C
  benchmark, live citation judging (0.7037 accuracy), LangSmith tracing
  verified live. Found + fixed a real `agents/qa_agent.py` crash on
  malformed live citations; the stringified-array sub-case was left as a
  known gap. RAGAS's full `evaluate()` deferred after running over an hour.
  324/324 tests passing.
- **Day 10** (today, `3117a43`): closed Day 9's stringified-citation gap
  with a recovery path + 2 regression tests. RAGAS deferred a second time —
  memory was worse than Day 9 at points (0.36GB vs 0.51GB), and the same
  `agents.qa_agent` import chain that segfaulted `pytest` also sits at the
  top of `ragas_suite.py`. Built `mcp/mcp_mlflow/server.py` (untested live)
  so Day 9's benchmark/citation-judge MLflow data is queryable going
  forward. OTel/Jaeger/Phoenix and the multi-agent orchestrator pushed to
  Day 11.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0-3b (ingestion → re-ranking) | Done | See Day 7 storyline |
| QA agent | Live-verified Day 9; Q10 stringified-citation gap closed today (not yet live-verified) | `agents/qa_agent.py`, `3117a43` |
| API | Unit-tested; still not run live with `uvicorn` | Carried over from Day 7 |
| Layer 4-5 (evaluation) | Config A/C + citation judge live-verified (Day 9); RAGAS deferred twice; Config B outstanding | `benchmark_results`, MLflow `reranker_benchmark`/`citation_judge` |
| Layer 1 (MCP tool-calling) | 3 of 4 servers built (`qdrant`, `airflow`, now `mlflow`); `postgres` still outstanding | `mcp/mcp_mlflow/server.py`, `.mcp.json` |
| Layers 6-8 (monitoring, observability, drift/HITL) | LangSmith tracing live-verified (Day 9); OTel/Jaeger/Phoenix, orchestrator, drift/HITL not started | Pushed to Day 11 |

**Test suite**: 324/324 as of Day 9; +2 today, not live-verified (see §2/§3).

**Open items for next session**:
1. Run a real `pytest` pass once free memory has actual headroom, to
   confirm the Q10 fix's 2 new tests pass and get a live total.
2. Run `run_day9_ragas.py`'s full RAGAS `evaluate()` call — deferred three
   times running (Day 8, Day 9, Day 10) on memory grounds.
3. Live-invoke `mcp/mcp_mlflow/server.py`'s two tools against
   `deploy-mlflow-1` to confirm they return Day 9's real run data.
4. Build the `mcp_postgres` server (AGENTS.md lists it; only `qdrant`/
   `airflow`/`mlflow` exist so far).
5. OTel + Jaeger + Arize Phoenix instrumentation (Layer 7 remainder).
6. Multi-agent orchestrator / `a2a_protocol` (Layer 1 remainder).
7. Run the full 50-query × top-100-pool benchmark (carried over from Day 9).
8. Build bge-reranker-v2-m3 (Config B) in a GPU environment (carried over).
9. Run `uvicorn api.main:app` live (carried over from Day 7).
