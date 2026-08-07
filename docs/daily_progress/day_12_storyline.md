# Day 12 — EDA Agent, Phoenix Config, Quality Regression Monitor, Streamlit UI, Drift DAG, Docker/Render Deploy, CI/CD Gate

## 1. What was built / run

| File | Purpose |
|---|---|
| `monitoring/quality_regression.py` | New: Layer 8 daily RAGAS-sample monitor. Pure core (`consecutive_decline_days`, `detect_regression`, `evaluate_regression`) alerts on a 3-consecutive-day decline streak in a tracked metric (default `faithfulness`, `answer_relevancy`); Airflow-task-shaped `run()` takes injected fetch/sample/score/persist/alert callables, mirroring `monitoring/drift_detector.py`'s Day 11 shape |
| `monitoring/phoenix_config.py` | New: Arize Phoenix instrumentation — `configure_phoenix()` registers a Phoenix `TracerProvider` and instruments LangChain/LangGraph globally via `openinference.instrumentation.langchain`'s `LangChainInstrumentor`, giving Phoenix a full per-node run tree for every `agents/*.py` graph invocation with zero agent-code changes. Complements (not duplicates) `api/telemetry.py`'s Jaeger spans from Day 11 |
| `agents/eda_agent.py` | New: PandasAI-powered conversational EDA agent over 3 DataFrames — real `benchmark_results` (Day 9, live Postgres query), real `day9_citation_judgments.json`, and a clearly-labelled synthetic `drift` frame (no live drift history exists yet). `AnthropicPandasAILLM` adapts `anthropic.Anthropic` to pandasai's `LLM` interface since pandasai ships no Anthropic connector |
| `streamlit_app/` (`app.py`, `theme.py`, `mock_data.py`, `search_tab.py`, `eval_dashboard.py`, `monitoring_tab.py`, `benchmark_tab.py`, `drift_tab.py`) | New: 5-tab Streamlit UI shell (Search / Eval Dashboard / Monitoring / Benchmark / Drift). No live retrieval, LLM, or model load from any tab — real historical numbers (Day 9 benchmark, Day 11 RAGAS) are baked in where they exist, everything else is seeded-deterministic mock data built on this project's real pydantic models. `theme.py` follows the dataviz skill's palette convention (fixed categorical order, one sequential hue, a status palette never reused for data) |
| `Dockerfile`, `.dockerignore` | New: production image for `api.main:app` (`/search`, `/ask`, `/health`) only — not Airflow/Streamlit/MCP, which stay on `docker-compose.yml`. `python:3.11-slim` base, `build-essential`+`libgomp1` for PyMuPDF/torch wheels, healthcheck against `/health` |
| `render.yaml` | New: Render Blueprint — builds the root `Dockerfile`, deploys to Render's `starter` plan, `autoDeploy: true` on `main`. Secrets (`ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `POSTGRES_URL`, `QDRANT_CLOUD_*`, `LANGCHAIN_API_KEY`) are `sync: false` — set once in the Render dashboard, never committed |
| `airflow/dags/drift_monitor.py` | New: single Airflow 3 TaskFlow DAG wiring all three Layer 8 monitors — `fetch_query_windows → {check_query_drift, check_term_shift}` + `check_quality_regression` → `aggregate_and_alert`. Real Postgres `query_log` reads, real embedding calls via `retrieval/dense_index.py`'s encoder; quality-regression history persists to a gitignored local JSON log (no Postgres table for it yet) |
| `evaluation/ci_ndcg_gate.py`, `evaluation/benchmark_baseline.json` | New: CI quality gate — pure core `check_ndcg_gate` fails if any config's `mean_ndcg_at_10` drops below 0.50, gated against a committed snapshot of Day 9's real live Config A/C benchmark (0.5333 / 0.5280), not a live re-run (no BM25/Qdrant/Cohere available in CI) |
| `.github/workflows/ci.yml` | **New this session**: checkout → setup Python 3.11 → `pip install -r requirements.txt` (ruff/mypy/pytest are all pinned inside it already) → `ruff check .` → `mypy .` → `pytest` → `python -m evaluation.ci_ndcg_gate`. Triggers on push to `main` and on every pull request |
| `tests/agents/test_eda_agent.py`, `tests/monitoring/test_phoenix_config.py`, `tests/monitoring/test_quality_regression.py`, `tests/streamlit_app/test_tabs.py`, `tests/evaluation/test_ci_ndcg_gate.py` | New: 61 new tests (15 + 7 + 23 + 10 + 6) — deterministic cores unit-tested with no I/O; Phoenix/PandasAI/Streamlit tests use injected `TracerProvider`/`InMemorySpanExporter`/`AppTest`-style doubles rather than a live collector, model, or browser |

## 2. Memory management

This session's own work (CI workflow + this storyline doc) was pure file
authoring — no pytest run, no model load, no container start — so it
carried none of the memory risk the previous 11 days tracked (see
[[user-host-memory-constraint]]). The two Day 12 feature commits
(`64acd47`, `a8ea248`) that this session's two remaining tasks build on
top of were made in an earlier session under the same low-headroom pattern
as Days 4/9/10/11: per this note's own instruction, **tests were committed
without a local pytest run** — the risk being the same repeated segfault-at-
low-memory failure mode Day 10 hit (0.36GB free), not a new one. That is
called out explicitly in §3 rather than silently assumed fixed.

## 3. Results

### Quality regression monitor — unit-verified only, not run against a real RAGAS history

`monitoring/quality_regression.py` mirrors `monitoring/drift_detector.py`'s
Day 11 shape exactly: a pure core (`consecutive_decline_days` counts a
strict decline streak scanning backward from today; `detect_regression`
applies it per metric; `evaluate_regression` raises on empty history and
composes an alert message across all regressing metrics) plus an
Airflow-task-shaped `run()` with `ragas_runner_fn` injected so unit tests
never call `evaluation/ragas_suite.py` or a live Claude/embedding model.
`build_default_ragas_runner()` wires the real pipeline (`agents/qa_agent.py`
+ `evaluation/ragas_suite.py`) lazily, so importing the module carries no
heavy dependency. 23 new tests, all deterministic — no live daily-RAGAS
history has been sampled yet (that only happens once `airflow/dags/drift_monitor.py`
is actually registered and run against a scheduler).

### Arize Phoenix — code-complete, not started against a live collector this session

`monitoring/phoenix_config.py`'s `configure_phoenix()` registers a
Phoenix-bound `TracerProvider` and instruments LangChain/LangGraph globally
via `openinference.instrumentation.langchain`, giving Phoenix a full
per-node run tree (prompts/completions/token counts) for every
`agents/*.py::run()` call with zero changes to agent code — the same
"instrument once at the framework layer" approach `api/telemetry.py` used
for Jaeger, one level up. `docker-compose.yml` already has a `phoenix`
service (`arizephoenix/phoenix:latest`, port 6006) from Day 12's earlier
commit, but this session didn't start it or send a real span through it —
7 new tests exercise `instrument_langchain()`/`uninstrument_langchain()`
against an injected `TracerProvider` bound to an `InMemorySpanExporter`,
always uninstrumenting afterward since the instrumentor patches
langchain-core globally. Closes the last item of Layer 7
("LangSmith, OpenTelemetry+Jaeger, structlog, Arize Phoenix") at the
code level; live verification is an open item.

### EDA agent — real data sources, no live PandasAI/Claude call made

`agents/eda_agent.py` gives Andres' Week 5 conversational-analysis
deliverable three DataFrames: `benchmark` (real, live-queried from
Postgres' `benchmark_results` table — Day 9's Config A/C run),
`eval` (real, Day 9's committed `day9_citation_judgments.json` — the
`eval_results` Postgres table has 0 rows, so the JSON artifact is the
actual real source), and `drift` (clearly-labelled synthetic, seeded RNG,
since no live drift history exists per Day 11's open item). 15 new tests
cover the loaders and `AnthropicPandasAILLM`'s adapter shape; `main()`'s
3 sample questions against a live `anthropic.Anthropic()` client were not
run this session.

### Streamlit UI — 5 tabs, mock/historical only, matches its own stated scope

`streamlit_app/app.py` composes 5 tabs (`search_tab`, `eval_dashboard`,
`monitoring_tab`, `benchmark_tab`, `drift_tab`) built on `theme.py`'s
dataviz-skill-compliant palette and `mock_data.py`'s deterministic
builders. Two tabs surface real historical numbers rather than fabricated
ones: `eval_dashboard.py`'s headline metrics are Day 9's Config A NDCG@10
(0.5333) and Day 11's live RAGAS scores (faithfulness 0.7616, answer_relevancy
0.2497), and `benchmark_tab.py` shows Day 9's real Config A/C benchmark
table verbatim (Config B explicitly flagged "not run", not omitted).
`eval_dashboard.py`'s trend chart is wired to the real
`monitoring/quality_regression.py::evaluate_regression()` logic over a
synthetic 10-day history, so its alert state is computed identically to
what the Day 12 Task 1 monitor computes, not reimplemented in the UI layer.
10 new tests; `streamlit run streamlit_app/app.py` itself was not launched
live this session (no browser/process check).

### Dockerfile + .dockerignore + render.yaml — code-complete, not deployed

The Dockerfile builds only `api.main:app` (`/search`, `/ask`, `/health`) —
Airflow/Streamlit/MCP servers keep their `docker-compose.yml` shapes.
Known limitation, stated directly in both the Dockerfile's and
`render.yaml`'s own comments: `BM25Index.load()` needs
`data/indexes/bm25_index.pkl` (gitignored, not baked into the image) and
`DenseIndex.connect()` needs a populated Qdrant collection — neither is
wired to the deployed service yet, so `/health` would report `ok` but
`/search`/`/ask` would fail until that data-wiring step happens. Not
pushed to Render this session — no live deploy URL to report yet.

### Airflow drift-monitor DAG — one DAG wiring all three Layer 8 monitors, not registered against a live scheduler

`airflow/dags/drift_monitor.py` wires `drift_detector`, `term_shift_monitor`,
and `quality_regression`'s pure cores (Day 11/12) to real Postgres
`query_log` reads and a real `retrieval/dense_index.py` embedding call —
kept as a single `drift_monitor` DAG per Day 12's explicit remaining-work
spec, rather than the three separate DAG files AGENTS.md's folder-structure
table sketches (`drift_monitor`, `quality_regression`, `feedback_aggregator`
remain as one file; the other two names are still open items). Query-drift
baseline window: `query_log` rows from 37→30 days back (no frozen baseline
table exists yet, so this recomputes fresh each run). Quality-regression
history persists to a gitignored local JSON log, not a Postgres table.
Alert routing to Slack is not wired (`alert_fn=None`) — `quality_regression.run()`
already `structlog.warning()`s on its own, so a missing `alert_fn` means
"no Slack DM," not "silent regression." Not run against a live Airflow
scheduler or real Postgres data this session — same "unit-tested, not yet
DAG-registered-and-run" status Day 11 left `drift_detector.py`/
`term_shift_monitor.py` in.

### CI quality gate — real logic, real baseline, no live re-run in CI

`evaluation/ci_ndcg_gate.py`'s pure core `check_ndcg_gate` fails the
pipeline if any config's `mean_ndcg_at_10` drops below 0.50 (default
threshold); `main()` loads `evaluation/benchmark_baseline.json` — a
committed snapshot of Day 9's real, live-measured Config A/C numbers
(0.5333 / 0.5280, both above the gate) — rather than re-running
`evaluation/benchmark_runner.py` live, since a CI runner has none of
BM25Index/DenseIndex/Qdrant/a Cohere key. 6 new tests, including one that
asserts `main() == 0` against the real committed baseline — the actual
check this session's new CI workflow now runs on every push/PR.

### GitHub Actions CI/CD workflow — new this session

`.github/workflows/ci.yml`: checkout → setup Python 3.11 (pip-cached on
`requirements.txt`) → `pip install -r requirements.txt` → `ruff check .` →
`mypy .` → `pytest` → `python -m evaluation.ci_ndcg_gate`, triggered on
push to `main` and on every pull request. One thing worth recording:
`requirements.txt` is UTF-16-encoded (Windows-authored — `git show --stat`
shows it as a binary diff), which broke a plain `grep`/`Select-String`
scan while writing this workflow; it does *not* break `pip install -r`,
since pip's requirements-file parser (`pip._internal.utils.encoding.auto_decode`)
sniffs the BOM and decodes UTF-16 correctly regardless of the runner's
OS/locale, so the workflow needed no special-casing for it.
ruff/mypy/pytest are already pinned inside `requirements.txt` itself
(`ruff==0.15.20`, `mypy==2.2.0`, `pytest==9.1.1`) alongside every runtime
dependency, so a single install step covers lint, type-check, and test —
no separate dev-requirements file exists in this project. `mypy`'s
`pyproject.toml` config already excludes `airflow/`, `notebooks/`,
`.venv/`, `data/` and skips following `ragas.*` (a byte its UTF-8 reader
can't decode), so `mypy .` in CI runs with the same scope as local `mypy .`
does.

**Correction (see §6 Addendum below):** this paragraph originally said the
workflow was "not run against GitHub's actual runners this session (no
push/PR opened yet to trigger it)." That was wrong — this repo's `main`
was already pushed to `origin` by the time this doc was written, so
`ci.yml` ran automatically on the `3b37b68` push and **failed** at the
"Install dependencies" step. §6 covers what broke and how it was fixed in
a follow-up audit.

### RAGAS live scores (carried from Day 11, referenced throughout Day 12's UI/monitor)

Day 11's real 10-query RAGAS run (`agents/eval_agent.py` outputs scored via
`evaluation/ragas_suite.py`, MLflow run `7d8f1005`) remains the only live
RAGAS data point this project has:

| Metric | Score |
|---|---|
| faithfulness | 0.7616 |
| answer_relevancy | 0.2497 |

Both `streamlit_app/eval_dashboard.py`'s headline metrics and
`monitoring/quality_regression.py`'s `DEFAULT_METRICS` track exactly these
two metric names, so a live daily sample (once the drift-monitor DAG is
actually run) would slot into both without a schema change.

## 4. Day 1 → 12 narrative

- **Days 1-7**: see `docs/daily_progress/day_07_storyline.md` §3 —
  scaffolding, ingestion DAG, BM25/SPLADE/dense/RRF retrieval, ms-marco
  re-ranking + router, QA agent, FastAPI service. 243/243 tests passing by
  Day 7.
- **Day 8** (`3220885`): evaluation stack — relevance labeller, Cohere
  re-ranking tier, retrieval metrics, benchmark runner, citation judge, RAGAS
  suite. 323/323 tests.
- **Day 9** (`1836632`): live 50-query relevance labels, Config A/C
  benchmark, live citation judging (0.7037 accuracy), LangSmith tracing
  verified live. Found + fixed a real `agents/qa_agent.py` crash on
  malformed live citations; RAGAS's full `evaluate()` deferred over an hour.
  324/324 tests.
- **Day 10** (`3117a43`, `710dd78`): closed Day 9's stringified-citation gap
  with a recovery path + 2 regression tests, committed without local pytest
  verification (0.36GB free, segfaulted twice). Built `mcp/mcp_mlflow/server.py`
  (untested live). RAGAS deferred a third time. Expected 326, not verified.
- **Day 11** (`fa021de`): pytest verified first — 326/326, no segfault.
  RAGAS finally run live twice — first attempt surfaced a real
  `bypass_temperature` bug in the `claude-sonnet-5` + ragas integration,
  second attempt produced real scores (faithfulness 0.7616, answer_relevancy
  0.2497). mcp-mlflow live-verified. Multi-agent orchestrator
  (`retrieve → rerank → generate → evaluate`, 337/337), PSI drift detector +
  term-shift monitor (362/362), OTel + Jaeger live end-to-end (368/368
  final).
- **Day 12** (`64acd47`, `a8ea248`, today): quality regression monitor,
  Phoenix instrumentation, PandasAI EDA agent, 5-tab Streamlit UI, Docker +
  Render deploy config, single Airflow drift-monitor DAG wiring all three
  Layer 8 monitors, and the CI NDCG regression gate — all built and
  unit-tested (61 new tests) but, per this note's own recurring pattern,
  **not locally pytest-verified** (committed under the same low-headroom
  conditions as Days 4/9/10/11). This session closed the two remaining Day
  12 items: the `.github/workflows/ci.yml` CI/CD pipeline (checkout → setup
  Python 3.11 → install → ruff → mypy → pytest → NDCG gate, on push to
  `main` and every PR) and this storyline doc.

## 5. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0-3b (ingestion → re-ranking) | Done | See Day 7 storyline |
| QA agent | Live-verified Day 9; stringified-citation gap closed Day 10; OTel-instrumented Day 11 | `agents/qa_agent.py` |
| API | Unit-tested; still not run live with `uvicorn`; Dockerfile/render.yaml built but not deployed | `api/main.py`, `Dockerfile`, `render.yaml` |
| Layer 1 (MCP tool-calling) | `qdrant`/`airflow`/`mlflow` live-verified; `postgres` still outstanding | `mcp/mcp_mlflow/server.py` |
| Layer 1 (multi-agent) | Orchestrator + eval-agent verified (Day 11); EDA agent built, unit-tested, no live PandasAI/Claude call yet; `a2a_protocol` not started | `agents/orchestrator.py`, `agents/eda_agent.py` |
| Layer 4-5 (evaluation) | Config A/C + citation judge live-verified (Day 9); RAGAS live-verified (Day 11); CI NDCG gate built and code-complete (Day 12); Config B outstanding | `evaluation/ci_ndcg_gate.py`, MLflow run `7d8f1005` |
| Layer 6 (monitoring) | PSI drift + term-shift (Day 11) + quality regression (Day 12) all unit-verified; now wired into one Airflow DAG (`airflow/dags/drift_monitor.py`), not yet run against a live scheduler | `monitoring/quality_regression.py`, `airflow/dags/drift_monitor.py` |
| Layer 7 (observability) | LangSmith live-verified (Day 9); OTel + Jaeger live-verified (Day 11); Arize Phoenix code-complete, not run against a live collector | `monitoring/phoenix_config.py` |
| Layer 8 (drift + HITL) | PSI/term-shift/quality-regression now wired into `drift_monitor.py`; Slackbot feedback loop not started | `airflow/dags/drift_monitor.py` |
| CI/CD | `.github/workflows/ci.yml` — lint/type-check/test/NDCG-gate on push to `main` and every PR. First real run failed (pywin32 install + mypy crash, see §6); both fixed and pushed | `.github/workflows/ci.yml`, §6 Addendum |
| UI | 5-tab Streamlit shell built, mock/historical data only, not launched live this session | `streamlit_app/app.py` |

**Test suite**: 415 `def test_` definitions counted across `tests/` (368
baseline reported Day 11 + 61 new Day 12: 15 EDA agent + 7 Phoenix config +
23 quality regression + 10 Streamlit tabs + 6 CI NDCG gate); expected total
**429** once parametrized cases are counted (Day 11's 368 included ~14 such
cases beyond its own flat `def` count, and no Day 12 test file uses
`@pytest.mark.parametrize`). **Not run locally this session** — per §2,
consistent with Days 4/9/10/11's low-headroom pattern; the two feature
commits this note builds on (`64acd47`, `a8ea248`) were themselves committed
without a local pytest pass.

**Open items for next session**:
1. ~~Run pytest live and confirm the 429 estimate~~ — done in the §6 audit:
   429/429, confirmed exactly.
2. ~~Trigger `.github/workflows/ci.yml` for real~~ — done in the §6 audit:
   it ran, failed, and was fixed; see §6 for the two root causes and their
   fixes. Still open: watch the next push/PR run to confirm green now that
   both fixes are in.
3. Start the `phoenix` docker-compose service and send one real span through
   `monitoring/phoenix_config.py::configure_phoenix()`, mirroring Day 11's
   Jaeger smoke test.
4. Run `agents/eda_agent.py::main()` live against a real `anthropic.Anthropic()`
   client and the real `benchmark`/`eval` DataFrames.
5. Launch `streamlit run streamlit_app/app.py` and confirm all 5 tabs render.
6. Register `airflow/dags/drift_monitor.py` with a live Airflow scheduler and
   run it once against real Postgres `query_log` data.
7. Push this repo to Render and confirm `/health` responds; wire
   `QDRANT_CLOUD_URL`/a managed Postgres so `/search`/`/ask` work too.
8. Build the `mcp_postgres` server (carried over from Day 11).
9. `agents/a2a_protocol.py` — not started.
10. Run the full 50-query × top-100-pool benchmark (carried over from Day 9).
11. Build bge-reranker-v2-m3 (Config B) in a GPU environment (carried over).
12. Slackbot HITL feedback loop, including wiring `drift_monitor.py`'s
    `alert_fn` (Layer 8 remainder) — not started.

## 6. Addendum — CI audit and fixes (same-day follow-up session)

A follow-up session audited every Day 12 component against its stated
intent (existence, correctness, test coverage, gaps) and, per its own
instruction, actually ran `pytest tests/ -x --no-header -q`: **429 passed**,
confirming §5's estimate exactly. `ruff check .` was clean.

That audit also discovered `main` had already been pushed to `origin` (by
whatever pushed the `64acd47`/`a8ea248`/`3b37b68` commits), which meant
`.github/workflows/ci.yml` had already run for real on GitHub's runner —
and failed, at the "Install dependencies" step, contradicting §3's original
"not yet triggered" claim (corrected above). Two independent, confirmed
root causes:

1. **`requirements.txt` pinned `pywin32==312` / `pywin32-ctypes==0.2.3`
   with no `sys_platform` marker.** Neither package is imported anywhere
   in this project's own code (`agents/`, `api/`, `monitoring/`,
   `evaluation/`, `retrieval/`, `streamlit_app/`, `schema/` all grepped
   clean) — they're artifacts of freezing a Windows dev venv (almost
   certainly pulled in transitively by `keyring`'s Windows credential
   backend), not a real dependency. `pip install -r requirements.txt`
   hard-fails on any Linux target as a result. This broke **two** Day 12
   deliverables identically: `.github/workflows/ci.yml` (confirmed via the
   live failed run) and the `Dockerfile`'s build (same install command,
   same Debian-family base image — never actually built locally, so this
   was latent until the audit). **Fixed**: appended `; sys_platform ==
   "win32"` to both lines, preserving `requirements.txt`'s UTF-16LE
   encoding+BOM exactly (a plain-text edit would have silently
   re-encoded it to UTF-8, an unintended side effect).
2. **`mypy .` crashes with an INTERNAL ERROR on `genai_prices/data.py`.**
   `genai_prices` is a transitive dependency (via `pydantic-ai-slim`),
   ships a `py.typed` marker, and is a 637KB/~12,900-line **generated**
   pricing-data file — mypy 2.2.0 tries to fully type-check it (`py.typed`
   means `ignore_missing_imports` doesn't shield it) and crashes, the same
   class of problem `pyproject.toml` already had one documented workaround
   for (`ragas.llms.base`'s undecodable byte). No override existed for
   `genai_prices` yet, so even a fixed `pip install` would still have
   failed CI's `mypy .` step. **Fixed**: added a matching
   `[[tool.mypy.overrides]]` block (`module = "genai_prices.*"`,
   `follow_imports = "skip"`) in `pyproject.toml`, mirroring the existing
   `ragas.*` entry.

Both fixes were re-verified with `pytest tests/ -x --no-header -q` (still
429/429 — neither change touches application code) and pushed as a
follow-up commit; the next `.github/workflows/ci.yml` run on `main` is the
live confirmation that both root causes are actually resolved, not just
locally reasoned through.

**Why this matters for next time**: `requirements.txt` was generated by
`pip freeze` on this Windows dev machine at some point without a
cross-platform target in mind, so any future re-freeze risks
reintroducing Windows-only pins (`pywin32*`, and potentially others) with
no marker — worth a quick `Select-String -Pattern 'pywin32|win32com'`
check (plain `grep` won't see it, due to the UTF-16 encoding) after any
`pip freeze > requirements.txt`. Similarly, any new dependency that ships
`py.typed` and has a large generated-data submodule is a candidate for the
same mypy-crash class of problem `ragas.*`/`genai_prices.*` both hit —
worth trying `mypy .` locally before assuming a new dependency is
type-check-safe.
