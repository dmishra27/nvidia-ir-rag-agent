# Day 13 — CI Green, Slackbot, HITL Feedback, Live Streamlit, README, Render Deploy

## 1. What was built / run

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml`, `Dockerfile` | Fixed: `--extra-index-url https://download.pytorch.org/whl/cpu` on the `pip install` line, per this session's Fix 1 spec |
| `requirements.txt` | Fixed: pyarrow, pillow, pdfplumber/pdfminer.six, openinference-semantic-conventions, datasets pins, and `langchain-google-vertexai` removed entirely — see §2 |
| `scripts/patch_ragas.py`, `requirements_notes.txt` | New: post-install patch removing ragas 0.4.3's one broken `ChatVertexAI` import; `requirements_notes.txt` is the file `AGENTS.md` has referenced since Day 1 and never existed until now — see §2 |
| `agents/qa_agent.py`, `agents/text_to_sql_agent.py`, `evaluation/citation_judge.py`, `evaluation/relevance_labeller.py` | Fixed: `anthropic.types` (`MessageParam`/`ToolParam`/`ToolChoiceToolParam`) instead of bare dict/list literals for `Messages.create()` calls — mypy strict debt, see §3 |
| `agents/qa_agent.py`, `agents/retrieval_agent.py`, `agents/orchestrator.py`, `monitoring/drift_detector.py`, `schema/models.py`, `retrieval/dense_index.py`, `retrieval/reranker_cohere.py`, `retrieval/biencoder_eval.py`, `retrieval/populate_qdrant.py`, `mcp/mcp_airflow/server.py`, `agents/eda_agent.py`, `evaluation/benchmark_runner.py`, `evaluation/ragas_suite.py` | Fixed: remaining mypy `--strict` errors (44 total) across every production module CI's mypy job touched for the first time ever — see §3 |
| `pyproject.toml` | Changed: `tests/` and root-level `run_*.py`/`verify_phase_c*.py` scripts scoped out of `mypy`'s file discovery (kept strict on the shipped package) — see §3 |
| `slackbot/app.py`, `slackbot/handlers.py`, `slackbot/feedback_handler.py`, `slackbot/__init__.py` | New: Slack Bolt app (Socket Mode), `/nvidia-search` slash command → `POST /ask` → Slack blocks with top-3 citations, and 👍/👎 reaction → `feedback_log` — see §4 |
| `schema/models.py` | New: `FeedbackLog` ORM model (`feedback_log` table already existed in `schema/schema.sql`, no model class yet) |
| `monitoring/feedback_aggregator.py`, `airflow/dags/feedback_aggregator.py` | New: weekly HITL feedback aggregation — pure core (`aggregate_feedback`) + Airflow-task-shaped `run()` in `monitoring/`, thin DAG wiring a real Postgres read + JSON history, mirroring `airflow/dags/drift_monitor.py`'s shape — see §5 |
| `streamlit_app/live_data.py` | New: live MLflow (`reranker_benchmark`/`citation_judge`/`ragas_eval`) + Postgres (`benchmark_results`) + JSON (`day9_citation_judgments.json`) fetchers |
| `streamlit_app/benchmark_tab.py`, `streamlit_app/eval_dashboard.py` | Changed: `mock_data.py` defaults replaced with `live_data.py` fetchers, each with a graceful last-known-values fallback; two new charts added — see §6 |
| `README.md`, `.env.example` | New: repo had no README at all before this session — see §7 |
| `tests/slackbot/`, `tests/monitoring/test_feedback_aggregator.py`, `tests/streamlit_app/test_live_data.py`, `tests/streamlit_app/test_tab_fallbacks.py` | New: 40 new tests (15 + 11 + 3 + 11) |

## 2. Fix 1 — torch CPU wheel turned out to be one of five stacked CI failures

The literal ask was a two-line diff: add `--extra-index-url
https://download.pytorch.org/whl/cpu` to `ci.yml` and `Dockerfile`'s `pip
install`. That line was correct and is still exactly what fixed the actual
torch error (`ERROR: Could not find a version that satisfies the
requirement torch==2.13.0+cpu` — pip's default index has no CPU-only
wheels for a CUDA-tagged torch version). But CI had never gotten past
`pip install` in its entire history (every previous run failed at that
exact step), so fixing it didn't turn CI green — it let pip's resolver run
far enough to hit four more pre-existing, previously-invisible problems,
one at a time, each requiring its own commit + push + watch cycle:

1. **`pyarrow==17.0.0` vs `langchain-google-vertexai==3.2.4`** (needs
   `pyarrow>=19.0.1,<24.0.0`) **vs `pandasai==3.0.0`** (needs
   `pyarrow<19.0.0,>=14.0.1`) — genuinely disjoint ranges, no version
   satisfies both. `langchain-google-vertexai` is never imported anywhere
   in this project's own code (grepped clean) and every version of it
   compatible with this project's `langchain-core==1.4.9` has the same
   `>=19.0.1` floor, so it was removed entirely rather than downgrading
   `pandasai` (a real, in-use dependency of `agents/eda_agent.py`).
   `pyarrow` went back to its original `17.0.0`.
2. **`openinference-semantic-conventions==0.1.30`** vs
   `openinference-instrumentation-langchain==0.1.70` (needs `>=0.1.31`) —
   bumped to `0.1.32` (latest available).
3. **`pillow==10.4.0`+`pdfplumber==0.11.10`** — `pdfplumber` 0.11.10
   tightened its own Pillow floor to `>=12.2.0`, conflicting with
   `pandasai`'s `pillow<11.0.0`. Downgraded to `pdfplumber==0.11.9` (the
   newest release still on the loose `Pillow>=9.1`) with its matching
   `pdfminer.six==20251230` pin (pdfplumber pins that exactly per-version).
4. **`datasets==2.21.0`** vs `ragas==0.4.3` (needs `datasets>=4.0.0`) —
   bumped to the minimum satisfying version, `4.0.0` (not latest, to keep
   the diff small); `evaluation/ragas_suite.py`'s only use of the package
   (`Dataset.from_dict`) is unaffected by the 2.x→4.x jump.
5. **`ragas.llms.base` unconditionally imports `ChatVertexAI` from
   `langchain_community.chat_models.vertexai`**, a submodule that doesn't
   exist in this project's pinned `langchain-community==0.4.2` (that
   integration moved to the standalone `langchain-google-vertexai`
   package — the same package #1 just removed for an unrelated reason).
   This one couldn't be fixed with a version pin: no `langchain-community`
   version both ships that submodule *and* satisfies everything else, and
   reintroducing `langchain-google-vertexai` reopens #1. `scripts/patch_ragas.py`
   patches the installed package after `pip install` (removes the one
   broken import line and its one usage), wired into `ci.yml` and
   `Dockerfile` right after the install step. This is the fix
   `requirements_notes.txt` documents — the file `AGENTS.md`'s "ragas:
   0.4.3 (patched — see requirements_notes.txt)" line has referenced since
   Day 1 and that never actually existed until now. (First version of the
   script tried `import ragas.llms` to locate the file to patch — which
   executes the exact broken import it exists to fix, failing identically.
   Fixed by locating the package via `importlib.util.find_spec("ragas")`,
   which resolves the path without running `__init__.py`.)

Each of these was found by watching the actual GitHub Actions run fail
(not guessed), fixed, pushed, and re-watched — six pushes total for what
started as a two-line ask. `uv pip compile --extra-index-url ... --index-strategy
unsafe-best-match` (mirrors pip's own resolution) was used locally after
the first two fixes to verify a full solution existed *before* pushing,
which caught nothing new but confirmed the fix set was complete before
spending a CI run on it.

## 3. mypy `--strict` ran end-to-end for the first time ever — 258 pre-existing errors

Once install/lint passed, `mypy .` ran in CI for the first time in this
project's history (every prior run died at install) and reported 258
errors across production code and `tests/`. This is pre-existing debt that
predates this session — CI's mypy job had simply never executed
successfully before, not even once, so nothing had ever verified the
`strict = true` in `pyproject.toml` was actually satisfied project-wide.

Per an explicit decision mid-session (asked rather than assumed, given the
scope jump from a 2-line torch fix to potentially hundreds of type-error
fixes): keep `--strict` on the actual shipped package
(`agents/`, `api/`, `retrieval/`, `monitoring/`, `evaluation/`, `schema/`,
`mcp/`) and fix its share of the debt (44 errors) for real; scope `tests/`
and the root-level `run_*.py`/`verify_phase_c*.py` one-off scripts out of
mypy's file discovery in `pyproject.toml`, matching the *already-existing*
`"run_*.py" = ["E402"]` ruff per-file-ignore's own reasoning ("not part of
the AGENTS.md package layout"). Test-file annotation debt is real but
explicitly deferred, not silently dropped.

The 44 production fixes, by category:

- **`anthropic.Messages.create()` calls** (`qa_agent`, `text_to_sql_agent`,
  `citation_judge`, `relevance_labeller`) — mypy's overload resolution
  can't structurally match a bare `dict`/`list` literal against the SDK's
  `ToolParam`/`MessageParam`/`ToolChoiceToolParam` TypedDicts when the
  target method is itself overloaded (a known category of mypy limitation
  with bidirectional inference through overloads). Fixed by typing the
  `tools`/`messages`/`tool_choice` locals explicitly with
  `anthropic.types`, not by suppressing the check.
- **`langgraph.StateGraph.add_node()` calls** (`qa_agent`,
  `retrieval_agent`, `orchestrator`) — mypy can't unify `add_node`'s
  `NodeInputT` TypeVar through its `StateNode` Union-of-10-Protocols
  parameter type for a plain `Callable[[State], State]`, even though each
  node's actual signature is correct (verified: the reported error code
  itself flips between `call-overload` and `arg-type` across otherwise-
  identical call sites, which is itself evidence of a resolver limitation,
  not a real type mismatch). Scoped `# type: ignore` with a comment
  explaining why, matching this project's existing precedent for exactly
  this situation (`pyproject.toml`'s `ragas.*`/`genai_prices.*` overrides).
- **`sys.stdout.reconfigure(...)`** (`eda_agent`, `benchmark_runner`,
  `ragas_suite`, `relevance_labeller`) — `sys.stdout`'s declared type
  (`TextIO`) doesn't have `.reconfigure`; guarded with
  `isinstance(sys.stdout, io.TextIOWrapper)` instead of assuming it.
- **`write_jsonl(path, records: list[BaseModel])`** — `list` is invariant,
  so `list[BenchmarkQuery]`/`list[RelevanceLabel]` callers didn't
  type-check against a `list[BaseModel]` parameter. Changed the parameter
  to `Sequence[BaseModel]` (covariant) — a real signature improvement, not
  a workaround.
- **`monitoring/drift_detector.py`** — bare `np.ndarray` needs a type
  argument under strict (`disallow_any_generics`); changed every signature
  to `npt.NDArray[np.float64]`.
- **`schema/models.py`** — `Column[list[str]]` annotation for the one
  `ARRAY(String)` column mypy couldn't infer a generic for on its own;
  `Engine`/`get_engine` return type; `get_session_factory`'s `engine`
  parameter type.
- **`retrieval/dense_index.py`** — `cast()` the intentionally-narrowed
  `QdrantSearchClient` Protocol at `DenseIndex.connect()` (the real
  `QdrantClient` is structurally far wider than the Protocol this project
  narrows it to specifically so unit tests can inject a fake without
  importing `qdrant_client`); `cast()` the `sentence-transformers`
  `.encode()` return; `cast(str, ...)` a payload's `chunk_id` *without* a
  `""` default. An earlier draft used `.get("chunk_id", "")` here and it
  broke `test_missing_payload_fields_default_to_none_and_empty_text` — this
  project's `Candidate` dataclass isn't runtime-validated, and downstream
  code relies on a missing `chunk_id` staying `None`, not silently
  becoming an empty string. Caught by the existing test suite before this
  was committed, not after — see §8.
- **`reranker_cohere`/`biencoder_eval`/`populate_qdrant`/`mcp_airflow`/
  `eda_agent`** — explicit return-type annotations/local-variable casts
  where a stub-untyped third-party call (`cohere`, `numpy.sum`, pandasai's
  `LLM` base class, `requests.Response.json()`) was flowing `Any` into a
  typed return; `pandasai`'s `LLM` base class itself resolves to `Any`
  (pandasai ships no type stubs), so subclassing it needs a scoped
  `# type: ignore[misc]` — the same "third-party library has no types"
  situation as the `add_node` case above, not a code issue.

Verified clean with a full local `mypy .` (`Success: no issues found in 59
source files`) before pushing, and confirmed green on the actual GitHub
Actions runner afterward — six pushes in, CI is now fully green: install →
patch ragas → ruff → mypy → pytest → NDCG regression gate, all passing.

## 4. Slack bot + HITL feedback

`slackbot/app.py` builds a `slack_bolt.App` in Socket Mode (no public URL
or request signing needed — `SLACK_BOT_TOKEN`/`SLACK_APP_TOKEN` were
already sitting in `.env`, so this was genuinely wired against real
workspace credentials, not placeholder tokens) and registers two handlers:

- **`slackbot/handlers.py`** — `/nvidia-search <question>` parses the text,
  calls `POST /ask` via a constructor-injected `AskClient` (so unit tests
  never open a real connection, per `AGENTS.md`'s mock-everything rule),
  and formats the response as Slack blocks: the answer, up to 3 citations
  (`claim` + `chunk_ids`), and a context footer carrying `query_id`.
- **`slackbot/feedback_handler.py`** — listens for `reaction_added`,
  filters to `thumbsup`/`thumbsdown` only, looks the `query_id` back out of
  the reacted-to message's own text (Slack reactions attach to a message,
  not to an application-level ID, so the footer `handlers.py` posts is
  what makes a 👍/👎 attributable to a query at all) via
  `conversations.history`, and writes one row to a new `FeedbackLog`
  SQLAlchemy ORM model (`schema/schema.sql`'s `feedback_log` table already
  existed from Day 1; no ORM class had been added for it yet) — no raw SQL
  anywhere, per `AGENTS.md`.

Per an explicit decision this session: built and unit-tested (15 tests,
Slack client/HTTP calls all mocked/faked) rather than smoke-tested against
the live workspace — the user runs `python -m slackbot.app` themselves
when ready to try it for real.

## 5. Feedback aggregator DAG

`monitoring/feedback_aggregator.py` is a pure, unit-tested core
(`aggregate_feedback`: total/up/down counts, down-rate, and the top-5
query_ids by thumbs-down count) plus an Airflow-task-shaped `run()` with
injected fetch/persist/alert callables — deliberately mirroring
`monitoring/quality_regression.py`'s exact shape from Day 12.
`airflow/dags/feedback_aggregator.py` is the thin DAG: a real Postgres read
of `FeedbackLog` (last 7 days) → `run()` → local JSON history persistence,
mirroring `airflow/dags/drift_monitor.py`'s already-established pattern
(project-root-on-`sys.path`, per-task `SessionFactory`, structlog). "Worth
alerting" requires *both* enough volume (`>=5` reactions — otherwise one
grumpy reactor on a quiet week isn't a signal) *and* a high down-rate
(`>=30%`), not either alone.

Not live-verified against a running Airflow scheduler this session — same
status Day 12's `drift_monitor.py` documented for itself. This project's
local dev venv deliberately doesn't install `apache-airflow` at all (`pip
show apache-airflow` confirms not found; `import airflow` "succeeding"
locally is actually Python treating this repo's own extension-less
`airflow/` directory as an empty implicit namespace package, not the real
one) — DAGs run inside `docker-compose.yml`'s `airflow` service image
(`apache/airflow:3.0.2`), not this project's `requirements.txt`. Because of
that, the real test coverage lives in `monitoring/feedback_aggregator.py`
(11 tests, zero I/O), and the DAG file itself is checked by matching
`drift_monitor.py`'s proven shape closely, not by a local DAG-parse run.

## 6. Streamlit wired to live data

`streamlit_app/live_data.py` adds five fetchers: `fetch_live_benchmark_summaries`
and `fetch_ragas_scores`/`fetch_citation_accuracy` query MLflow
(`reranker_benchmark`/`ragas_eval`/`citation_judge` experiments) via the
same `mlflow.tracking.MlflowClient` pattern `mcp/mcp_mlflow/server.py`
already established; `fetch_per_query_ndcg` queries Postgres'
`benchmark_results` table via the ORM; `load_citation_judgments` +
`citation_accuracy_by_query` read the real, committed
`evaluation/day9_citation_judgments.json` (27 judgments, 10 queries).
Every fetcher raises on failure rather than swallowing the error — the tab
modules decide the fallback, matching `benchmark_tab.py`'s pre-existing
`summaries_fn` injection design (its own Day 12 docstring already said "so
a real MCP-mlflow-backed fetch can replace it later behind the same
signature" — this is that swap).

`benchmark_tab.py` and `eval_dashboard.py` now default to the live
fetchers, each wrapped in a `_load_*` helper that falls back to
`mock_data.py`'s committed Day 9/11 numbers (the *same* real values, just
not live-queried) with a visible `st.info` banner on failure — verified
this degrades cleanly rather than crashing (both the live path, run
against this session's actually-running `docker-compose` Postgres/MLflow,
and the fallback path, unit-tested directly with always-raising injected
functions) so the tab still renders in CI, which has no live services at
all, per `AGENTS.md`.

Two new charts, both real live data, pushing the combined real-data chart
count across the two tabs to 5 (from 2 before this session):

- **`benchmark_tab.py`: per-query NDCG@10 box-plot** — all 45 real
  `benchmark_results` rows via a live Postgres read, not just the two
  existing MLflow-aggregate charts (quality bar chart, latency bar chart).
- **`eval_dashboard.py`: per-query citation-accuracy bar chart** — built
  from the real 27-judgment Day 9 file, alongside the existing headline
  metrics (now live-queried) and NDCG-vs-gate chart (also now live).

The quality-regression trend chart is the one piece left synthetic: no
real daily-quality history exists yet (`feedback_aggregator.py`'s DAG
sibling `drift_monitor.py`'s regression task has never run against a live
scheduler — see §5), so there's no real multi-day series to plot. It still
runs `monitoring/quality_regression.py`'s actual `evaluate_regression()`
logic, just over a clearly-labelled synthetic history, same as Day 12
shipped — the caption says so explicitly now.

**A note on where "live" MLflow data actually lives**: this project's
`docker-compose.yml` has no dedicated `mlflow` service (`MLFLOW_TRACKING_URI`
in `.env` has always just pointed at `http://localhost:5000` — "whatever's
running there," per `mcp/mcp_mlflow/server.py`'s own default). On this dev
machine, the container currently listening on port 5000 belongs to an
unrelated project's docker-compose config, not this repo's — but its
`reranker_benchmark`/`citation_judge`/`ragas_eval` experiments contain
exactly this project's real Day 9/11 numbers (verified: the metric values
match `mock_data.py`'s committed constants exactly), consistent with this
project's own code having always logged to "whatever's on the conventional
MLflow port" rather than a project-scoped container. Flagged here for
visibility, not changed — reading real data through the same real
mechanism this project's own MLflow-writing scripts already use.

## 7. README.md

The repo had no `README.md` at all before this session (verified: `find .
-maxdepth 1 -iname README.md` — nothing). Wrote one from scratch:
problem statement, an ASCII 8-layer architecture diagram plus a request-flow
diagram, full `.env` variable table, `docker-compose up` + per-service run
instructions, a real (not fabricated) example query/answer pulled from Day
9's committed `evaluation/day9_qa_states.json`, the Config A/C benchmark
table plus RAGAS/citation-judge scores, a description of all 5 Streamlit
tabs (with the new live/fallback behavior called out), the Slack bot +
HITL feedback flow, testing/CI status, Render deploy instructions +
known limitations, and links to `docs/daily_progress/`. `.env.example` was
added alongside it (referenced by the README's setup steps but didn't
exist either) with every variable name from the real `.env`, no real
values.

## 8. Render deploy — prepared, not deployed this session

Per an explicit decision: Render deploy needs a manual "connect this repo
as a Blueprint" step in Render's web dashboard plus pasting secrets there
(`render.yaml`'s `sync: false` list) — there is no API token or CLI
available in this environment to do that non-interactively, and it's an
outward-facing, real-money-adjacent action better left to the account
owner. `render.yaml` (Day 12) already builds the right `Dockerfile` and is
unchanged; the README documents the deploy steps and known limitation
(service goes live, full data-wired retrieval doesn't — see `render.yaml`'s
own comment) and has a placeholder line for the live URL once deployed.
**Open item**: add the real Render URL to `README.md` and verify `/health`
once deployed.

## 9. Test count / CI status

469 tests total (429 going into this session + 40 new: 15 slackbot + 11
feedback_aggregator + 3 live_data + 11 tab-fallback), all passing locally
(`pytest -q` — 469 passed, ~102s) and the same suite is what CI's `pytest`
step runs. `ruff check .` clean. `mypy .` clean (`Success: no issues found
in 59 source files`, strict on the shipped package). GitHub Actions'
`.github/workflows/ci.yml` run on the final push of this session's CI-fix
work was fully green end to end — the live confirmation, not just local
reasoning.

## 10. Open items / next steps

1. Render deploy — connect the repo, paste secrets, verify `/health`, add
   the URL to `README.md` (§8).
2. Wire `drift_monitor.py`'s and `feedback_aggregator.py`'s `alert_fn` to
   Slack now that `slackbot/` exists (`slack_sdk.WebClient.chat_postMessage`)
   — both already `structlog.warning()` on their own, so this is additive,
   not fixing a silent gap.
3. Register `airflow/dags/feedback_aggregator.py` (and `drift_monitor.py`,
   carried over from Day 12) against a live Airflow scheduler — needs the
   `docker-compose.yml` `airflow` service running, which the local venv's
   deliberate no-`apache-airflow` choice means can't be smoke-tested
   outside that container.
4. Config B (bge-reranker-v2-m3), ColBERT, `agents/a2a_protocol.py`, and
   the full 50-query benchmark — all carried over from earlier days,
   unchanged this session.
5. Test-file/root-script mypy debt (§3's scoped-out `tests/`/`run_*.py`) —
   real, deferred, not silently dropped.
