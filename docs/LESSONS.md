# Lessons learned

Troubleshooting notes and findings for anyone cloning this repo, written up
after the fact rather than left scattered across commit messages and daily
storylines. Each entry states the symptom, the cause, and the fix, with
commit hashes where one exists. Where a claim from the original brief for
this document could not be traced to a commit, test, or file in this repo,
that is stated explicitly rather than written up as fact — see the "Not
verified in this repo" notes below.

A note on sourcing: this document was written by reading `git log`,
`docker-compose.yml`, `render.yaml`, `Dockerfile`, `docs/uat/`, and the
`docs/daily_progress/` storylines. One source named in the brief for this
document — a "Functional Test Sign-Off v2.1" — is referenced by name in
`docker-compose.yml`'s DEF-17 renumbering comment (see below) but does not
exist as a file anywhere in this repository (checked by name, by content,
and by extension). DEF-17's original description ("the Qdrant Cloud
cluster deletion") is therefore only as reliable as that one comment; the
document itself could not be checked.

---

## 1. Infrastructure and deployment

### 1.1 Multi-stage Dockerfile stage ordering deployed the wrong service to Render

**Symptom**: after `Dockerfile` was restructured into a multi-stage build
(a shared `deps` layer feeding separate `api` and `streamlit` targets),
Render's deployed service silently became the Streamlit UI shell instead
of the public FastAPI app (custom HTML UI, `/health` JSON, Swagger docs).

**Cause**: `render.yaml` builds `./Dockerfile` with no `--target` set
(Render's Blueprint spec has no field for one), and a target-less
`docker build` on a multi-stage file defaults to building whichever stage
is declared *last* in the file. Commit `2394ad7` introduced the `deps` →
`api` / `streamlit` split but appended the `streamlit` stage after `api`,
so Render's default build silently picked up the UI shell instead of the
FastAPI service. `docker-compose.yml` was unaffected because it always
builds both stages by explicit `target:`.

**Fix** (`ae04380`, `fix(docker): restore api as final stage so Render
deploys FastAPI`): moved the `api` stage back below `streamlit` so it's
last in the file again, and added an explicit comment plus a `# DO NOT
append new stages below this one.` guard directly above it in the
Dockerfile, so a future edit doesn't reintroduce the same ordering bug
silently.

### 1.2 A "commit reported as live" was still serving pre-fix behaviour

**Symptom**: after the BM25 index was committed to git and the Dockerfile
was updated to `COPY` it into the image (`7b0b862`), the README was
updated to describe `/search`/`/ask` as fixed and pending a Render
redeploy. A follow-up live check shortly after found `/health` behaving
consistently with a redeploy having happened (it briefly stopped
responding, then came back `200`), but `/search` still returned `500` —
not the expected `200`/empty-results the fix should have produced.

**Cause**: not conclusively determined from outside the Render dashboard.
Commit `dd3202b` (`docs: BM25 Dockerfile fix pushed but not yet resolving
the live 500`) lists the candidate causes explicitly rather than guessing
one: the new image may not have finished building/deploying yet, Render's
`autoDeploy` may not have actually triggered on the push, or something
else in the request path was failing before the fix's code path was even
reached. The commit is explicit that this is unresolved, not that a stale
build cache was confirmed as the cause — that is a plausible read of "a
commit reported as live still served old behaviour," but it is not what
this repo's own commit trail asserts.

**Fix**: the README was corrected in the same commit to say `❌ still 500
as of the last check` instead of the earlier `⏳ pending redeploy`
framing — the practice change was to stop describing a fix as resolved
based on the code being pushed, and only describe it as resolved once a
live re-check actually confirmed it. The underlying `/search`/`/ask` 500
was later closed by a **different** fix entirely (Render's 512 MB memory
limit — see §1.3), confirmed live and documented in `README.md`'s Live
deployment table as of commit `8569bfb`.

### 1.3 Fitting the service into Render's 512 MB free tier by skipping torch imports in fallback mode

**Symptom**: every live `/search` call on the deployed Render service
failed. Render's own Events log reported the real cause directly: `Ran out
of memory (used over 512MB)`.

**Cause**: `api/dependencies.py`'s `get_dense_index()`/
`get_msmarco_reranker()` unconditionally called `DenseIndex.connect()` and
`MSMarcoReranker.load()` on first request regardless of `RERANKER_MODE` —
pulling in `qdrant_client` plus two `sentence-transformers`/torch model
loads well past 512 MB before a single request could be served, even
though `RERANKER_MODE=fallback` (the mode the deployed UI actually
targets) never needs either.

**Fix** (`66fa0b8`, `fix: skip torch/transformers imports in fallback mode
to fit Render 512MB free tier`):
- `api/dependencies.py`'s two providers return `None` under
  `RERANKER_MODE=fallback` instead of loading anything.
- `api/routers/search.py`/`ask.py` handle a `None` msmarco reranker
  (`RerankerRouter` already degrades gracefully when a tier is `None`).
- `agents/retrieval_agent.py`/`qa_agent.py`'s retrieve nodes treat a
  `None` dense index as a deliberate skip (`dense_results=[]`) rather than
  a failure, so BM25 results still get returned instead of discarded —
  this also fixed a pre-existing wart where the retrieve nodes'
  `dense_index or DenseIndex.connect()` pattern couldn't distinguish an
  explicitly-passed `None` from an omitted argument, and would have
  silently reconnected to Qdrant anyway, reintroducing the OOM.
- `render.yaml`'s `RERANKER_MODE` was changed from `live_fast` to
  `fallback` — the code fix alone is inert without this, since Render
  wasn't actually configured for the mode the deployed UI assumes.
`render.yaml`'s own comment documents this as load-bearing, not a quality
choice, and states the OOM was confirmed via Render's Events log before
the fix, not assumed.

### 1.4 MLflow 3.x rejects in-network Host headers; `--allowed-hosts` crash-loops uvicorn

**Symptom**: `api`/`streamlit` containers get a `403` when calling MLflow
over the docker-compose network (`Host: mlflow:5001`), even though
MLflow's own healthcheck passes — clients report the server as
unreachable despite the container being healthy.

**Cause**: MLflow 3.x's DNS-rebinding protection only allows `localhost`
Host headers by default. The documented fix, `--allowed-hosts`, is
unusable on the pinned `v3.14.0`: passing it (tested with both
`mlflow:5001,localhost:5001` and `mlflow,localhost`, on a freshly rebooted
host) crash-loops uvicorn workers and the port never serves. Dead workers
also become zombies (no reaper at PID 1), wedging the container against
`docker stop`. Forced kills can then corrupt the SQLite backend store,
which crash-loops even *without* the flag afterward — wiping the
`mlflow_data` volume restores it.

**Status**: open, not fixed. Documented in `docker-compose.yml`'s `mlflow`
service comment as **DEF-18** — originally recorded as DEF-17 in commit
`c55bb0c`'s message, then renumbered to DEF-18 in `98d0cc8` because DEF-17
was already assigned (per the "Functional Test Sign-Off v2.1" reference
this document could not independently verify — see the top of this
document) to the Qdrant Cloud cluster deletion. The Streamlit benchmark
tab falls back to committed historical results as a result — see
`streamlit_app/live_data.py`'s fallback path, closed under DEF-04 (§3.3
below) for the test side of the same MLflow-unreachable condition.
**Suggested next step**, per the same comment: add `init: true` to the
`mlflow` service so PID 1 reaps zombie workers, which at least removes the
`docker stop` wedge even though the root Host-header rejection stays open.

---

## 2. Retrieval findings from UAT

Source: `docs/uat/uat_day5_retrieval.md` (9 queries, live BM25 vs. dense
vs. RRF) and `docs/uat/uat_superiority_cases_executed.md` (15 queries
across 6 designed cases, also live). Both explicitly report results
honestly even where they contradicted the query's own design hypothesis.

### 2.1 RRF's corroboration bias

**Mechanism**: Reciprocal Rank Fusion (`retrieval/rrf_fusion.py`, `k=60`)
scores a chunk by summing `1/(60+rank)` across whichever of BM25's and
dense's top-100 pools it appears in. This means **two signals mildly
agreeing on a mediocre chunk can outscore one signal being highly
confident about the correct chunk** — RRF has no way to represent
"very confident" versus "weakly ranked," only rank position, and summing
two weak ranks can beat one strong one.

**Query 1 — "CUDA cudaMalloc function parameters"** (`uat_day5_retrieval.md`
Q2, reproduced live again in `uat_superiority_cases_executed.md` Q1): BM25
ranks the actual `cudaMalloc()` API reference chunk (`cc6c8e53...`) at
rank 2 — the correct answer. Dense never surfaces it in its own top-3.
Both BM25 and dense independently rank an unrelated `cudaStreamAddCallback`
chunk (`81b9c458...`) at rank 3, so its two-signal RRF score
(`2/(60+3) ≈ 0.0317`) beats the correct chunk's single-signal score
(`1/(60+2) ≈ 0.0161`) — RRF's rank 1 is the wrong chunk, even though BM25
alone had the right one at rank 2.

**Query 2 — "shader processor count"** (`uat_day5_retrieval.md` Q7): "shader
processor" is pre-CUDA-era GPU marketing terminology the corpus never
uses. BM25 has zero useful lexical signal on this query. Dense alone
correctly finds "An SM consists of: 128 CUDA cores..." at rank 1 — a real
semantic match across a total vocabulary gap. RRF loses this entirely:
because BM25 never ranked that chunk at all, it only carries a
single-list score and is crowded out by a "GPU Metrics" chunk both lists
rank moderately. The UAT document calls this "the clearest demonstration
of RRF's corroboration-bias limitation on vocabulary-mismatch queries" —
dense-only strictly beats hybrid here.

Both UAT documents' summary sections generalize this the same way:
**RRF is a clear net positive on semantic/conceptual queries** where BM25
and dense partially agree but each has blind spots (its intended use
case), but on queries where one signal has zero contribution (legacy
terminology, rare API symbols one signal misses entirely), RRF can
strictly underperform whichever single signal actually had the answer.

### 2.2 BM25 IDF dilution on `cudaDeviceSynchronize`

**Query — "cudaDeviceSynchronize return value"** (`uat_superiority_cases_executed.md`
Case 5, Q12), designed as a "dense failure / BM25 advantage" case and
reported honestly as contradicting that hypothesis: dense wins cleanly.

**Mechanism**: `cudaDeviceSynchronize` is a high-frequency term across
many API-reference chunks in this corpus (it appears in error-handling
notes, deprecation warnings, and usage examples throughout the CUDA
Runtime API reference, not just its own definition). BM25's IDF weighting
down-weights terms that appear in many documents — so the literal string
match that should make this an easy lexical win instead gets **diluted
across dozens of chunks that all mention the term without answering the
query**, and the chunk that actually states the return-value semantics
("returns an error if one of the preceding tasks has failed") never
reaches BM25's top-3. Dense finds that exact chunk at rank 1 via semantic
similarity to "return value," undiluted by how many other chunks happen to
share the literal function name. The UAT document's own framing:
"term-frequency dilution across many API-reference chunks sharing the
literal string `cudaDeviceSynchronize` hurt BM25 more than semantic
similarity hurt dense."

### 2.3 Low-information boilerplate in top-10 results

**Not verified in this repo.** The brief for this document described
"three classes of low-information boilerplate at 7.9% of top-10 results."
No file, commit, test, or UAT document in this repository states a 7.9%
figure or enumerates three boilerplate classes — the closest related
material is `uat_day5_retrieval.md`'s Q5 read ("every result here is
generic CUDA 'best practices' boilerplate or table-of-contents noise") and
several other UAT entries that call out table-of-contents chunks as noise
on a per-query basis, but nothing in the repo aggregates this into a
percentage or a taxonomy of boilerplate classes. This finding is not
written up here because it could not be traced to anything in the repo.

---

## 3. Test and regression issues

### 3.1 The Streamlit composition test under host memory pressure

**Partially verified.** `tests/streamlit_app/test_tabs.py`'s
`test_app_composes_all_five_tabs_without_exceptions` runs
`streamlit_app/app.py` — all 5 tabs, including `benchmark_tab.py`'s and
`eval_dashboard.py`'s plotly charts — through Streamlit's `AppTest`
harness with `default_timeout=30`. This is the one test in the suite that
instantiates the full tab composition rather than a single tab, so it is
structurally the most memory- and time-expensive test in that file.

Multiple `docs/daily_progress/` storylines (Days 6, 8, 9, 10, 12)
independently document this project's dev machine running with
0.22–0.79 GB of 7.65 GB free during active sessions, and record real
consequences of that pressure elsewhere in the suite — Day 10's storyline
records tests committed without a local `pytest` run after repeated
segfaults at 0.36 GB free. Given that pattern, a 30-second `AppTest`
timeout or an out-of-memory failure inside plotly's chart construction
during this specific test is a plausible failure mode.

**What could not be verified**: no commit message, test skip/xfail
marker, or storyline entry in this repo records an actual `AppTest`
timeout or a `MemoryError` from plotly occurring on this test. The
low-memory pattern is real and repeatedly documented; the specific failure
described in the brief for this document is not. Recorded here as a
plausible-but-unconfirmed risk on this specific test, not as a fixed
defect.

### 3.2 Shell environment variables leaking between test runs

**Not verified in this repo.** No commit, test file, or `conftest.py`
content found here documents four tests failing due to a leaked shell
environment variable being mistaken for a code regression.
`tests/api/test_main.py`'s DEF-03 tests (`test_reranker_mode_echoes_env_var_when_not_passed`,
`test_reranker_mode_echoes_default_when_not_passed_and_no_env_var`) are
the tests in this repo that exercise `RERANKER_MODE` env-var resolution,
and they use `pytest.MonkeyPatch.setenv()`/`delenv()`, which pytest resets
automatically after each test — the correct isolation pattern, and not
evidence of an actual leak having occurred. `tests/conftest.py` has no
project-wide env-clearing fixture, which would be consistent with a
`RERANKER_MODE` (or similar) value set directly in a developer's shell
leaking into a local `pytest` run — but no repo artifact confirms this
happened. Not written up as a resolved defect because no fix, commit, or
test change traces to it.

### 3.3 The `query_log` foreign key that nothing in the live request path populates

**Symptom**: `feedback_log.query_id` is a nullable foreign key to
`query_log.query_id` (`schema/schema.sql`). A feedback write that supplies
a real `query_id` from an actual `/search` or `/ask` response would fail
with a foreign-key violation, because nothing in the live request path
ever inserts a `query_log` row for that `query_id`.

**Cause**: `schema/schema.sql`'s `query_log` table exists and is read from
elsewhere in the project (e.g. `airflow/dags/drift_monitor.py`'s query-drift
window), but `api/routers/search.py` and `api/routers/ask.py` — the two
live endpoints that mint a `query_id` per request — never write a row to
it. The FK exists in the schema without a live writer on the other end of
it.

**Fix** (`e212c7f`, `docs: capstone README and setup guide; feat: web
feedback control`, `api/routers/feedback.py`): rather than change the
schema or add the missing `query_log` write (explicitly out of scope —
"Schema unchanged, per instruction"), `POST /feedback` looks the parent
row up first (`session.get(QueryLog, query_id)`) and stores `None` in its
place when no matching row exists, instead of passing the client-supplied
`query_id` straight through and letting Postgres reject the insert. This
mirrors the same tolerance `slackbot/feedback_handler.py` already had for
a `query_id` it couldn't resolve. The module's own docstring states this
plainly: "an infrastructure gap, not something either UI's user did
wrong" — and "not a fix for the missing `query_log` write, just this
endpoint refusing to be the thing that 500s because of it." The underlying
gap (no live writer for `query_log`) remains open.

---

## 4. Practices adopted as a result

- **Assert served content, not deploy status.** The stale-image episode in
  §1.2 (a README claiming a fix was "pending redeploy" while the live
  service still 500'd) led directly to the pattern used for the rest of
  this project's deploy claims: `README.md`'s Live deployment table states
  only what was independently re-checked at the time of writing (e.g. the
  `POST /search` row in commit `8569bfb` states the exact request made and
  the exact response scores received, not just "confirmed working"), and
  Day 14's storyline records refusing to write up `/health` as live after
  5 separate failed checks, asking how to proceed rather than writing it
  up as confirmed, and only updating the README once a later check
  actually returned `200`.
- **Verify every published figure against a repository artefact.**
  `reports/final_eval_report.md` (Day 14) links every number it states
  back to its real source — an MLflow run ID, a storyline section, or a
  committed JSON file — rather than restating a remembered figure.
  `evaluation/ci_ndcg_gate.py`'s CI check follows the same principle
  structurally: it gates against a committed snapshot of a real,
  live-measured benchmark run (`evaluation/benchmark_baseline.json`,
  Day 9's numbers) rather than a number written into the gate's own code.
