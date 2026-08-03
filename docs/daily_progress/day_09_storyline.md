# Day 9 — Live Benchmark Results, Citation Accuracy, LangSmith Tracing

## 1. What was built / run

| File | Lines | Purpose |
|---|---|---|
| `run_day9_relevance_labelling.py` | 67 | Task 1: BM25-only live retrieval + Claude relevance labelling, 50 queries |
| `run_day9_benchmark_ac.py` | 148 | Task 2+3: Config A + C over cached RRF pools, pool-wide Claude labelling |
| `run_day9_benchmark_c_retry.py` | 80 | Config C retry after Cohere trial rate limit (429) |
| `run_day9_benchmark_c_fixed_latency.py` | 83 | Config C rerun correcting a latency-measurement bug in the retry |
| `run_day9_ragas.py` | 110 | Task 4 attempt — QA states built; full `evaluate()` deferred (see §3) |
| `run_day9_citation_judge.py` | 91 | Task 5: 10 live QA outputs + citation judge |
| `run_day9_langsmith_check.py` | 56 | Task 6: traced LangGraph invocation + LangSmith API verification |
| `agents/qa_agent.py` (+`generate` fix) | 234 | Malformed-citation crash fixed, live-exercised for the first time |
| `tests/agents/test_qa_agent.py` (+1 test) | — | Regression test for the fix, 24 tests total |

Test suite: **324/324 passing** (323 + 1 new regression test).

New data artifacts committed: `evaluation/benchmark_queries.jsonl` (50),
`evaluation/relevance_labels.jsonl` (50), `evaluation/relevance_labels_superiority_pool.jsonl`
(45), `evaluation/day9_config_a_contexts.json`, `evaluation/day9_qa_states.json`,
`evaluation/day9_citation_judgments.json`.

## 2. Memory management

Free memory at session start: **0.06GB of 7.65GB** — worse than any prior
session, in the "near system-freeze" zone. Per user decision,
`deploy-streamlit-1` and `deploy-api-1` (unrelated-project containers, not
needed today) were stopped, freeing enough headroom to reach 0.51GB;
`deploy-mlflow-1` was kept running (`MLFLOW_TRACKING_URI=http://localhost:5000`
points at it, needed for Tasks 2/3/5's logging). All seven live steps below
ran without a crash, though free memory hovered in the 0.3-0.5GB range
throughout — this remains a hard ceiling for this host, not a one-off.
`deploy-streamlit-1`/`deploy-api-1` were left stopped at end of session.

Two other design adaptations kept every task's spirit ("Claude/Cohere API
calls only, no local model loading") technically true despite the tested
Day 8 code wiring in a torch-based dense encoder by default:
- **Task 1** used a `_NullDenseIndex` stand-in (returns `[]`) so
  `evaluation/relevance_labeller.py`'s `build_pairs()` ran on BM25-only
  retrieval — `retrieval/rrf_fusion.py`'s `fuse()` degenerates cleanly to a
  pure BM25 ranking when the second list is empty (an already-tested path).
  No changes to the tested module were needed.
- **Task 4/5** reused Task 2's live ms-marco-reranked contexts
  (`day9_config_a_contexts.json`) instead of re-running `agents/qa_agent.py`'s
  `retrieve`/`rerank` nodes (which need `DenseIndex.connect()` →
  e5-base-v2/torch) — only the `generate` node (a real, unmocked Claude call)
  ran, called directly against a manually-built `QAState`.

## 3. Results

### Task 1 — 50-query relevance label set

`evaluation/benchmark_queries.jsonl` (50 queries) and
`evaluation/relevance_labels.jsonl` (50 Claude-judged (query, passage) pairs,
BM25-only top-1 retrieval) are now real, committed artifacts — the first
time these files have existed. **28/50 (56%) judged relevant.**

### Task 2 + 3 — Config A / Config C benchmark

Scope note: only the 15 cached queries in
`docs/uat/uat_superiority_cases_raw.json` (top-3 RRF pool each, from Day 5's
UAT run) were available without a fresh dense-index load — this is a
15-query × 3-candidate smoke benchmark, not the full 50-query × top-100
benchmark `evaluation/benchmark_runner.py` is designed for (still deferred
until fresh live retrieval is memory-safe). A fresh Claude-judged relevance
set was built for this pool (45 (query, chunk) pairs, 11/45 relevant) —
Task 1's sparse 50-query labels don't cover these chunk_ids.

| Config | run_id | NDCG@10 | MRR | Prec@3 | latency_ms (avg) | cost_usd |
|---|---|---|---|---|---|---|
| A — ms-marco | `a55012a4` | 0.5333 | 0.5333 | 0.2444 | **48.90** | 0.0 |
| C — Cohere Rerank v3 | `c827cd71` | 0.5280 | 0.5333 | 0.2444 | **291.21** | 0.03 (15×$0.002) |

Both logged to MLflow experiment `reranker_benchmark`
(`http://localhost:5000/#/experiments/1`) and `benchmark_results`
(45 rows total, see §4 for the retry/correction runs also present there).
At this small scope, A and C are statistically indistinguishable on
ranking quality; ms-marco is ~6x lower latency and free. **Config B
(bge-reranker-v2-m3): not run** — hardware-blocked (OOM at model load,
8GB CPU-only constraint), per `CONFIG_B_DEFERRED_REASON` in
`evaluation/benchmark_runner.py`, deferred to a GPU environment.

**Two live bugs found and fixed during this run** (both in the throwaway
orchestration scripts, not the tested Day 8 library code):
1. Cohere's **trial API key is rate-limited to 10 calls/minute** — the
   first Config C attempt (`run_day9_benchmark_ac.py`) fired all 15 rerank
   calls in under a second and got a `429 TooManyRequestsError` on call 11.
   Fixed with a 6.5s-minimum-interval throttle (`run_day9_benchmark_c_retry.py`).
2. That throttle **slept inside the region `run_config()` times**, so the
   retry's logged `latency_ms` (~6.0-8.8s/query) was almost entirely
   rate-limit wait, not real API latency. `run_day9_benchmark_c_fixed_latency.py`
   reran Config C timing only the `rerank()` call itself — real Cohere
   latency is ~291ms/query, in line with expectations. Both runs remain in
   MLflow/`benchmark_results` (`db983987` = bad latency, `c827cd71` =
   corrected) rather than deleted, per this project's practice of
   documenting mistakes instead of erasing them.

### Task 4 — RAGAS: deferred (user decision, see below)

Building the 10 QA states for RAGAS scoring (`run_day9_ragas.py`) is what
surfaced the citation-parsing crash described in §5 below — it errored out
on the 2nd of 10 queries. Per user instruction, the run was interrupted and
the full RAGAS `evaluate()` call (faithfulness + answer_relevancy, which
would need many further live Claude + Cohere-embedding calls) is **deferred
to next session**, not attempted after the fix. `run_day9_ragas.py` is
committed as-written (including a from-scratch `CohereRagasEmbeddings`
adapter, since `langchain_community.embeddings.CohereEmbeddings` raises
`KeyError: 'user_agent'` at construction in the installed version — verified
directly, not assumed) and is ready to run next session.

### Task 5 — Citation judge: run successfully

With the crash fixed, `run_day9_citation_judge.py` regenerated all 10 QA
states (`generate` node only, live Claude calls) and judged every
(claim, cited chunk_id) pair:

**Overall citation_accuracy: 0.7037 (27 (claim, chunk) pairs judged)**, logged to
MLflow experiment `citation_judge`
(`http://localhost:5000/#/experiments/2`). Per-query accuracy ranged 0.0-1.0;
Q3 and Q10 produced zero judged citations (Q10 for the reason in §5 below —
not a citation-judge defect).

## 4. Bug found and fixed: `agents/qa_agent.py`'s `generate` node

Day 7 shipped `generate`'s citation parsing fully unit-tested but flagged
"live Claude call not yet exercised" — Day 9 is the first time it actually
ran against real model output, and it crashed on the 2nd live query:

```
TypeError: string indices must be integers, not 'str'
```

`tool_block.input.get("citations", [])` assumed every item was always a
`{claim, chunk_ids}` dict. Two distinct schema-drift shapes were observed
live, despite the forced `tool_choice`/`input_schema`:
- A citations list item was a **bare string** instead of a dict (query Q2
  reproduction case).
- The **entire `citations` value was a stringified JSON blob** rather than
  a native array (query Q10) — iterating it in the old unguarded code would
  have iterated *characters*, not list items.

**Fix**: `generate` now checks `isinstance(c, dict)` per citation item,
skips (with a `generate_malformed_citation_skipped` warning log) anything
that isn't, and keeps every well-formed citation rather than discarding the
whole answer. Verified against real Q10 output: 0 claims recovered (correct
— skipping is safe, not silently wrong) with no crash. Regression test
added: `test_skips_malformed_citation_entries_instead_of_crashing`
(`tests/agents/test_qa_agent.py`), 24 tests now in that file.

**Known follow-up, not fixed today**: the stringified-whole-array case
(Q10) is safely *skipped*, not *recovered* — a `try: json.loads(citations)`
fallback before iterating would recover it. Left for a future session since
today's priority was eliminating the crash, not maximizing citation
recovery.

## 5. Task 6 — LangSmith tracing: verified live

`LANGCHAIN_TRACING_V2=true` (plus `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=nvidia-ir-rag-agent`,
`LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com`) were already set in
`.env` — nothing to change. Verified live, not assumed: `run_day9_langsmith_check.py`
invoked `agents/qa_agent.py`'s compiled LangGraph graph (fake BM25/dense/router,
one real Claude call in `generate`), then a direct LangSmith API query
(`langsmith.Client().list_runs(project_name="nvidia-ir-rag-agent")`)
confirmed **5 real traced runs**, including a `LangGraph chain` run and its
child `retrieve`/`rerank`/`generate` spans, all `status=success`, timestamped
to this session — dashboard: `https://eu.smith.langchain.com/`.

## 6. Day 1 → 9 narrative

- **Days 1-7**: see `docs/daily_progress/day_07_storyline.md` §3 — scaffolding,
  ingestion DAG, BM25/SPLADE/dense/RRF retrieval, ms-marco re-ranking +
  router, QA agent, FastAPI service. 243/243 tests passing by Day 7.
- **Day 8** (`3220885`): evaluation stack built — relevance labeller, Cohere
  re-ranking tier, retrieval metrics, benchmark runner (Config A/C designed,
  Config B documented as deferred), citation judge, RAGAS suite. 323/323
  tests passing. All live artifact generation deferred to low free memory
  (0.06-0.79GB across sessions).
- **Day 9** (today): the Day 8 deferral resolved for benchmark + citation
  judge — real 50-query relevance labels, a live Config A/C benchmark
  (with two live bugs found and fixed: Cohere trial rate limiting, a
  latency-measurement bug), and live citation judging (0.7037 accuracy) all
  landed in MLflow/Postgres. RAGAS's full `evaluate()` run remains deferred
  (interrupted mid-run per user instruction) but its blocker — a real
  `agents/qa_agent.py` crash on live Claude output — is now fixed and
  regression-tested. LangSmith tracing confirmed live via direct API query.
  324/324 tests passing.

## 7. Cumulative project status

| Layer | Status | Evidence |
|---|---|---|
| Layer 0-3b (ingestion → re-ranking) | Done | See Day 7 storyline |
| QA agent | Live-verified, one crash found + fixed | `agents/qa_agent.py` — `generate` now handles real Claude schema drift |
| API | Unit-tested; still not run live with `uvicorn` | Carried over from Day 7 |
| Layer 4-5 (evaluation) | Config A/C + citation judge live-verified; RAGAS + Config B outstanding | `benchmark_results` (45 rows), MLflow experiments `reranker_benchmark` + `citation_judge` |
| Layers 6-8 (monitoring, observability, drift/HITL) | Layer 7's LangSmith tracing now live-verified; rest not started | `run_day9_langsmith_check.py` |

**Test suite**: 324/324 passing (323 through Day 8 + 1 citation-parsing
regression test).

**Open items for next session**:
1. Run `run_day9_ragas.py`'s full RAGAS `evaluate()` call (faithfulness +
   answer_relevancy over the 10 already-built QA states) — code is ready,
   just deferred by timeout today.
2. Recover the stringified-citations edge case (Q10) rather than skipping it.
3. Run the full 50-query × top-100-pool benchmark (all of Config A/C, not
   just the 15-query cached-pool smoke test) once fresh dense-index
   retrieval is memory-safe.
4. Build bge-reranker-v2-m3 (Config B / `live_quality`) in a GPU environment.
5. Run `uvicorn api.main:app` live (carried over from Day 7).
6. Consider a paid/production Cohere key if further live Config C runs are
   needed — the trial key's 10 calls/minute limit will recur.
7. Restart `deploy-streamlit-1`/`deploy-api-1` if those other-project
   containers are needed again (left stopped this session for memory).
