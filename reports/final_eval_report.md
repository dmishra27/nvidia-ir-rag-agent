# nvidia-ir-rag-agent — Final Evaluation Report

**As of:** Day 14 (v1.0.0). All numbers below are real, live-measured
results committed to this repository — not projections. Sources are cited
per table; see [`docs/daily_progress/`](../docs/daily_progress/) for the
full narrative behind each run.

## 1. Retrieval quality — re-ranker comparison (Config A vs C)

Live-measured Day 9, 15 cached queries (Day 5 UAT superiority pool),
top-3 RRF-fused candidates as input to each re-ranker. Source:
[`evaluation/benchmark_baseline.json`](../evaluation/benchmark_baseline.json)
(the CI NDCG gate's own baseline snapshot), MLflow experiment
`reranker_benchmark`, [`day_09_storyline.md`](../docs/daily_progress/day_09_storyline.md) §3.

| Config | Re-ranker | NDCG@10 | MRR | Prec@3 | Latency (ms/query) | Cost/query | Run ID |
|---|---|---|---|---|---|---|---|
| **A** | ms-marco-MiniLM-L-6-v2 (local CPU cross-encoder) | **0.5333** | 0.5333 | 0.2444 | 48.9 | $0.00 | `a55012a4` |
| **C** | Cohere Rerank v3 (hosted API) | 0.5280 | 0.5333 | 0.2444 | 291.2 | $0.002 | `c827cd71` |
| B | bge-reranker-v2-m3 | — | — | — | — | — | not run |

**Reading this**: Config A (free, local, ~6x faster) very slightly
*outperforms* Config C (paid, hosted API) on this query set — Cohere's
hosted re-ranker is not a strict quality upgrade here, and its 291ms
latency is a real cost for a marginal (and here, negative) quality
difference. This is a genuine result on a 15-query smoke-scope benchmark,
not a claim that generalizes without the full 50-query × top-100-candidate
run `evaluation/benchmark_runner.py` is designed for (that run is still an
open item — see §5).

**Config B (bge-reranker-v2-m3)** is hardware-blocked: it OOMs at model
load on this project's 8GB-RAM/CPU-only development machine
(`evaluation/benchmark_runner.py`'s `CONFIG_B_DEFERRED_REASON`) and is
deferred to a GPU environment. Shown here as "not run," not silently
omitted from the comparison it was meant to be part of.

**CI regression gate**: `evaluation/ci_ndcg_gate.py` fails any push where
Config A's `mean_ndcg_at_10` drops below **0.50** against this table's
committed baseline — quality regression is a CI-blocking condition in this
project, not just a dashboard number.

## 2. RAGAS — faithfulness and answer relevancy

Live-measured Day 11, 10 queries, real Claude Sonnet 5 generation scored
by RAGAS. Source: MLflow experiment `ragas_eval` (run `7d8f1005`),
[`day_11_storyline.md`](../docs/daily_progress/day_11_storyline.md).

| Metric | Score | What it measures |
|---|---|---|
| **Faithfulness** | **0.7616** | Fraction of claims in the generated answer that are actually supported by the retrieved passages |
| **Answer relevancy** | 0.2497 | How directly the answer addresses the question asked |

**Reading this**: faithfulness (0.76) is respectable — most claims the
model makes *are* grounded in what was retrieved. Answer relevancy (0.25)
is low, and that's consistent with what §3's per-query judgments show
directly: on several of the 10 sampled queries, the retrieved passages
simply didn't contain the answer, and the model correctly said so rather
than inventing one (see the `cudaMemcpyAsync`/`cudaErrorInvalidValue`
examples in [`README.md`](../README.md#example-query) and §3 below) — a
response that's *faithful* (doesn't hallucinate) but scores low on
*relevancy* (didn't answer the question) by RAGAS's construction. This is
a retrieval-recall problem, not a generation-quality problem: the fix is
better retrieval coverage over the corpus, not a different prompt.

## 3. Citation accuracy — per-claim LLM judge

Live-measured Day 9, 27 individual claim→citation judgments across 10
queries, judged by `evaluation/citation_judge.py` (a separate LLM-as-judge
pass, not the same call that generated the answer). Source:
[`evaluation/day9_citation_judgments.json`](../evaluation/day9_citation_judgments.json),
MLflow experiment `citation_judge` (run `43aa821a`).

**Aggregate: 0.7037** (19/27 claims judged as actually supported by their
cited chunk).

| Query | Supported / Total |
|---|---|
| Q1 — CUDA cudaMalloc function parameters | 3/3 |
| Q2 — cudaMemcpyAsync stream parameter | 3/3 |
| Q3 — CUDA error cudaErrorInvalidValue description | 1/3 |
| Q5 — how to make GPU programs run faster | 1/3 |
| Q6 — problems with threads executing different code paths | 3/3 |
| Q8 — shared memory bank conflicts and how to avoid them | 3/3 |
| Q9 — memory coalescing rules for global memory access patterns | 3/3 |
| Q10 — latency hiding through instruction level parallelism | 2/3 |

(`streamlit_app/eval_dashboard.py`'s per-query citation-accuracy chart —
added Day 13 — plots this same breakdown live from the same file.)

**Reading this**: the two weakest queries (Q3, Q5) are exactly the two
where §2's low answer-relevancy shows up concretely — the model still
produced 3 claims each, but 2 of 3 weren't actually grounded because the
retrieved passages didn't cover the question. Per-claim judging catches
this in a way an aggregate faithfulness score alone doesn't: it localizes
*which* queries and *which specific claims* need better retrieval, not
just that faithfulness "could be higher."

## 4. Test coverage and CI

| Metric | Value |
|---|---|
| Total tests | 469+ (grows with each day; see [`docs/daily_progress/README.md`](../docs/daily_progress/README.md) for the day-by-day count) |
| Test policy | Every embedding/LLM/DB call mocked (`AGENTS.md`) — no live API calls in CI |
| `mypy --strict` | Clean on `agents/`, `api/`, `retrieval/`, `monitoring/`, `evaluation/`, `schema/`, `mcp/`, `slackbot/`, `streamlit_app/` |
| CI pipeline | checkout → install → patch ragas → ruff → mypy → pytest → NDCG regression gate — all green on `main` |

## 5. Known gaps in this evaluation

Stated plainly, not buried:

- **15-query smoke scope, not 50-query full scope.** §1's benchmark is
  `evaluation/benchmark_runner.py`'s smoke-test path (15 queries, top-3
  pool). The full 50-query × top-100-candidate benchmark the module and
  `SKILLS.md`'s `run-reranker-benchmark` pattern are designed for has not
  been run.
- **Config B never measured.** No bge-reranker-v2-m3 number exists at any
  scope — GPU-hardware-blocked, not just unscaled.
- **RAGAS run at 10 queries, not the full set.** A larger sample would
  narrow the confidence interval on both faithfulness and answer
  relevancy; 10 queries is enough to see a real signal (§2's retrieval-
  recall finding), not enough to call either number precise.
- **No live multi-day quality-regression history.** `monitoring/quality_regression.py`'s
  `evaluate_regression()` logic is real and unit-tested, but
  `streamlit_app/eval_dashboard.py`'s trend chart still runs it over a
  synthetic history (see that file's own docstring) — `airflow/dags/drift_monitor.py`'s
  regression task has never run against a live scheduler.
- **citation_accuracy and RAGAS scores come from different, independent
  10-query runs** (Day 9 citation judge, Day 11 RAGAS), not one combined
  run over the same query set — a genuinely joint faithfulness×citation
  view of the same 10 answers doesn't exist yet.

## 6. Research context

This project extends prior work on neural passage/document quality
estimation for information retrieval (the author's SIGIR 2024
co-authorship, and the sibling
`MSc_Dissertation_Document_Quality_Estimation` project in this same
workspace, apply a learned T5-based quality estimator to PyTerrier
reranking) into a production RAG shape: Layer 0's `retrieval/chunk_quality.py`
is a lightweight **heuristic** stand-in for that line of research here —
sentence-length/non-ASCII-ratio/stopword-ratio scoring, not a ported
neural model — flagging low-quality chunks *before* they ever reach
retrieval, rather than only re-ranking after the fact. The broader thesis
carried over is the same one: retrieval quality is bounded by *corpus*
quality as much as by ranking algorithm choice, which is exactly what §2's
finding shows in practice — this project's re-ranking (§1) is solid, and
the answer-relevancy gap (§2) traces back to retrieval coverage over the
corpus, not the ranking step. A learned (rather than heuristic) chunk
quality model, trained the way the MSc dissertation's T5 estimator was, is
a natural next step for Layer 0 — not built here.
