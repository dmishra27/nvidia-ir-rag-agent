# Phase One — Building the System
### Everything before the capstone submission
**`1bc7f0c` (10 July 2026) → `8569bfb` (16 August 2026) · 60 commits · 38 days**

*Companion document to "What We Learned About Re-Ranking" (Phase Two)*

---

## What this phase was

Thirty-eight days from an empty repository to a deployed, tested, documented retrieval system with a formal sign-off behind it. The work divides into four distinct movements, and they are not the four you might guess — the last one, in particular, is the interesting one.

| Movement | Dates | What it was |
|---|---|---|
| **1 · Construction** | 10–13 July | Corpus, indexes, fusion, re-ranker, agents |
| **2 · Productionisation** | 2–9 August | Evaluation stack, observability, CI/CD, deployment |
| **3 · Formal testing** | 10–12 August | 42-case functional test, defect remediation, sign-off |
| **4 · Honesty work** | 13–16 August | Making the system's limitations legible before handing it over |

**The submission commit is not a feature.** `8569bfb` is *"fix(api): echo the resolved reranker_mode in `/search` and `/ask` (DEF-03)"* — the closure of a defect the functional testing had raised. The project was submitted on the back of a test cycle, not a feature push.

---

## Movement 1 — Construction (10–13 July)

Four days, and by the end of them the entire retrieval architecture existed.

**Day 1–2 — the corpus and the plumbing.** Project structure and schema first (`1bc7f0c`), then an Airflow 3 ingestion DAG with SQLAlchemy models and a **chunk quality scorer** (`42431ea`, 72 tests). Five NVIDIA PDFs were DVC-tracked and verified clean (`921281c`). A Text-to-SQL LangGraph agent arrived alongside (`aebd277`, 21 tests), and then the ingest runner produced the number that anchors everything downstream: **5,389 chunks in Postgres** (`daa4ae9`).

*Worth noting for Phase Two:* a chunk quality scorer existed from day one. The straddling-chunk defect that Family A's A1 would later diagnose slipped past it — which is not a criticism of the scorer so much as evidence that chunk quality is hard to score without a downstream task to measure it against.

**Day 3 — BM25.** The sparse index with a shared `Candidate` dataclass (`3fee692`), 108/108 tests passing, live search verified. The `Candidate` abstraction matters: it's what allows BM25, dense, and fused results to flow through the same interfaces later.

**Day 4 — dense retrieval and a detour.** Bi-encoder evaluation, SPLADE, MCP servers, Qdrant populated (`14bbcfa`). This is where the `all-MiniLM-L6-v2` dense baseline of **0.4469** was recorded — the comparison point the hypothesis plan would later reach for when asking whether a stronger encoder would move NDCG more than any fusion parameter.

A small commit here (`0568c5b`) fixes MCP servers routing structlog to stderr to stop JSON-RPC corruption on stdout. Minor, but a good example of the class of problem that only appears when components meet.

**Day 5 — fusion, and the first UAT.** Dense index, RRF fusion, hybrid search complete (`373c9f5`). Immediately followed by **UAT Round 1** (`4ba23cf`) — retrieval pipeline validation across 9 queries.

**13 July — Round 2, and the re-ranker.** **UAT Round 2** (`3c9e415`): 15 queries executed live against BM25, dense, and RRF. This is the run that produced the **six case types** every Family A experiment would later group by, and the qrels built from its top-3 pools — the instrument whose limits Phase Two spent three days discovering.

Same day, Day 6 (`d7a998f`): ms-marco re-ranker, router, LangGraph Retrieval Agent, 208/208 tests. And in the commit message itself: *"UAT regression deferred due to memory."*

**That is 13 July.** The 8 GB ceiling was already shaping what this project could measure, five weeks before it blocked hypothesis A7 entirely. The constraint is not an incident; it is a permanent feature of the environment.

---

### A three-week gap

Nothing between 13 July and 2 August. The deferred regression was finally run on 2 August (`139a9cc`, Q1 and Q12 verified) — and note *which* two queries were chosen. Q1 and Q10 were the pair the evaluation page had flagged as fusion failures; Q1 and Q12 were the pair chosen for regression. Q1 was already known to be the interesting one.

---

## Movement 2 — Productionisation (2–9 August)

Eight days that took a working retrieval pipeline and made it a service.

**Days 7–9 (2–3 Aug) — evaluation infrastructure.** QA agent, FastAPI endpoints, structlog instrumentation (`460c245`). Then the piece that matters most for Phase Two: **Day 8's evaluation stack** (`3220885`) — benchmark runner, RAGAS suite, citation judge. This is the module that produces the Config A / Config C NDCG figures, and therefore the authority on which config is which.

Day 9 (`1836632`) ran the live benchmark and recorded citation accuracy and LangSmith tracing. RAGAS was deferred on a timeout — the second deferral of the phase, again environmental.

**Days 10–12 (3–7 Aug) — observability and CI/CD.** An `mcp-mlflow` MCP server (`710dd78`); RAGAS live scores, orchestrator, drift detector, OpenTelemetry/Jaeger (`fa021de`); EDA agent, Phoenix config, quality regression monitor, Streamlit UI (`64acd47`); then CI/CD gate, drift DAG, Render deploy, Dockerfile (`a8ea248`, `3b37b68`).

**8 August — the dependency battle.** Six consecutive commits doing nothing but making CI pass: torch CPU wheel index (`171e004`), pyarrow bump (`bf4bc6a`), the full dependency graph the torch fix unmasked (`f8b30e5`), clearing `mypy --strict` debt in production code (`76c0867`), and two attempts at patching ragas to unblock pytest collection (`d5c16a9`, `6ba954f`).

Unglamorous, and worth including in any honest account. A quarter of that day went to dependency resolution.

**Day 13 and release.** Slackbot, HITL feedback, live Streamlit, README, Render deploy (`bf205f9`), then **`1c1c15d` — release v1.0.0**.

**And then the deployment reality.** The commits immediately after release tell their own story: guided HTML search UI, include BM25 index for Render, ship the BM25 index in the Docker image, *"document the real gap left"* (`7b0b862`), *"BM25 Dockerfile fix pushed but not yet resolving the live 500"* (`dd3202b`). Then on 9 August, the decisive one: **skip torch/transformers imports in fallback mode to fit Render's 512 MB free tier** (`66fa0b8`).

That commit is why the production deployment runs BM25 only. It is not a defect — it is a deliberate trade to fit the hosting tier, and the functional sign-off records it as such. But it means **the deployed system and the evaluated system are different systems**, which is a fact the next movement went to some trouble to make visible.

---

## Movement 3 — Formal testing (10–12 August)

A 42-case functional test suite across eight sections and two tiers: the live Render deployment, and the local Docker-compose stack running the full pipeline.

Baseline `20874a9`, certified at `cd6cf75`. Final result: **39 passed, 1 failed, 2 blocked**, 40 of 42 fully evidenced. Three sign-off revisions were issued in three days.

### What the testing found

**A repeated defect class — implemented, tested, never invoked.** `DEF-01` and `DEF-16` are the same failure in two modules. `configure_tracing()` in `api/telemetry.py` and `configure_phoenix()` in `monitoring/phoenix_config.py` were both fully implemented, correctly placed, and covered by unit tests — but neither was ever called at process start. Every span was inert. Every trace silently discarded.

Unit tests could not catch this *by design*: they inject their own tracer provider precisely so they don't depend on global state. Only an end-to-end test could see it. Fixed in `d3e5a9d` and `cd6cf75`.

The generalisable lesson, stated in the sign-off: **instrumentation needs an activation assertion at integration level, not merely unit coverage of the instrumented function.**

**Measurement replaced assumption, and overturned one verdict.** Version 1.0 of the sign-off had accepted four cases on inference or visual impression — brand colour by eye, page load by absence of a perceived problem, the reranker-mode parameter by reading the API spec, and LangSmith traces from a trace list. Version 2.0 measured all four directly. Three confirmed; **one failed**. Page load, once actually timed, came in at 72.56 s cold against a 60 s criterion — while the warm request measured 0.186 s, a 390× difference isolating the entire delay to container boot.

The sign-off's own comment on this is the sharpest line in the document: *"inference was right three times out of four, which is precisely why the fourth matters."*

**Cross-project coupling as the dominant reproducibility risk.** Port 8000 was occupied by an unrelated project returning plausible-but-wrong JSON to the test harness. MLflow experiments lived in a second project's volume. Streamlit tests failed intermittently when that second project was slow. Individually minor; collectively they meant the repository was not reliably reproducible from a clean clone. Partly addressed on 12 August (`2394ad7` — containerise api and streamlit, provision mlflow, decouple tests from live mlflow).

**Two findings that Phase Two would inherit directly.** §8.2 recorded that the cross-encoder *"appears to favour discursive explanation over terse reference material"* — it dropped the canonical `cudaMalloc` signature chunk out of its top five. And DEF-10 recorded that table-of-contents chunks pollute lexical rankings while the cross-encoder demotes them. Both become evidence in Family A: the first supports A4, the second explains A6.

---

## Movement 4 — Honesty work (13–16 August)

The four days between certification and submission were not spent adding features. They were spent making the system's limitations legible.

- **`79ac4f6`** *(10 Aug)* — honest retrieval mode labelling in the search UI
- **`107c060`** — derive the relevance number from the same normalised value as the bar, so the displayed figure and the visual agree
- **`e678ebe`** — an **evaluation page** surfacing the retrieval benchmark and method comparison
- **`85a2c83`** — a **corpus transparency panel** naming sources *and coverage gaps*
- **`4816d2d`** — footer states the deployed retrieval mode and links to the evaluation
- **`e212c7f`** — capstone README, setup guide, and a web feedback control
- **`8569bfb`** — echo the resolved `reranker_mode` (DEF-03), so responses are self-describing

The through-line is unmistakable. Every one of these commits makes it *harder* for a reader to over-estimate the system. The deployment runs BM25 only, so the footer says so. The corpus has coverage gaps, so a panel names them. The benchmark numbers exist, so an evaluation page shows them rather than a claim about them.

**And that last decision is what sets up Phase Two.** The evaluation page published a claim — that a cross-encoder *"should not be subject to"* the fusion failure mode. Publishing it is what made it testable, and what made its untested status visible enough to be worth attacking.

---

## What Phase One left on the table

Three things, all of which Phase Two picked up:

1. **The qrels were built from top-3 pools of the system being evaluated.** Circular by construction. Nobody noticed until experiments started returning results that didn't make sense.
2. **`candidate_pool_size` defaults to 100 but was benchmarked at 3.** A parameter with a 33× gap between its default and its tested value.
3. **The evaluation page's central claim about the re-ranker was asserted, not demonstrated** — and its own text recorded why it was shaky.

None of these are failures of Phase One. They are the natural residue of a build phase: you cannot validate an instrument until you have used it on something. What matters is that they were written down where someone could find them.

---

*Phase One status: v1.0.0 released, 42-case functional test signed off, deployed and documented. 60 commits, 10 July – 16 August 2026.*
