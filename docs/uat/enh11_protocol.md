# ENH-11 — Graded Relevance Judgements
### Protocol for building a retriever-independent, graded qrel set

**Reference:** ENH-NVIR-2026-011
**Status:** Complete — 325 of 325 judgements done (§7.1)
**Prepared:** 4 September 2026
**Unblocks (§8):** A2, A3, A3-2, A4, A6 re-runs · B4's aggregate test · B7's precision risk · D-QR re-test · every future NDCG figure

---

## 1. Why this exists

Every relevance label in this project was built by `run_day9_relevance_labelling.py` on 3 August, taking the **top 3 fused results** of the system under evaluation and marking which were relevant.

The human judgement was sound. **The sampling was not.** The pool a reviewer saw had already been filtered by the very system the labels would go on to evaluate.

Four consequences, all measured:

| | Effect |
|---|---|
| **A2** | Deepening the pool made quality appear to *decline* — chunks found at positions 4–50 could not be credited, because no label existed for them |
| **A3** | 8 of 15 queries survived the label filter. One whole case type had zero qualifying queries |
| **A3-2** | Q7/Q8's "dilution" turned out to be label sparsity: 5 of 6 competitor chunks were substantively on-topic and unlabelled |
| **A1** | Both queries that motivated Family A — R2-Q1 and R2-Q10 — carry **zero positive labels**, because fusion displaced their correct chunks before any human saw them |

That last one is the sharpest: **the bias under study erased its own evidence from the instrument used to study it.**

Since then, D-QR, B3, B4 and B7 have all had to use *target-chunk rank* instead of NDCG — a workaround that needs no labels but can only track one known chunk per query. It cannot answer "is this ranking good overall."

---

## 2. What "fixed" means

Three properties, each addressing a specific failure above.

**Retriever-independent pooling.** Candidates come from BM25, dense *and* RRF separately, unioned. No single configuration decides what a judge sees.

**Depth beyond 3.** The current cap is why deepening the pool looked like degradation.

**Graded 0–3, not binary.** Q7/Q8's competitors were *substantively relevant but not the labelled chunk*. Binary labels have no way to say that; graded ones do.

---

## 3. Scope and effort

**Honest estimate up front**, because this is the largest item in the project.

| | |
|---|---|
| Queries | 15 Round 2 + R1-Q7 = **16** |
| Pool per retriever | top 10 (see §3.1) |
| Unique chunks per query | ~20–25 after overlap |
| **Total judgements** | **~350** |
| At ~25 s each | **2.5–3 hours** of focused work |

Not a single sitting. §7 stages it.

### 3.1 Why top 10 and not top 20

Top 20 per retriever gives ~40 unique chunks per query and ~640 judgements — five hours, and the marginal chunks at ranks 15–20 are almost all grade 0.

Top 10 covers every rank the project's findings actually turn on: every displaced target in B3 sits at fused rank ≤17, and the deepest thing A3-2 examined was rank 9. If a later experiment needs depth 20, the pool can be extended and only the new chunks judged.

**Record the choice.** A future reader must know the pool was capped at 10, or they will misread an absent label as a judged zero.

---

## 4. The grading scale

| Grade | Meaning | Test |
|---|---|---|
| **3** | Directly answers the query | A developer reading only this chunk gets what they asked for |
| **2** | Substantively relevant | Answers part of it, or gives necessary context; a good result to see, not the best one |
| **1** | Topically related, doesn't answer | Same subject area, but reading it doesn't help with the question |
| **0** | Not relevant | Different topic, or matched on vocabulary alone |

**The 2 is the grade that matters.** It's the one binary labels couldn't express, and the reason A3-2's finding looked like a re-ranker defect. Q8's `b2e88edc…` — a walkthrough of eliminating bank conflicts via `CU_TENSOR_MAP_SWIZZLE_128B` — is a clear **3** on its own merits, despite not being the originally labelled chunk.

**The 1 vs 0 boundary** catches the straddling-chunk problem. A `cudaFreeMipmappedArray` See-also block listing `cudaMalloc` is **0** for a `cudaMalloc` query: it mentions the function and explains nothing. Vocabulary overlap is not relevance.

---

## 5. Judging discipline

Four rules. The first two matter most.

**Blind.** The interface must not show which retriever found a chunk, at what rank, or whether it was previously labelled. Knowing "dense ranked this first" biases the judgement toward 3.

**Randomised order.** Shuffle within each query, and don't judge queries in Q1–Q15 order every session. Fatigue correlates with position.

**Judge the chunk, not the query's intent.** Ask "does this text answer this question," not "was this what I meant when I wrote the query."

**Consistency check.** Re-present 10% of chunks at the end of each session, unmarked. Disagreement between the two passes is your error bar — with a single judge it's the only reliability estimate available, and it belongs in the findings.

---

## 6. Tooling

Two scripts. Neither needs an API key.

### Phase A — build the pool

> **Prompt for Claude Code:**
>
> Build the ENH-11 candidate pool. No judgements, no API calls — this produces the input for manual judging.
>
> **Script** `build_enh11_pool.py`:
> - For each of the 15 Round 2 queries plus R1-Q7 (same anchor set as `run_dqr_eval.py` and `run_b3_b4_fusion_eval.py`), retrieve **top 10 from BM25, top 10 from dense, top 10 from RRF** (`k=60`, pool 100)
> - Union them per query, deduplicating by `chunk_id`
> - Emit `evaluation/enh11_pool.json`: for each query, the query text and a list of `{chunk_id, chunk_text}` — **shuffled**, with **no rank, score, or source-retriever information**
> - Separately emit `evaluation/enh11_pool_provenance.json` with the retriever ranks per chunk, for post-judgement analysis only. **Do not merge these files.** Provenance in the judging input would bias grading; that is the failure ENH-11 exists to fix
> - Report unique-chunk counts per query and the total
>
> **Host constraints:** needs postgres + qdrant and one dense encoder load. This machine has hit 17 MB free mid-parse (DEF-23). Check RAM first, sequence the encoder load alone, and stop and report on a memory failure rather than retrying.
>
> Commit the script before running it and the output before any analysis, per `0149ca4`.

### Phase B — the judging interface

> **Prompt for Claude Code:**
>
> Build `judge_enh11.py` — a terminal tool for manual relevance judgement. No API calls; the human is the judge.
>
> **Behaviour:**
> - Reads `evaluation/enh11_pool.json`, writes `evaluation/enh11_qrels.json` incrementally after **every** judgement, so a session can stop at any point without loss
> - Shows: query text, then one chunk at a time. **Never** shows retriever, rank, score, or prior label
> - Accepts `0`/`1`/`2`/`3`, plus `s` to skip, `b` to go back one, `q` to save and quit
> - Displays the grading scale from §4 of `docs/uat/enh11_protocol.md` on screen
> - Tracks progress: "Query 4 of 16 · chunk 12 of 23 · 187 of ~350 judged"
> - **Consistency check:** re-present a random 10% of already-judged chunks at session end, unmarked, and record both passes in the output for disagreement analysis
> - Resumable: on restart, skip anything already judged unless it's a consistency re-check
>
> Output format per judgement: `{query_id, chunk_id, grade, timestamp, pass}` where `pass` is 1 or 2 (consistency re-check).
>
> Commit the script before any judging begins.

---

## 7. Staging

Don't attempt this in one sitting.

| Stage | Work | Output | Status |
|---|---|---|---|
| **1** | Build pool, build interface | Scripts committed, pool counts known | Done — `enh11_pool.json`, 325 chunks across 16 queries |
| **2** | Judge **5 queries** — suggest Q1, Q4, Q7, Q10, R1-Q7 | ~110 judgements. Validates the process | Done, folded into the live sessions below |
| **3** | **Review stage 2 before continuing.** Does the scale work? Are you consistent? Is the interface fighting you? | Adjust now, not after 350 | Done — no scale drift found |
| **4** | Judge the remaining 11 | ~240 judgements, across sessions | Done — via live judging + validated transcript recovery, see §7.1 |
| **5** | Consistency analysis, write findings | Disagreement rate, the qrel file itself | Done — see §7.1 |
| **6** | Re-run what was blocked | See §8 | Unblocked, not yet run |

**The stage-3 review is not optional.** Discovering at judgement 300 that your 2-vs-3 boundary drifted means redoing all of them.

### 7.1 Current state (13 September 2026) — complete

**All 325 of 325 judgements are on disk and committed** (`evaluation/enh11_qrels.json`, commit `d13eae0`), verified by direct inspection of the file — 325 pass-1 (primary) entries across all 16 queries, not taken from a pasted tool summary. That distinction matters here: earlier in this process a reported count was trusted without checking the file, and it was wrong (see the superseded consistency result below). This entry is written from the file itself.

Provenance, three sources, kept distinct:

**101 judged live via `judge_enh11.py`** (two original sessions, 9 and 12 September 2026) — 51 + 50 primary judgements, covering Q1–Q5 (Q5 partial).

**194 recovered from a separate Claude conversation transcript**, where the pool had been reasoned through chunk-by-chunk outside the tool (covering the remainder of Q5 through Q14, plus R1-Q7). Because that reasoning wasn't blind-tool-mediated, the hand-transcribed extraction (`enh11_recovery_grades.md`) was cross-validated row-by-row against `enh11_pool.json` before anything was entered: a row was written into `enh11_qrels.json` only if its chunk_id prefix resolved to exactly one chunk in that query's own pool and wasn't already on disk from the two live sessions above. Anything that didn't resolve cleanly was left out, not guessed (commit `40f8251`). See `enh11_recovery_grades.md` for the row-by-row detail.

R1-Q7's judgements include both chunks of particular project significance, each graded blind: `35b73f33…` (the canonical target — "An SM consists of: 128 CUDA cores") → **3**, and `b1b83570…` (the corroborated-but-wrong GPU-Metrics warp-occupancy chunk from the B3/B7 fusion analysis) → **0**.

**30 judged live in a final gap-fill session** (13 September 2026, commit `d13eae0`) — closed every remaining primary-judgement gap: the 12 chunks left over from the recovery cross-validation (one short per query section, two for Q8 and Q10) plus Q15 in full (17 chunks, whose recovery-transcript section had duplicate chunk_ids within itself and was marked unreliable and never used). This is also the session that ran the genuine consistency check.

**Consistency result: 3/3 exact agreement.** Three chunks re-presented unmarked at the end of the 13 September session — Q11/`0a880bdf…`, Q14/`937f086f…`, Q15/`23082588…` — each re-graded identically to its first pass. A sample of 3 is small; read this as "no disagreement observed in a small check," not as a reliability estimate with any statistical weight. It is the only inter/intra-rater signal this single-judge protocol produces (§9).

**This supersedes the earlier 22/22 consistency figure reported in a prior session.** That number was never written to disk — it described a consistency pass from a session that, on inspection of `enh11_qrels.json`, was never actually recorded. It should not be cited; the 3/3 figure above is the one genuine consistency result that exists.

**Same-day cross-reference — CC-VER-01.** Also closed 13 September 2026: a genuinely fresh clone (`%TEMP%\verify`) ran the full test suite — 589 passed, 0 failed, matching the working repo exactly — independently confirming that the tooling this protocol depends on (`build_enh11_pool.py`, `judge_enh11.py`, and the `evaluation/` test suite) reproduces from the repository alone, not just from this machine. See `docs/uat/correction_notice_a1.md` §6 and `docs/uat/clean_clone_test_findings.md`.

---

## 8. What unblocks now that this is complete

| Item | What is now possible |
|---|---|
| **A2** | Pool depth is re-testable against labels that credit deep finds. The original question — what should `candidate_pool_size` be — can now be answered, not just re-asked |
| **A3** | All 16 queries are usable instead of 8; Case 4 has data for the first time |
| **A3-2** | Q7/Q8's competitors have their actual grades on disk. Several came back 2s and 3s |
| **A4, A6** | Re-runnable with retained artifacts against a non-circular key |
| **B4** | Its aggregate NDCG test, previously recorded as *unevaluable*, is now evaluable |
| **B7** | Precision risk is quantifiable — how often a lone top-3 placement is wrong can now be measured |
| **D-QR** | Re-testable on labels with headroom, instead of the rank-only workaround |
| **Every NDCG figure** | No longer rests on labels drawn from the system's own top-3 output — real graded labels exist |

None of these re-runs have been executed yet — completing ENH-11 makes them possible, it doesn't run them.

---

## 9. Honest limits

**Single judge.** No inter-rater agreement, only self-consistency. State it plainly — the consistency re-check is a proxy, not a substitute.

**Domain expertise required.** Grading CUDA documentation relevance needs someone who can tell a swizzle pattern from a bank-conflict explanation. That's a strength here — the judge is the project author — but it isn't reproducible by an arbitrary reviewer.

**Pool depth 10 is a compromise.** Documented in §3.1. A chunk at rank 15 that would have graded 3 is invisible to this label set.

**16 queries is still small.** ENH-11 fixes the labels, not the sample size. Case-level claims stay directional.

---

## 10. What this changes about the project

Right now, every retrieval finding carries a caveat: *measured against labels built from the system's own output.* That caveat has constrained four Family A experiments, forced three later hypotheses onto a workaround metric, and left two verdicts explicitly provisional.

ENH-11 removes it.

The findings themselves may not change much — B3's displacement mechanism is arithmetic and won't care. But the difference between *"we think this is true and here's why we can't be sure"* and *"we measured this properly"* is the difference between a project that documents its limitations and one that resolves them.

---

*Related: `docs/uat/round3_family_a_findings.md` §3 (circularity) · `docs/uat/round3_b3_b4_findings.md` · `docs/uat/round3_b7_findings.md` §precision risk · `run_day9_relevance_labelling.py` (the original, circular labeller)*
