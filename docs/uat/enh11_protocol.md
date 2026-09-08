# ENH-11 — Graded Relevance Judgements
### Protocol for building a retriever-independent, graded qrel set

**Reference:** ENH-NVIR-2026-011
**Status:** Specified, not started
**Prepared:** 4 September 2026
**Blocks:** A2, A3, A3-2, A4, A6 re-runs · B4's aggregate test · B7's precision risk · D-QR re-test · every future NDCG figure

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

| Stage | Work | Output |
|---|---|---|
| **1** | Build pool, build interface | Scripts committed, pool counts known |
| **2** | Judge **5 queries** — suggest Q1, Q4, Q7, Q10, R1-Q7 | ~110 judgements. Validates the process |
| **3** | **Review stage 2 before continuing.** Does the scale work? Are you consistent? Is the interface fighting you? | Adjust now, not after 350 |
| **4** | Judge the remaining 11 | ~240 judgements, across sessions |
| **5** | Consistency analysis, write findings | Disagreement rate, the qrel file itself |
| **6** | Re-run what was blocked | See §8 |

**The stage-3 review is not optional.** Discovering at judgement 300 that your 2-vs-3 boundary drifted means redoing all of them.

---

## 8. What unblocks on completion

| Item | What becomes possible |
|---|---|
| **A2** | Re-test pool depth against labels that credit deep finds. The original question — what should `candidate_pool_size` be — is still unanswered |
| **A3** | All 16 queries usable instead of 8; Case 4 gets data for the first time |
| **A3-2** | Q7/Q8's competitors get their actual grades. Expect several 2s and 3s |
| **A4, A6** | Re-run with retained artifacts against a non-circular key |
| **B4** | Its aggregate NDCG test, recorded as *unevaluable*, becomes evaluable |
| **B7** | Precision risk quantifiable — how often *is* a lone top-3 placement wrong? |
| **D-QR** | Re-test on labels with headroom. 8 of 16 queries currently sit at rank 1 with nothing to improve |
| **Every NDCG figure** | Currently rests on labels drawn from the system's own top-3 output |

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
