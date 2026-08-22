# Phase Two — Interrogating the System
### Everything after the capstone submission
**`8569bfb` (16 August 2026) → `0149ca4` (20 August 2026) · 2 commits · 4 days**

*Companion document to "Phase One — Building the System" and "What We Learned About Re-Ranking"*

---

## Two commits in four days, and why that number is the point

Phase One produced 60 commits in 38 days. Phase Two produced **two**.

That ratio is not a slowdown. It is a change in the kind of work. Phase One was building — every day added a component, and components are commits. Phase Two was *asking whether the thing that was built actually does what it claims* — and questions don't produce commits, they produce findings. Most of those findings live in analysis output, documents, and notes rather than in source code.

But the ratio is also a warning, and this document says so plainly in its final section. Two commits is fewer than the work deserved, and the gap cost real time.

| Commit | Date | What it was |
|---|---|---|
| `609937c` | 18 Aug | Correct evaluation §4 with **measured** Q1 retrieval and re-ranking data |
| `0149ca4` | 20 Aug | A5 live top-100 orderings, head-weighted agreement, full-list Spearman verification |

---

## The pivot: from shipping to scrutiny

Phase One ended with an evaluation page that published the system's own benchmark numbers — a deliberate transparency decision. On that page sat a claim about the cross-encoder:

> *"a cross-encoder re-ranker that scores each candidate independently — rather than by rank position — should not be subject to this failure mode."*

The Retrieval Hypothesis Test Plan, authored 18 August, opens by pointing at that sentence: **"That word is doing a lot of work."**

The claim had never been tested. Worse, the same page recorded the reason it was shaky: on Q1, the canonical `cudaMalloc()` signature chunk *"never reached the candidate pool the cross-encoder saw at all."* A re-ranker cannot rescue what fusion already discarded — so the claim wasn't just unproven, it was arguably unprovable as stated.

**That is the pivot.** Phase One shipped a system and documented its limits honestly enough that the limits became attackable. Phase Two attacked them.

---

## What the plan set out

The Retrieval Hypothesis Test Plan (18 August) defines **31 hypotheses across six families**, each with an explicit prediction, a confirm condition, and a falsify condition written *before* execution:

| Family | Subject |
|---|---|
| **A** | Re-ranker isolation — does the cross-encoder do what the architecture assumes |
| **B** | RRF parameter sensitivity — `k`, weighting, fusion behaviour |
| **C** | Chunk quality and boilerplate — the ingestion side |
| **D** | Query characteristics as predictors — can we route by query shape |
| **E** | Corpus and index scale |
| **F** | Generation-side (blocked) |

The plan also inherits its reporting standard from Round 2, and states why:

> *"The value of Rounds 1 and 2 was not that 10 of 15 held — it was that the 5 failures were diagnosable because the predictions were written down beforehand."*

Family A was executed. Families B through F remain open.

---

## Commit 1 — `609937c`: a claim becomes a measurement

*18 August — "docs(eval): correct section 4 with measured Q1 retrieval and re-ranking data"*

This is hypothesis **A1** being written back into the repository.

A1 asked whether a cross-encoder can rescue what fusion loses, run against Q1's target chunk `cc6c8e53936d04e9b192a7d5` — the `cudaMalloc(void**, size_t)` signature block that BM25 had ranked **first with a score of 33.4 against 12.1** for its own rank 2, a 2.8× confidence gap that RRF discarded entirely because RRF looks only at rank position.

**Result: partially confirmed.** The cross-encoder lifted the target from rank 4 to rank 2 — real, measurable rescue — but a `cudaFreeArray` chunk still held rank 1.

**And the diagnosis mattered more than the verdict.** Reading the competitor's actual text revealed the chunker was producing chunks that *straddled a boundary*: the trailing "See also" boilerplate of one API entry glued to the opening header of the next. Dense with function names, empty of explanation — precisely the profile keyword matching over-rewards. A **chunking defect masquerading as a ranking problem**, and one the chunk quality scorer built on day one of Phase One had not caught.

The commit replaces an assertion on the evaluation page with a measurement. That is the entire spirit of Phase Two in a single diff.

---

## The uncommitted middle: A2, A3, A4, A6

Between the two commits sits the bulk of the analytical work, and **none of it produced a commit at the time.**

**A2 — pool depth.** Predicted quality would rise with depth then flatten. It *declined*. Root cause: the qrels were built from top-3 pools of the system being evaluated, so anything the re-ranker surfaced from positions 4–50 was scored as an error by construction. The metric was measuring the answer key running out of road.

**A3 — does re-ranking help more on some query types?** Predicted largest gains on semantic and vocabulary-gap queries. The largest gain in the whole dataset came from a *lexical* query. One case type couldn't be tested at all — both its queries were unlabelled. Only 8 of 15 queries survived the label filter; three case types rest on a single query each.

**A3 Part Two — the Q7/Q8 dilution.** Chased the one consistent pattern A3 surfaced. Pulled the full text of every chunk outranking the labelled one at depth 50: **five of six were substantively on-topic**. The apparent failure was label sparsity, not re-ranker weakness — the cross-encoder was finding genuinely good chunks a three-per-query answer key had no way to credit.

**A4 — does the cross-encoder harm exact lookups?** Deliberately the inverse of A3 on the same case type. On the single point of disagreement, the data sided with A4 — consistent with the functional sign-off's §8.2 observation, made a week earlier, that the cross-encoder *"appears to favour discursive explanation over terse reference material."*

**A6 — could we skip fusion for identifier queries?** Falsified cleanly. The BM25-only pool performed *worse*, revealing that dense retrieval had been quietly suppressing the table-of-contents distractors DEF-10 had documented during functional testing. **A component was doing a job nobody had credited it with**, and only removing it made that visible.

**A7 — blocked.** `bge-reranker-v2-m3` OOMs at model load on the 8 GB CPU-only host. Recorded as blocked with no verdict and an explicit instruction not to interpolate from the other two configs. The same ceiling had already deferred a UAT regression back on 13 July.

---

## Commit 2 — `0149ca4`: A5, and the lesson that produced it

*20 August — "A5: live top-100 Config A/C orderings, head-weighted agreement, full-list Spearman verification"*

A5 asked whether Config A (ms-marco, NDCG 0.5333) and Config C (Cohere Rerank v3, 0.5280) are actually interchangeable, given a gap of 0.0053. It took **three attempts**, and the sequence is the most instructive thing in Phase Two.

**Attempt 1 — confirmed, and worthless.** Run over the cached three-candidate pool. With three items, Spearman's ρ can only take four values: {1.0, 0.5, −0.5, −1.0}. Any single adjacent swap reads as 0.5 — *"below 0.7" by construction*. The hypothesis technically confirmed; the instrument was too coarse for that to mean anything.

**Attempt 2 — live top-100.** Median 0.595, five of six case types below 0.7. Two case-level readings that had pointed the *wrong way* at three candidates were revealed as pure pool-size artifacts — one had appeared to falsify outright.

**Attempt 3 — head-weighted.** Answered the objection that full-list correlation over-weights an irrelevant tail. Top-1 agreement **53%**; Overlap@10 **5.4 of 10**; head-restricted ρ **0.177**, *lower* than full-list; and RBO(0.9) at **0.556** failing to exceed the untruncated 0.604. If divergence were tail-confined, a top-weighted measure would have to read *higher*. It doesn't.

**Verdict: confirmed.** The two re-rankers genuinely disagree about what belongs at the top. The 1% aggregate gap conceals real per-query divergence, and choosing between them is not purely a cost decision.

### Why this commit exists at all

Attempt 2 ran successfully, computed its numbers in a temporary script, and **persisted nothing**. A later session searched the repository for those numbers, found no file, no commit, no MLflow run matching that shape — and correctly refused to proceed on figures it could not verify, declining to fabricate the missing ranks.

Re-running the entire pipeline from scratch reproduced the original figures **to three decimal places**. The numbers had been real all along.

Cost: two extra runs, an hour of compute on a memory-constrained machine, and a genuine scare about data integrity. Avoidable by writing one JSON file.

`0149ca4` is that lesson made permanent — four artifacts and three scripts, so the numbers can be reproduced rather than taken on trust. Its commit message says so explicitly.

---

## What Phase Two actually established

**Six hypotheses resolved, one hardware-blocked.** But the individual verdicts are not the finding.

**Four separate experiments were constrained by the same cause:** the committed benchmark only ever looked three chunks deep.

- The answer key was built from three-deep pools, so it cannot credit anything found deeper *(A2, A3 Part Two)*
- **Both** queries that motivated the entire investigation — Q1 and Q10, the two the evaluation page named as fusion failures — have **zero positive relevance labels**, because the bias under study had displaced their answers before human review *(A1, A3)*
- One whole case type was untestable, both its queries unlabelled *(A3)*
- Cached shortlists were three items long, so correlation couldn't take a meaningful value *(A5)*

The bias under study had systematically erased its own evidence from the instrument used to study it.

That reframes the phase. It is not *"seven findings about re-ranking."* It is **empirical evidence that the evaluation apparatus was constraining what could be concluded — plus what was found anyway.** The case for building graded, retriever-independent relevance judgements now rests on measurement rather than assertion.

---

## The honest assessment of this phase's record

Phase One committed 60 times in 38 days. Phase Two committed twice in four.

Some of that is legitimate — analysis produces findings, not source code, and the Hypothesis Test Plan is a document rather than a file in the repository. But not all of it is legitimate. **A2, A3, A4 and A6 produced substantive results and left no committed trace at the time.** The A5 incident demonstrated exactly what that costs: work that isn't persisted is, to anyone looking later, indistinguishable from work that was never done — or worse, from work that was invented.

The single most transferable lesson from these four days is the one that produced the second commit:

> **"I ran it and it said X" is not a result. "Here is the script, here is the output, run it yourself" is.**

Applied consistently, Phase Two would have produced perhaps eight commits rather than two, and none of the reconstruction would have been necessary.

---

## What remains open

- **Families B through F** — 24 of the plan's 31 hypotheses, untouched
- **A7** — blocked pending compute with more than 8 GB
- **Graded relevance judgements** — now the documented prerequisite for trusting any further re-ranking conclusion
- **The A2/A3/A4/A6 findings** — resolved analytically, still uncommitted

---

*Phase Two status: Family A closed (six resolved, one blocked). 2 commits, 16–20 August 2026. Full detail in "What We Learned About Re-Ranking."*
