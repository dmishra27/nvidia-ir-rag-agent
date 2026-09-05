# UAT Round 3 — Family A Findings

**Reference:** UAT-NVIR-2026-003-A
**Executed:** 18–20 August 2026
**Status:** Six hypotheses resolved · one hardware-blocked
**Plan:** `docs/uat/round3_hypothesis_test_plan.md`

---

## 1. Provenance — read this first

> ⚠️ **Correction in force — CORR-NVIR-2026-001 v2.0 (22 Aug 2026).** A1's write-up quoted a BM25 score
> belonging to a different query, and A1 ran against different cases from the ones the claim was
> documented on. Its measurements are real; its evidence base and mechanism are corrected below.
> Hypothesis B3 is unrunnable as written (this notice and R9 below originally called it "B4" — an
> off-by-one against the plan, corrected 31 Aug; B4 is a separate, valid hypothesis). **No other
> verdict in this document is affected**, and commit `609937c` was correct throughout. See
> `docs/uat/correction_notice_a1.md`.

This document records outcomes for Family A. **Not all of them are reproducible from this repository**, and the table below states which are which. That distinction is deliberate and non-negotiable: an unreproducible number should never be presented alongside a reproducible one without a flag, because a reader cannot tell them apart from the prose.

| Hypothesis | Verdict | Provenance | Reproducible? |
|---|---|---|---|
| **A1** | Partially confirmed — **wrong evidence, see CORR-001** | Committed — `609937c` (evaluation §4, measured Q1 data) | **Yes**, but tests the wrong queries |
| **A2** | Falsified, wrong mechanism | Analysis-only — computed in session, no script or output retained | **No** |
| **A3** | Falsified where testable | Analysis-only — computed in session, no script or output retained | **No** |
| **A3 Part 2** | Label-sparsity artifact | Analysis-only — computed in session, no script or output retained | **No** |
| **A4** | Indirectly supported | Analysis-only, plus committed prior evidence (Sign-Off §8.2) | **Partial** |
| **A5** | Confirmed | Committed — `0149ca4` (3 scripts, 4 JSON artifacts) | **Yes** |
| **A6** | Falsified | Analysis-only — computed in session, no script or output retained | **No** |
| **A7** | Blocked | N/A — never executed | N/A |

**What "analysis-only" means here.** The experiment was run, the output was read and interpreted, and the verdict below reflects that output. But no script was written to a file and no result was persisted, so the figures cannot be regenerated from this repository. They should be treated as recorded observations, not as verified measurements, and **re-run before being cited in any external context.**

**Why this matters, concretely.** A5's first top-100 run had exactly this status. A later session searched the repository, found no trace of it, and correctly refused to build on numbers it could not verify. Re-running from scratch reproduced the original figures to three decimal places — the numbers had been real all along — but establishing that cost two extra runs and an hour of compute on a memory-constrained host. `0149ca4` exists because of that lesson. The five analysis-only rows above are the same debt, unpaid.

**Re-running is not merely a matter of persistence.** Every Family A verdict was measured against qrels that §3 shows to be circular. When graded, retriever-independent judgements exist (ENH-11), these hypotheses need re-testing regardless of whether their scripts were saved.

---

## 2. Verdicts

### A1 — A cross-encoder rescues RRF corroboration failures when the pool is deep enough
**Partially confirmed — but executed against the wrong evidence.** *Provenance: committed, `609937c`. Corrected by CORR-NVIR-2026-001.*

**The measurements stand.** Run on Round 2 Q1 (`"CUDA cudaMalloc function parameters"`), target chunk `cc6c8e53936d04e9b192a7d5` — the `cudaMalloc(void**, size_t)` signature block. The cross-encoder promoted it from pool **rank 4 to rank 2**; a `cudaFreeArray` chunk retained rank 1.

**The mechanism as originally recorded was wrong.** The plan stated the target was BM25 rank 1 at score 33.4 against 12.1 for rank 2 — a 2.8× gap that fusion discarded. `uat_superiority_cases_raw.json` shows otherwise:

| BM25 rank | chunk_id | Score |
|---|---|---|
| 1 | `381cf7a1dddd75346b7446ee` — `cudaFreeMipmappedArray` See-also block | **12.1774** |
| 2 | `cc6c8e53936d04e9b192a7d5` — the target | **11.99** |

The target is at **rank 2, trailing by 1.5%**. The 33.4207 figure belongs to **Q8**, chunk `fd5aa3318c4def606709ac98`. Round 1's raw file shows identical ranks, so no configuration produces the plan's numbers.

**Corroboration bias was present — via a different route than the plan described.** The evaluation page (`609937c`) measured it: the target reaches BM25 rank 2 but **dense rank 75**, giving a single-signal RRF score of `1/(60+2)` ≈ 0.0161. A `cudaStreamAddCallback` chunk that *both* retrievers rank third scores `2/(60+3)` ≈ 0.0317 and takes fused rank 1. The target falls to fused rank 4. Two moderate agreements beat one solid single-signal result — but the asymmetry is dense's rank 75, not a BM25 confidence gap.

**And the cross-encoder failed on the same chunk by an unrelated mechanism.** After re-ranking, a `cudaFreeArray` chunk holds CE rank 1 (6.0145) ahead of the signature (5.4647). Per `609937c`: *"the cross-encoder never saw rank positions, only text — yet the same wrong chunk won under both mechanisms."* **Two independent ranking mechanisms, sharing no assumptions, converge on the same wrong chunk.** That is not a ranking problem twice over — it points at the chunk.

**What Q1 demonstrates, then.** The chunker produces chunks that *straddle a section boundary* — trailing "See also" cross-references from one API entry fused to the opening header of the next. Dense with identifiers, empty of prose. On a query built from function-name vocabulary, a chunk that is *nothing but function names* — including `cudaMalloc` itself, in its See-also list — is structurally advantaged.

So straddling chunks **corrupt lexical ranking directly and survive cross-encoding** — defeating one mechanism that reads rank positions and another that reads only text. That is the strongest single argument for the chunking defect available anywhere in Family A, and the same failure class as DEF-10 (table-of-contents chunks polluting BM25). Family C's C4 is confirmed. On the C1-versus-C5 trade-off, this evidence favours **merging** See-also blocks into their parent entry over filtering them: the cross-references are useful, but only when attached to explanatory text.

**Round 1's actual recommendation remains untested.** It nominated its own Q3 (`H100 HBM2e memory capacity`) and Q7 (`shader processor count`) as regression candidates once re-ranking landed — both cases where dense uniquely found the answer and RRF discarded it. Q1 is a poor substitute: the cross-encoder fails there because the competing chunk defeats text-based scoring too, so it cannot show whether re-ranking *corrects fusion*. That needs a case where the chunk is sound. See CORR-001 §5 for **A1-R**.

---

### A2 — Candidate pool depth determines whether re-ranking can help
**Falsified — but not by either predicted mechanism.** *Provenance: analysis-only, not reproducible.*

Predicted NDCG@10 would rise monotonically with pool depth then flatten at pool-recall saturation; falsification condition was flat gain across depths. Observed: **MRR declined monotonically with depth.** Neither branch.

**Root cause: the qrels are circular.** They were constructed by `run_day9_relevance_labelling.py` from the top-3 fused results of the system under evaluation. Deepening the pool surfaces chunks from positions 4–50; none can appear in a label set drawn from the top 3, so every one scores as an error. The metric was not measuring retrieval degrading — **it was measuring the answer key running out of range.**

The operational question A2 was written to answer — what `candidate_pool_size` should actually be, given a default of 100 and a benchmark exercised at 3 — **remains unanswered.**

---

### A3 — Re-ranking gain is largest on semantic queries, smallest on exact-identifier queries
**Falsified where testable; underpowered throughout.** *Provenance: analysis-only, not reproducible.*

Delta convention: RRF rank − cross-encoder rank at pool depth 20. Positive = re-ranking improved the labelled chunk's position.

| Case | Qualifying | Per-query delta | Mean |
|---|---|---|---|
| 1 · BM25 lexical | Q2 only (Q1, Q3 unlabelled) | +2 | +2.0 |
| 2 · Dense semantic | Q6 only | 0 | 0.0 |
| 3 · RRF hybrid | Q7, Q8, Q9 | −2, −2, 0 | −1.33 |
| 4 · BM25 failure / vocab gap | **none** (Q10, Q11 unlabelled) | — | *no data* |
| 5 · Dense failure / exact lookup | Q12 only | +1 | +1.0 |
| 6 · RRF mixed | Q14, Q15 | 0, −1 | −0.5 |

The largest positive delta in the dataset — **Q2 at +2** — came from Case 1, the category predicted to gain *least*. Case 2, predicted to gain most, was flat. Case 4, also predicted to gain most, **could not be tested at all**: both its queries are unlabelled.

**Statistical limitation, stated plainly.** 8 of 15 queries survived the label filter. Three case types rest on a single query; one on none. A single query changing sign overturns the Case 1 and Case 5 readings. **This is directional evidence, not a test.**

**Unpredicted pattern:** Case 3 was the only group where all queries moved consistently (−2, −2, 0). Investigated below.

---

### A3 Part 2 — The Q7/Q8 dilution
**Label-sparsity artifact. Not a re-ranker weakness, not a repeat of the A1 chunking defect.** *Provenance: analysis-only, not reproducible.*

Q7 and Q8 both held the labelled chunk at rank 1 after fusion; the cross-encoder demoted it progressively with depth (Q8: 1→3→4; Q7: 1→2→3→4). Full text of every competitor at depth 50 was pulled and read.

**Q7** — synchronization performance overhead. Labelled `ccd7708bb87a3ea7259deb4c` at rank 4 (CE 3.3664).

| Rank | Chunk | CE | Margin | Content |
|---|---|---|---|---|
| 1 | `494b8854…` | 5.2069 | +1.84 | *"the implicit synchronization of child kernels done when a thread block ends is more efficient compared to calling `cudaDeviceSynchronize()` explicitly"* |
| 2 | `1a561091…` | 4.4437 | +1.08 | §11.7 Grid Synchronization, Cooperative Groups |
| 3 | `08b3e89a…` | 3.5657 | +0.20 | `cudaDeviceSynchronize()` blocking semantics |

**Q8** — shared-memory bank conflicts. Labels `f6f2063c…` at rank 4 (5.1595) and `1fabb00b…` at rank 9 (3.2932).

| Rank | Chunk | CE | Margin | Content |
|---|---|---|---|---|
| 1 | `b2e88edc…` | 6.4416 | +1.28 | Bank-conflict elimination via `CU_TENSOR_MAP_SWIZZLE_128B` — arguably as good as the labelled chunk |
| 2 | `07060e4e…` | 6.1836 | +1.02 | Matrix-multiplication tutorial; bank conflicts mentioned once, in passing |
| 3 | `fd5aa331…` | 6.1774 | +1.02 | TMA Swizzle — *"To improve performance and reduce bank conflicts, we can change the shared memory layout by applying a 'swizzle' pattern"* |

**Five of six competitors are substantively on-topic.** Only Q8 rank-2 shows the keyword-riding character of the A1 defect — logged as a single supporting data point, not a pattern.

**Secondary finding worth carrying forward.** Chunk `fd5aa331…` had been dismissed in Day-5 testing as one of BM25's *"tangentially-related vocabulary traps"* on raw score behaviour alone. Its full text shows a real, distinct technique for the same problem. **The dismissal did not survive contact with the text.** Chunks written off on score behaviour warrant a content check before the verdict sticks.

---

### A4 — The cross-encoder actively harms exact-lookup queries
**Indirectly supported.** *Provenance: analysis-only for the A3 data; committed prior evidence in Sign-Off §8.2.*

A3 and A4 make deliberately opposite predictions on Case 1. On that single point of disagreement, the data sided with A4: Q2's +2 is what A4 predicts and A3 does not.

**Independent prior evidence exists and predates the hypothesis.** Functional Sign-Off §8.2 recorded that for the query "cudaMalloc", the full pipeline dropped the canonical signature chunk out of its top five entirely — a chunk BM25-only mode held at rank 2 — concluding the cross-encoder *"appears to favour discursive explanation over terse reference material."*

Rests on one query plus that observation. If it survives a proper sample, it argues for **query-dependent routing** (skip the re-ranker on identifier lookups) rather than reweighting — connecting to the deferred ENH-09 adaptive-router work.

---

### A5 — Cohere Rerank v3 and ms-marco disagree in a query-type-dependent way
**Confirmed.** *Provenance: committed, `0149ca4`.*

**Config mapping, per `evaluation/benchmark_runner.py` and Sign-Off §8.3:** Config A = `ms-marco-MiniLM-L-6-v2` (0.5333); Config C = Cohere Rerank v3 (0.5280). The plan's A5 entry names models and config labels in different orders without binding them, which invites misreading — the benchmark runner is the source of truth.

**Three attempts were required.**

*Attempt 1 (cached top-3 pool)* — confirmed, but uninformative. With three candidates, Spearman's ρ admits only {1.0, 0.5, −0.5, −1.0}. Any adjacent swap reads as 0.5, i.e. "below 0.7" **by construction**.

*Attempt 2 (live top-100)* — mean **0.604**, median **0.595**, min 0.210 (Q1), max 0.866 (Q2). Five of six case types below 0.7. Two case-level readings from Attempt 1 were revealed as pool-size artifacts: Case 2's apparent anti-correlation became moderate positive agreement, and Case 4 reversed from apparent falsification (0.750) to clear confirmation (0.543).

*Attempt 3 (head-weighted)* — answered the objection that full-list ρ over-weights an irrelevant tail:

| Measure | Result |
|---|---|
| Top-1 agreement | **53%** (8/15) |
| Overlap@10 | **5.40 / 10** |
| Overlap@5 | **2.67 / 5** |
| Spearman, top-10 only | **0.177** (lower than full-list) |
| RBO (p=0.9) | **0.556 mean / 0.590 median** |

**RBO is decisive.** It discounts the tail geometrically; if agreement were head-concentrated with tail noise, RBO would exceed full-list ρ. It does not — it sits marginally below (0.556 vs 0.604), exceeding full-list ρ in only 5 of 15 queries, mean gap −0.048. The tail-confined-divergence reading has no support.

**Conclusion:** the two re-rankers genuinely disagree at the head. The 0.0053 aggregate NDCG gap conceals substantial per-query divergence; the choice between them is **not purely a cost decision.**

**Heterogeneity worth flagging as hypothesis, not result.** Agreement concentrates where discrimination is easiest and collapses where it matters most — Case 4 (one dominant answer) reaches 100% top-1 match and RBO 0.791, while Case 5 (exact lookup) shows the worst agreement of all six including negative head correlation (ρ@10 −0.239, RBO 0.386). **n=2 per case.** Carry to the graded-relevance work.

---

### A6 — Re-ranking a BM25-only pool beats re-ranking an RRF pool on identifier queries
**Falsified.** *Provenance: analysis-only, not reproducible.*

BM25-only pool performed **worse** than the fused pool on Case 1 and Case 5.

**Mechanism.** Dense retrieval was performing an uncredited filtering function. DEF-10 documented table-of-contents and dot-leader chunks surfacing at the head of BM25-only rankings across three queries; dense retrieval reads them as meaning-poor and ranks them low, dragging their fused position down *before the re-ranker sees them*. Removing dense readmits those distractors to the shortlist.

Fusion's recall benefit outweighs its precision cost **even on the query class where the precision cost was expected to be worst.** The cheapest candidate fix for corroboration bias is not available.

Generalisable caution: **a component's value includes what it filters, not only what it contributes.**

---

### A7 — bge-reranker-v2-m3 changes the picture
**BLOCKED. No verdict.** *Provenance: never executed.*

Config B OOMs at model load on the 8 GB CPU-only host. Per the plan's explicit instruction, **its performance is not to be estimated from Configs A and C.** Two points and an interpolation is not a measurement.

Not a new constraint: commit `d7a998f` (13 July) records a UAT regression *"deferred due to memory"*, five weeks prior. During A5's top-100 run, free RAM dipped to 134 MB loading a *smaller* encoder.

Unblocking requires Colab, a rented GPU hour, or a larger host.

---

## 3. Cross-cutting finding — the instrument constrained the experiments

Four experiments were limited by one cause: **the committed benchmark only ever looked three chunks deep.**

- Qrels built from top-3 fused pools cannot credit anything found deeper — *A2, A3 Part 2*
- Cached shortlists three items long cannot yield meaningful rank correlation — *A5 Attempt 1*
- One whole case type was untestable, both queries unlabelled — *A3*
- **Both queries the plan named as motivating cases — Q1 and Q10 — carry zero positive labels**, because in each case fusion displaced the correct chunk out of the top 3 before human review — *A1, A3*. (Note: CORR-001 establishes these were the wrong motivating cases; Round 1's Q3 and Q7 were the documented ones.)

That last point is the sharpest. The bias under investigation had **systematically removed its own evidence from the instrument used to investigate it.**

**The HITL judgement was sound; the sampling was not.** A human read each candidate and marked relevance correctly. But the pool shown to that human had already been filtered by the system the labels would go on to evaluate — `run_day9_relevance_labelling.py` (3 Aug, 11:12) ran one minute before `run_day9_benchmark_ac.py` (11:13), from the same output.

**Consequence for ENH-11.** The case for graded, retriever-independent relevance judgements now rests on measurement rather than assertion. Requirements: pooled across multiple configurations so no single system gates what a human sees; depth beyond 3; graded (0–3) rather than binary, so "also relevant" is expressible.

---

## 4. Recommendations

| # | Action | Refs |
|---|---|---|
| **R1** | Build graded, retriever-independent relevance judgements before citing any Family A verdict externally. Four experiments are constrained by the current qrels. | ENH-11 |
| **R2** | Re-run A2, A3, A3-2, A4 and A6 with scripts and outputs persisted to the repository. Current verdicts are recorded observations, not verified measurements. | §1 |
| **R3** | Resolve `candidate_pool_size` empirically. A2 did not answer it; default 100, benchmarked at 3. | A2 |
| **R4** | Investigate the chunker's section-boundary handling. Straddling chunks are confirmed on Q1 and are Family C's C4. Evidence now favours **merge** over filter (C1 vs C5) — See-also blocks carry navigational value but corrupt ranking when standalone. | A1, C4 |
| **R5** | Do not treat Config A and Config C as interchangeable. Record which is deployed alongside every NDCG figure. | A5 |
| **R6** | Keep A7 visible as blocked. Do not interpolate. | A7 |
| **R7** | Bind model names to config labels in the same sentence, everywhere. The plan's A5 entry does not, and the resulting misreading would have inverted every A5 conclusion. | A5 |
| **R8** | Run **A1-R** — the corroboration-bias claim has never actually been tested. Round 1's Q3 and Q7 are the documented cases. Requires DEF-19 resolved for Q3. | CORR-001 §5 |
| **R9** | ~~Re-specify **B4**~~ **B3** before running Family B. Its premise — Q1's "2.8× BM25 gap" — does not exist. **Done 31 Aug**: re-specified around single-signal displacement, target-chunk rank as metric (no ENH-11 dependency), paired with B4 (score-normalised fusion) as the remedy. **Both run 2 Sep** (`round3_b3_b4_findings.md`): B3 confirmed binary / graded falsified as pre-registered; B4 a directional remedy with the predicted stability cost, not recommended. **B7 added + thresholds locked 4 Sep (`4a15435`), executed 5 Sep** (`round3_b7_findings.md`) as an offline re-score of the B3/B4 table under a binary single-retriever-preference rule (N=3, M=100): **confirmed as pre-registered** — fires on Q5 and R1-Q7 only, recovers both (R1-Q7 `10 → 1`, beating B4's 3), regresses none of the seven fused-rank-1 targets. Beats RRF and B4 on the two worst-displaced queries; not wired in — the precision cost of a lone top-3 placement is B3's blind spot and needs ENH-11. | CORR-001 §3.3 |
| **R10** | Adopt round-prefixed query identifiers (`R1-Q7`, `R2-Q7`) throughout. Overlapping numbering across rounds caused CORR-001. | CORR-001 §7 |

---

## 5. Artifacts

**Committed**

| Path | Contents |
|---|---|
| `evaluation/a5_top100_pools.json` | RRF top-100 candidate pools, 15 queries |
| `evaluation/a5_top100_orderings.json` | Full 100-item re-ranked orderings, both configs |
| `evaluation/a5_head_weighted_agreement.json` | ρ@10, Overlap@10/@5, top-1, RBO(0.9) |
| `evaluation/a5_full_list_spearman.json` | Full-list Spearman ρ, per query |
| `run_a5_top100_pipeline.py` | Live retrieval + dual rerank |
| `run_a5_head_weighted_analysis.py` | Head-weighted metrics |
| `run_a5_full_list_spearman.py` | Full-list verification |
| `609937c` | Evaluation §4 with measured Q1 data — **the authoritative account of Q1**; correct as committed |

**Not retained** — A2, A3, A3-2, A4, A6: no scripts, no output files. See §1.

**Prior instruments already in repository**

| Path | Relevance |
|---|---|
| `run_day9_relevance_labelling.py` | Built the qrels — the circularity in §3 is inspectable here |
| `run_day9_benchmark_ac.py` | Config A/C benchmark, `TOP_K = 3` |
| `docs/uat/uat_superiority_cases_executed.md` | Round 2 — the six case types |
| `docs/uat/uat_superiority_cases_raw.json` | Round 2 raw ranks and scores — **the source that exposed CORR-001** |
| `docs/uat/uat_day5_retrieval.md` | Round 1 — corroboration-bias findings and the Q3/Q7 regression nominations |

---

*Family A closed: six resolved, one blocked. Families B–F (24 hypotheses) remain open.*
