# Retrieval Hypothesis Test Plan
### nvidia-ir-rag-agent · UAT Round 3 — ranking, fusion and re-ranking experiments
*18 August 2026 · Reference UAT-NVIR-2026-003*

> Converted from `Retrieval_Hypothesis_Test_Plan_nvidia_ir_rag_agent.docx` for repository readability.
> The plan's status line reads "Version 1.0 — planned, none executed". Family A has since been
> executed; see `docs/uat/round3_family_a_findings.md`. **B3 and B4 were executed together on
> 2 September 2026** (`run_b3_b4_fusion_eval.py` → `evaluation/b3_b4_fusion_eval.json` →
> `docs/uat/round3_b3_b4_findings.md`): B3 confirmed in its binary form, its graded form
> falsified as pre-registered; B4 (score-normalised fusion) is a directional remedy that carries
> the predicted stability cost and is **not** recommended for wiring in. The rest of Families
> B–F remain open. Hypothesis **D-QR** (conditional query rewriting, §6) was added on 29 August
> 2026; implementation is `retrieval/query_rewrite.py`, evaluation is `run_dqr_eval.py` →
> `docs/uat/round3_dqr_findings.md`. Hypothesis **B3** (§4) was re-specified on 31 August 2026
> per CORR-NVIR-2026-001 §3.3 — its original score-ratio premise was void; it now tests
> single-signal displacement under fusion and is paired with B4. Hypothesis **B7** (§4) was
> added on 4 September 2026, specification only — not implemented, not executed — as the binary
> alternative to B4: a single-retriever-preference rule motivated by B3's finding that fusion
> displacement is binary, not graded.

---

| Field | Detail |
|---|---|
| Document | Retrieval Hypothesis Test Plan (UAT Round 3) |
| Reference | UAT-NVIR-2026-003 |
| Version | 1.0 — planned, none executed |
| Builds on | UAT Rounds 1 and 2 (24 queries) · Functional Test Sign-Off v2.2 §8.1–8.3 · Enhancement Completion Report ENH-NVIR-2026-002 |
| Corpus | 5,389 passages · CUDA C++ Programming Guide, Runtime API, Math API Reference, Best Practices Guide, Nsight Systems User Guide |
| Pipeline under test | BM25 (Postgres) + dense e5-base-v2 (Qdrant) → RRF k=60 → cross-encoder ms-marco-MiniLM-L-6-v2 |
| Author | Debabrata Mishra |


## 1.  Why This Round Exists

Rounds 1 and 2 established which retriever wins on which query type. They did not test the component the architecture most depends on. The cross-encoder appears in the existing evaluation only as a footnote on two queries, despite Sign-Off §8.2 finding that it changes rankings materially and not uniformly for the better.

One claim in particular is currently asserted but unproven. The evaluation page states that RRF cost the best available chunk on Q1 and Q10, and that "a cross-encoder re-ranker that scores each candidate independently — rather than by rank position — should not be subject to this failure mode." That word is doing a lot of work. It has not been demonstrated, and the same page records why: on Q1, the canonical cudaMalloc() signature chunk "never reached the candidate pool the cross-encoder saw at all". The re-ranker cannot rescue what fusion already discarded.

So the headline experiment of this round is not "does re-ranking help" but "can re-ranking help, given a candidate pool deep enough for it to see the answer". Everything else follows from that.


### 1.1  What is already established


| Finding | Evidence | Status |
|---|---|---|
| BM25 wins on exact API identifiers | Q1, Q3, Q13 | Established, 10 of 15 case hypotheses held |
| Dense wins on paraphrased/conceptual phrasing | Q6, Q10, Q11 | Established |
| RRF wins or ties on mixed-signal queries | Q7, Q8, Q9, Q15 | Established |
| RRF corroboration bias displaces a lone strong hit | Q1, Q10 | Observed, mechanism proposed, remedy untested |
| IDF dilution collapses BM25 on frequent identifiers | Q12 (cudaDeviceSynchronize) | Observed on one query |
| Low-information boilerplate is 7.9% of top-10 | 14 queries, 140 results hand-classified | Measured, three classes identified |
| Cross-encoder favours discursive prose over signatures | 2-query sample, Day 6 regression | Weakly observed, explicitly caveated |


### 1.2  Constraints this plan must respect

- The committed benchmark is 15 cached queries at top-3 RRF candidates. Any experiment varying pool depth cannot reuse that cache and must re-run retrieval live.
- Config B (bge-reranker-v2-m3) is hardware-blocked on an 8 GB CPU-only host and has never produced a figure at any scope. Hypotheses involving it are marked blocked rather than pending.
- The Qdrant collection holds 5,389 points, below the 10,000-vector HNSW threshold, so dense retrieval is exhaustive. Every NDCG figure produced here is an upper bound relative to approximate search, and must be reported as such.
- The deployed service runs BM25 only. All experiments here are local, and none should alter what the public URL serves.
- The NDCG gate sits at 0.50 with 5.6% to 6.7% headroom. Any experiment that changes the indexed corpus requires re-baselining rather than a pass/fail read against the current gate.

## 2.  Hypothesis Families

Thirty-three hypotheses across six families (D-QR added 29 August 2026; B7 added 4 September 2026). Each states what would confirm it and what would falsify it, because a hypothesis that cannot fail is not a hypothesis. Families A and C are the ones worth doing first; F is mostly deferred.


| Family | Theme | Count | Priority |
|---|---|---|---|
| A | Re-ranker isolation — the untested arm | 7 | P1 — the gap this round exists to close |
| B | RRF parameter sensitivity | 7 | P2 — cheap, and directly tests the corroboration mechanism |
| C | Chunk quality and boilerplate | 6 | P2 — closes DEF-10 with evidence rather than assertion |
| D | Query characteristics as predictors | 5 + D-QR | P3 — the most publishable, the least operationally urgent |
| E | Corpus and index scale | 4 | P3 — expensive, invalidates existing baselines |
| F | Generation-side ranking effects | 3 | P4 — blocked on API credit |


## 3.  Family A — Re-ranker Isolation

The central family. Every hypothesis here is currently untested, and A1 is the one the evaluation page has already promised an answer to.


### A1  A cross-encoder rescues RRF corroboration failures when the candidate pool is deep enough

*Correction annotation, 31 August 2026 — see CORR-NVIR-2026-001 §2. A1's re-ranking measurements stand; its evidence base does not, and is left here as originally written per plan-of-record convention. The "BM25 rank 1, score 33.4 against 12.1 for its own rank 2 — a 2.8× gap" in the Claim and Protocol below is a cross-query transposition: 33.4207 is R2-Q8's BM25 rank-1 score (`fd5aa331…`, the TMA Swizzle chunk). Q1's actual BM25 top two are `381cf7a1…` (a "See also" boilerplate block) at 12.1774 and the target `cc6c8e53…` at 11.99 — a 1.5% spread, target at rank 2, not rank 1. A1 was also run against R2-Q1 and R2-Q10, whereas the corroboration claim it set out to test was documented on Round 1's Q3 and Q7 (CORR-001 §2.2). The claim is re-specified as **A1-R** in §5, to run on R1-Q3 and R1-Q7 where the target chunk is sound; the single-signal-displacement question is separately re-specified as B3 in §4.*


| Field | Detail |
|---|---|
| Claim | On Q1 and Q10, where RRF displaced the single best chunk, a cross-encoder scoring candidates independently will restore it to rank 1 — provided the pool it sees actually contains that chunk. |
| Why it should hold | A cross-encoder computes a relevance score per (query, passage) pair. Rank position never enters the calculation, so the mechanism that caused the displacement cannot operate. |
| Why it might not | The chunk may be genuinely hard for the cross-encoder. Sign-Off §8.2 found it favours discursive prose over terse reference entries, and the Q1 answer is a terse signature block. |
| Confirms | Rank 1 after re-ranking is the chunk RRF displaced, on both queries, at pool depth 20 or greater. |
| Falsifies | The displaced chunk stays below rank 1 even when present in the pool, or the cross-encoder promotes a different chunk entirely. |


**Protocol**

for depth in 3 10 20 50 100:

pool = rrf_fuse(bm25(q, depth), dense(q, depth), k=60)[:depth]

record: is target_chunk in pool?  (pool recall)

reranked = cross_encoder.rerank(q, pool)

record: rank of target_chunk before and after

Run for Q1 with target chunk cc6c8e53936d04e9b192a7d5 (the cudaMalloc(void**, size_t) signature block, BM25 rank 1, score 33.4 against 12.1 for its own rank 2), and for Q10 with dense's latency-hiding match.

Report pool recall separately from re-ranking effect. If the target is absent at depth 3 but present at depth 20, that is itself the finding: the committed benchmark's top-3 scope makes re-ranking structurally unable to fix fusion errors, which reframes the smoke-scope caveat from a sampling limitation into a design one.


### A2  Candidate pool depth determines whether re-ranking can help at all

Claim: re-ranking gain is a function of pool recall, not of re-ranker quality, below some depth threshold. Confirms if NDCG@10 after re-ranking rises monotonically with pool depth up to the point where pool recall saturates, then flattens. Falsifies if gain is flat across depths, which would mean the re-ranker is reordering noise.

This is the most operationally useful hypothesis in the family. It answers "what should candidate_pool_size actually be", a parameter currently defaulted to 100 in the API schema but exercised at 3 in the committed benchmark.


### A3  Re-ranking gain is largest on semantic queries and smallest on exact-identifier queries

Claim: the cross-encoder adds most where lexical overlap is weakest. Test across the six Round 2 case types, reporting NDCG@10 delta (RRF versus RRF+rerank) per case rather than in aggregate.

Confirms if Case 2 (dense semantic) and Case 4 (BM25 failure / vocab gap) show the largest positive deltas, and Case 1 (BM25 lexical) and Case 5 (exact lookup) the smallest or negative. Falsifies if the deltas are uncorrelated with case type.

Note this hypothesis and A4 make opposite predictions on Case 1. Running both against the same data is deliberate; whichever survives tells you what the cross-encoder is actually doing.


### A4  The cross-encoder actively harms exact-lookup queries


| Field | Detail |
|---|---|
| Claim | On queries whose correct answer is a terse API signature block, re-ranking demotes it in favour of discursive prose that reads as more explanatory. |
| Existing evidence | Sign-Off §8.2, Day 6 ms-marco regression: re-ranking "tends to confirm or promote fluent, explanatory prose passages (definitions, 'returns an error if...' style text) rather than terse, structurally-formatted API reference entries". Explicitly caveated as a 2-query sample. |
| Test set | Q1, Q3, Q12, Q13, plus CH01–CH04 chip queries — all exact-identifier lookups with a known signature-block answer. |
| Confirms | Signature-block chunks fall in rank after re-ranking on a majority of these queries. |
| Falsifies | Signature blocks hold or improve rank, which would retire the §8.2 caveat. |

If confirmed, this is the strongest argument in the project for query-dependent routing: it would mean the re-ranker should be skipped, not merely reweighted, for identifier lookups. That connects directly to the deferred adaptive-router work (ENH-09).


### A5  Cohere Rerank v3 and ms-marco disagree in a query-type-dependent way

Claim: Config A (0.5333) and Config C (0.5280) differ by only 1% in aggregate NDCG, but that near-tie conceals divergent per-query behaviour. Test by computing rank correlation between the two re-rankers' outputs per query, not just their mean scores.

Confirms if Spearman correlation between the two orderings falls below roughly 0.7 on any case type. Falsifies if they agree closely everywhere, in which case the 1% aggregate gap is noise and the choice between them is a cost decision rather than a quality one.


### A6  Re-ranking a BM25-only pool beats re-ranking an RRF pool on identifier queries

Claim: for exact-identifier queries, feeding the cross-encoder BM25's own top-k — skipping fusion entirely — outperforms feeding it the fused pool, because fusion has already introduced the corroboration distortion A1 describes.

This is the cheapest possible fix for the corroboration bias and worth testing before anything more elaborate. It requires no new components: just a second pipeline path that bypasses RRF.

Confirms if NDCG@10 on Case 1 and Case 5 queries is higher for BM25→rerank than for RRF→rerank. Falsifies if fusion's recall benefit outweighs its precision cost even on identifier queries.


### A7  bge-reranker-v2-m3 changes the picture  (BLOCKED)

Config B has never run. It is hardware-blocked with an OOM at model load on an 8 GB CPU-only machine, and the evaluation page reports it as such rather than omitting it. Any hypothesis about its behaviour is speculation until it runs somewhere with more memory — a Colab session, a rented GPU hour, or a larger local machine.

Recorded here so that the gap stays visible. Do not estimate its performance from the other two configs.


## 4.  Family B — RRF Parameter Sensitivity

The corroboration bias is a consequence of RRF's parameterisation, not an inherent property of fusion. This family tests whether it can be tuned away, and at what cost.


### B1  Lowering k reduces corroboration bias

Claim: RRF at k=60 produces near-flat reciprocal ranks across the top positions — 1/61 through 1/64 span about 5% — which is what allows two mediocre chunks to outweigh one strong one. Lowering k steepens the curve and should restore the displaced chunk.


| k | 1/(k+1) | 1/(k+4) | Spread, rank 1 to 4 |
|---|---|---|---|
| 60 | 0.01639 | 0.01563 | 4.9% |
| 20 | 0.04762 | 0.04167 | 14.3% |
| 10 | 0.09091 | 0.07143 | 27.3% |
| 5 | 0.16667 | 0.11111 | 50.0% |

Confirms if Q1 and Q10 recover their displaced chunk at some k below 60, without NDCG@10 falling across the other 13 queries. Falsifies if recovering those two costs more elsewhere than it gains, which is the outcome I would actually expect: the flatness that causes the bias is the same flatness that makes fusion robust.

Sweep k over {5, 10, 20, 30, 60, 100} and report per-query, not just mean. The interesting result is the shape of the trade-off, not a single optimum.


### B2  Weighted RRF beats uniform on a corpus with asymmetric retriever strength

Claim: applying a weight per retriever — score = w_bm25/(k+rank_bm25) + w_dense/(k+rank_dense) — outperforms the uniform w=1 default, because Round 2 established that neither retriever dominates but each dominates a query class.

Sweep w_bm25 from 0.5 to 2.0 with w_dense fixed at 1.0. Confirms if any weighting beats uniform across all 15 queries. Falsifies if the best global weighting is uniform, which would mean the per-class advantages cancel — itself a useful finding, and an argument for routing rather than reweighting.


### B3  RRF displaces high-confidence single-signal results toward a central rank band

*Executed 2 September 2026 with B4 — see `docs/uat/round3_b3_b4_findings.md`. **Confirmed in
substance, binary form.** Every single-signal-dense target lands strictly below its dense rank
(5/5); every single-signal-BM25 mirror target is rescued from the retriever that buried it (3/3).
Displacement is bimodal on corroboration **presence** — fused rank 1–4 when the other retriever
also returns the target (`+1` to `+3`), fused rank 10–17 when it does not (`+9`, `+15`). The
graded form is **falsified as pre-registered**: displacement does not scale with corroboration
magnitude (Q12 BM25 rank 23 → `+1`; Q10 rank 14 and Q11 rank 6 → `+3` each). Two wording
corrections carried forward: "central band 2–10" → "bimodal, 1–4 vs 10–17"; "irrespective of the
supporting retriever's rank" holds for the finding retriever, not for the other retriever's
presence.*

*Re-specified 31 August 2026 per CORR-NVIR-2026-001 §3.3. The original B3 — "the magnitude of the corroboration effect is predictable from the winning retriever's rank-1-to-rank-2 score gap" — rested entirely on Q1's "33.4 versus 12.1, a 2.8× gap." That gap does not exist: the 33.4 figure belongs to R2-Q8, and Q1's actual BM25 top-two scores are 12.1774 and 11.99, a 1.5% spread (CORR-001 §2.1, §4.1). The score-ratio formulation is withdrawn. The underlying question — does fusion displace high-confidence single-signal results more readily? — is sound, and is re-specified below around single-signal displacement measured by target-chunk rank.*

> **Numbering note.** CORR-001 §3.3, `round3_family_a_findings.md` R9, and `completion_plan.md` originally referred to this hypothesis as "B4" — an off-by-one against this document, where B4 is "score-normalised fusion." Those three references have been corrected to "B3." B4 is the candidate *remedy* for the defect B3 measures; the two are cross-referenced at the end of both entries and are meant to be run as a pair.

| Field | Detail |
|---|---|
| Claim | RRF regresses a single-signal result toward a central band — roughly fused rank 2–10 on this corpus — largely irrespective of the rank its sole supporting retriever gave it. A chunk one retriever ranks 1 cannot reach fused rank 1 without corroboration; a chunk one retriever buries at rank 49–75 is pulled *up* into the same band. The mechanism: RRF reads rank position only and never sees scores, so one contribution of 1/(k+1) is outweighed by two mid-list contributions that sum higher, however confident the lone retriever is. |
| Motivating evidence — displacement | Three live-measured cases (`evaluation/dqr_eval.json`, baseline mode, pool 100, k=60) where dense uniquely holds the answer: **R1-Q7** `shader processor count` — dense rank 1, BM25 absent, **fused rank 10**; **R2-Q10** `latency hiding through instruction level parallelism` — dense rank 1, BM25 rank 14, **fused rank 4**; **R2-Q11** `occupancy versus performance tradeoffs` — dense rank 1, BM25 rank 6, **fused rank 4**. |
| Motivating evidence — mirror | Two cases where the buried signal is dense and fusion pulls the chunk back up: **R2-Q1** `CUDA cudaMalloc function parameters` — BM25 rank 2, dense rank 75, **fused rank 4**; **R2-Q3** `CUDA error cudaErrorInvalidValue description` — BM25 rank 1, dense rank 49, **fused rank 2**. Q1 shows the regression from both sides at once: fusion lifts the target from dense's 75 and pushes it down from BM25's 2, the two meeting at fused rank 4 (CORR-001 §4.1 documents the downward half as corroboration bias). |
| Why it should hold | RRF's score for a chunk is Σ 1/(k+rankᵢ) over the retrievers that returned it. At k=60 a lone rank-1 contribution is 1/61 ≈ 0.0164; two rank-3 contributions sum to ≈ 0.0317. A single strong retriever is arithmetically unable to beat any chunk that two retrievers place in their top ~5, and the strength of its conviction — carried only in scores RRF discards — never enters. |
| Why it might not | The five cases show the relationship is coarse, not strict: R2-Q10 (BM25 rank 14) and R2-Q11 (BM25 rank 6) displace to the *same* fused rank 4 despite different corroboration strength, while R1-Q7 (BM25 absent) is markedly worse at fused 10. Displacement may track only the presence or absence of *any* corroboration, not its degree — the 15-query sweep is what separates the two. |
| Confirms | On the single-signal-dense cases the target's fused rank is strictly worse than its dense rank, and the absent-corroboration case (R1-Q7) is worse-displaced than the weak-corroboration cases; on the mirror cases fusion improves the target's rank relative to the retriever that lost it; and in every displacement case the target's lone 1/(k+rank) contribution is smaller than the summed contribution of the chunk immediately above it in the fused list. |
| Falsifies | Displacement is uncorrelated with corroboration (absent and weak corroboration displace equally, at every depth) — or a displaced target reaches fused rank 1 with no corroboration — or the mirror cases show no fusion benefit — or single-signal targets across the wider set scatter from rank 1 to rank 30 with no central clustering, making "band" the wrong description. |

**Protocol**

1. Take the five live-measured cases above from `evaluation/dqr_eval.json` (baseline mode) as the motivating set — already retrieved at pool 100, RRF k=60, target chunks fixed in that file's `targets` block.
2. Extend to all 15 Round 2 queries (`uat_superiority_cases_raw.json`) plus R1-Q7. For each: retrieve BM25 top-100 and dense top-100 live, fuse with RRF k=60, and record the pre-identified target chunk's rank in the BM25 list, the dense list, and the fused list.
3. Classify each query by corroboration structure: **single-signal-dense** (dense rank ≤ 5, BM25 rank ≥ 6 or absent — covers R1-Q7, R2-Q10, R2-Q11), **single-signal-BM25** (the mirror: BM25 rank ≤ 5, dense rank ≥ 6 or absent — covers R2-Q1, R2-Q3), **corroborated** (both retrievers rank the target ≤ 5), **weak/neither** (neither retriever ranks it ≤ 5). The ≤ 5 / ≥ 6 split is a starting cut, to be reported alongside the raw ranks so a reader can re-bin.
4. For every single-signal case compute: (a) displacement = fused_rank − finding_retriever_rank (the retriever that ranks it ≤ 5); (b) corroboration strength = the *other* retriever's rank on that same chunk, as a continuous variable (6, 14, 49, 75, … ∞ if absent); (c) an RRF-score decomposition — the target's 1/(k+rank) contribution versus the per-retriever contributions of the chunk one rank above it in the fused list.
5. Plot displacement against corroboration strength across all single-signal cases. Report per-query, never aggregate — per §9.1 the effect is invisible in a mean.
6. **Metric is target-chunk rank, not NDCG.** `run_day9_relevance_labelling.py`'s qrels are circular (established by A2/A3) and no retriever-independent graded labels exist yet (ENH-11). Because the metric is the rank of a single pre-identified chunk, B3 needs no graded relevance judgements and is **not blocked behind ENH-11** — unlike Family A's NDCG hypotheses and Family D's router hypotheses. That is what makes B3 runnable now.
7. Persist the per-query JSON before any analysis, per `0149ca4`.

**If confirmed**, this yields a cheap runtime heuristic — when one retriever ranks a chunk in its top ~3 and the other ranks it outside its top ~10 (or not at all), prefer the single retriever's placement over the fused order — and it sets the target for **B4**: score-normalised fusion is the direct candidate remedy, because it restores the score signal whose absence is the mechanism measured here. **Run B3 and B4 as a pair**, on the same queries and target chunks, so B4's normalised-fusion ranks read directly against B3's measured RRF displacement.


### B4  Score-normalised fusion outperforms rank-based fusion on this corpus

*Executed 2 September 2026 with B3 — see `docs/uat/round3_b3_b4_findings.md`. **Directional
remedy, predicted stability cost, not recommended.** Min-max CombSUM reduces single-signal
displacement in 6 of 8 cases (fully recovers the mild cases Q12, Q3 to rank 1; thirds the severe
ones — Q5 17→5, R1-Q7 10→3) but never recovers a severely-displaced un-corroborated target to
rank 1, and it regresses a corroborated target (Q15 rank 1→3) plus two weak cases — exactly the
BM25-unbounded-magnitude fragility the counter-argument below names. Min-max beats z-score here.
The "beats RRF on NDCG@10 across all 15" test is **unevaluable** on this corpus (circular qrels,
A2/A3; ENH-11 not done); on target-chunk rank the result is B4's own predicted "Falsifies" shape
— wins on the displaced queries, loses in aggregate. Do not wire in; re-open with ENH-11.*

Claim: min-max or z-score normalising each retriever's scores before summing preserves the confidence signal RRF discards, and beats RRF on the queries where corroboration bias bites.

The counter-argument is the one that motivated RRF in the first place: BM25 produces unbounded term-weight sums, cosine similarity is bounded, and normalisation across incomparable distributions is fragile in a way rank fusion is not. Expect this to help on the displaced-chunk queries and hurt on stability.

Confirms if normalised fusion beats RRF on NDCG@10 across all 15. Falsifies if it wins on Q1 and Q10 but loses in aggregate — the likely outcome, and worth reporting as the trade-off it is.

**Relation to B3.** B3 measures a displacement defect that exists *precisely because* RRF fuses on rank position and never sees retriever scores. Score-normalised fusion is the direct candidate remedy: min-max or z-scored scores let a single highly-confident retriever hold rank 1 against weak corroboration. Run B4 on the same query set and target chunks as B3, and read each single-signal target's normalised-fusion rank against its measured RRF displacement. The two are better run together than independently — B3 quantifies the defect, B4 tests whether this fix closes it without the aggregate cost the counter-argument predicts.


### B5  Fusion adds nothing when one retriever returns an empty or near-empty result set

Claim: on out-of-corpus queries such as Q4 (shader processor count per streaming multiprocessor), where all methods returned weak results, fusion of two weak lists produces a confidently-ordered list of irrelevant chunks — worse than an explicit empty state, because it looks like an answer.

This connects to DEF-09's open retrieval half. Test by measuring whether the fused top-1 score distribution separates in-corpus from out-of-corpus queries at all. Confirms if there is a threshold that cleanly separates Q4 from the rest. Falsifies if the distributions overlap, which would mean a score threshold cannot be the empty-state trigger and something else is needed.


### B6  RRF corroboration bias is stronger with more retrievers, not weaker

Claim: adding a third retriever increases the probability that two mediocre chunks agree, so the bias worsens rather than averaging out.

Testable cheaply by adding a trivial third ranker (title-match, or a BM25 variant with different parameters) and re-measuring Q1 and Q10. Speculative, but it bears on whether the common instinct to "add more retrievers" is sound on a corpus like this one.


### B7  A binary single-retriever-preference rule recovers severe displacement without B4's cross-scale cost

*Added 4 September 2026. Specification only — not implemented, not executed. The claim, both
thresholds and the falsify branch are committed here before any result exists; see "Post-hoc
tuning" below.*

B3 (2 September, `7120deb`) established that RRF displacement is **binary, not graded**: a target
loses 1–3 fused ranks when the other retriever also returns it somewhere in the pool, and 9–15
when it does not. Q12's weakest corroboration — BM25 rank 23 — produced the *smallest*
displacement (`+1`); Q10 (BM25 rank 14) and Q11 (BM25 rank 6) displaced identically (`+3`).
Decomposition confirmed the mechanism in 7/7 cases: the target's lone `1/(k+rank)` contribution
is ≤ the summed contribution of the chunk one rank above it in the fused list. B4 —
score-normalised fusion, the pre-planned remedy — behaved as its own counter-argument predicted:
it improved 6 of 8 displaced cases but regressed a corroborated target (Q15, rank 1 → 3) plus two
weak ones, because BM25's unbounded term-weight sums and bounded cosine similarity do not share a
scale. Not recommended.

B7 is the alternative the measurement points to. If the defect is binary, the remedy can be
binary: no normalisation, no cross-distribution comparison, no reintroduced scale.

| Field | Detail |
|---|---|
| Claim | When one retriever ranks the target in its top **N = 3** and the other does not return it **anywhere in the retrieved pool (M = 100)**, substituting the single retriever's rank for the fused rank recovers the target on the severely-displaced queries — R1-Q7 (fused 10) and Q5 (fused 17) — without regressing any target currently at fused rank 1. |
| Motivating evidence | B3's two severe-displacement cases: **R1-Q7** `shader processor count` — dense rank 1, BM25 absent, fused rank 10; **Q5** — finding retriever top-3, other retriever absent, fused rank 17 (B4 recovered these only to 3 and 5). Both sit on the "no corroboration" side of B3's bimodal split. The three mild cases — Q10 (`+3`), Q11 (`+3`), Q12 (`+1`) — sit on the "corroboration present" side, at BM25 ranks 14 / 6 / 23, inside the pool and so outside this rule's trigger. |
| Why it should hold | B3 showed severe displacement occurs only when the other retriever contributes nothing to `Σ 1/(k+rankᵢ)`, leaving the target with a single small term while several corroborated-but-wrong chunks each carry two. In that regime the fused order carries strictly less information than the lone retriever's own order, which placed the known-correct target at rank 1 in every measured case. Taking that rank directly bypasses the fusion arithmetic exactly where B3 isolated it as pathological, and the rule is inert everywhere else — so, unlike B4, it never touches a corroborated target. |
| Why it might not | (1) Any rule that promotes a single-signal result reorders the list. A target held at fused rank 1 by *one* retriever alone — nothing corroborating it, but nothing else corroborated either, so it won by default — would be demoted to its raw retriever rank. Eight of the sixteen queries currently sit at fused rank 1; if any is a single-signal default win, B7 regresses it and falsifies on B4's exact grounds. (2) B3 measured displacement of *known-correct* targets only. It is silent on how often a lone top-3 placement is **wrong**; B7 would newly trust a retriever that confidently ranks a wrong chunk first, turning a recall problem into a precision problem B3's design cannot observe. (3) N is under-constrained — all five B3 single-signal cases had finding-retriever rank 1, so N ∈ {1, 3, 5} are indistinguishable on the current data. |
| Confirms | The rule fires on R1-Q7 and Q5 and lands each at its finding-retriever rank (R1-Q7 → 1, better than B4's 3), **and** a spot-check confirms the promoted top-3 chunk is B3's fixed target and not a lookalike, **and** no target currently at fused rank 1 regresses (the rule fires on none of the eight), **and** Q10 / Q11 / Q12 are unchanged. |
| Falsifies | Any fused-rank-1 target regresses (B4's failure mode, disqualifying on the same terms) — **or** R1-Q7 or Q5 does not improve — **or** the chunk B7 promotes to the top on either severe case is not the correct target, exposing the precision cost B3 could not measure. |

**Thresholds — committed now**

- **N = 3** (finding-retriever rank). The loosest value still describable as "top of list"; B3's data does not discriminate below N = 5, so this is recorded as weakly justified rather than derived.
- **M = 100 = pool depth** (the other retriever's rank of the target, or its absence). B3's bimodal split is presence-versus-absence *in the top-100 pool*, not presence within some shallower band: Q12's BM25 rank 23 counted as present and gave the mildest displacement on record. Any M materially below the pool edge would reclassify a functioning corroboration signal as absent and fire the rule on queries B3 showed are only mildly displaced. The pool edge is the only M the measurement supports — wider than B3's closing note, which had mooted "outside its top ~10" before the executed data moved the boundary out.

Under (N = 3, M = 100) the rule is predicted to fire on **R1-Q7 and Q5 only**. It will not fire on Q10 / Q11 / Q12 (other retriever present at BM25 14 / 6 / 23) or on B3's mirror cases Q1 / Q3 (dense present at 75 / 49, where fusion already helps and should be left alone).

**Post-hoc tuning invalidates the test.** N and M are free parameters and both are trivial to fit to a desired firing set once the per-query ranks are visible. Widening M to ~5 would capture Q10 / Q11 / Q12 — which the motivating discussion would like — but B3's rank-23 result is direct evidence against doing so. Either threshold changed after execution voids B7 as a pre-registered result; a re-run with different values is a new hypothesis and gets its own entry.

**Protocol**

1. Reuse B3's persisted per-query table (`evaluation/b3_b4_fusion_eval.json`): BM25 top-100, dense top-100, RRF k = 60, and the target chunk's rank in each of the three lists, for the 15 Round 2 queries (`uat_superiority_cases_raw.json`) plus R1-Q7. No new retrieval — B7 is a re-scoring of that file.
2. Apply the rule offline per query: if `min(bm25_rank, dense_rank) ≤ 3` **and** the target is absent from the other retriever's top-100, set `b7_rank = min(bm25_rank, dense_rank)`; otherwise `b7_rank = fused_rank`.
3. Report per-query: `fused_rank`, `b7_rank`, `delta` (positive = improvement). Never aggregate — per §9.1 the effect is invisible in a mean.
4. As **exploratory only**, outside the pass/fail read: tabulate what `M ∈ {10, 20, 50}` would do to the firing set and the per-query deltas, so the sensitivity the pre-registered M forecloses is at least visible.
5. Metric is **target-chunk rank**, per B3 and D-QR. `run_day9_relevance_labelling.py`'s qrels are circular (A2/A3) and no retriever-independent graded labels exist yet (ENH-11), so NDCG is unevaluable on this corpus and is not reported.
6. If `b3_b4_fusion_eval.json` is missing a field the rule needs, re-run retrieval per B3's protocol and persist the table before any analysis (`0149ca4`).

**Relation to B3 and B4.** B3 measured the defect and found it binary. B4 tried to fix it by restoring the discarded score signal and paid the cross-scale cost its own counter-argument predicted. B7 tries to fix it without reintroducing scores at all: it acts only in the regime B3 isolated as pathological — one retriever confident, the other silent — and is inert elsewhere, which is the structural reason it should not reproduce B4's regression of corroborated targets. On R1-Q7 the two remedies make a directly falsifiable contrast: B7's mechanical prediction is fused rank 1 (dense's own rank), against B4's measured 3. Run B7 against the same table as B3 and B4 so its per-query ranks read against both the measured displacement and B4's normalised-fusion attempt.


## 5.  Family C — Chunk Quality and Boilerplate

DEF-10 measured the problem at 7.9% of top-10 results across three classes. This family tests whether fixing it actually improves retrieval, which the defect record currently assumes rather than demonstrates.


### C1  Filtering boilerplate at ingestion improves NDCG@10


| Field | Detail |
|---|---|
| Classes to filter | (i) title pages and dot-leader contents chunks; (ii) running-header stubs — header plus chapter number plus page number, no body; (iii) See-also cross-reference blocks, dense with API names, near-empty of prose. |
| Corpus effect | Filtering reduces the 5,389 passage count. That figure appears in the page subtitle, the corpus transparency panel, the README, the evaluation page and possibly test fixtures — all require updating together. |
| Confirms | NDCG@10 rises across both re-ranker configs after re-indexing, with the gate re-baselined rather than read against the current 0.50. |
| Falsifies | NDCG holds or falls, which would mean the boilerplate was occupying positions that would otherwise go to something worse, or that the filter is catching legitimate short passages. |

The falsification case is not far-fetched. A See-also block for cudaMalloc genuinely does point at cudaMallocPitch, cudaMallocHost and cudaFree — it is navigationally useful even though it explains nothing. Removing it may cost recall on queries about the family of allocation functions.


### C2  See-also blocks hurt BM25 substantially more than dense

Claim: cross-reference blocks are dense with exact identifiers and empty of semantics, which is precisely the profile BM25 over-rewards and embeddings ignore.

CH01 is the evidence to start from: 5 of its 10 live BM25 results were See-also blocks, on a corpus-wide base rate of 7.9%. Test by classifying top-10 results per retriever across all 15 queries and comparing per-class rates.

Confirms if the See-also rate in BM25 top-10 is at least twice the rate in dense top-10. Falsifies if both retrievers surface them at similar rates, which would point at chunking rather than at lexical matching as the root cause.


### C3  Boilerplate pollution correlates with query-to-document-title vocabulary overlap

Claim: pollution is not uniform. It concentrates on queries whose wording overlaps a document's own title or running header, because every page of that document then carries matching tokens.

The observed instance: "Best practices for GPU memory optimization" returned 4 of 10 non-substantive results against the CUDA C++ Best Practices Guide, while exact-identifier queries returned none. Test by computing token overlap between each query and each source document's title, then correlating with per-query boilerplate rate.

If confirmed, this yields a cheap mitigation that needs no re-indexing: down-weight header and title tokens at query time rather than filtering chunks at ingest.


### C4  Chunk boundaries orphan See-also blocks into standalone chunks

Claim: the recursive splitter at 1500/150 cuts mid-section, leaving cross-reference lists as chunks with no anchoring prose. Evidence: CH01 ranks 5 and 7 consist only of See-also content.

Test by measuring what proportion of See-also-class chunks contain no sentence-terminated prose at all. Confirms if a substantial share are pure reference lists, which would make this a chunking defect rather than a filtering problem — and the fix a boundary rule, not a blocklist.


### C5  Merging orphaned blocks into their parent section beats deleting them

The alternative to C1's filter. Rather than dropping See-also blocks, attach them to the preceding section so their identifiers remain searchable but sit alongside explanatory prose.

Confirms if merged chunking beats both the unfiltered baseline and the C1 filtered version on NDCG@10. This is the outcome I would bet on, because it keeps the navigational value C1 risks discarding.


### C6  Chunk length correlates with retrieval quality in opposite directions per retriever

Claim: BM25 favours short chunks with high term density; dense embeddings degrade on very short chunks because there is too little text to embed meaningfully. If true, chunk-length distribution is a lever that affects the two retrievers oppositely, and any single chunking strategy is a compromise between them.

Test by bucketing chunks by token count and measuring per-bucket hit rates for each retriever across the 15-query set.


## 6.  Family D — Query Characteristics as Predictors

If which retriever wins is predictable from the query alone, routing becomes possible. This family is the empirical groundwork for the deferred adaptive router (ENH-09), and the most publishable material in the plan.


### D1  Maximum IDF of query terms predicts whether BM25 wins

Claim: BM25 wins when the query contains at least one rare identifier, and loses when its identifiers are common across the corpus. Q12 is the observed instance in the negative direction: cudaDeviceSynchronize appears in enough chunks that BM25's rarity weighting collapsed and dense won a query designed as an exact lookup.

Test by computing max-IDF over query terms against the BM25 index and correlating with the per-query winner from Round 2. Confirms if a max-IDF threshold separates BM25 wins from dense wins with reasonable accuracy across the 15. Falsifies if the winners are interleaved along the IDF axis.

This is the single most useful hypothesis for routing, because max-IDF is computable at query time in microseconds from the index you already have.


### D2  Query length predicts the winner independently of IDF

Claim: longer, more natural-language queries favour dense retrieval; short identifier-style queries favour BM25. Round 2 offers surface support — Q5 ("how to make GPU programs run faster", 7 words) went to dense, Q3 ("CUDA error cudaErrorInvalidValue description", 4 words) to BM25 — but length and IDF are confounded in that set.

Test requires deconfounding: construct query pairs with matched IDF and differing length, and vice versa. That means authoring new queries rather than reusing the 15, which is why this sits at P3.


### D3  Queries quoting source text verbatim are won by BM25 regardless of length

Claim: the operative variable is not paraphrase versus exact phrasing but whether the query's vocabulary appears literally in the target passage. The Round 2 note on Q6 is the tell: "this isn't a pure vocabulary-gap case as intended — 'different code paths' closely mirrors the source text's 'different execution paths', so BM25 also finds it."

That note is an acknowledgement that one of the dense-semantic cases was not testing what it intended. Worth correcting in this round by authoring genuine vocabulary-gap queries where no query term appears in the target chunk.


### D4  A two-feature router beats always-RRF

Claim: routing on max-IDF (D1) and literal-overlap (D3) alone — no model, just two computed features and a threshold — outperforms unconditional fusion across the 15-query set.

This is the minimum viable version of ENH-09 and the honest one: if two hand-computed features beat fusion, that is worth knowing before building a classifier. Confirms if routed NDCG@10 exceeds always-RRF. Falsifies if fusion's robustness beats the router's precision, which would retire the adaptive-routing idea on evidence rather than deferring it indefinitely.


### D5  Query type predicts boilerplate exposure

Follows from C3. If title-overlapping queries attract boilerplate, then the same routing features that pick a retriever could also trigger a boilerplate filter selectively, rather than applying it globally at ingest.


### D-QR  Conditional query rewriting helps vocabulary-gap queries and is neutral-to-harmful on exact-identifier queries

*Added 29 August 2026. Criterion 10's third best-practice point (query rewriting). Tested as a hypothesis, not shipped as a feature — this entry and its falsify branch decide whether the rewriting path is wired into retrieval at all.*

> **Resolved 29 August 2026; decisive claim corrected 30 August 2026 — weakly confirmed / inconclusive.** The identifier gate is provably zero-harm (Case 1 & 5 deltas exactly 0). Rewriting fired on only 3 of 16 queries, improved 1 (Q4, +3 fused ranks, still to rank 36), slightly regressed 1 (Q5, −1). Round 1's motivating `shader processor count` gap (R1-Q7) **reproduces exactly** against the current pipeline — dense ranks the "128 CUDA cores" chunk `35b73f33…` first (0.8281), BM25 does not surface it, and fusion discards it (target at fused rank 10, absent from `/search`'s top 10 under `live_fast`). The 29 August write-up read dense rank 1 as the gap having closed; that was wrong — dense rank 1 was Round 1's premise. Query rewriting cannot reach the failure: the expansion touches the dense query only, and the bridging term `CUDA` has a collapsed IDF, so forcing it onto BM25 fails too — it is a fusion defect (Family B). Recommendation unchanged: do **not** wire `retrieval/query_rewrite.py` into retrieval; keep it as the record; re-test with authored vocab-gap queries (D3) + graded labels (ENH-11). Full write-up: `docs/uat/round3_dqr_findings.md`.

| Field | Detail |
|---|---|
| Claim | A rewriting step ahead of retrieval — legacy-terminology expansion plus camelCase identifier splitting, applied to the **dense** query only while BM25 keeps the literal string — improves the fused rank of the correct chunk on vocabulary-gap queries (Case 4), and, **when gated to skip exact-identifier queries**, leaves Case 1 and Case 5 unchanged. Applied ungated (identifier queries rewritten too) it is neutral-to-harmful on Case 1 and Case 5. The gate, not the rewrite, is what makes it safe. |
| Why it should hold | Round 1's documented failure: `shader processor count` retrieves nothing because the corpus only ever says "CUDA cores"; dense found "An SM consists of: 128 CUDA cores" (`35b73f33…`) and RRF discarded it — "the most important finding of the UAT" (`uat_day5_retrieval.md`). A4 found indirect evidence that identifier lookups and conceptual queries need opposite handling: the cross-encoder helps the latter and demotes terse signature blocks on the former (Sign-Off §8.2, Q2's +2 delta). Rewriting rests on the same split — a paraphrase closes a gap where query and target share no tokens, and destroys the decisive token where they share it. |
| Why it might not | (a) Only Q4 among the 15 Round 2 queries carries a genuine terminology mismatch — Q10 (`latency hiding through instruction level parallelism`) and Q11 (`occupancy versus performance tradeoffs`) already share vocabulary with their targets, so a rewriter correctly no-ops and Case 4 shows no aggregate movement. (b) e5-base-v2 may already bridge "shader processor" → "CUDA core" semantically, making the expansion redundant. (c) camelCase splitting of `cudaDeviceSynchronize` may be neutral for e5 (subword tokenisation already handles it), so the ungated arm fails to harm Case 5 and the gate earns nothing. (d) The identifier regex may misfire on mixed queries (Q14, `pinned memory cudaMallocHost benefits`) — gating a query that also has conceptual content. |
| Confirms | (1) On Case 4, the target chunk's fused rank improves under gated rewriting on a majority of the case's queries (n=2: Q10, Q11; Q4 counted as the case's motivating query, plus R1-Q7 as supplementary). **and** (2) gated rewriting holds every Case 1 and Case 5 target within ±1 fused rank of baseline (the gate declines to rewrite them). Rewriting that helps Case 4 while the *ungated* arm harms Case 1/5 is the **expected confirming shape** — the response is to keep the gate, not to abandon rewriting or apply it everywhere. |
| Falsifies | Gated rewriting produces no Case 4 improvement on any query including Q4 and R1-Q7 (the gap is already closed, or the expansion is noise) — **or** the ungated arm is also neutral-to-positive on Case 1 and Case 5, making the gate pointless and unconditional rewriting the correct call — **or** gated rewriting moves a Case 1 / Case 5 target by more than ±1 rank despite the gate firing (the identifier regex is unreliable). Any of these retires conditional rewriting for this corpus as specified. |

**Protocol**

1. Classify each of the 15 Round 2 queries (`uat_superiority_cases_raw.json`) plus R1-Q7 (`shader processor count`, supplementary) as `exact_identifier` if it contains a CUDA API symbol / error code / struct token (`cuda[A-Z]…`, `cudaError…`, `dim3`, `__host__`-style), else `conceptual`.
2. Fix a retriever-independent target chunk per query from the Round 2 / Round 1 write-ups (e.g. Q1 → `cc6c8e53…`, Q4 → `35b73f33…`, Q10 → `f2730f1e…`). Record the list in the script.
3. Two rewrite strategies, dense-query only, BM25 always literal: (a) **legacy expansion** — a static map of terminology-era mismatches (`shader processor` / `streaming processor` / `shading unit` → `CUDA core`; `GPU program` → `kernel`; `video memory` / `VRAM` → `global memory`); (b) **identifier split** — camelCase CUDA identifiers split into words and appended.
4. Three modes per query: **baseline** (no rewrite), **gated** (both strategies, skipped entirely when `exact_identifier`), **ungated** (both strategies always).
5. Retrieve BM25 top-100 (literal) and dense top-100 (mode query) live, RRF-fuse (`k=60`). Record the target's rank in BM25, dense, and fused lists for every query × mode. Persist the full per-query table as JSON before any analysis.
6. Group by the six case types. Report per-query rank and per-query delta (baseline − mode; positive = improvement), then case means. State n per case — most are 2–3, Case 4 is 2 (Q10, Q11) plus Q4/R1-Q7 — and read case rollups as directional only, per the A3 standard.
7. NDCG is not used: `run_day9_relevance_labelling.py`'s qrels are circular (A2, A3) and no retriever-independent graded labels exist yet (ENH-11). Target-chunk rank is the metric.

**Persist** — script and per-query JSON committed before the findings write-up, per `0149ca4`.


## 7.  Family E — Corpus and Index Scale

These change the baseline rather than measure against it. Each invalidates existing figures, so they should be planned as a re-baselining exercise, not slipped into a maintenance pass.


### E1  Crossing the HNSW threshold lowers measured NDCG

Claim: the current collection of 5,389 points sits below Qdrant's default 10,000-vector indexing threshold, so dense retrieval runs exhaustively. Every NDCG figure the project reports is therefore an upper bound. Expanding the corpus past that threshold should lower measured NDCG with no change to retrieval logic.

Test by forcing HNSW construction on the current collection (Qdrant allows configuring the threshold) and re-measuring, rather than by expanding the corpus. That isolates the approximate-search effect from the corpus-content effect — two variables the naive experiment would confound.

This is the cheapest way to convert the standing caveat into a measured number, and it makes the evaluation page's honesty claim stronger: "an upper bound" becomes "an upper bound by approximately X%".


### E2  Corpus expansion changes which retriever wins

Claim: adding the missing documents (NVLink, H100, TensorRT) changes IDF distributions corpus-wide, which per D1 should shift the BM25-versus-dense boundary. A term that is rare in five documents may be common in eight.

Consequence worth stating plainly: this would invalidate Rounds 1 and 2, not extend them. Plan accordingly.


### E3  Two of the Round 1 failures are corpus gaps, not retrieval failures

Already established informally: Q4 and the NVLink queries failed because the answer is absent, not because retrieval erred. Formalise by re-running those queries after adding the relevant documents. Confirms if they pass unchanged with the documents present.

This matters for the pass-rate framing. The headline 15 of 24 conflates system failures with corpus gaps; 15 of 22 excluding them is the honest denominator, and E3 is what would justify the exclusion rather than merely asserting it.


### E4  Embedding model choice matters more than fusion strategy

Claim: swapping e5-base-v2 for a stronger or domain-adapted encoder moves NDCG more than any fusion parameter in Family B. Day 4 recorded 0.4469 for all-MiniLM-L6-v2 as a dense baseline, so at least one comparison point exists in the repository.

Worth running because it bounds the value of the whole of Family B. If the encoder is worth 5 points of NDCG and fusion tuning is worth 0.5, that is a useful thing to know before investing in the latter.


## 8.  Family F — Generation-Side Effects  (blocked)

All three require Anthropic API credit and are recorded for completeness.

- F1 — Retrieval rank order affects answer quality independently of which chunks are retrieved. Test by presenting the same top-5 chunks in different orders and measuring RAGAS faithfulness. The "lost in the middle" effect is documented in the literature; whether it bites at k=5 on this corpus is not.
- F2 — Citation accuracy degrades as candidate pool depth increases, because more chunks means more opportunity to cite a plausible-but-wrong one. The citation judge (0.7037 over 27 claims) is the existing instrument.
- F3 — Boilerplate chunks in the context window reduce faithfulness disproportionately, because a See-also list of API names invites the model to mention functions it has no explanatory basis for.

## 9.  Suggested Execution Order


| Order | Hypotheses | Effort | Why here |
|---|---|---|---|
| 1 | A1, A2 | Half a day | The claim already published on the evaluation page. Pool-depth sweep answers both at once, and A2 gives an operational parameter as a by-product. |
| 2 | A4, A6 | Half a day | Tests whether the cross-encoder is a net positive on identifier queries at all. A6 is the cheapest candidate fix for the corroboration bias and needs no new components. |
| 3 | B1, B3, B4, B7 | One day | Cheap parameter sweeps that directly probe the corroboration mechanism. B3 measures single-signal displacement and may yield a runtime heuristic; B4 (score-normalised fusion) is its candidate remedy and is run in the same pass, against the same target chunks; B7 is the binary alternative to B4 and re-scores B3's persisted output at near-zero added cost. |
| 4 | C2, C3, C4 | One day | Characterises the boilerplate problem per retriever and per query type before committing to the re-indexing that C1 requires. |
| 5 | C1 or C5 | One day | Only after C4 says whether the fix is filtering or re-chunking. Requires NDCG re-baselining and updating the passage count in five places. |
| 6 | D1, D3, D4 | One day | The routing groundwork. D1 alone may be enough to justify or retire ENH-09. |
| 7 | E1 | Half a day | Converts the standing exhaustive-search caveat into a measured figure. |
| 8 | A5, B2, C6, D2, D5 | Variable | Second-order refinements. Run if the earlier results make them interesting. |


### 9.1  Reporting standard

Every hypothesis in this plan should be reported the way Round 2 was: prediction stated first, outcome recorded whether or not it matched, and the mechanism explained where a prediction failed. The value of Rounds 1 and 2 was not that 10 of 15 held — it was that the 5 failures were diagnosable because the predictions were written down beforehand.

Two specific standards carried forward. Report per-query results, not just aggregate NDCG: the corroboration bias would have been invisible in a mean. And state the pool depth, the k value, and the re-ranker config alongside every figure, because this round makes all three variable where the existing benchmark held them fixed.


### 9.2  What would make this publishable

D1, D3 and D4 together constitute a small but complete empirical result: that retriever choice on a technical-documentation corpus is predictable from two computable query features, and that a threshold router on those features beats unconditional fusion. That is the kind of finding an IR venue takes seriously, and it needs graded relevance judgements to be defensible — which is the ENH-11 work currently deferred.

A1 and A4 together are a narrower but sharper contribution: that rank-based fusion and cross-encoder re-ranking have complementary failure modes on API reference text, and that the standard pipeline order (fuse, then re-rank) prevents the re-ranker from correcting the fusion error, because the error has already removed the candidate. That observation is worth writing up regardless of whether the rest of the plan runs.

END OF DOCUMENT
