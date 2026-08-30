# UAT Round 3 — Hypothesis D-QR Findings

**Reference:** UAT-NVIR-2026-003-D-QR
**Executed:** 29 August 2026
**Revised:** 30 August 2026 — decisive claim corrected; no measurement changed
**Status:** Resolved — **weakly confirmed / inconclusive** · recommendation unchanged (do not ship)
**Plan:** `docs/uat/round3_hypothesis_test_plan.md` §6 (D-QR)
**Criterion:** rubric point 10, third best-practice item (query rewriting)

> **Revision note — 30 August 2026.** The 29 August write-up concluded that Round 1's
> `shader processor count` finding (R1-Q7) *"does not reproduce as a vocabulary gap against
> e5-base-v2"*, treating dense rank 1 as evidence the gap had closed. **That reasoning was
> wrong, and it was the write-up's decisive claim.** Round 1's finding was never that dense
> ranked the chunk first — it was that *dense finds the chunk and RRF discards it*. Re-verified
> live today (postgres + qdrant up, `RERANKER_MODE=live_fast`): dense still ranks the correct
> chunk first, BM25 still does not surface it, and fusion still promotes a doubly-ranked
> mediocre chunk over it. R1-Q7 reproduces exactly. The "do not ship" recommendation stands,
> but now on a measured reason (§6) rather than on the gap having closed. No per-query
> measurement in §3 changed — only the interpretation; §§2, 4, 5, 6 and the footer are revised.
>
> **Second pass, same day.** Three characterisations were tightened against the full per-query
> mode comparison in `evaluation/dqr_eval.json`; again no measurement changed. (a) The gate is
> **net-zero, not net-positive** — across all 16 queries it changes two RRF ranks versus not
> gating (prevents a 1-rank loss on Q2, forecloses a 1-rank gain on Q1) and they cancel.
> (b) The rewrite's **measurable reach is one query, not three** — only Q4's target dense rank
> moves (16 → 10); R1-Q7 and Q5 fired the legacy rule and moved the target not at all, so Q5's
> 1-rank fused slip is not attributable to the rewrite. (c) On Q4, **fusion discarded half the
> gain** — dense improved six places, fused only three, because the target has no BM25 rank and
> carries a single fusion contribution. All three strengthen the recommendation not to ship.

---

## 1. Provenance

| Item | Detail |
|---|---|
| Implementation | `retrieval/query_rewrite.py` — committed `e03d230` |
| Unit tests | `tests/retrieval/test_query_rewrite.py` — 28 tests, committed `e03d230` |
| Eval harness | `run_dqr_eval.py` — committed `e03d230` (pre-run), fixed `2c302dc` |
| Raw output | `evaluation/dqr_eval.json` — committed `2c302dc`, **before** this analysis |
| Reproducible? | **Yes.** Script + output committed. Rewriting is rule-based and deterministic (no LLM — the Anthropic key is out of credit, and a rule-based rewriter reproduces from the repo anyway). Re-run: `python run_dqr_eval.py`. |
| Host note | Ran on the 8 GB CPU-only host at ~0.2 GB free. e5-base-v2 loaded and the run completed without OOM — one heavy stage, sequenced alone per the host rule. |

**Metric.** Target-chunk rank in the BM25, dense, and RRF top-100 lists, per query per mode. NDCG is **not** used: `run_day9_relevance_labelling.py`'s qrels are circular (Round 3 A2/A3) and no retriever-independent graded labels exist yet (ENH-11). The target chunk for each query is the one the Round 2 / Round 1 prose names as the correct answer — a fixed, retriever-independent anchor, the same device A1 used.

**Three modes.**

| Mode | Dense query | BM25 query |
|---|---|---|
| `baseline` | literal | literal |
| `gated` | literal + legacy-term expansion + camelCase split, **skipped entirely for exact-identifier lookups** | literal |
| `ungated` | literal + both strategies, **every** query | literal |

BM25 always gets the literal string; only the dense query is ever rewritten.

---

## 2. Verdict

**Weakly confirmed on the gate; inconclusive on the benefit; the motivating Round 1 finding reproduces exactly — as a fusion failure that query rewriting cannot reach.**

- **Confirm condition (2) — fully met.** Gated rewriting holds every Case 1 and Case 5 target at **exactly** baseline rank (delta 0), because the identifier gate fires on all five of those queries and hands both retrievers the literal string. The gate is provably zero-harm on identifier lookups — the one solid result here. But zero-harm is a safety property, not a demonstration of value: across all 16 queries the gate changes exactly two RRF ranks against not gating (Q1, Q2), and the two cancel — see the Falsify-(b) bullet and §6.
- **Confirm condition (1) — weak partial support.** Of the two queries with a genuine terminology mismatch, Q4 improved (+3 fused ranks) and R1-Q7 was unchanged — the rewrite reaches only the dense query, which already ranks the target first; the failure is downstream in fusion, where rewriting has no lever. 1 of 2 improved, 0 worsened. **Q4 is also the only query in the full 16 where the rewrite moves the target's dense rank at all** (16 → 10); every other dense rank is identical baseline-to-gated. Round 2's *nominal* Case 4 (Q10, Q11) has no terminology mismatch and the rewriter correctly no-ops on both.
- **Falsify condition (b) — partially triggered.** The ungated arm is net-neutral on Case 1 (mean delta 0.0: Q1 +1, Q2 −1, Q3 0) and Case 5 (0.0). It is not the clear "harms Case 1/5" the hypothesis predicted — and the gate's value over doing nothing is not merely thin, it nets to zero: across all 16 queries, gating changes exactly two RRF ranks against the ungated arm — it prevents a 1-rank regression on Q2 and forecloses a 1-rank improvement on Q1 (where ungated identifier-splitting moved dense 75 → 44). One prevented loss, one prevented gain. Every other identifier query is identical in both arms.
- **Falsify conditions (a) and (c) — not triggered.** Q4 did improve; the gate never moved a gated query at all.

**The sharpest finding is not in the confirm/falsify grid.** R1-Q7 — `shader processor count`, the query Round 1 called *"the most important finding of the UAT"* — **reproduces exactly, seven weeks on, on the query that first surfaced it.** Round 1's finding was never "dense ranks it first" — it was *dense finds the chunk and fusion discards it*. Verified live today (`RERANKER_MODE=live_fast`, postgres + qdrant up):

- Dense ranks the correct chunk `35b73f3371037bf5ba8fefc0` ("An SM consists of: ▶ 128 CUDA cores for arithmetic operations") at **rank 1, score 0.8281**.
- BM25 does **not** rank it in its top 20.
- The GPU-metrics chunk `a8c35fe2813fdaec5c5356ab` sits at **BM25 rank 2, dense rank 9** — two moderate contributions — and **wins fusion**. It is rank 1 in `/search` output under `live_fast`; the correct chunk is absent from the top 10 entirely.
- Dense scores across its top 20 span only **0.8281–0.8156, a 1.5% spread**. The correct chunk is *marginally* rather than confidently first — which is exactly what makes it easy for fusion to overrule.

This is Round 1's corroboration bias reproducing in its exact form. The encoder placing the chunk at dense rank 1 does not close the gap Round 1 documented — it *is* the condition Round 1 documented: a lone strong dense hit that fusion throws away because BM25 does not corroborate it. Query rewriting was never aimed at that failure and does not touch it (§6).

---

## 3. Per-query results

RRF-rank of the target chunk. `--` = absent from the top-100. Δ = `baseline − mode` (positive = target moved **up**).

| Q | Case | classify | strategy | BM25 | dense (b/g/u) | **RRF b / g / u** | Δ gated | Δ ungated |
|---|---|---|---|---|---|---|---|---|
| Q1 | 1 · BM25 lexical | exact_identifier | gated-skip | 2 | 75 / 75 / 44 | 4 / 4 / **3** | 0 | **+1** |
| Q2 | 1 · BM25 lexical | exact_identifier | gated-skip | 8 | 18 / 18 / 20 | 3 / 3 / **4** | 0 | **−1** |
| Q3 | 1 · BM25 lexical | exact_identifier | gated-skip | 1 | 49 / 49 / 25 | 2 / 2 / 2 | 0 | 0 |
| Q4 | 2 · dense semantic | conceptual | legacy-expansion (`+CUDA core`) | -- | 16 / **10** / 10 | 39 / **36** / 36 | **+3** | +3 |
| Q5 | 2 · dense semantic | conceptual | legacy-expansion (`+kernels`) | -- | 2 / 2 / 2 | 17 / **18** / 18 | **−1** | −1 |
| Q6 | 2 · dense semantic | conceptual | no-op | 2 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |
| Q7 | 3 · RRF hybrid | conceptual | no-op | 2 | 6 / 6 / 6 | 1 / 1 / 1 | 0 | 0 |
| Q8 | 3 · RRF hybrid | conceptual | no-op | 2 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |
| Q9 | 3 · RRF hybrid | conceptual | no-op | 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |
| Q10 | 4 · BM25 failure / vocab gap | conceptual | no-op | 14 | 1 / 1 / 1 | 4 / 4 / 4 | 0 | 0 |
| Q11 | 4 · BM25 failure / vocab gap | conceptual | no-op | 6 | 1 / 1 / 1 | 4 / 4 / 4 | 0 | 0 |
| Q12 | 5 · dense failure / exact lookup | exact_identifier | gated-skip | 23 | 1 / 1 / 2 | 2 / 2 / 2 | 0 | 0 |
| Q13 | 5 · dense failure / exact lookup | exact_identifier | gated-skip | 1 | 3 / 3 / 3 | 1 / 1 / 1 | 0 | 0 |
| Q14 | 6 · RRF mixed | exact_identifier | gated-skip | 1 | 1 / 1 / 3 | 1 / 1 / 1 | 0 | 0 |
| Q15 | 6 · RRF mixed | conceptual | no-op | 3 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |
| R1-Q7 | *(supp.)* 4 · vocab gap | conceptual | legacy-expansion (`+CUDA core`) | -- | **1** / 1 / 1 | 10 / 10 / 10 | 0 | 0 |

### Per-case rollup (official 15 only; n=2–3 per case — directional, per the A3 standard)

| Case | n | gated +/=/− (mean Δ) | ungated +/=/− (mean Δ) |
|---|---|---|---|
| 1 · BM25 lexical | 3 | 0 / 3 / 0 (0.0) | 1 / 1 / 1 (0.0) |
| 2 · dense semantic | 3 | 1 / 1 / 1 (+0.67) | 1 / 1 / 1 (+0.67) |
| 3 · RRF hybrid | 3 | 0 / 3 / 0 (0.0) | 0 / 3 / 0 (0.0) |
| 4 · BM25 failure / vocab gap | 2 | 0 / 2 / 0 (0.0) | 0 / 2 / 0 (0.0) |
| 5 · dense failure / exact lookup | 2 | 0 / 2 / 0 (0.0) | 0 / 2 / 0 (0.0) |
| 6 · RRF mixed | 2 | 0 / 2 / 0 (0.0) | 0 / 2 / 0 (0.0) |

The Case 2 mean (+0.67) is Q4's +3 divided across three queries, one of which (Q5) is −1. It is not a case-wide effect.

---

## 4. What actually happened

**The legacy rule fired on 3 of 16 queries; it reached the target on 1.** Everything else was `gated-skip` (6: the identifier queries) or `no-op` (7: no legacy term, no camelCase token). Only `shader processor` (Q4, R1-Q7) and `GPU programs` (Q5) matched the static map at all — but firing is not the same as moving the target. Comparing baseline to gated dense ranks across all 16 queries, **only Q4 moves** (16 → 10). R1-Q7 and Q5 fired the legacy rule and left the target's dense rank exactly where it was (1 and 2 respectively). Measurable reach on this set is one query.

**Q4 — the one genuine improvement — is counteracting phrasing dilution, not bridging a lexical gap.** Round 2 already diagnosed Q4: the short form `shader processor count` found the right chunk (that is R1-Q7, dense rank 1), and adding `per streaming multiprocessor` *"diluted it out of all three top-3s"*. Appending `CUDA core` pulls the dense rank back from 16 to 10 and the fused rank from 39 to 36 — real, directionally correct, and still nowhere near a rank a user would see. The gain is undoing dilution the query itself introduced, not closing a vocabulary gap the corpus created. **And most of what the rewrite bought, fusion discarded**: the dense rank improved six places (16 → 10), the fused rank only three (39 → 36). BM25 does not rank the target at all (`bm25_rank: null`), so the chunk carries a single `1/(60+rank)` fusion contribution and cannot convert a six-place dense gain into a comparable fused one. The rewriter succeeded on the query it could act on; the limiting factor is downstream, in fusion.

**R1-Q7 — the documented gap — is not closed; it reproduces.** Dense ranks the "128 CUDA cores" chunk (`35b73f33…`) first at 0.8281, BM25 does not surface it in the top 20, and fusion promotes `a8c35fe2…` (BM25 rank 2, dense rank 9) over it — fused rank 10 for the target, absent from `/search`'s top 10 under `live_fast`. That is the RRF corroboration-bias displacement Round 1 recorded, unchanged: a lone strong dense hit loses to a doubly-ranked mediocre chunk because BM25 contributes nothing. The write-up's earlier framing — "already closed by the encoder", "nothing to change" — was wrong: dense rank 1 is Round 1's *premise*, not a refutation of it.

**Rewriting cannot fix R1-Q7 even in principle — tested directly today.** `rewrite_query('shader processor count')` returns `bm25_query='shader processor count'` (unchanged) and `dense_query='shader processor count CUDA core'` — the expansion reaches the dense query only, by design. Running BM25 on the expanded string `'shader processor count CUDA core'` still does not surface `35b73f33…` in the top 20. `CUDA` appears across thousands of chunks; its IDF has collapsed — exactly the mechanism Round 1 recorded for `cudaDeviceSynchronize` on Q12. Vocabulary expansion cannot rescue a term whose discriminative weight is already gone, so even if the expansion *did* reach BM25 it would not help.

**Legacy expansion buys nothing on Q5.** Q5 (`how to make GPU programs run faster`) fired the `GPU programs → kernels` rule, and the target's dense rank did not move (2 → 2). The fused rank slipped one place (17 → 18) — but with the target's dense rank *and* its BM25 status both unchanged, that slip is other chunks reshuffling under the expanded dense query, not a change in how well the target is retrieved. Q5 is not evidence the rewrite harms the target; it is evidence the rule fired and reached nothing.

**The gate works exactly as specified — and its own effect nets to zero.** All six identifier queries (Q1, Q2, Q3, Q12, Q13, Q14 — note Q14 is a *mixed* query, `pinned memory cudaMallocHost benefits`, that the regex correctly caught) are handed the literal string on both sides under `gated`, so their deltas are identically 0. Under `ungated`, the camelCase split appended to the dense query moved things: Q1 improved by 1 fused rank (dense 75 → 44), Q2 regressed by 1 (dense 18 → 20), Q14's dense rank went 1 → 3 (absorbed by RRF). So gating changes exactly two RRF ranks against the ungated arm: it prevents the Q2 regression and it forecloses the Q1 improvement, in equal measure. Net zero — one prevented loss, one prevented gain.

---

## 5. Assessment against D-QR

| Branch | Predicted | Observed |
|---|---|---|
| Confirm (1): gated helps a majority of vocab-gap queries | Q4, R1-Q7, (Q10, Q11) improve | Q4 +3 (and the only query in the full 16 whose target dense rank the rewrite moves); R1-Q7 unchanged — rewrite reaches only the dense query (target already dense rank 1), the discard is in fusion; Q10/Q11 no-op. **1 improved, 0 worse, rest not applicable.** Weak. |
| Confirm (2): gated holds Case 1 & Case 5 within ±1 | deltas near 0 | **deltas exactly 0** (gate skips them). Met. |
| Falsify (a): no Case 4 improvement anywhere | — | Not triggered (Q4 +3). |
| Falsify (b): ungated also neutral-to-positive on Case 1 & 5 → gate pointless | — | **Partially triggered.** Ungated mean Δ = 0.0 on both cases. Gating changes exactly two RRF ranks against the ungated arm across all 16 queries — it prevents a 1-rank regression on Q2 and forecloses a 1-rank gain on Q1. They cancel: the gate's net measured value on this set is zero, not positive. |
| Falsify (c): gated moves a Case 1/5 target > ±1 despite the gate | — | Not triggered. |

**Reading.** The hypothesis is not falsified — Q4 improved and the gate is safe — but it is not meaningfully confirmed either. The pattern the plan anticipated (clear Case 4 help, clear Case 1/5 harm, therefore gate) did not materialise, for two reasons: **(b)** on the one real terminology gap (R1-Q7) the encoder ranks the target first *unaided* — but Round 1's finding was a *fusion* discard, not a dense-rank problem, and that discard still happens (target at fused rank 10, absent from `/search`'s top 10 under `live_fast`); the expansion is not redundant, it is aimed at the wrong stage. And **(c)** BM25 keeping the literal query means identifier queries are largely protected even ungated — only the dense half shifts, and RRF absorbs most of it. The plan's own "why it might not" (b) guessed the encoder would make the expansion *redundant*; what actually happened is subtler — the encoder does its part and fusion undoes it.

---

## 6. Recommendation

**Do not wire `retrieval/query_rewrite.py` into the live retrieval path.**

- The gate is sound but its net measured effect on this set is zero: across all 16 queries it changes two RRF ranks versus not gating — preventing a 1-rank regression on Q2, foreclosing a 1-rank improvement on Q1 — and they cancel.
- The rewrite moves the target's dense rank on exactly one query in the set (Q4, 16 → 10). Fusion then discards half of that: the fused rank improves only 39 → 36, because BM25 does not rank the target and it carries a single fusion contribution. The rewriter worked; the ceiling is downstream.
- The one fused improvement (Q4, +3 to rank 36) does not reach a rank a user sees, and it corrects dilution the query introduced rather than a corpus vocabulary gap.
- On Q5 the legacy rule fired and did not move the target's dense rank at all (2 → 2); the 1-rank fused slip is other chunks reshuffling, not a cost the rewrite imposed on the target. Blind expansion's measured effect here is nothing, in either direction.
- The finding that motivated the whole hypothesis — Round 1's `shader processor count` gap (R1-Q7) — **reproduces exactly**: dense ranks the correct chunk `35b73f33…` first at 0.8281, BM25 does not surface it in the top 20, fusion promotes `a8c35fe2…` (BM25 rank 2 / dense rank 9) over it, and the target lands at fused rank 10 — absent from `/search`'s top 10 under `live_fast`. Query rewriting cannot fix this: `rewrite_query('shader processor count')` expands the **dense** query only (`… CUDA core`), by design; and forcing the expansion onto BM25 fails too, because `CUDA` occurs across thousands of chunks and its IDF has collapsed — the same mechanism Round 1 recorded for `cudaDeviceSynchronize` on Q12. A term with no discriminative weight left cannot be revived by adding it. This is a fusion defect (Family B), measured today — a stronger reason not to ship than "the gap has closed", which was wrong.

Keep the module and this evaluation as the D-QR record. **Re-test only if** (a) authored genuine vocabulary-gap queries exist (plan D3: queries where no query term appears in the target chunk) and (b) retriever-independent graded relevance labels exist (ENH-11). The current 15-query set contains exactly one terminology mismatch (R1-Q7), and on it the bottleneck is fusion, not vocabulary — a stage query rewriting does not act on.

If rewriting is ever revisited, the evidence says: keep the identifier gate (it is free and it is correct), drop the blind legacy map in favour of expansions tied to measured corpus gaps, and treat the RRF corroboration-bias displacement on single-signal queries (R1-Q7 fused rank 10) as a separate problem — it is Family B's territory, not query rewriting's.

**Cross-reference — A1-R.** R1-Q7 is confirmed today as a live, reproducible RRF corroboration-bias case against the working pipeline: dense holds the target at rank 1, fusion discards it, and the chunk itself is sound (no straddling-boilerplate defect of the kind that makes Round 2's Q1 a poor test). `A1-R` in `docs/uat/correction_notice_a1.md` §5 is therefore runnable exactly as specified on R1-Q7. R1-Q3 (`H100 HBM2e memory capacity`) remains blocked behind DEF-19.

---

## 7. Small-n limitation

Stated plainly, as the A3 write-up does. Each Round 2 case rests on 2–3 queries; Case 4 is 2. The legacy rule fired on 3 of 16 queries but moved the target's dense rank on only one (Q4); Q5 and R1-Q7 fired and reached nothing. **This is directional evidence, not a test.** The verdict "weakly confirmed / inconclusive" should be read as "no reason to ship, no reason to abandon the idea — re-test with a proper instrument."

---

## 8. Artifacts

| Path | Contents |
|---|---|
| `retrieval/query_rewrite.py` | The rewriter — legacy expansion + camelCase split + identifier gate |
| `tests/retrieval/test_query_rewrite.py` | 28 unit tests, incl. classify/gate behaviour pinned on all 15 Round 2 queries |
| `run_dqr_eval.py` | Live eval harness (`--resummarize` re-rolls per-case with no retrieval) |
| `evaluation/dqr_eval.json` | Per-query BM25/dense/RRF target ranks × 3 modes, per-case rollup |
| `e03d230` | script + module + tests, pre-run |
| `2c302dc` | live output + harness fixes, pre-analysis |

---

*D-QR resolved: weakly confirmed / inconclusive. The identifier gate is correct and free; the rewriting benefit does not justify wiring it into retrieval on this corpus + encoder. Round 1's motivating finding reproduces exactly — dense still finds the chunk, fusion still discards it — but it is a fusion defect (Family B), not one query rewriting can reach: the expansion touches only the dense query, and the bridging term's IDF has already collapsed.*
