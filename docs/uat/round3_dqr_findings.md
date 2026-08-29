# UAT Round 3 — Hypothesis D-QR Findings

**Reference:** UAT-NVIR-2026-003-D-QR
**Executed:** 29 August 2026
**Status:** Resolved — **weakly confirmed / inconclusive**
**Plan:** `docs/uat/round3_hypothesis_test_plan.md` §6 (D-QR)
**Criterion:** rubric point 10, third best-practice item (query rewriting)

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

**Weakly confirmed on the gate; inconclusive on the benefit; the motivating Round 1 finding does not reproduce.**

- **Confirm condition (2) — fully met.** Gated rewriting holds every Case 1 and Case 5 target at **exactly** baseline rank (delta 0), because the identifier gate fires on all five of those queries and hands both retrievers the literal string. The gate is provably zero-harm on identifier lookups — that is the one solid result here.
- **Confirm condition (1) — weak partial support.** Of the two queries with a genuine terminology mismatch, Q4 improved (+3 fused ranks) and R1-Q7 was unchanged (already at dense rank 1 — nothing to fix). 1 of 2 improved, 0 worsened. Round 2's *nominal* Case 4 (Q10, Q11) has no terminology mismatch and the rewriter correctly no-ops on both.
- **Falsify condition (b) — partially triggered.** The ungated arm is net-neutral on Case 1 (mean delta 0.0: Q1 +1, Q2 −1, Q3 0) and Case 5 (0.0). It is not the clear "harms Case 1/5" the hypothesis predicted — so the gate's value over doing nothing is thin: it prevents exactly one 1-rank regression (Q2) and one absorbed dense perturbation (Q14).
- **Falsify conditions (a) and (c) — not triggered.** Q4 did improve; the gate never moved a gated query at all.

**The sharpest finding is not in the confirm/falsify grid.** R1-Q7 — `shader processor count`, the query Round 1 called *"the most important finding of the UAT"* — **does not reproduce as a vocabulary gap against e5-base-v2.** Dense retrieval ranks the "An SM consists of: 128 CUDA cores" chunk (`35b73f33…`) at **position 1 with no rewriting at all**. The Round 1 finding was measured against lexical-only behaviour; the current hybrid pipeline's dense arm already does the "shader processor" → "CUDA cores" bridging the rule-based expander was built to add. On R1-Q7 the expander's job is already done by the encoder.

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

**Rewriting fired on 3 of 16 queries.** Everything else was `gated-skip` (6: the identifier queries) or `no-op` (7: no legacy term, no camelCase token). The static legacy map is small by design, and only `shader processor` (Q4, R1-Q7) and `GPU programs` (Q5) matched anything in this set.

**Q4 — the one genuine improvement — is counteracting phrasing dilution, not bridging a lexical gap.** Round 2 already diagnosed Q4: the short form `shader processor count` found the right chunk (that is R1-Q7, dense rank 1), and adding `per streaming multiprocessor` *"diluted it out of all three top-3s"*. Appending `CUDA core` pulls the dense rank back from 16 to 10 and the fused rank from 39 to 36 — real, directionally correct, and still nowhere near a rank a user would see. The gain is undoing dilution the query itself introduced, not closing a vocabulary gap the corpus created.

**R1-Q7 — the documented gap — is already closed by the encoder.** Dense rank 1, no rewrite. The `+CUDA core` expansion changes nothing because there is nothing to change. Its fused rank of 10 is the RRF corroboration-bias displacement Round 1 recorded (BM25 contributes zero, so the lone dense signal loses to doubly-ranked mediocre chunks) — a fusion problem that query rewriting does not touch and was never meant to.

**Legacy expansion has a cost.** Q5 (`how to make GPU programs run faster`) fired the `GPU programs → kernels` rule and the fused rank slipped 17 → 18. One rank, but it is the wrong direction, and it came from applying an expansion where the encoder did not need help.

**The gate works exactly as specified.** All six identifier queries (Q1, Q2, Q3, Q12, Q13, Q14 — note Q14 is a *mixed* query, `pinned memory cudaMallocHost benefits`, that the regex correctly caught) are handed the literal string on both sides under `gated`, so their deltas are identically 0. Under `ungated`, the camelCase split appended to the dense query moved things: Q1 improved by 1 fused rank, Q2 regressed by 1, Q14's dense rank went 1 → 3 (absorbed by RRF). Net zero, one small regression prevented by the gate.

---

## 5. Assessment against D-QR

| Branch | Predicted | Observed |
|---|---|---|
| Confirm (1): gated helps a majority of vocab-gap queries | Q4, R1-Q7, (Q10, Q11) improve | Q4 +3; R1-Q7 already optimal; Q10/Q11 no-op. **1 improved, 0 worse, rest not applicable.** Weak. |
| Confirm (2): gated holds Case 1 & Case 5 within ±1 | deltas near 0 | **deltas exactly 0** (gate skips them). Met. |
| Falsify (a): no Case 4 improvement anywhere | — | Not triggered (Q4 +3). |
| Falsify (b): ungated also neutral-to-positive on Case 1 & 5 → gate pointless | — | **Partially triggered.** Ungated mean Δ = 0.0 on both. The gate prevents one 1-rank regression (Q2); that is the whole of its measured value on this set. |
| Falsify (c): gated moves a Case 1/5 target > ±1 despite the gate | — | Not triggered. |

**Reading.** The hypothesis is not falsified — Q4 improved and the gate is safe — but it is not meaningfully confirmed either. The pattern the plan anticipated (clear Case 4 help, clear Case 1/5 harm, therefore gate) did not materialise, for two reasons the plan's own "why it might not" section listed: **(b)** e5-base-v2 already bridges the one real terminology gap (R1-Q7), and **(c)** BM25 keeping the literal query means identifier queries are largely protected even ungated — only the dense half shifts, and RRF absorbs most of it.

---

## 6. Recommendation

**Do not wire `retrieval/query_rewrite.py` into the live retrieval path.**

- The gate is sound but its measured benefit on this set is preventing a single 1-rank regression.
- The one improvement (Q4, +3 fused ranks to rank 36) does not reach a rank a user sees, and it corrects dilution the query introduced rather than a corpus vocabulary gap.
- Blind legacy expansion carries a real if small cost (Q5, −1).
- The finding that motivated the whole hypothesis — Round 1's `shader processor count` gap — does not reproduce against the current dense encoder.

Keep the module and this evaluation as the D-QR record. **Re-test only if** (a) authored genuine vocabulary-gap queries exist (plan D3: queries where no query term appears in the target chunk) and (b) retriever-independent graded relevance labels exist (ENH-11). The current 15-query set contains exactly one terminology mismatch, and the encoder already handles its undiluted form.

If rewriting is ever revisited, the evidence says: keep the identifier gate (it is free and it is correct), drop the blind legacy map in favour of expansions tied to measured corpus gaps, and treat the RRF corroboration-bias displacement on single-signal queries (R1-Q7 fused rank 10) as a separate problem — it is Family B's territory, not query rewriting's.

---

## 7. Small-n limitation

Stated plainly, as the A3 write-up does. Each Round 2 case rests on 2–3 queries; Case 4 is 2. Rewriting fired on only 3 of 16 queries, and the single clear improvement (Q4) and single regression (Q5) are one query each. **This is directional evidence, not a test.** The verdict "weakly confirmed / inconclusive" should be read as "no reason to ship, no reason to abandon the idea — re-test with a proper instrument."

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

*D-QR resolved: weakly confirmed / inconclusive. The identifier gate is correct and free; the rewriting benefit does not justify wiring it into retrieval on this corpus + encoder. Round 1's motivating vocabulary-gap finding does not reproduce against e5-base-v2.*
