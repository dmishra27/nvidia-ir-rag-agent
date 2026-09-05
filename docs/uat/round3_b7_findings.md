# UAT Round 3 — Hypothesis B7 Findings

**Reference:** UAT-NVIR-2026-003-B7
**Executed:** 5 September 2026
**Status:** Resolved — **confirmed as pre-registered**: the rule fires on exactly the two
queries predicted (Q5, R1-Q7), recovers both, and regresses no fused-rank-1 target. The
precision cost B3's design cannot observe remains unmeasured and is the reason this is not
yet a ship recommendation.
**Plan:** `docs/uat/round3_hypothesis_test_plan.md` §4 (B7)
**Depends on:** B3 + B4 (`docs/uat/round3_b3_b4_findings.md`, 2 September 2026)

> **Offline re-scoring.** B7 adds no retrieval. It re-scores the persisted B3/B4 table
> (`evaluation/b3_b4_fusion_eval.json`) under one rule, so every rank here traces to a figure
> B3 already measured live. Thresholds N = 3 and M = 100 were committed on 4 September 2026
> (`4a15435`), before any result existed.

---

## 1. Provenance

| Item | Detail |
|---|---|
| Eval harness | `run_b7_single_retriever_rule.py` — committed `98e776e` (**before** the run) |
| Raw output | `evaluation/b7_single_retriever_rule.json` — committed `a3c89ea` (**before** this analysis) |
| Input | `evaluation/b3_b4_fusion_eval.json` — committed `0b13307` (B3/B4 raw, before their analysis) |
| Reproducible? | **Yes, trivially.** Pure function of the committed input JSON. No services, no model, no network. Re-run: `python run_b7_single_retriever_rule.py`. |
| Thresholds | **N = 3, M = 100**, locked `4a15435`. Not exposed as parameters in the harness. |
| Metric | Target-chunk rank, per B3 and D-QR. NDCG is unevaluable on this corpus (circular qrels, A2/A3; ENH-11 not done) and is not reported. |

**The rule** (plan §4, protocol step 2):

```
if min(bm25_rank, dense_rank) <= N   and   target absent from the other retriever's top-M pool:
    b7_rank = min(bm25_rank, dense_rank)      # the finding retriever's own rank
else:
    b7_rank = fused_rank                       # RRF untouched
```

With **M = 100 = the pool depth B3 retrieved**, "absent from the other retriever's top-M" is
exactly "rank is `null` in the eval file". No chunk in a top-100 pool can have rank > 100, so
for the pre-registered M only genuine absence satisfies the second clause. This is deliberate:
the plan entry records that Q12's BM25 rank 23 "counted as present and gave the mildest
displacement on record", so any M below the pool edge would reclassify a working corroboration
signal as absent.

**Query set.** The 15 Round 2 superiority queries (`docs/uat/uat_superiority_cases_raw.json`)
plus R1-Q7 (`shader processor count`, supplementary). Same 16 as B3/B4.

---

## 2. Result — per query

`delta` = `fused_rank − b7_rank`; positive = B7 improved the target's rank. "fired" = both
clauses of the rule held.

| Query | case | bm25 rank | dense rank | corroboration class | RRF fused | **B7** | delta | fired | B4 min-max (for contrast) |
|---|---|---|---|---|---|---|---|---|---|
| Q1  | C1 BM25 lexical | 2 | 75 | single-signal-bm25 | 4 | 4 | 0 | no | 3 |
| Q2  | C1 BM25 lexical | 8 | 18 | weak/neither | 3 | 3 | 0 | no | 4 |
| Q3  | C1 BM25 lexical | 1 | 49 | single-signal-bm25 | 2 | 2 | 0 | no | 1 |
| Q4  | C2 dense semantic | — | 16 | weak/neither | 39 | 39 | 0 | no | 41 |
| **Q5**  | C2 dense semantic | — | 2 | single-signal-dense | **17** | **2** | **+15** | **yes** | 5 |
| Q6  | C2 dense semantic | 2 | 1 | corroborated | 1 | 1 | 0 | no | 1 |
| Q7  | C3 RRF hybrid | 2 | 6 | single-signal-bm25 | 1 | 1 | 0 | no | 2 |
| Q8  | C3 RRF hybrid | 2 | 1 | corroborated | 1 | 1 | 0 | no | 1 |
| Q9  | C3 RRF hybrid | 1 | 1 | corroborated | 1 | 1 | 0 | no | 1 |
| Q10 | C4 BM25 failure / vocab gap | 14 | 1 | single-signal-dense | 4 | 4 | 0 | no | 3 |
| Q11 | C4 BM25 failure / vocab gap | 6 | 1 | single-signal-dense | 4 | 4 | 0 | no | 4 |
| Q12 | C5 dense failure / exact lookup | 23 | 1 | single-signal-dense | 2 | 2 | 0 | no | 1 |
| Q13 | C5 dense failure / exact lookup | 1 | 3 | corroborated | 1 | 1 | 0 | no | 1 |
| Q14 | C6 RRF mixed | 1 | 1 | corroborated | 1 | 1 | 0 | no | 1 |
| Q15 | C6 RRF mixed | 3 | 1 | corroborated | 1 | 1 | 0 | no | 3 |
| **R1-Q7** | C4 BM25 failure / vocab gap | — | 1 | single-signal-dense | **10** | **1** | **+9** | **yes** | 3 |

**The rule fired on Q5 and R1-Q7. On the other 14 queries `b7_rank == fused_rank` exactly** —
the rule is inert, by construction, wherever its trigger does not hold.

### 2.1 Why it fires on exactly these two

Both are single-signal-dense cases on the **"no corroboration"** side of B3's bimodal split —
the two queries where B3 measured severe displacement (fused rank 10 and 17, displaced +9 and
+15):

- **R1-Q7** `shader processor count` — dense rank 1, BM25 does not return the target at all.
  `min(∞, 1) = 1 ≤ 3`; BM25 absent. Fires. `b7_rank = 1`.
- **Q5** `how to make GPU programs run faster` — dense rank 2, BM25 does not return the target.
  `min(∞, 2) = 2 ≤ 3`; BM25 absent. Fires. `b7_rank = 2`.

It does **not** fire on the three mild single-signal-dense cases, because the other retriever
returns the target somewhere in its top-100:

- **Q10** — BM25 rank 14 (present) → rule inert, stays at fused rank 4.
- **Q11** — BM25 rank 6 (present) → stays at fused rank 4.
- **Q12** — BM25 rank 23 (present) → stays at fused rank 2.

Nor on B3's mirror cases **Q1** (dense rank 75, present) and **Q3** (dense rank 49, present) —
where fusion is already doing useful work lifting a chunk the other retriever buried, and the
plan explicitly says to leave those alone.

This is the pre-registered firing set — **R1-Q7 and Q5 only** — reproduced exactly.

---

## 3. Against the two pre-registered predictions

### Prediction 1 — "B7 fires on R1-Q7 and Q5 only"

**Confirmed.** Observed firing set `{Q5, R1-Q7}` = predicted firing set, exactly. No query
outside the pair fires; both members of the pair fire.

### Prediction 2 — "B7 mechanically predicts fused rank 1 on R1-Q7, against B4's measured 3"

**Confirmed.** `b7_rank` for R1-Q7 is **1** — dense's own rank, substituted wholesale for the
fused rank of 10. B4 min-max CombSUM recovered the same target only to rank 3 (z-score to 4).
On this one query the binary rule strictly beats the score-normalised remedy, and it does so
by the mechanism the plan named: dense placed the known-correct target first, BM25 contributed
nothing to `Σ 1/(k+rankᵢ)`, so the fused order carried strictly less information than dense's
own order. Taking dense's rank directly is the whole move.

Q5's `b7_rank` is **2** (dense rank 2, not 1 — the rule substitutes the finding retriever's
actual rank, and nothing at Q5 puts the target at dense rank 1). B4 min-max also reached 5
here, so B7 again beats it, by three ranks.

---

## 4. Against the falsify conditions

### 4.1 "Any target currently at fused rank 1 regresses"

**Not triggered.** Seven targets sit at RRF fused rank 1 in the persisted table — **Q6, Q7,
Q8, Q9, Q13, Q14, Q15**. (The plan entry's "eight of the sixteen" is one high; the executed
B3/B4 output records seven. The count does not change the outcome — see below.)

The rule fires on **none** of the seven. Every one of them has the other retriever returning
the target inside its top-100:

| fused-rank-1 query | bm25 rank | dense rank | other retriever present? |
|---|---|---|---|
| Q6  | 2 | 1 | yes (bm25 rank 2) |
| Q7  | 2 | 6 | yes (dense rank 6) |
| Q8  | 2 | 1 | yes (bm25 rank 2) |
| Q9  | 1 | 1 | yes (bm25 rank 1) |
| Q13 | 1 | 3 | yes (dense rank 3) |
| Q14 | 1 | 1 | yes (bm25 rank 1) |
| Q15 | 3 | 1 | yes (bm25 rank 3) |

So the rule's second clause fails on all seven and `b7_rank` stays at 1. The plan's
"why it might not" point (1) — *a target held at fused rank 1 by one retriever alone, nothing
corroborating it, demoted to its raw retriever rank* — **does not materialise in this data**:
there is no single-signal default win at fused rank 1. Every fused-rank-1 target here is
genuinely corroborated (both retrievers rank it, and six of the seven rank it ≤ 3 on both
sides). B7 does not touch a corroborated target, and none of the eight/seven is anything else.

Structurally, `b7_rank` can only differ from `fused_rank` when the rule fires, and the rule
fires only on Q5 and R1-Q7 (fused ranks 17 and 10). No path in the rule can move a fused-rank-1
target. Zero regressions is not a lucky draw on this input — it is what the rule's inertness
outside its trigger guarantees, given that no fused-rank-1 target satisfies the trigger.

### 4.2 "R1-Q7 or Q5 does not improve"

**Not triggered.** R1-Q7 delta `+9` (10 → 1), Q5 delta `+15` (17 → 2). Both improve, both
substantially — each target moves from outside any plausible top-3 or top-10 candidate pool to
the top 2.

### 4.3 "The chunk B7 promotes is not the correct target" — the precision risk B3 is silent on

**Not triggered on the two firing cases — but this run says nothing about the general case,
and that limit is the headline caveat.**

State it plainly, the way B3 and D-QR state theirs:

1. **B3 measured only known-correct targets.** Every rank in `b3_b4_fusion_eval.json` is the
   rank of a *pre-identified correct chunk* (`_rank_of(target, results)`). The file does not
   store the full ranked chunk-id lists, so this re-scoring cannot see what *else* sits at
   dense rank 1 for a query where the rule fires. `b7_rank = min(bm25_rank, dense_rank)` places
   the **fixed target** at that rank *by construction of the input*.

2. **For the two firing cases, the promoted chunk is independently confirmed to be the target.**
   R1-Q7: `docs/uat/round3_dqr_findings.md` verified that dense ranks `35b73f33…`
   ("An SM consists of: 128 CUDA cores") first for `shader processor count` — that is B3's
   fixed target. Q5: the target `8f2dbd94…` (the Autotuning section) sits at dense rank 2 in
   the same anchor set. So the plan's falsify branch (3) — *"the chunk B7 promotes to the top
   on either severe case is not the correct target"* — does **not** fire.

3. **But as a runtime rule, B7 would promote the finding retriever's rank-1 chunk whether or
   not it is correct.** It newly trusts a retriever that confidently ranks a *wrong* chunk
   first — turning a recall problem (B3's displacement) into a precision problem this data set
   is structurally unable to observe. B3's design tracked correct targets only; it never
   sampled how often a lone top-3 placement is wrong. Quantifying that false-positive rate
   needs retriever-independent graded labels over the full pool — **ENH-11** — and until then
   B7's clean pass here is a demonstration that the rule *can* recover severe displacement
   without B4's cross-scale regression, **not** evidence that it is safe to wire in.

### 4.4 Summary against the plan's "Confirms" clause

| "Confirms" sub-clause | Outcome |
|---|---|
| Rule fires on R1-Q7 and Q5 | **Holds** |
| Each lands at its finding-retriever rank (R1-Q7 → 1, better than B4's 3) | **Holds** — R1-Q7 → 1, Q5 → 2 |
| Spot-check: promoted chunk is B3's fixed target, not a lookalike | **Holds for both firing cases** (R1-Q7 confirmed via D-QR write-up; Q5 target at dense rank 2 per anchor set). General false-positive rate: **unevaluable here**, ENH-11. |
| No fused-rank-1 target regresses (rule fires on none of them) | **Holds** — fires on 0 of 7 |
| Q10 / Q11 / Q12 unchanged | **Holds** — all three inert (other retriever present at bm25 14 / 6 / 23) |

| "Falsifies" sub-clause | Triggered? |
|---|---|
| Any fused-rank-1 target regresses | **No** |
| R1-Q7 or Q5 does not improve | **No** |
| Promoted chunk on a severe case is not the target | **No** (for the two firing cases) |

**B7 confirms as pre-registered.**

---

## 5. Exploratory only — what `M < 100` would do

Outside the pass/fail read (plan protocol step 4). N held at 3. **Changing M after execution
voids B7 as a pre-registered result** (plan §4, "Post-hoc tuning invalidates the test"); this
table exists only so the sensitivity the locked M forecloses is visible.

| M | fires on | fused-rank-1 regressions | note |
|---|---|---|---|
| **100** (locked) | Q5, R1-Q7 | none | the pre-registered result above |
| 50 | Q1, Q5, R1-Q7 | none | picks up Q1 (mirror case, dense rank 75) — fusion already helps it; b7 would move it 4 → 2 |
| 20 | Q1, Q3, Q5, Q12, R1-Q7 | none | adds Q3 (mirror, dense 49) and Q12 (mild, bm25 23) |
| 10 | Q1, Q3, Q5, Q10, Q12, R1-Q7 | none | adds Q10 (mild, bm25 14); still misses Q11 (bm25 6 ≤ 10) |
| 5 | Q1, Q3, Q5, Q7, Q10, Q11, Q12, R1-Q7 | **Q7 (1 → 2)** | captures all three mild cases Q10/Q11/Q12 — **and regresses Q7**, B4's exact failure mode |

Two things this shows:

- **M ≈ 5 is what the motivating discussion "would like"** — it captures Q10 (+3), Q11 (+3),
  Q12 (+1) — but it also fires on Q7, whose target BM25 ranks 2 and dense ranks 6, and demotes
  it from fused rank 1 to 2. That is precisely the regression class B7 was designed to avoid,
  and B3's rank-23 Q12 result is the direct evidence against widening M this far. The locked
  M = 100 is the only value in the sweep that touches the two severe cases and nothing else.
- **The mirror cases Q1 and Q3 come in first** as M drops (at M = 50 and 20), before any of
  the mild single-signal-dense cases. Those are cases where fusion is already lifting a
  buried chunk — moving them to the raw retriever rank is not obviously an improvement even
  though the delta is nominally positive (Q1 4 → 2, Q3 2 → 1). The pre-registered M leaves
  them alone.

---

## 6. Small-n limit

Sixteen queries; the rule fires on two. Every figure in §2 is a single-query measurement, read
directionally, the same standard B3 and D-QR set. There is no aggregate — per plan §9.1 the
effect would be invisible in a mean, and with n = 2 firing cases a mean would be actively
misleading. What B7 establishes is narrow and specific: **on the two queries where B3 measured
severe un-corroborated displacement, a zero-parameter rank substitution recovers the target to
the top 2, and it provably does not disturb any of the other 14 queries.** It does not
establish that the rule is safe against wrong-but-confident single-retriever placements —
that question is B3's blind spot, inherited here, and it needs ENH-11.

---

## 7. What this means for the pipeline

1. **B7 is the remedy the B3 measurement actually points to.** B3 found displacement binary;
   B7 is a binary fix, and it behaves exactly as the "why it should hold" argument predicted —
   it recovers the severe cases (R1-Q7 to rank 1, beating B4), and it is inert on all 14
   non-triggering queries, so it cannot reproduce B4's regression of Q15 or its two weak-case
   slips. On this corpus, on target-chunk rank, B7 dominates B4: same or better on every
   query, no normalisation, no cross-distribution comparison.
2. **It is not yet shippable, for one reason.** B3's design only ever tracked known-correct
   targets, so neither B3 nor this re-scoring can say how often a lone top-3 placement is
   *wrong*. B7 as live code would promote the finding retriever's rank-1 chunk unconditionally
   in its trigger regime. The false-positive cost is unmeasured and unmeasurable without
   retriever-independent graded labels.
3. **Recommendation:** record B7 as a confirmed, cheap, correct-on-the-recall-axis heuristic
   that beats both RRF and B4's score-normalised fusion on the two queries where fusion fails
   hardest. Do **not** wire it into retrieval yet. Re-open it alongside **ENH-11**, which is
   what would let the precision question be answered rather than flagged. If it is prototyped
   before then, gate it behind a metric that can see wrong promotions (an A/B on answer
   quality, or the citation judge) rather than trusting the rank recovery alone.
4. **Thresholds stay as committed.** N = 3, M = 100. The exploratory sweep (§5) confirms the
   plan's pre-registration reasoning: any M materially below the pool edge fires on mildly
   displaced or already-helped queries, and at M = 5 it regresses Q7. A re-run with different
   thresholds is a new hypothesis and gets its own entry.

---

## 8. Reproduction

```
python run_b7_single_retriever_rule.py     # offline re-score of b3_b4_fusion_eval.json -> evaluation/b7_single_retriever_rule.json
```

No services, no model, no network. Pure function of `evaluation/b3_b4_fusion_eval.json`
(committed `0b13307`). Runs in well under a second.
