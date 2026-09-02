# UAT Round 3 — Hypotheses B3 + B4 Findings

**Reference:** UAT-NVIR-2026-003-B3-B4
**Executed:** 2 September 2026
**Status:** Resolved — **B3 confirmed in its binary form, graded form falsified (as pre-registered); B4 a directional remedy with the predicted stability cost**
**Plan:** `docs/uat/round3_hypothesis_test_plan.md` §4 (B3, B4)
**Depends on:** CORR-NVIR-2026-001 §3.3 (B3 re-specification, 31 Aug 2026)

> **Run as a pair.** B3 measures a displacement defect that exists precisely because RRF fuses on
> rank position and never sees retriever scores. B4 (score-normalised fusion) is the direct
> candidate remedy. Both were measured in one retrieval pass over the same queries and the same
> fixed target chunks, so every single-signal target's normalised-fusion rank reads directly
> against its measured RRF displacement.

---

## 1. Provenance

| Item | Detail |
|---|---|
| Eval harness | `run_b3_b4_fusion_eval.py` — committed `f5c1cc0` (**before** the run) |
| Raw output | `evaluation/b3_b4_fusion_eval.json` — committed `0b13307` (**before** this analysis) |
| Reproducible? | **Yes.** Script + output committed. Retrieval is deterministic (BM25 pickle + e5-base-v2 on a fixed Qdrant collection); fusion arithmetic is pure. Re-run: `python run_b3_b4_fusion_eval.py`. |
| Host note | 8 GB CPU-only host at ~0.22 GB free. e5-base-v2 loaded, 16 queries embedded, encoder released before any fusion arithmetic — one heavy stage, sequenced alone per the host rule. No OOM. 1m55s wall. |
| Cross-check | RRF ranks reproduce `evaluation/dqr_eval.json` baseline mode **exactly** for every query (Q1=4, Q3=2, Q5=17, Q10=4, Q11=4, Q12=2, R1-Q7=10). The two harnesses share no fusion code path beyond `retrieval/rrf_fusion.fuse`. |

**Metric.** Target-chunk rank in the BM25, dense, RRF, min-max-CombSUM and z-score-CombSUM top-100
lists, per query. NDCG is **not** used: `run_day9_relevance_labelling.py`'s qrels are circular
(Round 3 A2/A3) and no retriever-independent graded labels exist yet (ENH-11). Because the metric
is the rank of a single pre-identified chunk, B3 and B4 need no graded judgements and are **not**
blocked behind ENH-11 — that is what made them runnable now. The target chunk per query is the one
the Round 2 / Round 1 prose names as the correct answer — the same anchor A1 and D-QR used.

**Query set.** The 15 Round 2 superiority queries (`docs/uat/uat_superiority_cases_raw.json`) plus
R1-Q7 (`shader processor count`, supplementary). Literal query to both retrievers — no rewriting
(that was D-QR). Pool 100 per retriever, RRF `k=60`.

**Corroboration classes** (B3 step 3, `rank ≤ 5` cut):

| Class | Definition | Queries (n) |
|---|---|---|
| single-signal-dense | dense rank ≤ 5, BM25 rank ≥ 6 or absent | Q5, Q10, Q11, Q12, R1-Q7 (5) |
| single-signal-BM25 | BM25 rank ≤ 5, dense rank ≥ 6 or absent | Q1, Q3, Q7 (3) |
| corroborated | both retrievers rank the target ≤ 5 | Q6, Q8, Q9, Q13, Q14, Q15 (6) |
| weak/neither | neither retriever ranks it ≤ 5 | Q2, Q4 (2) |

The `≤ 5 / ≥ 6` cut is arbitrary at the boundary. **Q7** (BM25 2, dense 6) is placed in
single-signal-BM25 by one rank; dense rank 6 contributes `1/66`, within 1.5% of what rank 5 would,
so Q7 is effectively corroborated and — as §2 shows — behaves like the corroborated cases, not the
displaced ones. Raw ranks are in the JSON for anyone who wants to re-bin.

**Small-n.** Single-signal-dense rests on 5 queries, two of them (Q5, R1-Q7) sharing the
"corroboration absent" bin. Single-signal-BM25 rests on 3. Per-query rank is the defensible unit;
every class statement below is directional, read the same way A3 and D-QR are.

---

## 2. B3 — does RRF displace single-signal results?

### 2.1 The single-signal-dense group (the core claim)

Displacement = `fused_rank − dense_rank` (dense is the finding retriever). Positive = pushed down.

| Query | dense rank (finding) | BM25 rank (corroboration) | RRF fused rank | displacement | corroboration |
|---|---|---|---|---|---|
| Q12 `cudaDeviceSynchronize return value` | 1 | 23 | **2** | **+1** | present, weak |
| Q10 `latency hiding through ILP` | 1 | 14 | **4** | **+3** | present, weak |
| Q11 `occupancy versus performance` | 1 | 6 | **4** | **+3** | present, borderline |
| R1-Q7 `shader processor count` | 1 | absent | **10** | **+9** | **absent** |
| Q5 `how to make GPU programs run faster` | 2 | absent | **17** | **+15** | **absent** |

**Every single-signal-dense target lands strictly below its dense rank. 5 of 5.** The displacement
is not marginal on the un-corroborated cases: R1-Q7 falls from dense rank 1 to fused rank 10, Q5
from dense rank 2 to fused rank 17.

### 2.2 The mirror — single-signal-BM25

Displacement = `fused_rank − bm25_rank` (BM25 is the finding retriever). The "does fusion rescue a
chunk the other retriever buried?" direction.

| Query | BM25 rank (finding) | dense rank (buried) | RRF fused rank | displacement | rescued from |
|---|---|---|---|---|---|
| Q3 `cudaErrorInvalidValue description` | 1 | 49 | 2 | +1 | dense rank 49 → fused 2 |
| Q1 `CUDA cudaMalloc function parameters` | 2 | 75 | 4 | +2 | dense rank 75 → fused 4 |
| Q7 `CUDA thread synchronization overhead` | 2 | 6 | 1 | −1 | dense rank 6 → fused 1 |

**All three mirror targets are improved relative to the retriever that lost them** — dense ranked
Q1's target 75th and Q3's 49th; fusion put them at 4 and 2. Q1 is the canonical corroboration-bias
case (CORR-001 §4.1): fusion simultaneously lifts the target from dense's 75 and pushes it down
from BM25's 2, the two meeting at fused rank 4. The mirror cases show displacement from *both*
sides at once.

### 2.3 RRF-score decomposition — the mechanism

For every displacement case, the target's `Σ 1/(k+rankᵢ)` contribution versus the same
decomposition for the chunk one rank above it in the fused list (the chunk that beat it):

| Query | target contribution | chunk-above contribution | relation |
|---|---|---|---|
| Q1 | 0.02354 (`1/62 + 1/135`) | 0.02715 (`1/69 + 1/79`) | target **<** above |
| Q3 | 0.02557 (`1/61 + 1/109`) | 0.02844 (`1/67 + 1/74`) | target **<** above |
| Q10 | 0.02991 (`1/74 + 1/61`) | 0.03037 (`1/69 + 1/63`) | target **<** above |
| Q12 | 0.02844 (`1/83 + 1/61`) | 0.02967 (`1/70 + 1/65`) | target **<** above |
| R1-Q7 | 0.01639 (`0 + 1/61`) | 0.01954 (`1/117 + 1/91`) | target **<** above |
| Q5 | 0.01613 (`0 + 1/62`) | 0.01613 (`1/62 + 0`) | target **=** above (tie-break) |
| Q11 | 0.03154 (`1/66 + 1/61`) | 0.03154 (`1/61 + 1/66`) | target **=** above (tie-break) |

In 5 of 7 cases the target's contribution is strictly smaller than the chunk immediately above it —
the target is genuinely outweighed. In Q5 and Q11 the chunk above has an *exactly equal*
contribution and the ordering is decided by `fuse()`'s first-seen tie-break (BM25 list iterated
first). The target is never *ahead* of the chunk above it — RRF has no channel that could put it
there, because the lone retriever's confidence lives only in the scores RRF discards.

The clearest single contrast in the set is **Q2 vs R1-Q7**. Q2's target is ranked mediocrely by
*both* retrievers (BM25 8, dense 18) and RRF *promotes* it to fused rank 3 — two mid-list
contributions (`1/68 + 1/78 ≈ 0.0275`) outscore many lone strong ones. R1-Q7's target is ranked
**1st** by dense and unseen by BM25, and RRF *demotes* it to fused rank 10 (`1/61 ≈ 0.0164`). A
chunk both retrievers rank ~10th outranks a chunk one retriever ranks 1st. That is the defect,
stated at its sharpest.

### 2.4 The pre-registered tension — explicit answer

> Plan, B3 "Why it might not": *"R2-Q10 (BM25 rank 14) and R2-Q11 (BM25 rank 6) displace to the
> same fused rank 4 despite different corroboration strength … Displacement may track only the
> presence or absence of any corroboration, not its degree — the 15-query sweep is what separates
> the two."*

**The binary version survives. The graded version is falsified.**

- **Binary** — displacement tracks whether corroboration exists at all: **survives.** The three
  single-signal-dense cases *with* corroboration displace `+1 / +3 / +3`; the two *without*
  displace `+9 / +15`. Clean separation, no overlap. Extending the motivating trio with Q12 and
  Q5 sharpens the split rather than blurring it.
- **Graded** — displacement scales with *how much* corroboration: **falsified.** Among the
  corroboration-present cases, BM25 rank 23 (Q12) gives displacement **+1**, BM25 rank 14 (Q10)
  gives **+3**, BM25 rank 6 (Q11) gives **+3**. The relationship is non-monotone and, at the weak
  end, inverted: the *weakest* corroboration (Q12) produces the *smallest* displacement, because
  dense rank 1 plus a real BM25 rank-23 contribution (`1/61 + 1/83`) still sums high enough for
  fused rank 2. Q10 and Q11 — BM25 ranks 14 and 6, more than a factor of two apart — displace
  identically. A 2-rank spread across the present group at n=3, partly tie-break driven (Q11), is
  not distinguishable from noise.

This is the same failure the *original* B3 suffered, in a new form. The void score-ratio premise
("2.8× BM25 gap") tried to predict displacement magnitude from confidence magnitude; the rank
form tries to predict it from corroboration magnitude. Neither holds. What predicts displacement
is the **binary**: does the other retriever return the target anywhere in its pool.

### 2.5 Is "central band 2–10" the right description?

Not quite. Single-signal fused ranks: dense group `{2, 4, 4, 10, 17}`, BM25 group `{1, 2, 4}`.
Six of eight land in `[1, 4]` — *below* the plan's proposed band floor of 2 — and the two
un-corroborated cases sit at 10 and 17, one above the band ceiling. The phenomenon is **bimodal,
not a single band**:

- single-signal target **with** corroboration anywhere in the other pool → fused rank **1–4**
  (displaced only 1–3 from its finding rank);
- single-signal target **with no** corroboration → fused rank **10–17** (displaced 9–15).

No single-signal target reaches fused rank 1 without corroboration (closest: Q12 at rank 2, and
it has BM25 rank 23). Among the un-corroborated cases the displacement magnitude (+9 vs +15)
tracks how many corroborated chunks that particular query has competing for the top — a
query-density property, not anything about the target.

### 2.6 B3 verdict

**Confirmed in substance, with two corrections to the plan's wording.**

| Plan "Confirms" clause | Outcome |
|---|---|
| single-signal-dense target's fused rank strictly worse than its dense rank | **Holds 5/5** |
| absent-corroboration case worse-displaced than weak-corroboration cases | **Holds** — `+9, +15` vs `+1, +3, +3` |
| mirror cases: fusion improves the target vs the retriever that lost it | **Holds 3/3** |
| target's lone `1/(k+rank)` < summed contribution of the chunk above it | **Holds 5/7 strictly; 2/7 equal** (tie-break, target still not ahead) |

| Plan "Falsifies" clause | Triggered? |
|---|---|
| displacement uncorrelated with corroboration (absent ≈ weak, every depth) | No — binary separation is clean |
| a displaced target reaches fused rank 1 with no corroboration | No |
| mirror cases show no fusion benefit | No |
| single-signal targets scatter 1–30 with no clustering | **Partially** — "central band 2–10" is imprecise; the real shape is bimodal (1–4 with corroboration, 10–17 without) |

**Corrections carried forward:** (1) replace "central band, roughly fused rank 2–10" with
"bimodal: fused 1–4 when the other retriever also returns the target, 10–17 when it does not";
(2) the claim that displacement is "largely irrespective of the rank its sole supporting retriever
gave it" is right for the *finding* retriever's rank but the *other* retriever's rank matters as a
binary, not a magnitude.

---

## 3. B4 — does score-normalised fusion fix it?

Min-max CombSUM and z-score CombSUM, each retriever normalised over its own top-100 pool, a chunk
absent from a pool taking that scale's floor (`0.0` for min-max, `min(z)` for z-score — never the
mean). `fused = norm_bm25 + norm_dense`, sorted descending. This is the classical score fusion
RRF was designed to avoid (Lee 1997; Montague & Aslam 2001), and the counter-argument in the
plan's B4 entry — BM25's unbounded term-weight sums make cross-distribution normalisation fragile
— is exactly what we test.

### 3.1 Effect on the B3 defect (single-signal cases, n=8)

Recovery = `rrf_rank − normalised_rank`. Positive = the remedy moved the target back up.

| Query | class | RRF | min-max | z-score | min-max recovery | z-score recovery |
|---|---|---|---|---|---|---|
| Q12 | ss-dense | 2 | **1** | **1** | +1 (to rank 1) | +1 (to rank 1) |
| Q10 | ss-dense | 4 | **3** | **3** | +1 | +1 |
| Q11 | ss-dense | 4 | 4 | 4 | 0 | 0 |
| R1-Q7 | ss-dense | 10 | **3** | **4** | +7 | +6 |
| Q5 | ss-dense | 17 | **5** | **5** | +12 | +12 |
| Q3 | ss-bm25 | 2 | **1** | **1** | +1 (to rank 1) | +1 (to rank 1) |
| Q1 | ss-bm25 | 4 | **3** | 5 | +1 | −1 |
| Q7 | ss-bm25 | 1 | 2 | 2 | −1 | −1 |

**Min-max CombSUM improves 6 of 8, leaves 1 unchanged (Q11), worsens 1 (Q7 by one rank).**
z-score improves 5, unchanged 1, worsens 2 (Q1, Q7) — worse than min-max because its absent-pool
floor (a large negative `min(z)`) over-penalises any single-signal chunk against a fully
corroborated one.

But the recovery is **partial**. Only the two mild cases (Q12, Q3) reach rank 1. The severely
displaced un-corroborated cases improve sharply yet still fall short: R1-Q7 `10 → 3`, Q5
`17 → 5`. The reason is structural and the same as RRF's: a chunk both retrievers rank in their
top 5 scores roughly `0.9 + 0.9 = 1.8` in min-max space, while a lone dense rank-1 target scores
`1.0 + 0.0 = 1.0`. Score fusion shrinks the corroboration gradient; it does not remove it. A
confident single retriever still cannot hold rank 1 against two corroborating mid-ranks.

### 3.2 Stability cost (corroborated + weak/neither, n=8)

| Query | class | RRF | min-max | z-score | note |
|---|---|---|---|---|---|
| Q6, Q8, Q9, Q13, Q14 | corroborated | 1 | 1 | 1 | unchanged |
| **Q15** | corroborated | **1** | **3** | **3** | target demoted 2 ranks under both normalisers |
| Q2 | weak/neither | 3 | 4 | 5 | slips 1–2 ranks |
| Q4 | weak/neither | 39 | 41 | 32 | min-max worse, z-score better — noise at this depth |

**Q15 is the predicted failure.** Its target (BM25 3, dense 1) is RRF rank 1. Under score fusion,
two chunks with larger raw BM25 magnitudes at ranks 1–2 overtake it — BM25's unbounded score
distribution does exactly what the counter-argument said it would. Q2 slips too. So score fusion
buys ~6 mild single-signal recoveries at a cost of one clear corroborated regression (Q15, −2) and
two weak-case slips.

### 3.3 B4 verdict

**A directional remedy that carries the stability cost the plan's B4 entry predicted; not a clean
win, and the aggregate NDCG question the plan's "Confirms" clause asks cannot be answered here.**

- On the B3 defect: min-max CombSUM reduces single-signal displacement in 6 of 8 cases and is the
  better of the two normalisers. It fully recovers the mild cases (Q12, Q3 → rank 1) and roughly
  thirds the displacement on the severe ones (Q5 17→5, R1-Q7 10→3).
- It does **not** close the defect. No severely-displaced un-corroborated target reaches rank 1;
  summed normalised scores keep the same "two corroborators beat one confident signal" bias, with
  a shallower gradient.
- It regresses a corroborated target (Q15, rank 1 → 3) and nudges two weak cases down — the exact
  fragility the counter-argument named, driven by BM25's unbounded magnitudes.
- The plan's B4 "Confirms" test ("normalised fusion beats RRF on NDCG@10 across all 15") is
  **unevaluable** on this corpus: the qrels are circular (A2/A3) and ENH-11 is not done. On
  target-chunk rank, the honest read is B4's own predicted "Falsifies" shape: *it wins on the
  displaced-chunk queries and loses in aggregate — the trade-off it is, not a fix.*

---

## 4. What this means for the pipeline

1. **The displacement defect is real, systematic, and cheap to detect.** Every single-signal
   target is displaced; the two un-corroborated ones badly (fused rank 10, 17 — outside a
   top-3 or even top-10 candidate pool). B3's proposed runtime heuristic — *when one retriever
   ranks a chunk in its top ~3 and the other ranks it outside its top ~10 or not at all, prefer
   the single retriever's placement* — would fire correctly on R1-Q7, Q5, Q10, Q12 and lift their
   targets substantially. Whether it costs precision on non-target chunks is an ENH-11 question
   (needs graded labels), unchanged.
2. **Score-normalised fusion is not the drop-in fix.** Min-max CombSUM helps the flagged queries
   but under-recovers the severe cases and regresses at least one corroborated query. Shipping it
   wholesale trades one failure mode for another. If it is used at all, min-max beats z-score here.
3. **The candidate-pool-depth finding from A1/A2 stands reinforced.** R1-Q7 and Q5 at fused rank
   10 and 17 confirm that a top-3 fused candidate pool structurally cannot contain these targets,
   so no downstream re-ranker can rescue them — the fusion error has already removed the
   candidate. This is the A1/A4 "complementary failure modes" point, measured again from the
   fusion side.
4. **Recommendation:** do not wire score-normalised fusion into retrieval on this evidence. Record
   B3's displacement measurement and B4's partial-recovery result. Re-open both alongside ENH-11
   (graded, retriever-independent labels), which is what would let the aggregate-cost question be
   answered rather than reasoned around. The single-retriever-preference heuristic (point 1) is
   the cheaper thing to prototype next and does not need normalised fusion at all.

---

## 5. Reproduction

```
python run_b3_b4_fusion_eval.py           # live retrieval + all five fusions -> evaluation/b3_b4_fusion_eval.json
python run_b3_b4_fusion_eval.py --resummarize   # recompute summary block from the retained per-query records, no retrieval
```

Requires postgres + qdrant up (`docker compose up -d postgres qdrant`), the BM25 pickle at
`data/indexes/bm25_index.pkl`, and Python 3.11 in the project venv. e5-base-v2 loads once
(~0.2 GB peak on top of the services); the encoder is released before the fusion arithmetic runs.
