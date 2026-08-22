# Correction Notice — Hypothesis A1 and Related Documents

**Reference:** CORR-NVIR-2026-001
**Version:** 2.0 — supersedes v1.0 of the same date
**Raised:** 22 August 2026
**Severity:** Major — affects a stated mechanism and one unrun hypothesis. **No measured result is invalidated.**
**Affects:** Retrieval Hypothesis Test Plan (A1, B4) · `docs/uat/round3_family_a_findings.md` · `docs/explainers/family_a_explained.md`

> **v2.0 revision note.** v1.0 of this notice concluded that Round 2 Q1 could not demonstrate
> corroboration bias, on the grounds that BM25 had ranked a See-also block above the target. Inspection
> of `609937c` and `api/static/evaluation.html` shows that conclusion was **wrong**. Corroboration bias
> *was* the fusion mechanism on Q1, via a different route than the plan described — see §2.3 and §4.1.
> v1.0 also listed `609937c` as requiring correction; it does not. **That commit was correct on 18 August
> and remains the most accurate account of Q1 in the repository.** The 33.4 misattribution (§2.1) and the
> wrong-queries finding (§2.2) are unchanged.

---

## 1. Summary

Hypothesis A1's write-up quoted a BM25 score belonging to a different query, and A1 was run against different cases from the ones the claim it tested had actually been documented on. A1's measurements are real and its verdict stands.

**Two things are wrong, and one thing v1.0 of this notice got wrong.**

1. **The 33.4 score is misattributed.** It belongs to Q8, not Q1. Q1's target sits at BM25 rank 2, not rank 1, trailing by 1.5% rather than leading by 2.8×. *(§2.1)*
2. **A1 ran against the wrong queries.** The claim under test — Round 1's recommendation that a cross-encoder "should not be subject to RRF's corroboration bias" — was documented on Round 1's Q3 and Q7, which Round 1 explicitly nominated as regression candidates. The Round 3 plan reattached it to Round 2's Q1 and Q10. *(§2.2)*
3. **v1.0 of this notice then over-corrected**, concluding that corroboration bias was absent from Q1. It was present — the evaluation page had already measured it precisely. *(§2.3)*

The repository was never wrong about any of this. Commit `609937c` recorded the correct figures on 18 August and retracted the "should not be subject to" claim on measured evidence. **The error existed only in the planning document and in the explanatory documents built from it.**

---

## 2. What the error is

### 2.1 The score attribution

The plan states, twice:

> *"Run for Q1 with target chunk `cc6c8e53936d04e9b192a7d5` (the `cudaMalloc(void**, size_t)` signature block, BM25 rank 1, score 33.4 against 12.1 for its own rank 2)"*

> *"On Q1, BM25 scored 33.4 for the target against 12.1 for its rank 2: a 2.8x gap."*

**Neither statement is supported by the raw data.** From `docs/uat/uat_superiority_cases_raw.json`, Round 2 Q1 (`"CUDA cudaMalloc function parameters"`):

| BM25 rank | chunk_id | Score | Chunk |
|---|---|---|---|
| 1 | `381cf7a1dddd75346b7446ee` | **12.1774** | `cudaFreeMipmappedArray` — a "See also" cross-reference block |
| 2 | `cc6c8e53936d04e9b192a7d5` | **11.99** | The `cudaMalloc(void**, size_t)` signature — *the target* |
| 3 | `81b9c458ed8d5bbf219819b8` | 11.1421 | `cudaStreamAddCallback` note |

The target is at **rank 2, not rank 1**, and trails rank 1 by **1.5%** — not a 2.8× lead.

`uat_day5_raw.json` shows byte-identical ranks and scores for the same query in Round 1, so no alternative configuration produces the plan's figures.

**Where 33.4207 actually comes from:** `docs/uat/uat_superiority_cases_executed.md:159` — BM25 rank 1 on **Q8** (`"shared memory bank conflicts and how to avoid them"`), chunk `fd5aa3318c4def606709ac98`, the TMA Swizzle chunk, against 29.1927 at rank 2.

The `12.1` in the plan is real but is Q1's **rank-1** score, not its rank-2 score. Two queries' figures were fused into a single claim.

### 2.2 The deeper error — wrong queries entirely

The score is the visible symptom. The cause is an identifier collision between rounds.

`docs/uat/uat_day5_retrieval.md` — Round 1, 12 July — closes with:

> *"**Recommendation for Layer 3b**: a cross-encoder re-ranker reading full chunk text (not just rank position) should recognize "shader processor" ≈ "CUDA cores" the way dense embeddings do, and should not be subject to RRF's corroboration bias since it scores each candidate independently — **Q7 and Q3 are good regression-test candidates** once re-ranking lands."*

This is the sentence the entire Round 3 plan was built to attack. It refers to **Round 1's Q7 and Q3**:

| Round 1 | Query | Round 1 finding |
|---|---|---|
| **Q3** | `H100 HBM2e memory capacity` | *"dense uniquely found the HBM2/ECC chunk; RRF loses it entirely"* |
| **Q7** | `shader processor count` | *"dense uniquely found the '128 CUDA cores' chunk; RRF loses it entirely"* — described as **"the most important finding of the UAT"** |

Both are cases where **one signal ranked the correct chunk highly and fusion discarded it** because, in Round 1's own words, *"a chunk only one signal ranks (even at rank 1) is worth less under RRF than two chunks both signals rank moderately."* That is corroboration bias, precisely stated.

By the time the Round 3 plan was written on 18 August, the claim had been reattached to **Round 2's** Q1 (`cudaMalloc`) and Q10 (`latency hiding`). Different queries, different round, and — critically — **a different mechanism.**

### 2.3 Corroboration bias *was* present on Q1 — v1.0 of this notice was wrong about this

v1.0 reasoned that because BM25 ranked a See-also block above the target, BM25 was already wrong and fusion merely inherited a bad ranking — so Q1 could not demonstrate a fusion defect.

**That reasoning does not survive the measured data.** `api/static/evaluation.html`, as written by `609937c`, gives the arithmetic:

| Chunk | BM25 | Dense | RRF score |
|---|---|---|---|
| `cc6c8e53…` — the `cudaMalloc` signature, **the target** | rank 2 | **rank 75** (cosine 0.872) | `1/(60+2)` ≈ **0.0161** |
| `81b9c458…` — a `cudaStreamAddCallback` chunk | rank 3 | rank 3 | `2/(60+3)` ≈ **0.0317** |

Fused rank 1 went to `81b9c458…`, a chunk **neither retriever ranked first**, because both ranked it third and it therefore carried two contributions. The target carried one.

That is corroboration bias in its exact form. The plan's error was in *which* asymmetry drives it: not a 2.8× BM25 confidence gap, but the target's **dense rank of 75** isolating it to a single signal. Two moderate agreements at rank 3 beat one solid single-signal result at rank 2.

**Consequence for v1.0's claim.** BM25's own rank 1 (`381cf7a1…`, the See-also block) is *not* the fused rank 1 (`81b9c458…`). Different chunks. So fusion did not inherit BM25's ranking — it promoted a third chunk over both. v1.0 conflated the two and overstated the BM25 problem.

**What remains true from §2.1 and §2.2.** The score is still misattributed, the target is still at rank 2 not rank 1, and A1 was still run against different queries from the ones the claim was documented on. Q1 turns out to be a *legitimate* corroboration-bias case — it simply isn't the one Round 1 nominated, and the plan reached it via a figure that doesn't exist.

---

## 3. What is and is not affected

### 3.1 Not affected — all measured results stand

| | Status |
|---|---|
| Capstone submission (`8569bfb`, 16 Aug) | **Unaffected.** The plan post-dates it by two days. |
| Config A / C NDCG (0.5333 / 0.5280) | Unaffected — computed independently |
| Round 1 and Round 2 UAT results | Unaffected — these are the source of truth that exposed the error |
| A2, A3, A3 Part 2, A4, A6, A7 | Unaffected — none depends on the Q1 score |
| A5 (all three attempts) | Unaffected — recomputed and committed at `0149ca4` |
| A1's re-ranking measurements | **Unaffected** — target at pool rank 4, promoted to rank 2, `cudaFreeArray` chunk retaining rank 1 are real observations |
| A1's chunking diagnosis | **Unaffected, and strengthened** — see §4.2 |
| **`609937c`** | **Unaffected — correct as committed.** Recorded BM25 rank 2, dense rank 75, fused rank 4 on 18 August, and retracted the "should not be subject to this failure mode" claim on measured evidence. v1.0 of this notice wrongly flagged it for review. |

### 3.2 Affected

| Item | Correction required |
|---|---|
| **Plan, A1** | Wrong score, wrong rank, wrong queries. Re-specified in §5. |
| **Plan, B4** | Premise void — see §3.3 |
| **`round3_family_a_findings.md`** | A1 entry repeats the score and mechanism |
| **`family_a_explained.md`** | A1 section and Part Six Q&A repeat both |

### 3.3 Hypothesis B4 is unrunnable as written

B4 proposes testing whether displacement under fusion correlates with the BM25 rank-1-to-rank-2 score ratio, motivated entirely by Q1's *"2.8× gap."* **That gap does not exist**, so B4 has no motivating case.

Family B has not been run, so nothing is invalidated. But B4 must be re-specified before it is. The underlying question — *does fusion displace high-confidence single-signal results more readily?* — remains sound and is arguably better served by Round 1's Q3 and Q7, where dense uniquely held the answer.

---

## 4. Corrected findings

### 4.1 What Round 2 Q1 actually shows

Three distinct things, none of which is the plan's account.

**A · Fusion exhibits corroboration bias.** The target reaches BM25 rank 2 but dense rank 75, giving it a single-signal RRF score of 0.0161. A `cudaStreamAddCallback` chunk that both retrievers rank third scores 0.0317 on two contributions and takes fused rank 1. The target falls to fused rank 4. *(§2.3)*

**B · Lexical ranking is separately corrupted by chunk boundaries.** BM25's own rank 1 is `381cf7a1…`, a `cudaFreeMipmappedArray` See-also block — trailing cross-reference boilerplate fused to the next entry's header, containing no explanatory prose. It out-scores the real signature 12.1774 to 11.99. On a query built from function-name vocabulary, **a chunk that is nothing but function names — including `cudaMalloc` itself, in its See-also list — is structurally advantaged over the page that explains it.**

**C · The cross-encoder fails on the same chunk, by an unrelated mechanism.** The evaluation page records that after re-ranking, a `cudaFreeArray` chunk holds CE rank 1 at score 6.0145, ahead of the signature at 5.4647. As `609937c` puts it: *"Corroboration bias isn't the mechanism — the cross-encoder never saw rank positions, only text — yet the same wrong chunk won under both mechanisms. Both retrieval strategies failed."*

**Point C is the strongest finding available from Q1, and neither the plan nor v1.0 of this notice stated it.** Two independent ranking mechanisms — one rank-based, one text-based, sharing no assumptions — converge on the same wrong chunk. That is not a ranking problem twice over. **It points at the chunk.** The evaluation page reaches the same conclusion by inspecting text: three of the four chunks dominating Q1's pool come from the same source and share the straddling-boilerplate shape.

### 4.2 The chunking diagnosis strengthens

Straddling chunks are not merely a problem that *survives* fusion. They **corrupt lexical ranking directly** — BM25 prefers them to the correct answer on its own terms — and they **survive cross-encoding**, which reads full text and shares none of fusion's rank-position assumptions.

A defect that defeats two independent ranking mechanisms is not a ranking problem. It is a claim about the chunk, and it is independent of which retrieval architecture you choose. It also connects cleanly to DEF-10 from the functional sign-off (table-of-contents chunks polluting lexical rankings) — the same failure class, a different chunk shape.

Family C's C4 is confirmed by Q1. The C1-versus-C5 trade-off — filter See-also blocks, or merge them into their parent entry — is now the live question, and §4.1 suggests merging may be preferable: the block's cross-references are genuinely useful, but only when attached to explanatory text rather than standing alone.

### 4.3 Corroboration bias is real, and Round 1's nominated cases are still untested

Round 1 documented it on **3 of 9 queries** (its Q2, Q3, Q7). Round 2's Q1 is a fourth instance (§2.3). The phenomenon is well evidenced.

What remains untested is Round 1's specific recommendation: that a cross-encoder, scoring text rather than rank position, should **recover** chunks fusion discards. Round 1 nominated its Q3 and Q7 for exactly this test.

Q1 is a poor test of that recommendation, because §4.1 point C shows the cross-encoder failing there for reasons unrelated to fusion — the chunk defeats text-based scoring too. **A test of whether the cross-encoder corrects fusion needs a case where the chunk itself is sound**, which is precisely what Round 1's Q3 and Q7 offer: dense found the right chunk, fusion lost it, and nothing suggests the chunk is malformed.

That is A1-R, in §5.

---

## 5. Re-specification — A1-R

> **A1-R · A cross-encoder recovers single-signal answers that RRF discards**
>
> **Motivating evidence.** Round 1 found that on `H100 HBM2e memory capacity` (R1-Q3) and `shader processor count` (R1-Q7), dense retrieval uniquely surfaced the correct chunk and RRF lost it entirely — because a chunk ranked by one signal scores below two chunks ranked moderately by both. Round 1 nominated exactly these two as regression candidates once re-ranking landed. This closes that loop.
>
> **Claim.** A cross-encoder scoring each candidate against full chunk text, independent of rank position, should restore the dense-unique chunk to the top of the final ranking — provided the candidate pool is deep enough to contain it.
>
> **Why not Q1.** Q1 is a genuine corroboration-bias case, but a poor test of this claim: the cross-encoder fails there because the competing chunk defeats text-based scoring as well (§4.1 point C). Testing whether re-ranking *corrects fusion* requires a case where the chunk itself is sound. R1-Q3 and R1-Q7 are those cases.
>
> **Protocol.**
> 1. For R1-Q3 and R1-Q7, retrieve BM25 top-100 and dense top-100; record the rank at which the dense-unique target chunk appears in each.
> 2. Fuse with RRF (`k=60`) and record the target's fused rank — confirming Round 1's displacement finding at depth, not just at top-3.
> 3. Re-rank the fused pool with Config A and Config C; record the target's final rank under each.
> 4. Repeat at pool depths 10, 20, 50, 100 to separate pool-recall failure from re-ranking failure.
>
> **Confirms if** the target reaches rank 1–3 post-re-ranking at any pool depth where it is present in the pool.
>
> **Falsifies if** the target remains displaced at all depths where the pool contains it — which would mean the cross-encoder shares the fusion bias rather than correcting it, and would make the Round 1 recommendation wrong.
>
> **Records rather than concludes** if the target is absent from the pool at every depth. That is a pool-recall failure and says nothing about re-ranking.
>
> **Prerequisite — R1-Q3 is currently untestable.** Its target is HBM2/ECC content from the Hopper architecture whitepaper, which §6 shows was never ingested. Either resolve DEF-19 first, or run A1-R on R1-Q7 alone and record the limitation.
>
> **Persist everything.** Scripts and per-query JSON output committed before analysis, per the lesson recorded at `0149ca4`.

---

## 6. Related defects found during this investigation

Not part of the A1 correction, recorded here because they were surfaced by it.

### DEF-19 · `data/raw/` is orphaned — Round 1 coverage gaps are ingestion defects

Six PDFs are DVC-tracked in `data/raw/` (`921281c`, 10 July): the A100 datasheet, Ampere and Hopper architecture whitepapers, and three CUDA guides. **The ingestion pipeline never reads this directory** — both `run_ingest_direct.py` and `airflow/dags/ingest_nvidia_docs.py` download from a hardcoded URL list into a temp staging directory.

Consequently, Round 1's two "coverage gap" conclusions are wrong about their cause:

| Round 1 query | Recorded as | Actually |
|---|---|---|
| `NVLink 4.0 bandwidth specifications` | *"the corpus has no GPU-architecture whitepaper"* | Two whitepapers are in `data/raw/`, never ingested |
| `best practices for optimising neural network training` | *"the corpus is CUDA systems documentation, not ML-framework documentation"* | The cuDNN Developer Guide was declared in the ingest list; its URL 404s |

Round 1 was right that these were not retrieval-algorithm defects. It was wrong that they were scope decisions. **Both are ingestion failures recorded as design boundaries.**

### DEF-20 · Silent partial-corpus ingestion

Both ingest paths declare **eight** source documents. Three return HTTP 404 and have done since at least October 2025 (`Last-Modified` on the error page):

- `cuDNN-Developer-Guide.pdf`
- `TensorRT-Developer-Guide.pdf`
- `Thrust_Quick_Start_Guide.pdf`

`download_pdfs()` catches the failure, logs a warning, and continues. Parsing skips anything not `downloaded`. **The pipeline reports success with five documents.** Database confirms five `doc_id` values summing to exactly 5,389 chunks.

NVIDIA serves an identical 78,745-byte HTML error page for all three (same ETag). The pipeline's `raw[:5] != b"%PDF-"` check is the only thing preventing three copies of an error page entering the corpus as content. **That check should be documented as deliberate.**

*Fix:* assert the expected document count and fail the run on shortfall; hash downloaded content and log it so corpus drift is detectable (`_doc_id` hashes the URL, not the content, so revisions are currently invisible).

### DEF-21 · A test encodes an ingestion defect as intended scope

`tests/api/test_main.py:194`:

```python
for out_of_scope_topic in ["NVLink", "H100", "TensorRT"]:
```

All three are absent for defect reasons — NVLink and H100 via DEF-19, TensorRT via DEF-20. The test passes and is correct about current behaviour, but **locks a plumbing failure in as a design boundary**, making it substantially harder to notice or reverse.

### DEF-22 · Qrel entries for uningested content

`evaluation/relevance_labeller.py:153` includes `"cuDNN convolution algorithm selection"` and `"NCCL all-reduce collective communication pattern"` in its query set. cuDNN was declared but 404'd; **NCCL was never in the source list at all.** Neither can have a relevant chunk in the index.

Whether these produced zero labels or something worse should be checked, as it bears on the label-sparsity findings in A2 and A3.

---

## 7. How this happened, and what it does and doesn't say

The plan's A1 entry fused a score from Round 2 Q8 with queries from Round 2 that had inherited a claim documented on Round 1's Q3 and Q7. Two rounds use overlapping Q-numbering for different query sets, and one query — `CUDA cudaMalloc function parameters` — appears in both. Every explanatory document built on the plan reproduced the error faithfully.

**The scope of the failure is narrower than it first appeared, and worth stating precisely.**

The repository was correct throughout. `609937c`, committed 18 August, measured Q1 against live indexes, recorded BM25 rank 2 and dense rank 75, published the RRF arithmetic, and explicitly retracted the "should not be subject to this failure mode" claim. It also recorded the two-mechanism convergence that §4.1 identifies as Q1's strongest finding. **Nothing user-facing was ever wrong.**

What was wrong was a planning document, and the explanatory documents written from it — including this notice at v1.0, which over-corrected in the opposite direction and had to be revised in turn.

Two lessons, of unequal weight.

**The smaller one.** A figure in prose carries no provenance. `uat_superiority_cases_raw.json` disproves the 33.4 claim in one line and has been committed since 13 July. The lesson at `0149ca4` — *"here is the script, here is the output, run it yourself"* — extends to planning documents, which are exactly the kind of authoritative-looking artifact nobody thinks to verify.

**The larger one, from v1.0's own failure.** Having found the plan wrong, v1.0 constructed a replacement mechanism from the data immediately to hand (BM25's ranks) without checking `609937c`, which had already measured the whole picture including the dense rank 75 that actually explains the displacement. **The correction repeated the error it was correcting** — reasoning from a partial view rather than checking the record — and produced a confident, wrong conclusion in the document arguing for verification.

There is a symmetry here with Family A's central finding, and it should be stated rather than smoothed over. Family A found that four experiments had been constrained by an evaluation instrument nobody had checked. This notice found that a mechanism narrative had been built on a figure nobody had checked. And v1.0 of this notice built a replacement narrative on evidence it hadn't finished checking. The failure mode is not carelessness about numbers; it is **stopping the investigation at the first coherent story.**

*Mitigating detail, in fairness to the process:* each error was caught by the practice it argues for, and caught within days rather than shipping.

## 8. Actions

| # | Action | Priority |
|---|---|---|
| **C1** | Correct A1 in `round3_family_a_findings.md` — replace mechanism per §4, mark verdict *"partially confirmed, wrong evidence"* | High |
| **C2** | Correct the A1 section and Part Six Q&A in `family_a_explained.md` | High |
| **C3** | ~~Verify `609937c`~~ — **closed, no action.** Commit is correct as written and is the authoritative account of Q1. | Done |
| **C4** | Re-specify B4 in the plan; note the void premise | High |
| **C5** | Add a correction note to the plan's A1 entry pointing here | Medium |
| **C6** | Run A1-R per §5 — the first genuine test of the corroboration-bias claim | Medium |
| **C7** | Raise DEF-19 through DEF-22 in the defect register | Medium |
| **C8** | Adopt a round-prefixed query convention (`R1-Q7`, `R2-Q7`) throughout | Low |
| **C9** | Carry §4.1 point C — two independent mechanisms converging on the same wrong chunk — into Family C as direct evidence for C4. It is the strongest single argument for the chunking defect and is currently recorded only on the evaluation page. | Medium |

---

*v2.0, 22 August 2026, superseding v1.0 of the same date. No measured result is invalidated by this notice. A1's verdict stands. Its evidence base was wrongly chosen; its mechanism was wrongly described by the plan and then wrongly re-described by v1.0 of this notice. The repository — `609937c` — was correct throughout and remains the authoritative account of Q1.*
