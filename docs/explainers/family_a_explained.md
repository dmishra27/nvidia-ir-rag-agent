# What We Learned About Re-Ranking
### A complete plain-English account of the Family A experiments on `nvidia-ir-rag-agent`
*Hybrid retrieval over NVIDIA CUDA documentation · UAT Round 3 — ranking, fusion and re-ranking*

> ⚠️ **Revised 22 August 2026** per correction notice CORR-NVIR-2026-001 v2.0. Hypothesis A1 was
> originally written up using a BM25 score belonging to a different query, and against different queries
> from the ones the claim was documented on. A1's measurements stand; its mechanism is corrected here,
> and the chunking finding it produced is considerably *stronger* as a result. No other section is
> affected, and the repository's own evaluation page was correct throughout.

---

## Part One: What the system is, and why it exists

Imagine you have a large pile of technical documentation — in this case, NVIDIA's CUDA programming guides, API references, and best-practice manuals. Someone asks a question: *"What does `cudaMalloc` actually do?"* or *"How do I make my GPU programs run faster?"*

A search system has to do two things. First, **find** the handful of paragraphs most likely to contain the answer. Second, **use** those paragraphs to write a reply. This project is about the first half — the finding — because if you retrieve the wrong paragraphs, no amount of clever writing afterwards will save you. The model will confidently explain something that isn't the answer.

The corpus here is **5,389 chunks** of text, sliced out of the NVIDIA documentation. Everything below is about how we pick three to ten of those 5,389 when a question arrives.

### The three tools in the toolbox

**BM25** is keyword matching, done well. It is decades old and still excellent. If you search for `cudaErrorInvalidValue`, BM25 finds every chunk containing that exact string and ranks them by how unusual the term is and how often it appears. Its blind spot is vocabulary: ask "how do I make my GPU programs run faster" and BM25 looks for the literal words "faster" and "programs," which may appear nowhere in a document that is entirely about performance optimisation.

**Dense retrieval** solves exactly that. Every chunk is converted by an embedding model (`intfloat/e5-base-v2`) into a list of numbers that encodes its *meaning* rather than its words, and stored in a vector database (Qdrant, 768-dimension cosine vectors). The question gets the same treatment, and we find the chunks whose numbers sit closest to the question's. This handles paraphrase beautifully. Its blind spot is the mirror image of BM25's: exact identifiers. To an embedding model, `cudaMalloc` and `cudaMallocPitch` look nearly identical, because they *are* nearly identical as text — but they're different functions, and returning the wrong one is a real failure.

**RRF — Reciprocal Rank Fusion** — is the compromise. Run both retrievers, then merge their two ranked lists into one. The merging rule is deliberately simple: it ignores each retriever's confidence scores entirely and looks only at *rank position*, via `score = Σ 1/(k + rank)` with `k = 60`. A chunk that BM25 puts first and dense puts twentieth gets combined based on "first" and "twentieth," not on how strongly BM25 felt about it.

This is a feature, not an oversight — BM25 scores and embedding similarities live on incompatible scales, and comparing them directly is fragile. But it has a cost, and that cost is where Family A begins. With `k = 60`, the reciprocal ranks of the top few positions are nearly identical: 1/61, 1/62, 1/63, 1/64 span about 5%. **Two mediocre chunks that both retrievers rank in the top few can therefore outweigh one chunk that a single retriever is overwhelmingly confident about.** That's the *corroboration bias*, and it is the flaw the entire family circles around.

*(This isn't theoretical arithmetic. The production deployment runs BM25 only, and its returned scores were measured at exactly 0.016393, 0.016129, 0.015873 — precisely 1/61, 1/62, 1/63. Single-ranker RRF scores are near-flat by construction: a 3% spread across ten results.)*

### The fourth tool, and the reason for this investigation

After fusion, a **cross-encoder re-ranker** takes a second look. Where BM25 and dense retrieval each score chunks in isolation and hope the rankings line up, a cross-encoder reads the question and a candidate chunk *together* and produces a relevance score for that specific pair. Far more accurate, far slower — too slow for 5,389 chunks, fine over a shortlist of a hundred.

Three configurations were on the table:

| | Model | Where it runs | Aggregate NDCG@10 | Status |
|---|---|---|---|---|
| **Config A** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Locally, ~90 MB | **0.5333** | Ran |
| **Config B** | `bge-reranker-v2-m3` | Needs more RAM than the host has | — | **Never ran — see A7** |
| **Config C** | Cohere Rerank v3 | Hosted API | **0.5280** | Ran |

NDCG is the standard measure here: roughly, "how close to perfect was this ranking," where a perfect ranking puts the best chunk first. Both configurations clear the project's quality gate of NDCG@10 ≥ 0.50 — but with headroom of only **6.7% and 5.6%** respectively. The gate is doing real work, not acting as a formality.

The two configurations differ by **0.0053** — about one percent. A gap that small looks like a tie. Most teams would stop there and pick whichever is cheaper.

### Where this round sits

This is the third round of user acceptance testing on the project, and the rounds build on each other:

| | What it established |
|---|---|
| **Round 1** *(9 queries)* | Retrieval pipeline validation — does the thing work end to end |
| **Round 2** *(15 queries, 6 case types)* | Which retriever wins on which query type. 10 of 15 predictions held; the 5 failures were diagnosable *because the predictions had been written down beforehand* |
| **Round 3** *(this document)* | The component Rounds 1 and 2 never tested: the re-ranker |

That last point is the gap. Rounds 1 and 2 compared BM25 against dense against RRF. **The cross-encoder appeared in the existing evaluation only as a footnote on two queries** — despite the functional sign-off already having found that it changes rankings materially, and *not uniformly for the better*.

---

## Part Two: The architecture, and where each experiment attaches

```
                          ┌──────────────────────────┐
                          │   User question          │
                          │  "why is my kernel slow?"│
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │  BM25  (sparse)       │             │  Dense  (semantic)    │
        │  keyword / IDF match  │             │  e5-base-v2 → Qdrant  │
        │  over 5,389 chunks    │             │  5,389 × 768-dim      │
        │                       │             │  (exhaustive search — │
        │                       │             │   below HNSW 10k)     │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │  ranked list                        │  ranked list
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  RRF fusion   k = 60         │  ◄── A6 asks: skip this
                        │  score = Σ 1/(k + rank)      │       for identifier
                        │  ignores confidence scores   │       queries?
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  CANDIDATE POOL              │  ◄── A2 asks: how deep?
                        │  depth = 3 … 100             │       (default 100,
                        │  ** the variable under test**│        benchmarked at 3)
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Cross-encoder re-rank       │  ◄── A1 can it rescue?
                        │  Config A / B / C            │  ◄── A3 which query types?
                        │  scores (query, chunk) pairs │  ◄── A4 does it ever harm?
                        │                              │  ◄── A5 do A and C agree?
                        └──────────────┬───────────────┘  ◄── A7 what about B?
                                       ▼
                        ┌──────────────────────────────┐
                        │  Final top-N → LLM → answer  │
                        └──────────────────────────────┘
```

**The single most important thing this diagram shows** is that the candidate pool sits *between* fusion and re-ranking. Whatever fusion discards never reaches the re-ranker. The re-ranker cannot rescue what it never sees.

That is the crux of the whole family. The project's evaluation page had asserted that a cross-encoder — which scores candidates independently rather than by rank position — *"should not be subject to this failure mode."* As the plan put it: **that word is doing a lot of work.** And the same page quietly recorded why the claim was shaky: on Q1, the canonical `cudaMalloc()` signature chunk *"never reached the candidate pool the cross-encoder saw at all."*

So the headline question wasn't *does re-ranking help*. It was **can re-ranking help, given a shortlist deep enough for it to see the answer** — and everything in Family A follows from that.

---

## Part Three: The measuring instrument — qrels and the human in the loop

Before any experiment, one question has to be answered: **how do you know a ranking is good?**

You need an answer key. In information retrieval it's called a **qrel** file — short for *query relevance judgements*. It is a list of pairs: for each question, which chunks are genuinely relevant. Every metric in this project — NDCG, MRR, precision — is computed by comparing what the system returned against that answer key.

### How this project's qrels were built

This is where the human-in-the-loop part comes in, and it's worth being precise about what was and wasn't human-judged.

```
   1. System runs the 15 test queries through BM25 + Dense + RRF
                            │
                            ▼
   2. For each query, take the TOP 3 fused results          ◄── MACHINE decided
                            │                                    what gets shown
                            ▼
   3. A human reads each chunk and marks: relevant?         ◄── HUMAN judged
                            │                                    what's relevant
                            ▼
   4. Positive labels become the qrel answer key
      → 15 queries × 3 candidates judged
      → variable positives per query (0, 1, or 2)
```

The human judgement in step 3 was sound. **The sampling in step 2 was not.** The pool a human was shown had already been filtered by the very system the qrels would go on to evaluate. That single design decision propagated into three separate experiments, and it is the connective tissue of this entire family.

### What the resulting answer key actually looks like

Because the pool was three deep, the label set is thin and uneven. Only **8 of 15 queries** survived the filter of having at least one positive label:

| Query | Case type | Positive labels | Consequence |
|---|---|---|---|
| Q1 | BM25 lexical | **0** | Excluded from A3 entirely |
| Q2 | BM25 lexical | 1 | Usable |
| Q3 | BM25 lexical | **0** | Excluded from A3 entirely |
| Q7 | RRF hybrid | 1 (`ccd7708b…`) | Usable |
| Q8 | RRF hybrid | **2** (`f6f2063c…`, `1fabb00b…`) | Usable, two-deep |
| Q10 | BM25 failure / vocab gap | **0** | Excluded — whole case type has no data |
| Q11 | BM25 failure / vocab gap | **0** | Excluded — whole case type has no data |
| … | … | … | … |

### The sharpest illustration in the document

The evaluation page had named exactly **two** queries where RRF cost the system its best chunk: **Q1 and Q10**. Those two queries are the reason Family A exists.

**Both of them ended up with zero positive relevance labels.**

Not by coincidence. On Q1, the correct `cudaMalloc` signature chunk had been displaced out of the top 3 before a human ever reviewed the pool, so it was never labelled. On Q10, the same. **The two queries the plan built its case on are precisely the two the answer key cannot see** — displaced by the very ranking behaviour they were chosen to illustrate.

*(Per the correction notice: these were not in fact the best cases for the plan's claim — Round 1's own regression candidates were. But the label consequence stands regardless of which queries motivated the round.)*

**The bias under study had systematically erased its own evidence from the instrument used to study it.** That is not one unlucky query. That is a measurement system with a structural blind spot aimed directly at the thing being measured.

### A worked example: Q8's two labels

Q8 asks about avoiding shared-memory bank conflicts. Two chunks were judged relevant during HITL review — `f6f2063c3d7aac6f24143411` and `1fabb00b…` — both from the top 3 the human was shown.

Now watch what happens when the pool deepens to 50 and the cross-encoder re-ranks everything (full detail in A3 Part Two below): the primary label lands at **rank 4**, the secondary at **rank 9**, and three unlabelled chunks sit above them — one of which is a detailed walkthrough of eliminating bank conflicts via a swizzle pattern.

Is that a retrieval failure? By the metric, yes: the labelled chunks moved down, so NDCG falls. **By any reasonable human reading, no** — a genuinely useful chunk was promoted above a labelled one, and the answer key simply had no vocabulary for saying so.

### The fix, and why it's now evidence-backed

The remedy is **graded relevance judgements** built from a deeper, retriever-independent pool — ideally pooled across multiple configurations so no single system decides what a human gets to see, and graded (0–3) rather than binary so "also good" is expressible.

That work was already on the backlog as a deferred enhancement. What Family A did was convert it from a nice-to-have into a documented blocker, with four independent lines of evidence.

---

## Part Four: The seven experiments, with worked examples

Each hypothesis was written down *before* running it, with explicit confirm and falsify conditions. This is the reporting standard Round 2 established and Round 3 inherited: **prediction stated first, outcome recorded whether or not it matched, and the mechanism explained where a prediction failed.**

The plan says why that matters, and it's worth quoting: *"The value of Rounds 1 and 2 was not that 10 of 15 held — it was that the 5 failures were diagnosable because the predictions were written down beforehand."*

---

### A1 — Can the cross-encoder rescue what fusion loses?

> **Prediction:** give the re-ranker a deeper shortlist and it will recover the good chunk that fusion buried.
> **Run on:** Round 2's Q1 (target `cc6c8e53936d04e9b192a7d5`, the `cudaMalloc(void**, size_t)` signature block) and Q10 (dense's latency-hiding match). *These were the wrong cases — see below.*
> **Confirms if:** the target climbs to rank 1 once the pool is deep enough to contain it.
> **Falsifies if:** the target stays buried regardless of depth.

**The worked example — Q1, the `cudaMalloc` query.**

**What was originally written here was wrong, and the correction is more interesting than the error.**

The plan recorded that BM25 had been emphatic about the right answer — the signature chunk at rank 1 with a score of 33.4 against 12.1 for rank 2, a 2.8× gap that fusion then discarded. That would be a textbook demonstration of corroboration bias: rank-based fusion ignoring a confidence signal it cannot see.

Checking the raw results shows something else entirely:

| BM25 rank | Chunk | Score |
|---|---|---|
| 1 | `381cf7a1…` — a `cudaFreeMipmappedArray` "See also" block | **12.1774** |
| 2 | `cc6c8e53…` — **the `cudaMalloc` signature, the target** | **11.99** |

The target was at **rank 2, trailing by 1.5%**. The 33.4 belongs to a different query altogether — Q8, about shared-memory bank conflicts. Two queries' numbers had been fused into one claim.

**What was actually going on.** Fusion *did* discard the target — but through the other retriever, not BM25. Dense ranked the signature chunk **75th**. So it entered fusion with one signal's support:

| Chunk | BM25 | Dense | RRF score |
|---|---|---|---|
| The `cudaMalloc` signature — **the target** | rank 2 | **rank 75** | `1/(60+2)` ≈ **0.0161** |
| A `cudaStreamAddCallback` chunk | rank 3 | rank 3 | `2/(60+3)` ≈ **0.0317** |

Fused rank 1 went to a chunk **neither retriever ranked first** — it just had two moderate votes instead of one good one. The target fell to rank 4. That is corroboration bias, exactly as described; the plan simply had the wrong asymmetry driving it.

**What happened when the cross-encoder got a deeper pool:** the target climbed from **rank 4 to rank 2**. Real, measurable rescue. But a `cudaFreeArray` chunk still held rank 1, scoring 6.0145 against the signature's 5.4647.

**And that is the finding worth keeping.** The cross-encoder never saw a rank position. It read text, scored each chunk against the query directly, and shares none of fusion's assumptions. Yet **the same wrong chunk won under both mechanisms.** Two independent ranking systems, no common machinery, converging on the same mistake.

When that happens, the problem is not the ranker. It is the chunk.

**Verdict: partially confirmed.** The re-ranker helps and cannot fully undo the damage.

**The diagnosis, which the correction strengthens.** Why was a cross-reference block beating the `cudaMalloc` signature on a `cudaMalloc` query? Reading the actual chunk text gave the answer: the **chunker was cutting in the wrong places**. It produced chunks that *straddled a boundary* — the trailing "See also" boilerplate of one API entry glued to the opening header of the next, with no explanatory prose at all.

Now look at what such a chunk contains on this particular query. A `cudaFreeMipmappedArray` See-also block lists `cudaMalloc`, `cudaMallocPitch`, `cudaFree`, `cudaMallocArray`. **On a query built from function-name vocabulary, a chunk that is nothing but function names — including the queried function itself — is structurally advantaged over the page that actually explains it.**

So a straddling chunk beats the correct answer under keyword matching *and* under a cross-encoder reading full text. It defeats both. That is a far stronger claim than "fusion has a flaw," it is independent of which retrieval architecture you pick, and it is the same failure class the functional sign-off had already documented as table-of-contents chunks polluting BM25 rankings.

*A chunking defect masquerading as a ranking problem* — found by asking why a prediction only half-worked, and sharpened by checking the number the prediction rested on.

**A caution the plan raised, and where the evidence now points.** The obvious fix — delete See-also blocks at ingestion — is not free. A See-also block for `cudaMalloc` genuinely does point at `cudaMallocPitch`, `cudaMallocHost` and `cudaFree`. It is navigationally useful even though it explains nothing, and removing it may cost recall on queries about the whole family of allocation functions.

The corrected Q1 evidence favours **merging** over filtering: the cross-references earn their place when attached to the explanatory text they belong to, and cause harm only when standing alone as a chunk in their own right. *Filter versus merge versus leave alone is a real trade-off, and this is one data point toward merge — not a settled answer.*

**And the hypothesis A1 was written to test is still open.** Round 1 had documented corroboration bias properly, on 3 of its 9 queries, and nominated two of them — `H100 HBM2e memory capacity` and `shader processor count`, both cases where dense uniquely found the answer and fusion discarded it — as *"good regression-test candidates once re-ranking lands."* Those nominations were lost when the claim was carried into the Round 3 plan and reattached to different queries. Running the test on the cases Round 1 actually identified would close a loop the project opened in July.

---

### A2 — How deep should the candidate pool be?

> **Prediction:** re-ranking quality rises with pool depth (more candidates → better odds the answer is included), then flattens once pool recall saturates.
> **Confirms if:** NDCG@10 rises monotonically with depth, then plateaus.
> **Falsifies if:** gain is flat across depths — meaning the re-ranker is reordering noise.

The most practically useful question in the family. `candidate_pool_size` is defaulted to **100** in the API schema but exercised at **3** in the committed benchmark. Somebody should know which is right.

**What happened:** neither branch. Quality *declined* as the pool deepened — MRR fell monotonically with depth, the opposite of the prediction and not the predicted failure mode either.

**Why — and this is the uncomfortable part.** Return to Part Three. The answer key was built from top-3 pools. Deepen the search to 20 or 50 and the re-ranker starts surfacing chunks from positions 4 through 50. Some are genuinely good. **None of them are in the answer key**, so every single one is scored as an error.

The metric wasn't measuring retrieval quality declining. **It was measuring the answer key running out of road.** Deeper pools weren't finding worse chunks; they were finding chunks the instrument was structurally incapable of crediting.

**Verdict: falsified — but not by the predicted mechanism.** The reason it was falsified was worth more than a confirmation would have been, and it is the finding that reframed the family.

---

### A3 — Does re-ranking help more on some kinds of question than others?

> **Prediction:** cross-encoders add most where lexical overlap is weakest.
> **Confirms if:** Case 2 (dense semantic) and Case 4 (vocab gap) show the largest positive deltas; Case 1 (BM25 lexical) and Case 5 (exact lookup) the smallest or negative.
> **Falsifies if:** deltas are uncorrelated with case type.

The fifteen queries are grouped into the six Round 2 case types. Delta convention: **RRF rank − cross-encoder rank**, so positive means re-ranking improved the labelled chunk's position.

| Case | Qualifying queries | Per-query delta | Mean |
|---|---|---|---|
| 1 · BM25 lexical | Q2 only *(Q1, Q3 have no label)* | Q2: **+2** | +2.0 |
| 2 · Dense semantic | Q6 only | Q6: **0** | 0.0 |
| 3 · RRF hybrid | Q7, Q8, Q9 | −2, −2, 0 | **−1.33** |
| 4 · BM25 failure / vocab gap | **none** *(Q10, Q11 have no label)* | — | *no data* |
| 5 · Dense failure / exact lookup | Q12 only | Q12: **+1** | +1.0 |
| 6 · RRF mixed | Q14, Q15 | 0, −1 | −0.5 |

**What happened:** the prediction pointed the wrong way. The single **largest positive delta in the entire dataset was Q2 (+2) — a Case 1 lexical query**, precisely the category predicted to benefit least. Case 2, predicted to gain most, was flat. Case 5 was positive rather than negative.

**And Case 4 — the other category A3 predicted would benefit most — had no qualifying query at all.** Half of A3's positive prediction could not even be tested, because the two queries in that case type are Q10 and Q11, and Q10 is one of the two queries fusion had already stripped of its correct answer before labelling.

**The honest caveat, which matters as much as the result.** Only 8 of 15 queries survived the label filter. Three of six case types rest on a **single query**. One rests on none. A single query changing sign would overturn the Case 1 and Case 5 readings entirely.

This is directional evidence, not a test. Saying so plainly is the difference between a finding and a story.

**The unpredicted pattern:** Case 3 was the only group where every query moved the same way — a consistent negative-to-zero effect across all three. Neither A3 nor A4 predicted it, which is exactly why it was worth chasing.

---

### A3, Part Two — Chasing the negative result

Q7 and Q8 both had their labelled chunk at **rank 1 after fusion**. As the pool deepened, the cross-encoder pushed it steadily down — Q8 went 1 → 3 → 4, Q7 went 1 → 2 → 3 → 4. On its face, the re-ranker actively making things worse.

**The diagnostic question:** *what exactly is beating the labelled chunk?* Two possible answers with opposite implications — either the competitors are straddling-boilerplate junk (the Q1 defect again, re-ranker is fine) or they're genuinely good chunks the answer key never covered (re-ranker is right, measurement is wrong).

Only reading the full text can settle it.

**Q7 — a query about synchronization performance overhead.** Labelled chunk `ccd7708bb87a3ea7259deb4c` lands at rank 4, cross-encoder score 3.3664.

| Rank | Chunk | CE score | Margin | What it actually contains |
|---|---|---|---|---|
| 1 | `494b8854…` | 5.2069 | +1.84 | Code sample, then: *"the implicit synchronization of child kernels done when a thread block ends is more efficient compared to calling `cudaDeviceSynchronize()` explicitly"* — a direct statement about synchronization performance overhead |
| 2 | `1a561091…` | 4.4437 | +1.08 | §11.7 Grid Synchronization, Cooperative Groups |
| 3 | `08b3e89a…` | 3.5657 | +0.20 | *"`cudaDeviceSynchronize()` blocks the calling CPU thread until all CUDA calls previously issued by the thread are completed"* |

All three are substantively about synchronization. None is a passing mention.

**Q8 — a query about shared-memory bank conflicts.** Primary label `f6f2063c…` at rank 4 (5.1595); secondary label `1fabb00b…` all the way down at rank 9 (3.2932).

| Rank | Chunk | CE score | Margin | What it actually contains |
|---|---|---|---|---|
| 1 | `b2e88edc…` | 6.4416 | +1.28 | Walks through eliminating bank conflicts via `CU_TENSOR_MAP_SWIZZLE_128B` — a concrete avoidance technique, **arguably as good an answer as the labelled chunk** |
| 2 | `07060e4e…` | 6.1836 | +1.02 | Matrix-multiplication shared-memory tutorial; mentions bank conflicts **once, in passing** |
| 3 | `fd5aa331…` | 6.1774 | +1.02 | TMA Swizzle — *"To improve performance and reduce bank conflicts, we can change the shared memory layout by applying a 'swizzle' pattern… to avoid shared memory bank conflicts"* |

**A finding inside the finding.** Chunk `fd5aa331…` had been dismissed in earlier testing as one of BM25's *"tangentially-related vocabulary traps"* — a verdict based on raw BM25 score dominance alone. Reading the full text shows it is a real, different technique for the same problem. **The earlier dismissal did not survive contact with the actual text.** Worth remembering whenever a chunk is written off on score behaviour rather than content.

**Verdict:** five of six competitors across Q7 and Q8 are substantively, specifically on-topic. Only Q8's rank-2 — the tutorial mentioning bank conflicts once — shows the keyword-riding character of the Q1 defect, and it is logged as a single supporting data point, not a pattern.

**So the "dilution" is a label-sparsity artifact, not a re-ranker weakness and not a repeat of the chunking defect.** The cross-encoder was finding legitimately good chunks that a three-per-query answer key had no way to credit. Same root cause as A2, wearing a different disguise.

---

### A4 — Does the cross-encoder actively *harm* exact lookups?

> **Prediction:** the re-ranker should be *skipped*, not merely down-weighted, for identifier lookups.

A3 and A4 make **deliberately opposite predictions on Case 1**. Running both against the same data means whichever survives tells you what the cross-encoder is actually doing — a small piece of experimental design worth pointing at, because it converts a vague question into a decidable one.

**There was prior evidence for A4 before A4 was written.** The functional sign-off (§8.2) had already recorded that for the query "cudaMalloc," the full pipeline and the BM25-only deployment agreed on only two of their top five chunks. The full pipeline made one clear improvement — promoting the host/device semantics section to rank 1 and eliminating a table-of-contents chunk. But it also **dropped the canonical `cudaMalloc(void**, size_t)` signature chunk out of its top five entirely**, a chunk the BM25-only mode had at rank 2. The sign-off's conclusion: *"the cross-encoder appears to favour discursive explanation over terse reference material."*

That is A4's claim, observed independently and before the hypothesis was formalised.

**What happened in A3's data:** on the single point of disagreement, the data sided with A4. Q2's **+2** — the largest positive delta observed — is what A4 predicts and A3 does not.

**Verdict: indirectly supported**, resting on one query plus the prior sign-off observation, held loosely. If it survives a proper sample it argues for **query-dependent routing**: detect identifier lookups at query time and bypass the re-ranker entirely. The plan identifies A1 and A4 together as the sharpest contribution available here — *that rank-based fusion and cross-encoder re-ranking have complementary failure modes on API reference text, and that the standard pipeline order (fuse, then re-rank) prevents the re-ranker from correcting the fusion error, because the error has already removed the candidate.*

---

### A5 — Are the two re-rankers actually interchangeable?

> **Prediction:** the 1% aggregate NDCG gap conceals divergent per-query behaviour.
> **Confirms if:** Spearman correlation between the two orderings falls below ~0.7 on any case type.
> **Falsifies if:** they agree closely everywhere — in which case the gap is noise and the choice is a cost decision, not a quality one.

This experiment took three attempts, and the sequence is instructive.

#### Attempt 1 — confirmed, and worthless

Run over the cached benchmark pool: **three candidates per query**. With three items, Spearman's ρ can only take four values: **{1.0, 0.5, −0.5, −1.0}**. There is no 0.8. There is no 0.65.

Any single adjacent swap — possibly two near-tied scores, not a real disagreement — automatically reads as 0.5, which is *"below 0.7" by construction*. Four of six case types came in below threshold and the hypothesis technically confirmed. **The instrument was too coarse for that to mean anything.**

#### Attempt 2 — live top-100 pool

Re-ran retrieval fresh and re-ranked full 100-candidate pools with both configs. Continuous range restored.

Overall: **mean 0.604, median 0.595, min 0.210 (Q1), max 0.866 (Q2)**. Five of six case types below 0.7.

The comparison against Attempt 1 is the interesting part:

| Case | ρ @ top-3 | ρ @ top-100 | What changed |
|---|---|---|---|
| 1 · BM25 lexical | 0.333 | 0.604 | **Weakens.** The n=3 reading included a full reversal (Q1: ρ = −1.0); at n=100 that's gone, replaced by weak-but-positive 0.210 — still the lowest value in the dataset |
| 2 · Dense semantic | 0.000 | 0.520 | **Weakens substantially.** Apparent anti-correlation (Q5, Q6 at −0.5) becomes moderate positive agreement. The "negative correlation" story does not survive scale-up |
| 3 · RRF hybrid | 0.667 | 0.691 | **Replicates almost exactly.** Most stable case across both resolutions |
| 4 · Vocab gap | 0.750 | 0.543 | **Reverses.** At n=3 this looked like it *falsified*; at full depth it clearly confirms. The small-pool reading was actively misleading, not merely noisy |
| 5 · Exact lookup | 0.500 | 0.564 | **Confirms, consistent direction** |
| 6 · RRF mixed | 1.000 | 0.703 | **Weakens dramatically.** Perfect agreement at n=3; at n=100 the mean clears 0.7 by 0.003 while one of its two queries (Q15: 0.595) sits below |

Two of six case-level *stories* had pointed the wrong way at low resolution. The headline verdict didn't change — but the explanations underneath it did, completely.

#### Attempt 3 — the objection that had to be answered

A sceptic could still object: correlation over 100 items treats a disagreement at positions 95 and 96 as seriously as one at positions 1 and 2. But **only the head affects answer quality** — NDCG@10 doesn't care how the tail is ordered. Perhaps the two models agree where it matters and diverge in an irrelevant tail, which would mean the "they disagree" conclusion is over-claimed and it really is just a cost decision. The ~1% NDCG gap arguably *favours* that reading.

Four head-weighted measurements, all pointing the same way:

| Measure | Result | What it means |
|---|---|---|
| Top-1 agreement | **53%** (8/15) | Barely above a coin flip on which single chunk is best |
| Overlap @ 10 | **5.4 / 10** | Roughly half of each model's top ten is absent from the other's entirely |
| Overlap @ 5 | **2.67 / 5** | Same picture, tighter window |
| Spearman, top-10 only | **0.177** | *Lower* than the full list, not higher |
| RBO (p = 0.9) | **0.556 / 0.590** | Does **not** exceed full-list 0.604 / 0.595 |

**The last row is decisive.** RBO is built to discount the tail geometrically. If agreement really were concentrated at the head with noise below, RBO would have to read noticeably *higher* than plain correlation. It doesn't move — it sits marginally lower, exceeding full-list ρ in only 5 of 15 queries, mean gap −0.048. The tail-only story has no support.

**Verdict: confirmed.** The two re-rankers genuinely disagree about what belongs at the top. The 1% aggregate gap conceals substantial per-query divergence, and choosing between Config A and Config C is **not** purely a cost decision.

**The most interesting detail is also the least certain.** Disagreement isn't uniform — it concentrates exactly where discrimination is hardest:

- **Case 4** (one dominant, unambiguous answer): near-consensus — 100% top-1 match, RBO 0.791
- **Case 5** (exact-lookup queries, where getting the top result right matters most): **worst agreement of all six**, including outright negative head correlation (ρ@10 = −0.239, RBO 0.386)

That is the opposite of what a "they only differ on filler" story predicts. It also rests on **two queries per case**, so it is flagged as a hypothesis for the graded-relevance work, not a result. It is simultaneously the most quotable finding in the family and the least statistically supported — and both halves of that sentence belong in any retelling.

---

### A6 — Could we just skip fusion for identifier queries?

> **Prediction:** for exact-identifier queries, feed the cross-encoder BM25's own top-k and bypass fusion entirely — since fusion has already introduced the corroboration distortion A1 describes.
> **Confirms if:** NDCG@10 on Case 1 and Case 5 is higher for BM25→rerank than RRF→rerank.
> **Falsifies if:** fusion's recall benefit outweighs its precision cost even on identifier queries.

The cheapest conceivable fix for corroboration bias — no new components, just a second pipeline path that skips one box in the diagram. Worth testing before anything elaborate.

**What happened: falsified, cleanly.** The BM25-only pool performed *worse*.

**Why it's interesting.** Dense retrieval had been doing a job nobody credited it with. It wasn't only *contributing* chunks — it was quietly **suppressing distractors** that BM25 ranked highly on keyword overlap alone.

The functional sign-off had documented exactly the distractors in question. Table-of-contents and dot-leader index chunks surfaced at the head of BM25-only rankings across three separate queries; the same report notes that the cross-encoder demotes them once the full pipeline runs. Dense retrieval, reading those chunks as meaning-poor, ranks them low — which drags their fused position down before the re-ranker ever sees them. **Remove dense from the pipeline and those distractors come flooding back into the shortlist.**

Fusion's recall benefit outweighed its precision cost **even on the query type where the precision cost was supposed to be worst**.

A clean negative result that changed how the pipeline is understood — and a caution against removing a component because you've only measured what it adds, never what it filters.

---

### A7 — The one we couldn't run

Config B, `bge-reranker-v2-m3`, has never produced a single number. It runs out of memory at model load on an 8 GB CPU-only host — the same machine that, during these very experiments, dipped to 134 MB free RAM while loading a *smaller* encoder.

This is not a new constraint. The project's own commit history records a UAT regression *"deferred due to memory"* back at Day 6, months before A7. **The hardware ceiling has been shaping what this project can measure for its entire life**, and A7 is simply the most explicit instance.

**It is recorded as blocked, not omitted, and it gets no verdict.** The plan's own instruction was explicit: do not estimate its performance from the other two configurations. Two data points and a plausible-sounding interpolation is not a measurement.

Unblocking it needs a Colab session, a rented GPU hour, or a larger machine. Until then the gap stays visible.

Leaving a hole is more honest than filling it with a guess. *"We didn't have the hardware and we declined to fabricate a number"* is a better answer than one nobody can defend.

---

## Part Five: The thread running through all of it

Seven hypotheses, six with verdicts, one blocked. But the individual results are not the real finding.

**Four separate experiments were constrained by the same underlying cause.** A2's decline with depth, A3's untestable Case 4, A3 Part Two's Q7/Q8 dilution, and A5's coarse first attempt were all consequences of one constraint: **the committed benchmark only ever looked three chunks deep.**

- The answer key was built from three-deep pools → it cannot credit anything found deeper *(A2, A3-2)*
- The two queries that motivated the entire family, Q1 and Q10, both have **zero positive labels** — because the bias under study had already displaced their answers before human review *(A1, A3)*
- One whole case type could not be tested at all, because both its queries are unlabelled *(A3)*
- The cached shortlists were three items long → correlation couldn't take a meaningful value *(A5)*

Every experiment in this family was, to some degree, measuring the limits of its own instrument.

That reframes the whole exercise. It isn't *"here are seven findings about re-ranking."* It's **"here is empirical evidence that our evaluation apparatus was constraining what we could conclude, and here is what we found anyway."** The case for building proper graded relevance judgements now rests on measurement rather than assertion — four independent lines of evidence, converging.

### Five smaller lessons, each earned the hard way

**The chunker was a suspect the whole time — and the obvious next accusation was wrong.** A1's straddling-chunk diagnosis explained one failure. The natural move was to blame it for Q7/Q8 too. Reading the actual text showed that was wrong: five of six competitors were genuinely relevant. The discipline of checking rather than pattern-matching kept a single-instance defect from being inflated into a systemic one.

**Verdicts survived while their explanations didn't.** A3's headline was falsified but its case-level story shifted between resolutions. A5 confirmed at n=3 and again at n=100 — yet two of six case readings had pointed the *wrong way* in the coarse version, one of them appearing to falsify outright. *The direction of a finding and the confidence you should place in it are separate questions.*

**A chunk dismissed on score behaviour deserves a second look at its text.** `fd5aa331…` was written off as a "vocabulary trap" on BM25 score dominance alone. It was a real technique for the same problem. Scores describe how a retriever behaved; only the text says what the chunk is.

**Bind labels to referents explicitly, or someone will guess wrong.** The plan's A5 entry is titled *"Cohere Rerank v3 and ms-marco disagree…"* and then states *"Config A (0.5333) and Config C (0.5280)."* It never says which model is which config — and the two orderings invite the reader to fuse them into a false mapping. That misreading was made, and it would have inverted the interpretation of every A5 number had it not been caught against the codebase. The benchmark runner is unambiguous: **Config A is ms-marco, Config C is Cohere.** The fix is not to blame the reader; it is to treat the code that produces the figures as the source of truth, and to bind names to referents in the same sentence.

**A number in prose is not a result either.** The plan asserted a BM25 score of 33.4 for Q1's target. It belonged to Q8. Every document built on the plan inherited it, including this one — while `uat_superiority_cases_raw.json`, which disproves it in one line, had been committed since 13 July. The lesson at `0149ca4` was "here is the script, here is the output, run it yourself." This extends it: **figures cited in a planning document need the same provenance discipline as figures in a benchmark**, because a plan is exactly the kind of authoritative-looking artifact nobody thinks to check. Mitigating detail, in fairness: it was caught by the practice it argues for.

**A result that isn't saved is indistinguishable from one that was invented.** The top-100 run computed its numbers in a temporary script and persisted nothing. A later session searched the repository, found no trace, and — correctly — refused to proceed on numbers it couldn't verify. Re-running from scratch reproduced the original figures **to three decimal places**, vindicating them entirely.

Two extra runs, an hour of compute, and a genuine scare about data integrity, all avoidable by writing one JSON file. Every artifact and script is now committed. **In a research context, "I ran it and it said X" is not a result. "Here is the script, here is the output, run it yourself" is.**

---

## Part Six: If someone asks

**"What was Family A about?"**
Whether the re-ranking step in our hybrid search pipeline actually does what the architecture assumes. Earlier UAT rounds had compared BM25 against dense against RRF; the cross-encoder appeared in the evaluation only as a footnote on two queries. Round 3 is seven pre-registered hypotheses about that component, each with explicit confirm and falsify conditions written before running.

**"Explain the pipeline in one breath."**
Question goes to BM25 and to a dense vector search in parallel. Their two ranked lists are fused by reciprocal rank fusion, which deliberately ignores confidence scores and looks only at rank position. That produces a candidate pool, and a cross-encoder re-ranks the pool by scoring each question-chunk pair directly. The critical structural fact is that the pool sits *between* fusion and re-ranking — so anything fusion discards is invisible to the re-ranker.

**"What did you find?"**
Headline: two re-rankers with near-identical average NDCG disagree substantially about what belongs at the top — 53% top-1 agreement, half of each model's top ten missing from the other's — so choosing between them isn't just a cost decision. But the more valuable finding was structural. Four separate experiments were constrained by the same cause: our answer key was built from the top three results of the system being evaluated, so it couldn't credit anything found deeper. We were measuring our measuring instrument.

**"Give me the single clearest example of that."**
The two queries the plan built its case on — Q1 and Q10 — both ended up with zero positive relevance labels, because in each case fusion had displaced the correct chunk out of the top three before any human reviewed the pool. The behaviour under study had erased its own evidence from the instrument used to study it.

**"Did you get anything wrong?"**
Yes, twice, and the second one is the more interesting. A1's write-up quoted a BM25 score that belonged to a different query. When I corrected that, I built a replacement explanation from the data in front of me and concluded the fusion mechanism wasn't involved at all — without checking the evaluation page, which had already measured the whole picture, including the dense rank of 75 that actually explains the displacement. So the correction repeated the error it was correcting: stopping at the first coherent story. The measurements were never affected, and the chunking finding came out considerably stronger — a chunk that defeats both keyword matching and a cross-encoder is a much better argument than one that only defeats fusion. But the pattern is the same one this whole round was about, and I'd rather show it than tidy it away.

**"What would you do differently?"**
Build relevance judgements independently of the system being evaluated — pooled across configurations, graded rather than binary. That circularity constrained four experiments. And persist every intermediate artifact, because an unsaved result is unverifiable, which in practice means worthless.

**"What's the most interesting thing you found?"**
That the two re-rankers agree most on easy queries and least on exact-identifier lookups — precisely where getting the top result right matters most. If it holds under a proper sample it's an argument for routing between them by query type rather than picking one. It rests on two queries, so it's a hypothesis, not a result.

**"Did anything you expected to be true turn out false?"**
Several, and the failures were more useful than the confirmations. A6 predicted that skipping fusion would help identifier queries; it hurt, revealing that dense retrieval had been silently filtering table-of-contents distractors nobody had credited it for. A3 predicted re-ranking would help semantic queries most; the largest gain came from a lexical one.

**"Why is one experiment blank?"**
Config B needs more memory than the machine has — it OOMs at model load on an 8 GB CPU-only host, the same ceiling that forced a UAT regression to be deferred back at Day 6. We recorded it as blocked rather than estimating from the other two, because two points and an interpolation isn't a measurement.

---

*Family A status: six hypotheses resolved, one hardware-blocked. All artifacts and analysis scripts committed and reproducible.*

*Sources: Retrieval Hypothesis Test Plan (UAT Round 3, 18 Aug 2026); Functional Test Sign-Off FTS-NVIR-2026-001 v2.1; `docs/uat/uat_superiority_cases_executed.md`; committed A5 artifacts and analysis scripts.*
