# ENH-11 Recovery Grades — Chat Transcript Extraction

**Purpose:** 224 judgements were reasoned through in a Claude chat session but never entered into `judge_enh11.py`. Only 101 of 325 are on disk (`evaluation/enh11_qrels.json`, commit `e500314`) — Q1 (24), Q2 (23), Q3 (24), Q4 (23), Q5 partial (7 of 23).

This document is a best-effort extraction of the remaining ~224 grades from that transcript, in judging order, by query.

**⚠️ Before entering any of this: validate against `enh11_pool.json`.** This was hand-extracted from a long conversation and may contain transcription errors — a mis-copied chunk_id, an off-by-one, or a duplicate. For each entry below:
1. Confirm the `chunk_id` prefix actually exists in that query's pool in `enh11_pool.json`
2. If a chunk_id doesn't match anything in the query's pool, **skip it and flag it** rather than guessing or forcing a match
3. Cross-check the final per-query count against `enh11_pool.json`'s stated pool size for that query — if the recovered count is short, the missing chunk(s) need re-judging fresh, not invented

**How to use:** the entries below give `chunk_id_prefix → grade`. Chunk IDs are truncated to the first 8 hex characters as they appeared on-screen — `judge_enh11.py` should match on prefix against the full IDs in `enh11_pool.json`. If a prefix matches more than one chunk in a query's pool, flag it for manual disambiguation rather than guessing.

---

## Q5 — remaining chunks (pool size 23, 7 already on disk)

| chunk_id prefix | grade |
|---|---|
| 92be6ceb | 0 |
| 4af51a0c | 0 |
| 78600f19 | 1 |
| e0d64832 | 1 |
| 69c2f815 | 1 |
| a8c35fe2 | 0 |
| 877ff33f | 0 |
| fca91c87 | 0 |
| 61323053 | 0 |
| 490bb9d3 | 1 |
| ab506142 | 0 |
| 66de81a3 | 3 |
| 2613f4d3 | 3 |
| 8fa1dd72 | 2 |
| f9713c3e | 1 |
| 6083e1f4 | 1 |

*(If pool size 23 minus 7 on-disk = 16 expected, and 16 rows are listed above — count matches.)*

---

## Q6 — full query (pool size 20)

| chunk_id prefix | grade |
|---|---|
| 455e5ba8 | 3 |
| c30bae85 | 0 |
| 43142ca7 | 0 |
| df5d7d80 | 3 |
| eecd9b0d | 3 |
| 48efb723 | 0 |
| 01146335 | 0 |
| 47cdfe34 | 0 |
| 98db470e | 3 |
| 3e21f636 | 3 |
| a49007ac | 0 |
| e96b85bb | 0 |
| 3805d3ce | 3 |
| a31fa76b | 2 |
| d5d95f33 | 1 |
| a737c10d | 0 |
| fee8e256 | 0 |
| 5dc04244 | 2 |
| 3c9b5ecf | 3 |
| 1c12de6e | 2 |

*(20 rows — matches pool size.)*

---

## Q7 — full query (pool size 22)

| chunk_id prefix | grade |
|---|---|
| 1a561091 | 2 |
| 35cbea8c | 1 |
| 04aa0e0f | 1 |
| a138e58c | 1 |
| ca63ac8d | 1 |
| 494b8854 | 3 |
| 08b3e89a | 3 |
| d5188d35 | 2 |
| 330acac2 | 3 |
| db84382d | 0 |
| a8b2f5f6 | 0 |
| 6bf18455 | 1 |
| 1dc3cddd | 1 |
| f51fff5a | 2 |
| 6c42edd6 | 2 |
| ccd7708b | 3 |
| 8597895a | 3 |
| c250c2f1 | 2 |
| dae68367 | 1 |
| 2651954b | 1 |
| 97b0b58e | 0 |

*(21 rows recovered — pool says 22. One chunk short; likely the query's first chunk `cc01e5e6` which appears as the opening confirmation of Q8 rather than inside Q7's own block. Add manually: `cc01e5e6 → 1` if it belongs to Q7, or verify against pool — see note below.)*

---

## Q8 — full query (pool size 14)

| chunk_id prefix | grade |
|---|---|
| cc01e5e6 | 1 |
| f51fff5a | 3 |
| fd5aa331 | 3 |
| 18c3a318 | 2 |
| 4731ad59 | 3 |
| 4949c564 | 0 |
| 1fabb00b | 3 |
| 6996517a | 2 |
| 4e0a7be7 | 2 |
| 5a6c6b2e | 2 |
| b2e88edc | 3 |
| e3add51e | 1 |
| f6f2063c | 3 |

*(13 rows — pool says 14. `f51fff5a` appears twice across Q7/Q8 in this transcript, once as `f51fff5a→2` under Q7 and once as `f51fff5a→3` under Q8 — these are almost certainly two different chunks that happen to share the same 8-char prefix; disambiguate by full ID against the pool before entering either.)*

---

## Q9 — full query (pool size 15)

| chunk_id prefix | grade |
|---|---|
| 052db955 | 0 |
| 1c016f02 | 3 |
| 60c05157 | 1 |
| fd5aa331 | 0 |
| 18c3a318 | — *(not in Q9; ignore — belongs to Q8, listed above)* |
| b5ca753b | 2 |
| 3c8573a7 | 3 |
| 879c3af2 | 3 |
| 91945270 | 3 |
| dca95bd9 | 3 |
| 31e19331 | 2 |
| 348eab59 | 2 |
| 17984bf0 | 0 |
| 4949c564 | 3 |
| 48ba7e1f | 3 |

*(14 valid rows after removing the stray line — pool says 15. One short; likely `eecd9b0d` which opens Q10's first confirmation line rather than closing inside Q9 — see Q10 note.)*

---

## Q10 — full query (pool size 19)

| chunk_id prefix | grade |
|---|---|
| eecd9b0d | 0 |
| db1c0602 | 3 |
| 92dfc007 | 3 |
| 74c8525c | 0 |
| 7d9f5945 | 3 |
| 8605ef31 | 1 |
| 091e25e1 | 0 |
| b5ca753b | 1 |
| 31e19331 | 2 |
| 2f30edbe | 1 |
| f2730f1e | 3 |
| 4f75e69f | 0 |
| 5f6edc00 | 0 |
| 60c05157 | 0 |
| e952bc06 | 2 |
| 23082588 | 0 |
| 15f22f2d | 0 |
| 6083e1f4 | 2 |

*(18 rows — pool says 19, one short. Likely the query's final confirmation chunk `84d5aa0f` which appears as the opening confirmation of Q11 — see Q11.)*

---

## Q11 — full query (pool size 13)

| chunk_id prefix | grade |
|---|---|
| 84d5aa0f | 2 |
| 84905b60 | 2 |
| d5818ab3 | 1 |
| 60395666 | 3 |
| 35dfc678 | 1 |
| 9e7d5603 | 0 |
| db1c0602 | 3 |
| a1f985ea | 1 |
| feec243a | 0 |
| 9d1f3a9e | 3 |
| d827aaac | 0 |
| 74c8525c | 2 |
| 23082588 | 2 |

*(13 rows — matches pool size.)*

---

## Q12 — full query (pool size 24)

| chunk_id prefix | grade |
|---|---|
| 0a880bdf | 2 |
| b621d87b | 0 |
| 02bb6a20 | 3 |
| 68ddca9d | 1 |
| 34006d7f | 0 |
| 9e5aeb95 | 0 |
| 001a3f1d | 3 |
| 2d6afa49 | 0 |
| 281a9cb4 | 0 |
| 19cfdd35 | 0 |
| 69f1777b | 0 |
| 2ca0b687 | 0 |
| bdfe11a2 | 0 |
| f5cde567 | 2 |
| 7168ba67 | 2 |
| 871a5345 | 0 |
| d16ef105 | 0 |
| 854c58b3 | 0 |
| 7cea617b | 3 |
| eec8e219 | 0 |
| 379a1c65 | 0 |
| 494b8854 | 2 |
| b4658cf3 | 0 |
| ab1e570a | 0 |

*(24 rows — matches pool size.)*

---

## Q13 — full query (pool size 19)

| chunk_id prefix | grade |
|---|---|
| b17f4430 | 0 |
| 1153e0ef | 3 |
| 90e4cc61 | 0 |
| 5f3c6f39 | 0 |
| 5379a2db | 0 |
| d120d5f4 | 0 |
| 8e43b6ce | 1 |
| 93719765 | 0 |
| 9e66efd2 | 0 |
| a612172b | 1 |
| 1191a052 | 0 |
| fe95ffdb | 0 |
| 4bb29425 | 0 |
| da1c5672 | 3 |
| 71f15d77 | 3 |
| 1a57340e | 0 |
| 36037247 | 2 |
| 5a3771e9 | 0 |
| b71921df | 2 |

*(19 rows — matches pool size.)*

---

## Q14 — full query (pool size 19)

| chunk_id prefix | grade |
|---|---|
| df535745 | 0 |
| 93151749 | 2 |
| f9c6c072 | 0 |
| c9f1f71b | 1 |
| c52d9f12 | 3 |
| 225d92e7 | 3 |
| 83ee70cd | 2 |
| c9c4861f | 3 |
| 6a54459b | 0 |
| 189299cc | 1 |
| 5e385631 | 3 |
| ae09deef | 0 |
| cc123568 | 2 |
| f10fc7d7 | 3 |
| b15c7064 | 2 |
| c5e45561 | 1 |
| 0e8d9704 | 0 |
| 3a2215c9 | 2 |
| eb2ca045 | 1 |

*(19 rows — matches pool size.)*

---

## Q15 — full query (pool size 17)

| chunk_id prefix | grade |
|---|---|
| 937f086f | 3 |
| a6910bcc | 0 |
| d827aaac | 0 |
| a6910bcc | 0 |
| 60395666 | 3 |
| 84905b60 | 2 |
| 35dfc678 | 1 |
| 9e7d5603 | 0 |
| feec243a | 0 |
| d827aaac | 0 |
| 9d1f3a9e | 3 |
| a581c578 | 3 |
| 6bf18455 | 2 |
| 23082588 | 1 |
| db1c0602 | 3 |
| 9d1f3a9e | 1 |
| 35dfc678 | 1 |
| e3add51e | 3 |
| a1f985ea | 3 |

*(Warning: this query showed several chunks that appeared to reuse chunk_ids already seen in Q10/Q11's pool, e.g. `d827aaac`, `9d1f3a9e`, `35dfc678`, `84905b60`. Given the shared-corpus nature of this pool, the same underlying chunk can legitimately appear in multiple queries' pools — but the repeated values above within Q15 itself suggest a transcription duplication error. Treat this section with extra caution: re-derive Q15 from the transcript directly if possible, since it's the least reliable section of this extraction.)*

---

## R1-Q7 — full query (pool size 26)

| chunk_id prefix | grade |
|---|---|
| 875a0e2c | 3 |
| b4d92cd0 | 0 |
| 9e370910 | 0 |
| d30e9742 | 0 |
| 6083e1f4 | 0 |
| c0a6fa3a | 0 |
| 8ddd5e09 | 0 |
| b1b83570 | 0 |
| a8c35fe2 | 0 |
| b25551d0 | 0 |
| aed2815d | 2 |
| d43ec568 | 0 |
| ab1f0660 | 3 |
| 3ce249ad | 0 |
| 143be4b8 | 0 |
| 1adbb45f | 0 |
| f4f414ab | 0 |
| 35f3ce1b | 3 |
| 35b73f33 | 3 |
| b9a40bfc | 0 |
| 9295d323 | 0 |
| fc217634 | 0 |
| 4c6f0721 | 0 |
| f2730f1e | 0 |
| f886d5a0 | 0 |
| 2dddc971 | 0 |

**These two rows are the ones that matter most for the project's findings** — both were graded blind, independently confirming the fusion-bias measurement:
- `35b73f33` → **3** (the canonical correct target chunk — "An SM consists of: 128 CUDA cores")
- `b1b83570` → **0** (the corroborated-but-wrong GPU-Metrics warp-occupancy chunk that wins fusion)

*(26 rows — matches pool size.)*

---

## Consistency re-check (22 items, all `pass: 2`)

These re-grade chunks already judged once above (query re-stated per item). Enter as `pass: 2` records if the tool supports specifying pass explicitly; otherwise these are what the tool's own `--consistency` re-check step would regenerate automatically once the primary 325 are in — **it may be simpler to let `judge_enh11.py` run its own consistency sampling fresh after all primary judgements are entered, rather than manually replaying these 22.**

| # | Query | chunk_id prefix | grade (2nd pass) |
|---|---|---|---|
| 1 | Q14 | b15c7064 | 2 |
| 2 | Q14 | (pipeline chunk, ~236) | 0 |
| 3 | Q6 | 43142ca7 | 0 |
| 4 | Q10 | 379a1c65 *(from Q12 context — verify)* | 3 |
| 5 | Q10 | 7d9f5945 | 3 |
| 6 | Q12 | 7cea617b | 3 |
| 7 | Q8 | f6f2063c | 1 |
| 8 | Q6 | 6083e1f4 | 0 |
| 9 | Q7 | fee8e256 *(verify — may be Q6)* | 3 |
| 10 | Q7 | 330acac2 | 2 |
| 11 | Q13 | 36037247 | 2 |
| 12 | Q14 | cc123568 | 3 |
| 13 | Q6 | eecd9b0d | 3 |
| 14 | Q6 | 3c9b5ecf | 0 |
| 15 | Q10 | 2f30edbe | 1 |
| 16 | Q9 | 31e19331 | 0 |
| 17 | R1-Q7 | fc217634 | 2 |
| 18 | Q10 | 84d5aa0f | 3 |
| 19 | Q6 | 3805d3ce | 0 |
| 20 | Q11 | 74c8525c | 0 |

*(Only 20 clearly recovered of 22 — this section is the least reliable in the extraction. Recommend: skip manual re-entry of this section entirely and let the tool generate a fresh consistency sample once all 325 primary judgements are on disk. A freshly-drawn sample is methodologically cleaner than replaying a hand-transcribed one anyway.)*

---

## Summary and recommended entry order

| Query | Rows recovered | Pool size | Status |
|---|---|---|---|
| Q5 (remainder) | 16 | 16 needed | Clean |
| Q6 | 20 | 20 | Clean |
| Q7 | 21 | 22 | 1 short — verify |
| Q8 | 13 | 14 | 1 short — verify |
| Q9 | 14 | 15 | 1 short — verify |
| Q10 | 18 | 19 | 1 short — verify |
| Q11 | 13 | 13 | Clean |
| Q12 | 24 | 24 | Clean |
| Q13 | 19 | 19 | Clean |
| Q14 | 19 | 19 | Clean |
| Q15 | ~17 (duplicates present) | 17 | **Unreliable — re-derive or re-judge** |
| R1-Q7 | 26 | 26 | Clean, includes both flagged chunks |
| Consistency | 20 of 22 | 22 | Skip — regenerate fresh instead |

**Recommended approach for Claude Code:**
1. Load `enh11_pool.json` as ground truth for what chunk_ids exist per query
2. For each clean query above, cross-check every row's chunk_id prefix resolves to exactly one chunk in that query's pool, then write the judgement directly into `enh11_qrels.json`'s structure (bypassing the interactive terminal, since this is a data-entry/recovery operation, not fresh judgement) — commit as its own clearly-labelled commit, e.g. `test(uat): ENH-11 recovered judgements Q5–Q14, R1-Q7 (from chat transcript, see enh11_recovery_grades.md)`
3. For the five "1 short" queries (Q7, Q8, Q9, Q10), identify the specific missing chunk_id by diffing the pool against what's recovered, and have the person re-judge just that single chunk fresh through the normal tool interface
4. For Q15, re-derive from the transcript directly rather than trusting this extraction, or re-judge the whole query fresh (17 chunks, ~10 minutes)
5. Once all 325 are in, run `judge_enh11.py`'s own consistency-check step fresh rather than replaying the 22 above
6. Commit the qrels file, update `docs/uat/enh11_protocol.md` §7 to reflect true completion only once verified counts match `enh11_pool.json` exactly per query

**Do not treat this document as ground truth for anything before it has been cross-validated against `enh11_pool.json`.** It is a recovery aid, not a verified record.
