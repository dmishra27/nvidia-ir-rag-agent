# Prompt Variant Comparison — QA Agent Generation Step

### nvidia-ir-rag-agent · criterion 2 (RAG evaluation)

**Written before running anything**, per this project's own standard (`0149ca4`): the design, the confirm/falsify conditions, and the protocol are fixed here first; the results section below is filled in only after the script has run and its output is committed.

---

## 1. Why this exists

`agents/qa_agent.py`'s `make_generate_node()` had exactly one hardcoded prompt, one model, evaluated by both `run_day9_ragas.py` and `run_day9_citation_judge.py` — they differ in *metric*, not in generation *approach*. The rubric's criterion 2 wants multiple RAG approaches evaluated with the best one used. This closes that gap.

## 2. The two variants

Both live in `agents/qa_agent.py`'s `PROMPT_VARIANTS` dict, selected via `make_generate_node(prompt_variant=...)`.

- **`baseline`** — the original instruction, byte-for-byte unchanged from what ran through Day 9. Kept exactly as shipped so the comparison is against the real thing, not a reworded stand-in.
- **`cite_verify`** — a genuinely different generation strategy, not a reworded instruction: before making the `answer_with_citations` tool call, the model is asked to draft its claims, then re-check each one against its candidate chunk_id(s) — confirming the passage actually *states* what the claim asserts, not merely that it shares a topic — and to drop or re-cite anything that doesn't hold up. `baseline` asks for citation and answer in one undifferentiated pass; `cite_verify` inserts an explicit verification step ahead of the same tool call.

## 3. Hypothesis

**Claim.** `cite_verify` raises citation accuracy and RAGAS faithfulness relative to `baseline`, at some cost to answer_relevancy and/or response length — an asked-for verification pass tends to produce more hedged, narrower answers, not necessarily more directly responsive ones.

**Confirms if** `cite_verify` shows higher citation accuracy or faithfulness than `baseline` on a majority of the 10 queries, even if answer_relevancy drops.

**Falsifies if** `cite_verify` shows no improvement in citation accuracy or faithfulness over `baseline` — in which case the verification instruction bought nothing and `baseline` should ship as the default.

**A third, explicitly allowed outcome:** `cite_verify` could improve on *every* axis, or *baseline* could win outright. Either is reportable; the standard here (per this project's own evaluation work) is to report whichever happened, not to engineer a win for the new variant.

## 4. Protocol

1. Reuse the same 10 saved Config A reranked contexts (`evaluation/day9_config_a_contexts.json`, first 10 of its 15 queries) that `run_day9_ragas.py` and `run_day9_citation_judge.py` already scored under `baseline` — this makes the comparison apples-to-apples against the existing historical numbers, and needs no live retrieval (per Part A's no-live-retrieval constraint).
2. For each query, call `make_generate_node(prompt_variant=v)()` for `v` in `{baseline, cite_verify}` — two live Claude Sonnet calls per query, 20 total.
3. Score both variants' full 10-query sets with the existing RAGAS suite (`faithfulness`, `answer_relevancy` — same metric selection Day 9 used, no `ground_truth` available) and the existing citation judge (`evaluation/citation_judge.py`), unmodified.
4. Persist: both variants' QA states, per-query RAGAS scores (not just the mean), and every citation judgment to a single committed JSON file, before any comparison is written up.
5. Log two separate MLflow runs (`prompt_variant=baseline`, `prompt_variant=cite_verify`) under a new `prompt_variant_comparison` experiment.
6. Report per-query numbers for both variants side by side, state the trade-off explicitly (which axis moved which way), and name the variant that ships as `DEFAULT_PROMPT_VARIANT`.

**Small-n caveat, stated up front:** this is 10 queries, one run each, no repeated sampling. A query or two flipping sign would change which variant looks better on any single metric. This is directional evidence in the shape of this project's other single-pass evaluations (Day 9's RAGAS/citation-judge runs, A3's 8-query reading), not a statistically powered test.

---

## 5. Results

*(Filled in after `run_prompt_variant_comparison.py` has run and `evaluation/prompt_variant_comparison.json` is committed.)*
