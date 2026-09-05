"""Hypothesis B7 — a binary single-retriever-preference rule recovers severe
RRF displacement without B4's cross-scale cost.

`docs/uat/round3_hypothesis_test_plan.md` §4:

  B7  When one retriever ranks the target in its top N = 3 and the other
      does not return it anywhere in the retrieved pool (M = 100),
      substitute the single retriever's rank for the fused rank. Predicted
      to fire on R1-Q7 (fused 10) and Q5 (fused 17) only, and to regress
      no target currently at fused rank 1.

Thresholds N = 3 and M = 100 were committed on 4 September 2026 (`4a15435`)
before any result existed; the reasoning is in the plan entry under
"Thresholds — committed now". They are LOCKED. This script does not expose
them as tunable parameters. The exploratory `M in {5, 10, 20, 50}` table
(protocol step 4) is reported strictly outside the pass/fail read, to make
visible the sensitivity the pre-registered M forecloses.

Scope: a pure offline re-scoring of `evaluation/b3_b4_fusion_eval.json`
(B3 + B4, committed `0b13307`). No new retrieval, no encoder load. The
metric is target-chunk rank, per B3 and D-QR: `run_day9_relevance_labelling.py`'s
qrels are circular (Round 3 A2/A3) and no retriever-independent graded
labels exist yet (ENH-11), so NDCG is unevaluable on this corpus and is
not reported.

The rule (protocol step 2):

    if min(bm25_rank, dense_rank) <= N  and  the target is absent from the
    other retriever's top-M pool:
        b7_rank = min(bm25_rank, dense_rank)     # the finding retriever's own rank
    else:
        b7_rank = fused_rank                      # RRF unchanged

With M = 100 = pool depth, "absent from the other retriever's top-M" is
exactly "absent from that retriever's persisted pool" — B3 retrieved
top-100 per retriever, so a rank of `null` in the eval file is the only
way the second clause is satisfied.

Writes `evaluation/b7_single_retriever_rule.json`. No analysis here — see
`docs/uat/round3_b7_findings.md`.
"""

from __future__ import annotations

# Consistency with the other run_*.py harnesses; this script imports no
# third-party package but the interpreter guard is cheap and uniform.
from utils.require_python import require_python

require_python()

import json
from pathlib import Path

IN_PATH = Path("evaluation/b3_b4_fusion_eval.json")
OUT_PATH = Path("evaluation/b7_single_retriever_rule.json")

# Locked 4 September 2026 (4a15435), pre-registered. Do not parameterise.
N = 3    # finding-retriever rank: "top of list"
M = 100  # the other retriever's pool depth; absence within it is the trigger

# Exploratory only (protocol step 4) — never enters the pass/fail read.
EXPLORATORY_M = [5, 10, 20, 50]


def _min_rank(bm25_rank: int | None, dense_rank: int | None) -> int | None:
    present = [r for r in (bm25_rank, dense_rank) if r is not None]
    return min(present) if present else None


def _apply_rule(rec: dict, n: int, m: int) -> dict:
    """Return {fired, b7_rank, delta, finding_retriever, other_retriever_rank}.

    `m` is the other-retriever pool depth: the rule's second clause holds
    when the target's rank in the other retriever's list is None (absent)
    or strictly greater than `m`. B3 retrieved top-100, so for m = 100 only
    absence can satisfy it; smaller m values are the exploratory sweep.
    """
    bm25_rank = rec["bm25_rank"]
    dense_rank = rec["dense_rank"]
    fused_rank = rec["fused_rank"]["rrf"]

    mn = _min_rank(bm25_rank, dense_rank)
    if mn is None:
        return {
            "fired": False,
            "b7_rank": fused_rank,
            "delta": 0,
            "finding_retriever": None,
            "other_retriever_rank": None,
        }

    if bm25_rank is not None and (dense_rank is None or bm25_rank <= dense_rank):
        finding, other_rank = "bm25", dense_rank
    else:
        finding, other_rank = "dense", bm25_rank

    other_absent_or_beyond_m = other_rank is None or other_rank > m
    fired = mn <= n and other_absent_or_beyond_m

    b7_rank = mn if fired else fused_rank
    return {
        "fired": fired,
        "b7_rank": b7_rank,
        "delta": fused_rank - b7_rank,  # positive = improvement
        "finding_retriever": finding,
        "other_retriever_rank": other_rank,
    }


def main() -> None:
    src = json.loads(IN_PATH.read_text(encoding="utf-8"))
    records = src["per_query"]

    if src["pool_size"] != M:
        raise SystemExit(
            f"B7 pins M = pool depth = 100; eval file pool_size is {src['pool_size']}. "
            "A mismatch would silently change the rule's trigger — stopping."
        )

    per_query: list[dict] = []
    fired_ids: list[str] = []
    regressions: list[str] = []
    fused_rank1_ids: list[str] = []

    for rec in records:
        qid = rec["query_id"]
        fused_rank = rec["fused_rank"]["rrf"]
        res = _apply_rule(rec, N, M)

        if fused_rank == 1:
            fused_rank1_ids.append(qid)
        if res["fired"]:
            fired_ids.append(qid)
            if fused_rank == 1 and res["b7_rank"] > 1:
                regressions.append(qid)
        # a fused-rank-1 target can only regress if the rule fires on it;
        # the rule never touches fused_rank otherwise, so no other path
        # can move a rank-1 target.

        row = {
            "query_id": qid,
            "query": rec["query"],
            "case_label": rec["case_label"],
            "supplementary": rec.get("supplementary", False),
            "target_chunk": rec["target_chunk"],
            "bm25_rank": rec["bm25_rank"],
            "dense_rank": rec["dense_rank"],
            "corroboration_class": rec["corroboration_class"],
            "finding_retriever": res["finding_retriever"],
            "other_retriever_rank": res["other_retriever_rank"],
            "fused_rank": fused_rank,
            "b7_fired": res["fired"],
            "b7_rank": res["b7_rank"],
            "delta": res["delta"],
            "b4_minmax_rank": rec["fused_rank"]["minmax_combsum"],
            "b4_zscore_rank": rec["fused_rank"]["zscore_combsum"],
        }
        per_query.append(row)
        print(
            f"{qid:6s} {rec['case_label']:32s} "
            f"bm25={_fmt(rec['bm25_rank'])} dense={_fmt(rec['dense_rank'])} | "
            f"rrf={fused_rank:>2} -> b7={res['b7_rank']:>2} (d={res['delta']:+d}) "
            f"{'FIRED' if res['fired'] else ''}"
        )

    # --- exploratory M sweep (protocol step 4) — outside the pass/fail read ---
    exploratory: list[dict] = []
    for m in EXPLORATORY_M:
        m_fired: list[dict] = []
        m_regressions: list[str] = []
        for rec in records:
            res = _apply_rule(rec, N, m)
            if res["fired"]:
                m_fired.append(
                    {
                        "query_id": rec["query_id"],
                        "fused_rank": rec["fused_rank"]["rrf"],
                        "b7_rank": res["b7_rank"],
                        "delta": res["delta"],
                    }
                )
                if rec["fused_rank"]["rrf"] == 1 and res["b7_rank"] > 1:
                    m_regressions.append(rec["query_id"])
        exploratory.append(
            {
                "M": m,
                "fires_on": [r["query_id"] for r in m_fired],
                "per_query": m_fired,
                "fused_rank1_regressions": m_regressions,
            }
        )

    predictions = {
        "fires_on_R1-Q7_and_Q5_only": {
            "predicted": ["R1-Q7", "Q5"],
            "observed": sorted(fired_ids),
            "match": sorted(fired_ids) == sorted(["R1-Q7", "Q5"]),
        },
        "R1-Q7_reaches_fused_rank_1": {
            "predicted_b7_rank": 1,
            "observed_b7_rank": next(
                r["b7_rank"] for r in per_query if r["query_id"] == "R1-Q7"
            ),
            "b4_minmax_rank_for_contrast": next(
                r["b4_minmax_rank"] for r in per_query if r["query_id"] == "R1-Q7"
            ),
            "match": next(
                r["b7_rank"] for r in per_query if r["query_id"] == "R1-Q7"
            ) == 1,
        },
    }

    falsify = {
        "any_fused_rank1_target_regresses": {
            "fused_rank1_queries": fused_rank1_ids,
            "count": len(fused_rank1_ids),
            "regressions": regressions,
            "triggered": bool(regressions),
        },
        "R1-Q7_or_Q5_fails_to_improve": {
            "R1-Q7_delta": next(r["delta"] for r in per_query if r["query_id"] == "R1-Q7"),
            "Q5_delta": next(r["delta"] for r in per_query if r["query_id"] == "Q5"),
            "triggered": (
                next(r["delta"] for r in per_query if r["query_id"] == "R1-Q7") <= 0
                or next(r["delta"] for r in per_query if r["query_id"] == "Q5") <= 0
            ),
        },
        "promoted_chunk_is_not_the_target": {
            "note": (
                "Unevaluable from this file alone. b3_b4_fusion_eval.json stores only the "
                "target chunk's own rank in each retriever list (from _rank_of(target, ...)), "
                "not the full ranked chunk-id lists, so b7_rank = min(bm25_rank, dense_rank) "
                "places the FIXED target at that rank by construction. Independent confirmation "
                "for R1-Q7: docs/uat/round3_dqr_findings.md verified dense ranks 35b73f33... "
                "('128 CUDA cores') first for 'shader processor count'. For Q5 the target "
                "8f2dbd94... sits at dense rank 2 per the same anchor set."
            ),
            "triggered": False,
        },
    }

    precision_caveat = (
        "B3 measured displacement of KNOWN-CORRECT targets only. This re-scoring inherits that "
        "design limit: it says nothing about how often a lone top-3 placement is WRONG. B7 as a "
        "runtime rule would promote the finding retriever's rank-1 chunk whether or not it is "
        "correct, converting a recall problem into a precision problem this data cannot observe. "
        "Quantifying the false-positive rate needs retriever-independent graded labels (ENH-11)."
    )

    out = {
        "hypothesis": "B7",
        "plan": "docs/uat/round3_hypothesis_test_plan.md §4",
        "source": str(IN_PATH),
        "source_commit_note": "b3_b4_fusion_eval.json committed 0b13307 (before B3/B4 analysis)",
        "method": "offline re-scoring of the B3/B4 persisted table; no retrieval, no encoder",
        "metric": "target-chunk rank (per B3 and D-QR); NDCG unevaluable pre-ENH-11",
        "thresholds": {
            "N": N,
            "M": M,
            "locked_commit": "4a15435 (4 September 2026), pre-registered",
            "rule": (
                "if min(bm25_rank, dense_rank) <= N and target absent from the other "
                "retriever's top-M pool: b7_rank = min(bm25_rank, dense_rank); else b7_rank = fused_rank"
            ),
        },
        "n_queries": len(per_query),
        "small_n_limit": (
            "16 queries (15 Round 2 superiority + R1-Q7 supplementary). The rule fires on at most "
            "a handful; every statement is per-query, read directionally, the same standard as B3 "
            "and D-QR. No aggregate is computed — per plan §9.1 the effect is invisible in a mean."
        ),
        "predictions": predictions,
        "falsify_conditions": falsify,
        "precision_caveat": precision_caveat,
        "per_query": per_query,
        "exploratory_M_sweep": {
            "note": (
                "Protocol step 4 — exploratory only, outside the pass/fail read. Shows what "
                "M < 100 would do to the firing set. N held at 3 throughout. The pre-registered "
                "result is M = 100 above; changing M post hoc voids B7 as a pre-registered test "
                "(plan §4, 'Post-hoc tuning invalidates the test')."
            ),
            "sweep": exploratory,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nFired on: {sorted(fired_ids) or '(none)'}")
    print(f"Fused-rank-1 regressions: {regressions or '(none)'}")
    print(f"Saved -> {OUT_PATH}")


def _fmt(rank: int | None) -> str:
    return "--" if rank is None else str(rank)


if __name__ == "__main__":
    main()
