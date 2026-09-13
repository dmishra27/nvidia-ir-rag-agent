"""Hypothesis B4 -- aggregate NDCG@10, re-scored against ENH-11's graded,
retriever-independent qrels.

`docs/uat/round3_b3_b4_findings.md` §3.3 recorded the plan's B4 "Confirms"
test -- "normalised fusion beats RRF on NDCG@10 across all 15" -- as
**unevaluable**: `run_day9_relevance_labelling.py`'s qrels are circular
(Round 3 A2/A3) and no retriever-independent graded labels existed. ENH-11
(`evaluation/enh11_qrels.json`, 325/325 judged, `docs/uat/enh11_protocol.md`)
removes that blocker. This script is the first thing to score against it.

Scope: a pure offline re-scoring of two already-committed static files --
`evaluation/b3_b4_fusion_eval.json` (B3/B4, committed `0b13307`) and
`evaluation/enh11_qrels.json` (ENH-11, committed `d13eae0`), cross-referenced
with `evaluation/enh11_pool_provenance.json` (committed alongside the pool,
`3636dd5`). No live retrieval, no encoder load, no postgres/qdrant. Host:
8 GB total, frequently < 500 MB free (DEF-23, DEF-26) -- checked at 0.40 GB
free immediately before this ran, and nothing here loads more than three
small JSON files into memory.

What is, and is not, reconstructable from static files -- read this before
the numbers below, it changes what they mean:

  RRF's full top-10 ranked list IS reconstructable. `enh11_pool_provenance.
  json` records each pooled chunk's `rrf_rank` from the SAME retrieval
  config B3/B4 used (pool 100, k=60) -- `build_enh11_pool.py` pooled "top 10
  from RRF" with that exact config, six days after the B3/B4 run. Cross-
  checked here against every target chunk's `fused_rank.rrf` in
  `b3_b4_fusion_eval.json`: 15 of 16 match exactly; the one exception (Q4)
  is a pool-depth artifact, not a disagreement -- Q4's target sits at RRF
  rank 39, which was never pooled, so provenance correctly has no entry for
  it. That match is the evidence the two harnesses saw the same indexes.
  This lets us score REAL, full, multi-relevant-chunk NDCG@10 for RRF, using
  the project's own `evaluation.retrieval_metrics.ndcg_at_k` against the
  complete per-query qrels.

  minmax_combsum's and zscore_combsum's full top-10 lists are NOT
  reconstructable from any static file. `_combsum()` in
  `run_b3_b4_fusion_eval.py` ranks the union of each retriever's full
  top-100 by a SCORE sum (min-max or z-score normalised); ENH-11 never
  pooled or graded those two rankings, and no raw per-chunk BM25/dense
  scores survive on disk to redo the normalisation -- `b3_b4_fusion_eval.
  json` kept only the designated target chunk's score and rank under each
  method. Reconstructing even an approximate top-10 would mean guessing a
  score distribution, which is fabrication, not re-scoring, so this script
  does not attempt it.

  What IS honest for minmax/zscore: the target chunk's already-on-disk rank
  under each method, now scored against its REAL grade instead of an
  assumed one, as single-relevant-item NDCG@10 -- the standard metric when
  exactly one gold chunk per query is known (NDCG@10 = 1/log2(rank+1) when
  the one known chunk is relevant and ranked <=10, else 0; the grade's
  magnitude cancels against its own IDCG, so this form distinguishes "the
  target is relevant at all" from rank, not grade 1 vs 2 vs 3 -- which is
  exactly why the binary cut below adds information the graded form can't).
  It is computed identically for RRF, so all three methods sit on the same
  basis for this one number. It is reported ALONGSIDE, never merged with,
  RRF's full-list number: crediting RRF for every relevant chunk it
  surfaces while crediting minmax/zscore for only the one pre-identified
  chunk would bias the comparison against normalised fusion -- exactly the
  failure this script exists to avoid.

Two relevance mappings, both reported throughout:
  graded  -- qrels grade (0-3) used as-is as the gain.
  binary  -- grade >= 2 ("substantively relevant" or better) -> 1, else 0.

Aggregate is the 15 Round 2 queries (`docs/uat/uat_superiority_cases_raw.
json`); R1-Q7 is the supplementary anchor case and is reported separately,
per the plan and per B3/B4's own convention.

Writes `evaluation/b4_enh11_ndcg_rescore.json`. No analysis here -- see
`docs/uat/round3_b3_b4_findings.md` §3.3 (to be updated).
"""

from __future__ import annotations

# Consistency with the other run_*.py harnesses; this script imports no
# third-party package but the interpreter guard is cheap and uniform.
from utils.require_python import require_python

require_python()

import json
import statistics
from pathlib import Path

from evaluation.retrieval_metrics import ndcg_at_k
from retrieval.candidates import Candidate

B34_PATH = Path("evaluation/b3_b4_fusion_eval.json")
QRELS_PATH = Path("evaluation/enh11_qrels.json")
PROVENANCE_PATH = Path("evaluation/enh11_pool_provenance.json")
CASES_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
OUT_PATH = Path("evaluation/b4_enh11_ndcg_rescore.json")

K = 10
BINARY_CUT = 2  # grade >= this counts as relevant for the binary mapping
METHODS = ("rrf", "minmax_combsum", "zscore_combsum")


def _load_round2_ids() -> list[str]:
    rows = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return [r["query_id"] for r in rows]


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """query_id -> {chunk_id: grade}, primary judgements (pass 1) only."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for row in data["judgements"]:
        if row.get("pass") != 1:
            continue
        out.setdefault(row["query_id"], {})[row["chunk_id"]] = row["grade"]
    return out


def _load_rrf_top10(path: Path) -> dict[str, list[str]]:
    """query_id -> chunk_ids in ascending rrf_rank order, rank <= 10 only."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for q in data["queries"]:
        ranked = [
            c for c in q["chunks"] if c.get("rrf_rank") is not None and c["rrf_rank"] <= K
        ]
        ranked.sort(key=lambda c: c["rrf_rank"])
        out[q["query_id"]] = [c["chunk_id"] for c in ranked]
    return out


def _binary(relevance: dict[str, int]) -> dict[str, int]:
    return {cid: 1 if grade >= BINARY_CUT else 0 for cid, grade in relevance.items()}


def _full_rrf_ndcg(chunk_ids: list[str], relevance: dict[str, int]) -> float:
    """Real, full top-10 NDCG@10 -- every graded chunk in the query's pool
    counts toward both DCG (if ranked) and IDCG (always)."""
    ranked = [
        Candidate(chunk_id=cid, text="", score=0.0, rank=i)
        for i, cid in enumerate(chunk_ids, start=1)
    ]
    return ndcg_at_k(ranked, relevance, k=K)


def _single_target_ndcg(rank: int | None, target_grade: int | None) -> float:
    """Single-relevant-item NDCG@10: the one known gold chunk, scored at its
    known rank under this method, against its own grade as the sole entry
    in the relevance map (so IDCG is that grade, never another chunk's)."""
    if target_grade is None or rank is None or rank > K:
        return 0.0
    # A sparse ranked list: unknown placeholders up to rank-1 (gain 0, absent
    # from the relevance map), the target at its real rank.
    placeholder_ids = [f"_unknown_{i}" for i in range(rank - 1)]
    ranked = [
        Candidate(chunk_id=cid, text="", score=0.0, rank=i)
        for i, cid in enumerate(placeholder_ids, start=1)
    ]
    ranked.append(Candidate(chunk_id="_target_", text="", score=0.0, rank=rank))
    relevance = {"_target_": target_grade}
    return ndcg_at_k(ranked, relevance, k=K)


def main() -> None:
    round2_ids = _load_round2_ids()
    b34 = json.loads(B34_PATH.read_text(encoding="utf-8"))
    qrels_graded = _load_qrels(QRELS_PATH)
    qrels_binary = {qid: _binary(rel) for qid, rel in qrels_graded.items()}
    rrf_top10 = _load_rrf_top10(PROVENANCE_PATH)

    per_query: list[dict] = []
    target_grade_check: list[dict] = []

    for rec in b34["per_query"]:
        qid = rec["query_id"]
        target = rec["target_chunk"]
        rel_graded = qrels_graded.get(qid, {})
        rel_binary = qrels_binary.get(qid, {})
        target_grade = rel_graded.get(target)

        target_grade_check.append(
            {
                "query_id": qid,
                "target_chunk": target,
                "grade": target_grade,
                "in_pool": target in rel_graded,
            }
        )

        rrf_ids = rrf_top10.get(qid, [])
        full_ndcg_graded = _full_rrf_ndcg(rrf_ids, rel_graded)
        full_ndcg_binary = _full_rrf_ndcg(rrf_ids, rel_binary)

        single_graded = {
            m: _single_target_ndcg(rec["fused_rank"][m], target_grade) for m in METHODS
        }
        single_binary = {
            m: _single_target_ndcg(
                rec["fused_rank"][m], 1 if (target_grade or 0) >= BINARY_CUT else 0
            )
            for m in METHODS
        }

        per_query.append(
            {
                "query_id": qid,
                "query": rec["query"],
                "case_label": rec["case_label"],
                "supplementary": rec.get("supplementary", False),
                "target_chunk": target,
                "target_grade": target_grade,
                "fused_rank": rec["fused_rank"],
                "rrf_full_ndcg10": {"graded": round(full_ndcg_graded, 4), "binary": round(full_ndcg_binary, 4)},
                "single_target_ndcg10": {
                    "graded": {m: round(v, 4) for m, v in single_graded.items()},
                    "binary": {m: round(v, 4) for m, v in single_binary.items()},
                },
            }
        )
        print(
            f"{qid:6s} {rec['case_label']:32s} grade={target_grade} | "
            f"rrf_full(g={full_ndcg_graded:.3f},b={full_ndcg_binary:.3f}) | "
            f"single g: rrf={single_graded['rrf']:.3f} mm={single_graded['minmax_combsum']:.3f} "
            f"z={single_graded['zscore_combsum']:.3f}"
        )

    def _mean(rows: list[dict], key_path: tuple[str, ...]) -> float:
        vals = []
        for r in rows:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        return round(statistics.fmean(vals), 4) if vals else 0.0

    round2_rows = [r for r in per_query if r["query_id"] in round2_ids]
    r1q7_rows = [r for r in per_query if r["query_id"] == "R1-Q7"]

    def _aggregate(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "rrf_full_ndcg10": {
                "graded": _mean(rows, ("rrf_full_ndcg10", "graded")),
                "binary": _mean(rows, ("rrf_full_ndcg10", "binary")),
            },
            "single_target_ndcg10": {
                "graded": {
                    m: _mean(rows, ("single_target_ndcg10", "graded", m)) for m in METHODS
                },
                "binary": {
                    m: _mean(rows, ("single_target_ndcg10", "binary", m)) for m in METHODS
                },
            },
        }

    aggregate_round2 = _aggregate(round2_rows)
    aggregate_r1q7 = _aggregate(r1q7_rows)

    out = {
        "hypothesis": "B4",
        "plan": "docs/uat/round3_hypothesis_test_plan.md §4 (B4)",
        "source_files": {
            "b3_b4_fusion_eval": str(B34_PATH),
            "enh11_qrels": str(QRELS_PATH),
            "enh11_pool_provenance": str(PROVENANCE_PATH),
        },
        "method_note": (
            "rrf_full_ndcg10 is real multi-relevant-chunk NDCG@10: RRF's top-10 "
            "reconstructed from enh11_pool_provenance.json's rrf_rank, scored "
            "against every graded chunk in the query's enh11 pool. "
            "single_target_ndcg10 is single-relevant-item NDCG@10 (the one "
            "pre-registered target chunk only), computed identically for all "
            "three fusion methods because minmax_combsum/zscore_combsum have "
            "no persisted full top-10 list to score for real -- see module "
            "docstring. The two are not comparable to each other; only "
            "single_target_ndcg10 is comparable ACROSS methods."
        ),
        "binary_cut": f"grade >= {BINARY_CUT}",
        "k": K,
        "round2_query_ids": round2_ids,
        "target_grade_check": target_grade_check,
        "per_query": per_query,
        "aggregate": {
            "round2_15_queries": aggregate_round2,
            "r1_q7_supplementary": aggregate_r1q7,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")
    print("\nRound-2 (15 queries) aggregate:")
    print(json.dumps(aggregate_round2, indent=2))
    print("\nR1-Q7 (supplementary):")
    print(json.dumps(aggregate_r1q7, indent=2))


if __name__ == "__main__":
    main()
