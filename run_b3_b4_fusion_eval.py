"""Hypotheses B3 + B4 — single-signal displacement under fusion, and its
score-normalised remedy, evaluated live in one pass.

`docs/uat/round3_hypothesis_test_plan.md` §4:

  B3  RRF displaces high-confidence single-signal results toward a central
      rank band, largely irrespective of the rank its sole supporting
      retriever gave it. Metric: target-chunk rank. Re-specified 31 Aug
      2026 (CORR-NVIR-2026-001 §3.3) after its score-ratio premise was
      found void.
  B4  Score-normalised fusion (min-max / z-score CombSUM) preserves the
      confidence signal RRF discards and should recover the displaced
      target. Run on the SAME queries and target chunks as B3 so each
      single-signal target's normalised-fusion rank reads directly
      against its measured RRF displacement.

For each of the 15 Round 2 superiority queries
(`docs/uat/uat_superiority_cases_raw.json`) plus R1-Q7 (`shader processor
count`, supplementary), retrieve BM25 top-100 and dense top-100 live with
the LITERAL query (no rewriting — that was D-QR), then fuse four ways and
record the fixed retriever-independent target chunk's rank in each list:

  bm25            — BM25Okapi over chunk_text (local pickle)
  dense           — e5-base-v2 + Qdrant cosine
  rrf             — reciprocal rank fusion, k=60 (the pipeline default)
  minmax_combsum  — per-retriever min-max scaled scores, summed
  zscore_combsum  — per-retriever z-scored scores, summed

Normalisation (B4). Each retriever's own top-100 pool defines the scale:
min-max -> n = (s - min) / (max - min), range [0, 1]; z-score ->
z = (s - mean) / std. A chunk absent from a retriever's pool contributes
that scale's floor for the missing retriever — 0.0 for min-max, min(z in
pool) for z-score — i.e. "no better than the worst chunk that retriever
returned", never the mean. fused = norm_bm25 + norm_dense, sorted
descending. This is min-max / z-score CombSUM (Lee 1997; Montague & Aslam
2001); RRF exists precisely to avoid it, which is the counter-argument
B4 states.

Metric is target-chunk rank, not NDCG: `run_day9_relevance_labelling.py`'s
qrels are circular (Round 3 A2/A3) and no retriever-independent graded
labels exist yet (ENH-11). Lower rank = better; `None` = target absent
from the top-`POOL_SIZE`. The target chunk per query is the one the Round
2 / Round 1 prose names as the correct answer (same anchor as A1, D-QR).

Corroboration classification (B3 step 3). single-signal-dense: dense
rank <= 5 and BM25 rank >= 6 or absent. single-signal-BM25: the mirror.
corroborated: both <= 5. weak/neither: neither <= 5. The <= 5 / >= 6 cut
is a starting bin; raw ranks are persisted so a reader can re-bin.

Memory: loads e5-base-v2 once and embeds all 16 queries up front, then
frees the encoder before any fusion arithmetic. One heavy stage, per the
host rule. Writes `evaluation/b3_b4_fusion_eval.json`. No analysis here —
see `docs/uat/round3_b3_b4_findings.md`.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version.
from utils.require_python import require_python

require_python()

import gc
import json
import statistics
import sys
from pathlib import Path

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse

CASES_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
OUT_PATH = Path("evaluation/b3_b4_fusion_eval.json")
POOL_SIZE = 100
RRF_K = 60
CORROB_CUT = 5  # rank <= CUT counts as "this retriever ranks the target well"

# Retriever-independent target chunk per query — copied verbatim from
# run_dqr_eval.py (itself taken from the Round 2 / Round 1 write-ups).
TARGETS: dict[str, str] = {
    "Q1": "cc6c8e53936d04e9b192a7d5",   # cudaMalloc(void**, size_t) signature
    "Q2": "242e353090d3f493c8ef64dc",   # "cudaMemcpyAsync() is a non-blocking variant" definition
    "Q3": "0904f18029b044943773fe74",   # "Returns the description string for an error code"
    "Q4": "35b73f3371037bf5ba8fefc0",   # "An SM consists of: 128 CUDA cores" (Round 1 Q7 target)
    "Q5": "8f2dbd94357e8ac31b8a595c",   # Autotuning section (dense rank 2)
    "Q6": "3c9b5ecfb38e9ee9377cf4f6",   # "Branching and Divergence"
    "Q7": "ccd7708bb87a3ea7259deb4c",   # "atomic memory operations ... much better performance"
    "Q8": "1fabb00b8793f603b648e7d1",   # "To minimize bank conflicts ..."
    "Q9": "3c8573a7f22cd37a771b4816",   # "Coalesced Access to Global Memory"
    "Q10": "f2730f1e6b30f8901b027a33",  # "instructions required to hide a latency of L clock cycles"
    "Q11": "0a880bdf76e6612e502eb50a",  # "Execution Configuration Optimizations ... keep the multiprocessors busy"
    "Q12": "02bb6a205ba73aa9763b937c",  # "cudaDeviceSynchronize() returns an error if ..."
    "Q13": "a612172bf233e826b41e390c",  # dim3 struct field reference
    "Q14": "5e38563157daf351863b54ec",  # "10.1.1. Pinned Memory"
    "Q15": "a1f985eaed4eb1f19e46d20a",  # "register count affects occupancy ... allocation granularity"
    "R1-Q7": "35b73f3371037bf5ba8fefc0",  # supplementary — the short "shader processor count" phrasing
}

CASE_LABELS = {
    "case1_bm25_lexical_superiority": "Case 1 — BM25 lexical",
    "case2_dense_semantic_superiority": "Case 2 — dense semantic",
    "case3_rrf_hybrid_superiority": "Case 3 — RRF hybrid",
    "case4_bm25_failure_dense_advantage": "Case 4 — BM25 failure / vocab gap",
    "case5_dense_failure_bm25_advantage": "Case 5 — dense failure / exact lookup",
    "case6_rrf_hybrid_advantage_mixed": "Case 6 — RRF mixed",
}


def load_queries() -> list[dict]:
    rows = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    queries = [
        {"query_id": r["query_id"], "query": r["query"], "case": r["case"], "supplementary": False}
        for r in rows
    ]
    queries.append(
        {
            "query_id": "R1-Q7",
            "query": "shader processor count",
            "case": "case4_bm25_failure_dense_advantage",
            "supplementary": True,
        }
    )
    return queries


def _rank_of(chunk_id: str, results: list[Candidate]) -> int | None:
    for c in results:
        if c.chunk_id == chunk_id:
            return c.rank
    return None


def _score_of(chunk_id: str, results: list[Candidate]) -> float | None:
    for c in results:
        if c.chunk_id == chunk_id:
            return c.score
    return None


# --------------------------------------------------------------------------
# B4 — score-normalised CombSUM fusion
# --------------------------------------------------------------------------

def _minmax_norm(results: list[Candidate]) -> tuple[dict[str, float], float]:
    """chunk_id -> min-max scaled score over this pool; plus the floor (0.0)."""
    if not results:
        return {}, 0.0
    scores = [c.score for c in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return {c.chunk_id: 1.0 for c in results}, 0.0
    return {c.chunk_id: (c.score - lo) / span for c in results}, 0.0


def _zscore_norm(results: list[Candidate]) -> tuple[dict[str, float], float]:
    """chunk_id -> z-scored score over this pool; plus the floor (min z)."""
    if not results:
        return {}, 0.0
    scores = [c.score for c in results]
    mean = statistics.fmean(scores)
    sd = statistics.pstdev(scores)
    if sd == 0:
        return {c.chunk_id: 0.0 for c in results}, 0.0
    z = {c.chunk_id: (c.score - mean) / sd for c in results}
    return z, min(z.values())


def _combsum(
    bm25_results: list[Candidate],
    dense_results: list[Candidate],
    norm_fn,
    top_k: int = POOL_SIZE,
) -> list[Candidate]:
    """Sum each retriever's normalised score; absent -> that pool's floor."""
    bm25_norm, bm25_floor = norm_fn(bm25_results)
    dense_norm, dense_floor = norm_fn(dense_results)

    texts: dict[str, str] = {}
    order: list[str] = []
    for results in (bm25_results, dense_results):
        for c in results:
            if c.chunk_id not in texts:
                texts[c.chunk_id] = c.text
                order.append(c.chunk_id)

    total = {
        cid: bm25_norm.get(cid, bm25_floor) + dense_norm.get(cid, dense_floor) for cid in order
    }
    ranked = sorted(order, key=lambda cid: total[cid], reverse=True)[:top_k]
    return [
        Candidate(chunk_id=cid, text=texts[cid], score=total[cid], rank=rank)
        for rank, cid in enumerate(ranked, start=1)
    ]


# --------------------------------------------------------------------------
# B3 — corroboration structure + displacement + RRF decomposition
# --------------------------------------------------------------------------

def _classify_corroboration(bm25_rank: int | None, dense_rank: int | None) -> str:
    bm25_well = bm25_rank is not None and bm25_rank <= CORROB_CUT
    dense_well = dense_rank is not None and dense_rank <= CORROB_CUT
    if bm25_well and dense_well:
        return "corroborated"
    if dense_well and not bm25_well:
        return "single-signal-dense"
    if bm25_well and not dense_well:
        return "single-signal-bm25"
    return "weak/neither"


def _rrf_contribution(bm25_rank: int | None, dense_rank: int | None, k: int) -> dict:
    b = 1.0 / (k + bm25_rank) if bm25_rank is not None else 0.0
    d = 1.0 / (k + dense_rank) if dense_rank is not None else 0.0
    return {"bm25": b, "dense": d, "sum": b + d}


def evaluate(queries: list[dict], bm25: BM25Index, dense: DenseIndex) -> list[dict]:
    # --- heavy stage: one live retrieval pass, encoder loaded once ---
    bm25_pools = {q["query_id"]: bm25.search(q["query"], top_k=POOL_SIZE) for q in queries}
    dense_pools = {q["query_id"]: dense.search(q["query"], top_k=POOL_SIZE) for q in queries}
    return bm25_pools, dense_pools  # type: ignore[return-value]


def analyse(
    queries: list[dict],
    bm25_pools: dict[str, list[Candidate]],
    dense_pools: dict[str, list[Candidate]],
) -> list[dict]:
    records: list[dict] = []
    for q in queries:
        qid = q["query_id"]
        target = TARGETS[qid]
        bm25_hits = bm25_pools[qid]
        dense_hits = dense_pools[qid]

        bm25_rank = _rank_of(target, bm25_hits)
        dense_rank = _rank_of(target, dense_hits)

        rrf_hits = fuse(bm25_hits, dense_hits, top_k=POOL_SIZE, k=RRF_K)
        minmax_hits = _combsum(bm25_hits, dense_hits, _minmax_norm)
        zscore_hits = _combsum(bm25_hits, dense_hits, _zscore_norm)

        rrf_rank = _rank_of(target, rrf_hits)
        minmax_rank = _rank_of(target, minmax_hits)
        zscore_rank = _rank_of(target, zscore_hits)

        corrob = _classify_corroboration(bm25_rank, dense_rank)

        # finding retriever = the one that ranks the target <= CUT (for a
        # single-signal case there is exactly one); for corroborated/weak
        # cases, whichever ranks it better, recorded but not the headline.
        ranks = {"bm25": bm25_rank, "dense": dense_rank}
        present = {r: v for r, v in ranks.items() if v is not None}
        if corrob == "single-signal-dense":
            finding, other = "dense", "bm25"
        elif corrob == "single-signal-bm25":
            finding, other = "bm25", "dense"
        elif present:
            finding = min(present, key=lambda r: present[r])
            other = "bm25" if finding == "dense" else "dense"
        else:
            finding = other = None

        finding_rank = ranks[finding] if finding else None
        other_rank = ranks[other] if other else None
        best_single = min(present.values()) if present else None

        displacement = (
            rrf_rank - finding_rank if (rrf_rank is not None and finding_rank is not None) else None
        )
        disp_vs_best = (
            rrf_rank - best_single if (rrf_rank is not None and best_single is not None) else None
        )

        # RRF-score decomposition: target vs the chunk one rank above it in
        # the fused list (the chunk that "beat" it).
        decomp = None
        if rrf_rank is not None and rrf_rank >= 2:
            above_cid = rrf_hits[rrf_rank - 2].chunk_id  # rrf_rank is 1-indexed
            decomp = {
                "chunk_above_id": above_cid,
                "chunk_above_rank_bm25": _rank_of(above_cid, bm25_hits),
                "chunk_above_rank_dense": _rank_of(above_cid, dense_hits),
                "target_rrf_contribution": _rrf_contribution(bm25_rank, dense_rank, RRF_K),
                "chunk_above_rrf_contribution": _rrf_contribution(
                    _rank_of(above_cid, bm25_hits), _rank_of(above_cid, dense_hits), RRF_K
                ),
            }

        rec = {
            "query_id": qid,
            "query": q["query"],
            "case": q["case"],
            "case_label": CASE_LABELS.get(q["case"], q["case"]),
            "supplementary": q.get("supplementary", False),
            "target_chunk": target,
            "target_score_bm25": _score_of(target, bm25_hits),
            "target_score_dense": _score_of(target, dense_hits),
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "corroboration_class": corrob,
            "finding_retriever": finding,
            "finding_rank": finding_rank,
            "corroboration_strength_rank": other_rank,  # the OTHER retriever's rank; None = absent
            "best_single_rank": best_single,
            "fused_rank": {
                "rrf": rrf_rank,
                "minmax_combsum": minmax_rank,
                "zscore_combsum": zscore_rank,
            },
            "displacement_rrf_vs_finding": displacement,
            "displacement_rrf_vs_best_single": disp_vs_best,
            "b4_recovery_vs_rrf": {
                "minmax": (rrf_rank - minmax_rank)
                if (rrf_rank is not None and minmax_rank is not None)
                else None,
                "zscore": (rrf_rank - zscore_rank)
                if (rrf_rank is not None and zscore_rank is not None)
                else None,
            },
            "rrf_decomposition": decomp,
        }
        records.append(rec)
        print(
            f"{qid:6s} {rec['case_label']:32s} "
            f"bm25={_fmt(bm25_rank)} dense={_fmt(dense_rank)} | "
            f"rrf={_fmt(rrf_rank)} mm={_fmt(minmax_rank)} z={_fmt(zscore_rank)} | "
            f"{corrob}"
        )
    return records


def _fmt(rank: int | None) -> str:
    return "--" if rank is None else str(rank)


def summarise(records: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["corroboration_class"], []).append(r["query_id"])
    single = [
        {
            "query_id": r["query_id"],
            "class": r["corroboration_class"],
            "finding_retriever": r["finding_retriever"],
            "finding_rank": r["finding_rank"],
            "corroboration_strength_rank": r["corroboration_strength_rank"],
            "rrf_rank": r["fused_rank"]["rrf"],
            "displacement_rrf_vs_finding": r["displacement_rrf_vs_finding"],
            "minmax_rank": r["fused_rank"]["minmax_combsum"],
            "zscore_rank": r["fused_rank"]["zscore_combsum"],
        }
        for r in records
        if r["corroboration_class"] in ("single-signal-dense", "single-signal-bm25")
    ]
    return {
        "classes": {k: sorted(v) for k, v in by_class.items()},
        "single_signal_cases": single,
    }


def main() -> None:
    resummarize = "--resummarize" in sys.argv
    if resummarize:
        out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        out["summary"] = summarise(out["per_query"])
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Re-summarised {OUT_PATH} from {len(out['per_query'])} records.")
        return

    queries = load_queries()
    print(f"Loaded {len(queries)} queries (15 Round 2 + R1-Q7 supplementary).\n")

    print("Loading BM25 index (local pickle)...")
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)

    print("Connecting to live Qdrant + loading e5-base-v2 query encoder (heavy step)...\n")
    dense = DenseIndex.connect()

    bm25_pools, dense_pools = evaluate(queries, bm25, dense)

    del dense
    gc.collect()
    print("\nEncoder released. Running fusion arithmetic (no model in memory)...\n")

    records = analyse(queries, bm25_pools, dense_pools)

    out = {
        "hypotheses": ["B3", "B4"],
        "plan": "docs/uat/round3_hypothesis_test_plan.md §4",
        "metric": "target-chunk rank in each of {bm25, dense, rrf, minmax_combsum, zscore_combsum} top-100",
        "pool_size": POOL_SIZE,
        "rrf_k": RRF_K,
        "corroboration_cut": CORROB_CUT,
        "b4_normalisation": {
            "minmax": "per-retriever (s - min) / (max - min) over its top-100 pool; absent chunk -> 0.0",
            "zscore": "per-retriever (s - mean) / pstdev over its top-100 pool; absent chunk -> min(z in pool)",
            "combine": "sum of the two normalised scores (CombSUM), sorted descending",
        },
        "targets": TARGETS,
        "per_query": records,
        "summary": summarise(records),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
