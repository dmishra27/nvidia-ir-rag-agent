"""Hypothesis D-QR — conditional query rewriting, evaluated live.

For each of the 15 Round 2 superiority queries (`docs/uat/uat_superiority_
cases_raw.json`) plus R1-Q7 (`shader processor count`, the documented
vocabulary-gap case, supplementary), retrieve three ways and record the
rank of a fixed retriever-independent target chunk in the BM25, dense and
RRF lists:

  baseline  — no rewrite (bm25 == dense == literal query)
  gated     — retrieval/query_rewrite.rewrite_query(q, gate=True):
              legacy-term expansion + camelCase split on the DENSE query,
              skipped entirely for exact-identifier lookups
  ungated   — rewrite_query(q, gate=False): both strategies, every query

Metric is target-chunk rank, not NDCG: `run_day9_relevance_labelling.py`'s
qrels are circular (Round 3 A2/A3) and no retriever-independent graded
labels exist yet (ENH-11). Lower rank = better; `None` = target absent
from the top-`POOL_SIZE`.

Memory: loads e5-base-v2 once (the only heavy step). Per the host rule,
this is the single heavy stage; BM25 is a local pickle. Writes
`evaluation/dqr_eval.json` and prints a per-query / per-case summary. No
analysis here — see `docs/uat/round3_dqr_findings.md`.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version.
from utils.require_python import require_python

require_python()

import gc
import json
import sys
from pathlib import Path

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.query_rewrite import classify, rewrite_query
from retrieval.rrf_fusion import fuse

CASES_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
OUT_PATH = Path("evaluation/dqr_eval.json")
POOL_SIZE = 100
RRF_K = 60

# Retriever-independent target chunk per query, taken from the Round 2 and
# Round 1 write-ups (the chunk each prose verdict names as the best answer).
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
    # R1-Q7 — the short "shader processor count" phrasing Round 1 nominated as
    # the vocabulary-gap regression case. Grouped with Round 2's Case 4 but
    # flagged so it is not counted among the official 15.
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


def evaluate(queries: list[dict], bm25: BM25Index, dense: DenseIndex) -> list[dict]:
    # BM25 uses the literal query in every mode -> retrieve once per query.
    bm25_pools = {q["query_id"]: bm25.search(q["query"], top_k=POOL_SIZE) for q in queries}

    # Dense query strings vary by mode; cache search by the exact string.
    dense_cache: dict[str, list[Candidate]] = {}

    def dense_search(text: str) -> list[Candidate]:
        if text not in dense_cache:
            dense_cache[text] = dense.search(text, top_k=POOL_SIZE)
        return dense_cache[text]

    records: list[dict] = []
    for q in queries:
        qid = q["query_id"]
        target = TARGETS[qid]
        bm25_hits = bm25_pools[qid]
        bm25_rank = _rank_of(target, bm25_hits)

        gated = rewrite_query(q["query"], gate=True)
        ungated = rewrite_query(q["query"], gate=False)
        modes = {
            "baseline": q["query"],
            "gated": gated.dense_query,
            "ungated": ungated.dense_query,
        }

        rec: dict = {
            "query_id": qid,
            "query": q["query"],
            "case": q["case"],
            "case_label": CASE_LABELS.get(q["case"], q["case"]),
            "supplementary": q.get("supplementary", False),
            "target_chunk": target,
            "classify": classify(q["query"]),
            "gated_dense_query": gated.dense_query,
            "gated_strategy": gated.strategy,
            "gated_detail": gated.detail,
            "ungated_dense_query": ungated.dense_query,
            "ungated_strategy": ungated.strategy,
            "ungated_detail": ungated.detail,
            "bm25_rank": bm25_rank,
            "modes": {},
        }
        for mode, dense_query in modes.items():
            dense_hits = dense_search(dense_query)
            fused = fuse(bm25_hits, dense_hits, top_k=POOL_SIZE, k=RRF_K)
            rec["modes"][mode] = {
                "dense_query": dense_query,
                "dense_rank": _rank_of(target, dense_hits),
                "rrf_rank": _rank_of(target, fused),
            }
        base_rrf = rec["modes"]["baseline"]["rrf_rank"]
        for mode in ("gated", "ungated"):
            m_rrf = rec["modes"][mode]["rrf_rank"]
            rec["modes"][mode]["rrf_delta_vs_baseline"] = _delta(base_rrf, m_rrf)
        records.append(rec)
        print(
            f"{qid:6s} {rec['case_label']:24s} "
            f"bm25={_fmt(bm25_rank)} "
            f"rrf base={_fmt(base_rrf)} gated={_fmt(rec['modes']['gated']['rrf_rank'])} "
            f"ungated={_fmt(rec['modes']['ungated']['rrf_rank'])}  [{gated.strategy}]"
        )
    return records


def _delta(base: int | None, other: int | None) -> int | None:
    """base - other; positive = target moved up (improved). None if either missing."""
    if base is None or other is None:
        return None
    return base - other


def _fmt(rank: int | None) -> str:
    return "--" if rank is None else str(rank)


def summarise(records: list[dict]) -> dict:
    """Per-case gated/ungated RRF-rank movement. Supplementary queries
    (R1-Q7) are excluded from the case rollup and reported on their own."""
    official = [r for r in records if not r.get("supplementary")]
    by_case: dict[str, list[dict]] = {}
    for r in official:
        by_case.setdefault(r["case_label"], []).append(r)

    def _roll(recs: list[dict], mode: str) -> dict:
        scored = [
            r["modes"][mode]["rrf_delta_vs_baseline"]
            for r in recs
            if r["modes"][mode]["rrf_delta_vs_baseline"] is not None
        ]
        return {
            "n": len(recs),
            "n_scored": len(scored),
            "improved": sum(1 for d in scored if d > 0),
            "unchanged": sum(1 for d in scored if d == 0),
            "worsened": sum(1 for d in scored if d < 0),
            "mean_delta": round(sum(scored) / len(scored), 2) if scored else None,
        }

    case_summary = {
        case_label: {mode: _roll(recs, mode) for mode in ("gated", "ungated")}
        for case_label, recs in by_case.items()
    }
    supplementary = {
        r["query_id"]: {
            mode: r["modes"][mode]["rrf_delta_vs_baseline"] for mode in ("gated", "ungated")
        }
        for r in records
        if r.get("supplementary")
    }
    return {"by_case": case_summary, "supplementary": supplementary}


def _print_case_summary(case_summary: dict) -> None:
    print("\nPer-case RRF-rank movement vs baseline (official 15 only):")
    for case_label, modes in case_summary["by_case"].items():
        g, u = modes["gated"], modes["ungated"]
        print(
            f"  {case_label:38s} n={g['n']}  "
            f"gated +{g['improved']}/={g['unchanged']}/-{g['worsened']} (mean {g['mean_delta']})   "
            f"ungated +{u['improved']}/={u['unchanged']}/-{u['worsened']} (mean {u['mean_delta']})"
        )
    for qid, deltas in case_summary["supplementary"].items():
        print(f"  {qid} (supplementary): gated delta {deltas['gated']}, ungated delta {deltas['ungated']}")


def main() -> None:
    resummarize = "--resummarize" in sys.argv

    if resummarize:
        # Memory-safe: recompute per_case from an existing per_query dump, no retrieval.
        out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        supp_ids = {q["query_id"] for q in load_queries() if q.get("supplementary")}
        for r in out["per_query"]:
            r["case_label"] = CASE_LABELS.get(r["case"], r["case"])
            r["supplementary"] = r["query_id"] in supp_ids
        out["per_case"] = summarise(out["per_query"])
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Re-summarised {OUT_PATH} from {len(out['per_query'])} retained per-query records.")
        _print_case_summary(out["per_case"])
        return

    queries = load_queries()
    print(f"Loaded {len(queries)} queries (15 Round 2 + R1-Q7 supplementary).\n")

    print("Loading BM25 index (local pickle)...")
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)

    print("Connecting to live Qdrant + loading e5-base-v2 query encoder (heavy step)...\n")
    dense = DenseIndex.connect()

    records = evaluate(queries, bm25, dense)

    del dense
    gc.collect()

    out = {
        "hypothesis": "D-QR",
        "metric": "target-chunk rank in BM25 / dense / RRF top-100; delta = baseline_rrf - mode_rrf (positive = improved)",
        "pool_size": POOL_SIZE,
        "rrf_k": RRF_K,
        "targets": TARGETS,
        "per_query": records,
        "per_case": summarise(records),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")
    _print_case_summary(out["per_case"])


if __name__ == "__main__":
    main()
