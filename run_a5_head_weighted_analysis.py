"""Experiment A5 (follow-up -- head-weighted agreement).

Resolves whether the ~0.595 median full-list Spearman rho between Config A
(ms-marco) and Config C (Cohere Rerank v3) reflects genuine head disagreement
or close head agreement with divergence confined to an irrelevant tail --
using the retained full top-100 orderings from run_a5_top100_pipeline.py
(evaluation/a5_top100_orderings.json). No new retrieval or API calls.

Per query, over the same retained orderings:
  - Spearman rho restricted to the head: computed over the union of each
    config's top-10 chunk_ids, using each config's full-100 rank for every
    id in that union. This is well-defined (not just for the intersection)
    because both configs rerank the *same* fused pool -- every chunk_id in
    one config's top-10 necessarily appears somewhere in the other's full
    ranking.
  - Overlap@10: |top-10(A) intersect top-10(C)|.
  - Overlap@5 and whether top-1 matches.
  - RBO (rank-biased overlap, p=0.9) over the full top-100 orderings, as a
    properly top-weighted measure that doesn't split disagreement evenly
    across all 100 ranks the way full-list Spearman does.

Reports per-query values grouped by the six Round 2 case types, plus
case-level means and the overall distribution for each metric.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

ORDERINGS_PATH = Path("evaluation/a5_top100_orderings.json")
RESULTS_PATH = Path("evaluation/a5_head_weighted_agreement.json")


def rbo(order_a: list[str], order_c: list[str], p: float = 0.9) -> float:
    """Rank-biased overlap (Webber, Moffat & Zobel 2010), extrapolated form.

    Assumes equal-length full orderings covering the same ground set (true
    here: both configs rerank the identical RRF pool), which makes the
    extrapolation term at depth k exact rather than an estimate.
    """
    if len(order_a) != len(order_c):
        raise ValueError("rbo() here assumes equal-length full orderings over the same pool")
    k = len(order_a)
    seen_a: set[str] = set()
    seen_c: set[str] = set()
    weighted_sum = 0.0
    for d in range(1, k + 1):
        seen_a.add(order_a[d - 1])
        seen_c.add(order_c[d - 1])
        x_d = len(seen_a & seen_c)
        weighted_sum += (x_d / d) * (p**d)
    x_k = len(seen_a & seen_c)
    return (x_k / k) * (p**k) + ((1 - p) / p) * weighted_sum


def spearman_top10(order_a: list[str], order_c: list[str]) -> float:
    rank_a = {cid: r for r, cid in enumerate(order_a, start=1)}
    rank_c = {cid: r for r, cid in enumerate(order_c, start=1)}
    head_ids = sorted(set(order_a[:10]) | set(order_c[:10]))
    if len(head_ids) < 2:
        return float("nan")
    ranks_a = [rank_a[cid] for cid in head_ids]
    ranks_c = [rank_c[cid] for cid in head_ids]
    rho, _ = spearmanr(ranks_a, ranks_c)
    return float(rho)


def compute_row(query_id: str, rec: dict) -> dict:
    order_a, order_c = rec["config_a"], rec["config_c"]
    top10_a, top10_c = set(order_a[:10]), set(order_c[:10])
    top5_a, top5_c = set(order_a[:5]), set(order_c[:5])
    return {
        "query_id": query_id,
        "case": rec["case"],
        "query": rec["query"],
        "spearman_top10": spearman_top10(order_a, order_c),
        "overlap_at_10": len(top10_a & top10_c),
        "overlap_at_5": len(top5_a & top5_c),
        "top1_match": order_a[0] == order_c[0],
        "rbo_p90": rbo(order_a, order_c, p=0.9),
    }


def print_report(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["query_id"])
    print(f"{'query':6} {'case':42} {'rho@10':>7} {'ovl@10':>7} {'ovl@5':>6} {'top1':>5} {'rbo.9':>7}")
    for r in rows:
        print(
            f"{r['query_id']:6} {r['case']:42} {r['spearman_top10']:7.3f} {r['overlap_at_10']:7d} "
            f"{r['overlap_at_5']:6d} {str(r['top1_match']):>5} {r['rbo_p90']:7.3f}"
        )

    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)

    print("\nCase-level means:")
    for case, case_rows in sorted(by_case.items()):
        n = len(case_rows)
        print(
            f"{case:42} n={n} "
            f"rho@10={statistics.mean(r['spearman_top10'] for r in case_rows):.3f} "
            f"ovl@10={statistics.mean(r['overlap_at_10'] for r in case_rows):.2f} "
            f"ovl@5={statistics.mean(r['overlap_at_5'] for r in case_rows):.2f} "
            f"top1={sum(r['top1_match'] for r in case_rows) / n:.2f} "
            f"rbo.9={statistics.mean(r['rbo_p90'] for r in case_rows):.3f}"
        )

    print("\nOverall distribution:")
    for key, label in [
        ("spearman_top10", "rho@10"),
        ("overlap_at_10", "ovl@10"),
        ("overlap_at_5", "ovl@5"),
        ("rbo_p90", "rbo.9"),
    ]:
        vals = sorted(r[key] for r in rows)
        print(
            f"{label}: mean={statistics.mean(vals):.3f} median={statistics.median(vals):.3f} "
            f"min={min(vals):.3f} max={max(vals):.3f}"
        )
    n_match = sum(r["top1_match"] for r in rows)
    print(f"top1_match: {n_match}/{len(rows)} ({n_match / len(rows):.0%})")


def main() -> None:
    data = json.loads(ORDERINGS_PATH.read_text(encoding="utf-8"))
    rows = [compute_row(query_id, rec) for query_id, rec in data.items()]
    print_report(rows)
    RESULTS_PATH.write_text(json.dumps(sorted(rows, key=lambda r: r["query_id"]), indent=2), encoding="utf-8")
    print(f"\nSaved per-query results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
