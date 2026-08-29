"""Experiment A5 verification: full-list (all 100 items) Spearman rho between
Config A (ms-marco) and Config C (Cohere), per query, from the retained
orderings in evaluation/a5_top100_orderings.json. No new retrieval/rerank
calls.

This exists to check a specific number the head-weighted argument in
run_a5_head_weighted_analysis.py depends on: that analysis claims RBO(0.9)
(mean 0.556 / median 0.590) does *not* read meaningfully higher than the
full-list Spearman rho, and treats that as evidence against "close head
agreement, tail-only divergence." The full-list rho it was compared against
(median ~0.595) was carried over from an earlier, unverified run -- this
script recomputes it directly from the same retained data the rest of A5
now uses, so the comparison rests on one consistent dataset instead of two.
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
HEAD_WEIGHTED_PATH = Path("evaluation/a5_head_weighted_agreement.json")
RESULTS_PATH = Path("evaluation/a5_full_list_spearman.json")


def full_list_spearman(order_a: list[str], order_c: list[str]) -> float:
    if set(order_a) != set(order_c):
        raise ValueError("full-list Spearman here assumes both orderings cover the same pool")
    rank_a = {cid: r for r, cid in enumerate(order_a, start=1)}
    rank_c = {cid: r for r, cid in enumerate(order_c, start=1)}
    # Pair every item by its rank in each list, in a fixed (order_a) sequence.
    xs = [rank_a[cid] for cid in order_a]
    ys = [rank_c[cid] for cid in order_a]
    rho, _ = spearmanr(xs, ys)
    return float(rho)


def main() -> None:
    data = json.loads(ORDERINGS_PATH.read_text(encoding="utf-8"))
    rbo_by_query = {}
    if HEAD_WEIGHTED_PATH.exists():
        for row in json.loads(HEAD_WEIGHTED_PATH.read_text(encoding="utf-8")):
            rbo_by_query[row["query_id"]] = row["rbo_p90"]

    rows = []
    for query_id, rec in data.items():
        rho = full_list_spearman(rec["config_a"], rec["config_c"])
        rows.append(
            {
                "query_id": query_id,
                "case": rec["case"],
                "spearman_full100": rho,
                "rbo_p90": rbo_by_query.get(query_id),
            }
        )
    rows.sort(key=lambda r: r["query_id"])

    print(f"{'query':6} {'case':42} {'rho_full100':>12} {'rbo.9':>8} {'rbo>rho':>8}")
    for r in rows:
        higher = "" if r["rbo_p90"] is None else ("yes" if r["rbo_p90"] > r["spearman_full100"] else "no")
        rbo_str = f"{r['rbo_p90']:.3f}" if r["rbo_p90"] is not None else "n/a"
        print(f"{r['query_id']:6} {r['case']:42} {r['spearman_full100']:12.3f} {rbo_str:>8} {higher:>8}")

    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)

    print("\nCase-level means:")
    for case, case_rows in sorted(by_case.items()):
        n = len(case_rows)
        mean_rho = statistics.mean(r["spearman_full100"] for r in case_rows)
        rbo_vals = [r["rbo_p90"] for r in case_rows if r["rbo_p90"] is not None]
        mean_rbo = statistics.mean(rbo_vals) if rbo_vals else float("nan")
        print(f"{case:42} n={n} rho_full100={mean_rho:.3f} rbo.9={mean_rbo:.3f}")

    vals = [r["spearman_full100"] for r in rows]
    print("\nOverall distribution (full-list Spearman, all 100 items):")
    print(f"mean={statistics.mean(vals):.3f} median={statistics.median(vals):.3f} min={min(vals):.3f} max={max(vals):.3f}")

    rbo_vals_all = [r["rbo_p90"] for r in rows if r["rbo_p90"] is not None]
    if rbo_vals_all:
        print("\nOverall distribution (RBO p=0.9, already computed):")
        print(
            f"mean={statistics.mean(rbo_vals_all):.3f} median={statistics.median(rbo_vals_all):.3f} "
            f"min={min(rbo_vals_all):.3f} max={max(rbo_vals_all):.3f}"
        )
        n_higher = sum(1 for r in rows if r["rbo_p90"] is not None and r["rbo_p90"] > r["spearman_full100"])
        print(f"\nRBO(.9) > full-list Spearman in {n_higher}/{len(rows)} queries")
        print(
            f"mean gap (RBO - rho_full100) = "
            f"{statistics.mean(r['rbo_p90'] - r['spearman_full100'] for r in rows):.3f}"
        )

    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
