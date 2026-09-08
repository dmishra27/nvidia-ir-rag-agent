"""ENH-11 Phase A — build the graded-relevance candidate pool.

`docs/uat/enh11_protocol.md` §6 Phase A. This script produces the *input*
for manual judging. It makes no relevance judgements and calls no API.

For each of the 15 Round 2 superiority queries
(`docs/uat/uat_superiority_cases_raw.json`) plus R1-Q7 (`shader processor
count`, supplementary) — the same 16-query anchor set as `run_dqr_eval.py`
and `run_b3_b4_fusion_eval.py` — retrieve three ways with the LITERAL
query:

  bm25   top 10 — BM25Okapi over chunk_text (local pickle)
  dense  top 10 — e5-base-v2 + Qdrant cosine
  rrf    top 10 — reciprocal rank fusion, k=60, over each retriever's
                  top-100 pool (the pipeline default; matches B3/B4)

The three top-10 lists are unioned per query and deduplicated by
`chunk_id`. Two files are emitted and MUST NOT be merged:

  evaluation/enh11_pool.json
      The judging input. Per query: the query text and a list of
      {chunk_id, chunk_text}, shuffled, with NO rank, score, or
      source-retriever information (enh11_protocol.md §5, "Blind").
      chunk_text is the authoritative full text from Postgres.

  evaluation/enh11_pool_provenance.json
      For post-judgement analysis only. Per query, each pooled chunk's
      rank in BM25 / dense / RRF (1-indexed within each retriever's
      top-100; null = absent from that top-100), plus which retrievers'
      top-10 put it in the pool. Provenance reaching the judging
      interface is the exact contamination ENH-11 exists to remove.

Pool depth is capped at top 10 per retriever (enh11_protocol.md §3.1).
A future reader must know this: a chunk_id absent from enh11_pool.json
was never judged — it is NOT a judged zero.

Memory (host rule + DEF-23): the dense encoder is the only heavy step.
It is loaded once, used for one retrieval pass, then released before any
fusion arithmetic or file I/O. On MemoryError the script stops and
reports rather than retrying.

Commit this script before running it, and the two output files before any
analysis — per 0149ca4.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version.
from utils.require_python import require_python

require_python()

import gc
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse
from schema.models import Chunk, get_engine, get_session_factory

load_dotenv()

CASES_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
POOL_JSON = Path("evaluation/enh11_pool.json")
PROVENANCE_JSON = Path("evaluation/enh11_pool_provenance.json")

TOP_N = 10  # pool cap per retriever — enh11_protocol.md §3.1
FUSION_POOL_DEPTH = 100  # BM25/dense depth fed to RRF — matches B3/B4
RRF_K = 60  # pipeline default

R1_Q7 = {"query_id": "R1-Q7", "query": "shader processor count", "supplementary": True}


def load_queries() -> list[dict]:
    """15 Round 2 anchor queries + R1-Q7 supplementary — identical selection
    to run_dqr_eval.py / run_b3_b4_fusion_eval.py."""
    rows = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    queries = [
        {
            "query_id": r["query_id"],
            "query": r["query"],
            "case": r["case"],
            "supplementary": False,
        }
        for r in rows
    ]
    queries.append({**R1_Q7, "case": "case4_bm25_failure_dense_advantage"})
    return queries


def _rank_of(chunk_id: str, results: list[Candidate]) -> int | None:
    for c in results:
        if c.chunk_id == chunk_id:
            return c.rank
    return None


def retrieve_pools(
    queries: list[dict], bm25: BM25Index, dense: DenseIndex
) -> tuple[dict[str, list[Candidate]], dict[str, list[Candidate]]]:
    """The one heavy stage: encoder in memory. Retrieve BM25 and dense
    top-`FUSION_POOL_DEPTH` for every query, then return so the caller can
    release the encoder before doing anything else."""
    bm25_pools = {q["query_id"]: bm25.search(q["query"], top_k=FUSION_POOL_DEPTH) for q in queries}
    dense_pools = {q["query_id"]: dense.search(q["query"], top_k=FUSION_POOL_DEPTH) for q in queries}
    return bm25_pools, dense_pools


def build_query_pool(
    bm25_hits: list[Candidate], dense_hits: list[Candidate]
) -> tuple[list[str], dict[str, dict]]:
    """Union the three top-10 lists (BM25, dense, RRF-over-top-100),
    dedup by chunk_id preserving first-seen order, and record provenance.

    Returns (ordered_unique_chunk_ids, provenance_by_chunk_id) where each
    provenance entry is {bm25_rank, dense_rank, rrf_rank, pooled_from}.
    """
    rrf_full = fuse(bm25_hits, dense_hits, top_k=FUSION_POOL_DEPTH, k=RRF_K)

    top10 = {
        "bm25": bm25_hits[:TOP_N],
        "dense": dense_hits[:TOP_N],
        "rrf": rrf_full[:TOP_N],
    }

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for name in ("bm25", "dense", "rrf"):
        for c in top10[name]:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                ordered_ids.append(c.chunk_id)

    top10_ids = {name: {c.chunk_id for c in hits} for name, hits in top10.items()}
    provenance = {
        cid: {
            "bm25_rank": _rank_of(cid, bm25_hits),
            "dense_rank": _rank_of(cid, dense_hits),
            "rrf_rank": _rank_of(cid, rrf_full),
            "pooled_from": [name for name in ("bm25", "dense", "rrf") if cid in top10_ids[name]],
        }
        for cid in ordered_ids
    }
    return ordered_ids, provenance


def fetch_chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    """Authoritative full chunk text from Postgres, joined by chunk_id."""
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        rows = session.execute(
            select(Chunk.chunk_id, Chunk.chunk_text).where(Chunk.chunk_id.in_(chunk_ids))
        ).all()
    texts = {r.chunk_id: r.chunk_text for r in rows}
    missing = chunk_ids - texts.keys()
    if missing:
        raise SystemExit(
            f"{len(missing)} pooled chunk_id(s) absent from Postgres chunks table: "
            f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}. "
            "Pool and corpus are out of sync — not writing partial output."
        )
    return texts


def _git_commit_of(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", path],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    queries = load_queries()
    print(f"Loaded {len(queries)} queries (15 Round 2 + R1-Q7 supplementary).")
    print(f"Pool = union of BM25 top-{TOP_N}, dense top-{TOP_N}, RRF top-{TOP_N} "
          f"(RRF k={RRF_K} over each retriever's top-{FUSION_POOL_DEPTH}).\n")

    print("Loading BM25 index (local pickle)...")
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)

    print("Connecting to live Qdrant + loading e5-base-v2 query encoder (heavy step)...\n")
    try:
        dense = DenseIndex.connect()
        bm25_pools, dense_pools = retrieve_pools(queries, bm25, dense)
    except MemoryError:
        print(
            "\nMemoryError during the dense-encoder stage. Stopping without retry "
            "(host rule / DEF-23). Free memory and re-run — no output was written.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    del dense
    gc.collect()
    print("Encoder released. Building pools + provenance (no model in memory)...\n")

    pool_queries: list[dict] = []
    prov_queries: list[dict] = []
    all_chunk_ids: set[str] = set()
    per_query_ordered: dict[str, list[str]] = {}
    per_query_prov: dict[str, dict] = {}

    for q in queries:
        qid = q["query_id"]
        ordered_ids, provenance = build_query_pool(bm25_pools[qid], dense_pools[qid])
        per_query_ordered[qid] = ordered_ids
        per_query_prov[qid] = provenance
        all_chunk_ids.update(ordered_ids)

    texts = fetch_chunk_texts(all_chunk_ids)

    total_unique = 0
    for q in queries:
        qid = q["query_id"]
        ordered_ids = per_query_ordered[qid]
        provenance = per_query_prov[qid]
        n = len(ordered_ids)
        total_unique += n

        # --- judging input: shuffled {chunk_id, chunk_text}, nothing else ---
        rng = random.Random(f"enh11-pool:{qid}")
        chunks = [{"chunk_id": cid, "chunk_text": texts[cid]} for cid in ordered_ids]
        rng.shuffle(chunks)
        pool_queries.append({"query_id": qid, "query": q["query"], "chunks": chunks})

        # --- provenance: rank-ordered, deliberately NOT the shuffled order ---
        big = FUSION_POOL_DEPTH + 1
        prov_chunks = sorted(
            (
                {
                    "chunk_id": cid,
                    "bm25_rank": provenance[cid]["bm25_rank"],
                    "dense_rank": provenance[cid]["dense_rank"],
                    "rrf_rank": provenance[cid]["rrf_rank"],
                    "pooled_from": provenance[cid]["pooled_from"],
                }
                for cid in ordered_ids
            ),
            key=lambda r: (
                r["rrf_rank"] or big,
                r["bm25_rank"] or big,
                r["dense_rank"] or big,
                r["chunk_id"],
            ),
        )
        top10_counts = {
            name: sum(1 for cid in ordered_ids if name in provenance[cid]["pooled_from"])
            for name in ("bm25", "dense", "rrf")
        }
        prov_queries.append(
            {
                "query_id": qid,
                "query": q["query"],
                "case": q["case"],
                "supplementary": q["supplementary"],
                "unique_chunk_count": n,
                "top10_present": top10_counts,
                "chunks": prov_chunks,
            }
        )
        print(
            f"{qid:6s} unique={n:3d}  "
            f"(bm25 {top10_counts['bm25']} | dense {top10_counts['dense']} | rrf {top10_counts['rrf']})"
        )

    print(f"\nTotal unique chunks across {len(queries)} queries: {total_unique}")

    pool_doc = {
        "artifact": "ENH-11 Phase A candidate pool — judging input",
        "protocol": "docs/uat/enh11_protocol.md §6 Phase A",
        "pool_cap_per_retriever": TOP_N,
        "grading_scale": "docs/uat/enh11_protocol.md §4 (grades 0-3)",
        "note": (
            "Chunks are shuffled per query. No rank, score, or source-retriever "
            "information is present, by design (enh11_protocol.md §5 'Blind'). A "
            "chunk_id absent here was never judged — not a judged zero (§3.1). "
            "Provenance lives in evaluation/enh11_pool_provenance.json and must "
            "never be shown during judging."
        ),
        "queries": pool_queries,
    }
    prov_doc = {
        "artifact": "ENH-11 Phase A candidate pool — provenance (post-judgement analysis ONLY)",
        "protocol": "docs/uat/enh11_protocol.md §6 Phase A",
        "warning": (
            "NEVER expose this file to the judging interface. Provenance in the "
            "judging input is the exact contamination ENH-11 exists to remove "
            "(enh11_protocol.md §1, §5)."
        ),
        "generated": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "script_git_commit": _git_commit_of("build_enh11_pool.py"),
            "anchor_set": (
                "15 Round 2 (docs/uat/uat_superiority_cases_raw.json) + R1-Q7 "
                "'shader processor count' (supplementary)"
            ),
            "params": {
                "top_n_per_retriever": TOP_N,
                "fusion_pool_depth": FUSION_POOL_DEPTH,
                "rrf_k": RRF_K,
            },
            "retrievers": {
                "bm25": f"retrieval.bm25_index.BM25Index ({DEFAULT_INDEX_PATH})",
                "dense": "retrieval.dense_index.DenseIndex — e5-base-v2 + Qdrant 'nvidia_ir_chunks'",
                "rrf": "retrieval.rrf_fusion.fuse(bm25_top100, dense_top100, k=60)",
            },
            "chunk_text_source": "Postgres chunks.chunk_text (authoritative full text), joined by chunk_id",
            "rank_domain": (
                "bm25_rank / dense_rank / rrf_rank are 1-indexed positions within each "
                f"retriever's top-{FUSION_POOL_DEPTH}; null = absent from that top-"
                f"{FUSION_POOL_DEPTH}. pooled_from lists which retrievers' top-{TOP_N} "
                "placed the chunk in the pool."
            ),
            "shuffle": (
                "enh11_pool.json chunk order is random.Random('enh11-pool:'+query_id)"
                ".shuffle(...) — deterministic and reproducible from this script."
            ),
        },
        "total_unique_chunks": total_unique,
        "queries": prov_queries,
    }

    POOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps(pool_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    PROVENANCE_JSON.write_text(json.dumps(prov_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {POOL_JSON}")
    print(f"Saved -> {PROVENANCE_JSON}")
    print("\nNext: commit both files before any analysis (0149ca4), then Phase B (judge_enh11.py).")


if __name__ == "__main__":
    main()
