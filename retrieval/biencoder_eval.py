"""Evaluate candidate bi-encoders in dense-only mode over a judged query set.

Compares all-MiniLM-L6-v2 (speed baseline), intfloat/e5-base-v2 (quality
candidate), and BAAI/bge-m3 (dense-mode only, quality ceiling) by embedding
every chunk in Postgres and running cosine-similarity search for 10 hand
judged queries. The winner (highest mean NDCG@10) becomes the dense
component for RRF fusion.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from schema.models import Chunk, get_engine, get_session_factory

load_dotenv()
log = structlog.get_logger()

torch.set_num_threads(os.cpu_count() or 4)

RESULTS_PATH = Path("biencoder_eval_results.json")
TOP_K = 10


@dataclass(frozen=True)
class BiEncoderConfig:
    label: str
    hf_id: str
    param_count: str
    query_prefix: str = ""
    passage_prefix: str = ""


MODELS: list[BiEncoderConfig] = [
    BiEncoderConfig(
        label="all-MiniLM-L6-v2",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        param_count="22M",
    ),
    BiEncoderConfig(
        label="e5-base-v2",
        hf_id="intfloat/e5-base-v2",
        param_count="109M",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    BiEncoderConfig(
        label="bge-m3",
        hf_id="BAAI/bge-m3",
        param_count="570M",
    ),
]


@dataclass(frozen=True)
class EvalQuery:
    query: str
    # chunk_id -> graded relevance: 2 = exact/direct match, 1 = related context
    relevance: dict[str, int]


# ---------------------------------------------------------------------------
# Judged query set — 10 queries hand-curated against the 5,389-chunk corpus
# (CUDA C++ Programming Guide, CUDA C++ Best Practices Guide, CUDA Math API
# Reference, CUDA Runtime API Reference, Nsight Systems User Guide) by
# keyword search over chunk_text, then manually graded for relevance.
# ---------------------------------------------------------------------------

QUERIES: list[EvalQuery] = [
    EvalQuery(
        "NVLink bandwidth",
        {
            "97724de868d508c6796761ea": 2,
            "d9a29915d184c931b72b7c4d": 2,
        },
    ),
    EvalQuery(
        "warp divergence",
        {
            "3c9b5ecfb38e9ee9377cf4f6": 2,
        },
    ),
    EvalQuery(
        "shared memory bank conflicts",
        {
            "18c3a31876811522d4624338": 2,
            "2af2a6d68ac02bf8e1e497b6": 2,
            "1fabb00b8793f603b648e7d1": 2,
            "b2e88edc44a1bbffcdf95e77": 1,
            "f6f2063c3d7aac6f24143411": 1,
        },
    ),
    EvalQuery(
        "Nsight Systems timeline view",
        {
            "f97913ebc90e0cd6115c8eab": 2,
            "26e4bfdcf367327432c7ad70": 1,
        },
    ),
    EvalQuery(
        "CUDA kernel launch grid and block dimensions",
        {
            "dae82243c874335c58aa2eac": 2,
            "de728073e9ac45415dc227b6": 2,
            "057ff658082ffbb6a540fd32": 1,
            "3f9cc2dec97efbd20ff30738": 1,
        },
    ),
    EvalQuery(
        "FP8 tensor core performance",
        {
            "589b8ba34fb76ef7c2defd0d": 2,
            "342b83d013245cf167657e47": 2,
        },
    ),
    EvalQuery(
        "occupancy calculator CUDA",
        {
            "482adc6bf4e72c1fc04b542b": 2,
            "35dfc6787549aaeda1677954": 2,
            "ec8029317b8292517dacd53d": 1,
            "d13f2bd5a87e1e8fc0f4fece": 1,
        },
    ),
    EvalQuery(
        "CUDA unified memory managed allocation",
        {
            "4e89e946202c5cfec0c12501": 2,
            "7f9e634f3510db828e6088fd": 2,
            "3ddd571f8cec6d1fc5ad1de4": 1,
            "7e26a8ede712bcfa914fc6a1": 1,
        },
    ),
    EvalQuery(
        "cudaStreamSynchronize stream synchronization",
        {
            "c7b53a7eb487fe71105f8819": 2,
            "e22cf03a80b046d28bca450d": 2,
            "08b3e89a40fb9324527888fb": 1,
            "dfa8b65207ac79e515ce47ca": 1,
        },
    ),
    EvalQuery(
        "cudaMallocAsync memory pool allocation",
        {
            "b9f6333b7c7155466507d096": 2,
            "0308df5e01a079bf11a7ad45": 2,
            "e958dbc3a53359972682a078": 1,
            "bcc333ff83bd271b96dfd3d9": 1,
        },
    ),
]


def _dcg(gains: list[int]) -> float:
    return sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_chunk_ids: list[str], judgments: dict[str, int], k: int = TOP_K) -> float:
    gains = [judgments.get(cid, 0) for cid in ranked_chunk_ids[:k]]
    dcg = _dcg(gains)
    ideal_gains = sorted(judgments.values(), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    return dcg / idcg if idcg > 0 else 0.0


def load_corpus() -> tuple[list[str], list[str]]:
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        rows = session.execute(select(Chunk.chunk_id, Chunk.chunk_text)).all()
    return [r.chunk_id for r in rows], [r.chunk_text for r in rows]


def load_corpus_with_doc_ids() -> tuple[list[str], list[str], list[str]]:
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        rows = session.execute(select(Chunk.chunk_id, Chunk.chunk_text, Chunk.doc_id)).all()
    return [r.chunk_id for r in rows], [r.chunk_text for r in rows], [r.doc_id for r in rows]


def judged_chunk_ids() -> set[str]:
    ids: set[str] = set()
    for eq in QUERIES:
        ids.update(eq.relevance.keys())
    return ids


def build_stratified_sample(
    chunk_ids: list[str],
    texts: list[str],
    doc_ids: list[str],
    target_size: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Sample ~target_size chunks, proportionally stratified by doc_id.

    All chunks referenced in QUERIES' relevance judgments are force-included
    so a sampled run can't be unfairly penalized for excluding a known
    relevant chunk; the remainder is filled by proportional random draw
    per document.
    """
    import random

    rng = random.Random(seed)
    by_doc: dict[str, list[int]] = {}
    for i, doc_id in enumerate(doc_ids):
        by_doc.setdefault(doc_id, []).append(i)

    forced = judged_chunk_ids()
    selected_idx: set[int] = {i for i, cid in enumerate(chunk_ids) if cid in forced}

    remaining_target = max(target_size - len(selected_idx), 0)
    total = len(chunk_ids)
    for doc_id, idxs in by_doc.items():
        pool = [i for i in idxs if i not in selected_idx]
        doc_share = round(remaining_target * len(idxs) / total)
        rng.shuffle(pool)
        selected_idx.update(pool[:doc_share])

    ordered = sorted(selected_idx)
    return [chunk_ids[i] for i in ordered], [texts[i] for i in ordered]


def evaluate_model(
    config: BiEncoderConfig,
    chunk_ids: list[str],
    texts: list[str],
    batch_size: int = 64,
    sampled: bool = False,
    full_corpus_size: int | None = None,
) -> dict[str, Any]:
    log.info(
        "biencoder_eval_start",
        stage="biencoder_eval",
        query_id="eval",
        model=config.label,
        hf_id=config.hf_id,
        batch_size=batch_size,
    )
    model = SentenceTransformer(config.hf_id, device="cpu")

    passages = [config.passage_prefix + t for t in texts]
    t0 = time.perf_counter()
    corpus_emb = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embed_seconds = time.perf_counter() - t0

    per_query_ndcg: dict[str, float] = {}
    latencies_ms: list[float] = []
    sample_top5: list[str] = []

    for eq in QUERIES:
        t0 = time.perf_counter()
        q_emb = model.encode(
            [config.query_prefix + eq.query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        scores = corpus_emb @ q_emb
        top_idx = np.argsort(-scores)[:TOP_K]
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        ranked_ids = [chunk_ids[i] for i in top_idx]
        per_query_ndcg[eq.query] = round(ndcg_at_k(ranked_ids, eq.relevance), 4)
        if eq.query == "NVLink bandwidth":
            sample_top5 = ranked_ids[:5]

    result = {
        "model": config.label,
        "hf_id": config.hf_id,
        "param_count": config.param_count,
        "corpus_embed_seconds": round(embed_seconds, 2),
        "mean_ndcg_at_10": round(float(np.mean(list(per_query_ndcg.values()))), 4),
        "mean_query_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_query_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "per_query_ndcg": per_query_ndcg,
        "sample_top5_nvlink_bandwidth": sample_top5,
        "sampled": sampled,
        "sample_size": len(chunk_ids) if sampled else None,
        "full_corpus_size": full_corpus_size if sampled else None,
    }
    log.info(
        "biencoder_eval_done",
        stage="biencoder_eval",
        query_id="eval",
        model=config.label,
        mean_ndcg_at_10=result["mean_ndcg_at_10"],
        p95_query_latency_ms=result["p95_query_latency_ms"],
        corpus_embed_seconds=result["corpus_embed_seconds"],
    )
    return result


# Models excluded from the comparison after a confirmed hardware failure, with
# the reason documented so the storyline/results stay honest about why.
DEFERRED_MODELS: dict[str, str] = {
    "bge-m3": (
        "Blocked by hardware: OOM during model load on this 8GB CPU-only host "
        '("memory allocation of 67067869 bytes failed") even at batch_size=8 over '
        "a 1,000-chunk doc-stratified sample. The failure happened while loading "
        "model weights/tokenizer, before any batching — not a speed problem that "
        "smaller batches could fix. Deferred until GPU or cloud compute is available."
    ),
}


def _load_existing_results() -> dict[str, dict[str, Any]]:
    """Load previously saved per-model results, keyed by model label, if any."""
    if not RESULTS_PATH.exists():
        return {}
    data = json.loads(RESULTS_PATH.read_text())
    return {r["model"]: r for r in data.get("results", [])}


def _write_results(num_chunks: int, results_by_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = [results_by_label[c.label] for c in MODELS if c.label in results_by_label]
    deferred = [
        {"model": c.label, "hf_id": c.hf_id, "param_count": c.param_count, "reason": DEFERRED_MODELS[c.label]}
        for c in MODELS
        if c.label not in results_by_label and c.label in DEFERRED_MODELS
    ]
    output: dict[str, Any] = {
        "num_chunks": num_chunks,
        "num_queries": len(QUERIES),
        "results": results,
        "deferred": deferred,
    }
    if results and len(results) + len(deferred) == len(MODELS):
        winner = max(results, key=lambda r: r["mean_ndcg_at_10"])
        output["winner"] = winner["model"]
        output["winner_rationale"] = (
            f"{winner['model']} achieved the highest mean NDCG@10 "
            f"({winner['mean_ndcg_at_10']}) among the models that could run on this "
            f"hardware (bge-m3 excluded — see 'deferred')."
        )
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=[c.label for c in MODELS], default=None,
        help="Run only this model and merge into any existing results file.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Evaluate on a doc-stratified random sample of this many chunks instead of "
        "the full corpus (all judged chunks are force-included). Use for large models "
        "where full-corpus embedding is too slow for the available hardware.",
    )
    args = parser.parse_args()

    full_chunk_ids, full_texts, doc_ids = load_corpus_with_doc_ids()
    full_corpus_size = len(full_chunk_ids)
    log.info(
        "biencoder_eval_corpus_loaded", stage="biencoder_eval", query_id="eval",
        num_chunks=full_corpus_size,
    )

    if args.sample_size:
        chunk_ids, texts = build_stratified_sample(
            full_chunk_ids, full_texts, doc_ids, args.sample_size
        )
        log.info(
            "biencoder_eval_sampled", stage="biencoder_eval", query_id="eval",
            sample_size=len(chunk_ids), full_corpus_size=full_corpus_size,
        )
    else:
        chunk_ids, texts = full_chunk_ids, full_texts

    results_by_label = _load_existing_results()
    configs = [c for c in MODELS if c.label == args.model] if args.model else MODELS

    for config in configs:
        result = evaluate_model(
            config,
            chunk_ids,
            texts,
            batch_size=args.batch_size,
            sampled=bool(args.sample_size),
            full_corpus_size=full_corpus_size,
        )
        results_by_label[config.label] = result
        _write_results(full_corpus_size, results_by_label)

    output = _write_results(full_corpus_size, results_by_label)
    results = output["results"]

    print(f"\n{'Model':<18}{'Params':<8}{'NDCG@10':<10}{'p95 ms':<10}{'Embed s':<10}{'Corpus':<10}")
    print("-" * 66)
    winner_label = output.get("winner")
    for r in results:
        marker = " *" if r["model"] == winner_label else ""
        corpus_note = f"sample {r['sample_size']}" if r.get("sampled") else "full"
        print(
            f"{r['model']:<18}{r['param_count']:<8}{r['mean_ndcg_at_10']:<10}"
            f"{r['p95_query_latency_ms']:<10}{r['corpus_embed_seconds']:<10}{corpus_note:<10}{marker}"
        )
    if winner_label:
        print(f"\nWinner: {winner_label} — {output['winner_rationale']}")
    else:
        done = {r["model"] for r in results}
        remaining = [c.label for c in MODELS if c.label not in done]
        print(f"\n{len(results)}/{len(MODELS)} models done. Remaining: {remaining}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
