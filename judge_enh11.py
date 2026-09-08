"""judge_enh11.py — terminal tool for manual ENH-11 relevance judging.

`docs/uat/enh11_protocol.md` §6 Phase B, with the grading scale from §4 and
the judging discipline from §5. No API calls: the person at the terminal is
the judge.

Reads `evaluation/enh11_pool.json` (the blind pool built by
build_enh11_pool.py — query text + shuffled {chunk_id, chunk_text}, nothing
else) and writes `evaluation/enh11_qrels.json`.

Discipline enforced here (§5):
  - Blind. The screen never shows which retriever found a chunk, its rank,
    its score, or any grade it was previously given — not on a `b` (back),
    not in the consistency re-check.
  - Order. Chunks are shown in the pool file's existing per-query shuffle
    order. Query order is the pool file's order by default; --shuffle-queries
    randomises it for the session (§5: "don't judge queries in Q1-Q15 order
    every session").
  - Consistency. At session end a random 10% of the chunks judged *this
    session* are re-presented unmarked; each re-judgement is stored as a
    separate `pass: 2` record so pass-1 vs pass-2 disagreement — the only
    reliability estimate a single judge has — can be read off later.

Persistence: `enh11_qrels.json` is rewritten atomically (temp file +
os.replace) after every single judgement, so the session can be interrupted
at any point — Ctrl-C, `q`, closed terminal — with zero loss. On restart,
anything already carrying a `pass: 1` record is skipped.

Per-judgement record (exactly, per §6): {query_id, chunk_id, grade,
timestamp, pass}, pass in {1, 2}.

Tooling only — nothing to run until judging begins. Commit before that.
"""

from __future__ import annotations

# F-04 / F-07: consistency with every other host-run entrypoint in this repo.
from utils.require_python import require_python

require_python()

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Chunk text from the CUDA docs carries em-dashes, ≈, ×, µ, non-breaking
# spaces and the like; the stock Windows console is cp1252 and would raise
# UnicodeEncodeError on the first such chunk. Match the other root scripts
# that "set stdout encoding before their imports" (pyproject ruff note).
for _stream in (sys.stdout, sys.stderr):
    _rc = getattr(_stream, "reconfigure", None)
    if _rc is not None:
        _rc(encoding="utf-8", errors="replace")

POOL_PATH = Path("evaluation/enh11_pool.json")
QRELS_PATH = Path("evaluation/enh11_qrels.json")

RULE = "─" * 72

# §4 — the four-point scale, shown on every prompt so it need not be memorised.
SCALE: dict[int, tuple[str, str]] = {
    3: ("directly answers the query", "read only this chunk and you have what you asked for"),
    2: ("substantively relevant", "answers part of it, or gives needed context — good to see, not the best"),
    1: ("topically related", "same subject area, but reading it does not help with the question"),
    0: ("not relevant", "different topic, or matched on vocabulary alone"),
}
SCALE_BLOCK = "\n".join(
    f"   {g}   {short:<24}  {hint}" for g, (short, hint) in sorted(SCALE.items(), reverse=True)
) + "\n   s   skip (unjudged)          b   back one          q   save & quit"

VALID_KEYS = {"0", "1", "2", "3", "s", "b", "q"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _new_doc() -> dict[str, Any]:
    return {
        "artifact": "ENH-11 graded relevance judgements",
        "protocol": "docs/uat/enh11_protocol.md §6 Phase B",
        "pool_file": str(POOL_PATH),
        "grading_scale": {str(g): short for g, (short, _hint) in SCALE.items()},
        "pass_semantics": {
            "1": "primary judgement",
            "2": "end-of-session consistency re-check (enh11_protocol.md §5)",
        },
        "sessions": [],
        "judgements": [],
    }


def _pass1_keys(doc: dict[str, Any]) -> set[tuple[str, str]]:
    return {(j["query_id"], j["chunk_id"]) for j in doc["judgements"] if j["pass"] == 1}


def _upsert_pass1(doc: dict[str, Any], query_id: str, chunk_id: str, grade: int) -> bool:
    """Store/replace the primary grade for a chunk. Returns True if it revised
    an existing grade (a `b` then re-judge), False if it is new."""
    for j in doc["judgements"]:
        if j["pass"] == 1 and j["query_id"] == query_id and j["chunk_id"] == chunk_id:
            j["grade"] = grade
            j["timestamp"] = _now()
            return True
    doc["judgements"].append(
        {"query_id": query_id, "chunk_id": chunk_id, "grade": grade, "timestamp": _now(), "pass": 1}
    )
    return False


def _append_pass2(doc: dict[str, Any], query_id: str, chunk_id: str, grade: int) -> None:
    """Consistency re-checks are never collapsed — each is its own data point."""
    doc["judgements"].append(
        {"query_id": query_id, "chunk_id": chunk_id, "grade": grade, "timestamp": _now(), "pass": 2}
    )


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")  # noqa: S605,S607 — fixed args, no shell input


def _render(header: str, query_text: str, chunk_text: str, breadcrumb: str | None) -> None:
    _clear()
    print(RULE)
    print(header)
    if breadcrumb:
        print(breadcrumb)
    print(RULE)
    print(f"\nQUERY:\n  {query_text}\n")
    print(f"CHUNK:\n{chunk_text}\n")
    print(RULE)
    print(SCALE_BLOCK)


def _pause(msg: str) -> None:
    _clear()
    print(RULE)
    print(msg)
    print(RULE)
    try:
        input("\n[enter] to continue  ")
    except EOFError:
        pass


def _prompt() -> str:
    while True:
        try:
            raw = input("grade> ").strip().lower()
        except EOFError:
            print()
            return "q"  # stdin closed mid-session -> save & quit, lose nothing
        if raw in VALID_KEYS:
            return raw
        print("  enter 0 / 1 / 2 / 3, or s (skip) / b (back) / q (quit)")


def _print_disagreement(doc: dict[str, Any]) -> None:
    """pass-1 vs latest pass-2, paired by (query_id, chunk_id), across all sessions."""
    p1 = {(j["query_id"], j["chunk_id"]): j["grade"] for j in doc["judgements"] if j["pass"] == 1}
    p2: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for j in doc["judgements"]:
        if j["pass"] == 2:
            p2.setdefault((j["query_id"], j["chunk_id"]), []).append((j["timestamp"], j["grade"]))
    if not p2:
        return
    exact = within1 = off_more = 0
    mismatches: list[tuple[str, str, int, int]] = []
    for key, entries in p2.items():
        if key not in p1:
            continue
        g1 = p1[key]
        g2 = sorted(entries)[-1][1]
        delta = abs(g1 - g2)
        if delta == 0:
            exact += 1
        elif delta == 1:
            within1 += 1
        else:
            off_more += 1
        if delta != 0:
            mismatches.append((key[0], key[1], g1, g2))
    total = exact + within1 + off_more
    if total:
        print(
            f"  self-consistency (all sessions): {exact}/{total} exact · "
            f"{within1} within 1 · {off_more} off by >1"
        )
        for qid, cid, g1, g2 in mismatches:
            print(f"    {qid:6s} {cid[:8]}  pass1={g1} → pass2={g2}")


def _build_plan(
    pool_queries: list[dict[str, Any]],
    order: list[str],
    done: set[tuple[str, str]],
) -> tuple[list[tuple[str, str, str, str, int, int]], list[str]]:
    """Flat session sequence of unjudged (query_id, chunk_id, query_text,
    chunk_text, chunk_index, chunk_total), plus the ids of queries that have
    remaining work, in plan order."""
    by_id = {q["query_id"]: q for q in pool_queries}
    plan: list[tuple[str, str, str, str, int, int]] = []
    plan_query_ids: list[str] = []
    for qid in order:
        q = by_id[qid]
        chunks = q["chunks"]
        remaining = [
            (idx, c) for idx, c in enumerate(chunks, start=1) if (qid, c["chunk_id"]) not in done
        ]
        if remaining:
            plan_query_ids.append(qid)
        for idx, c in remaining:
            plan.append((qid, c["chunk_id"], q["query"], c["chunk_text"], idx, len(chunks)))
    return plan, plan_query_ids


def _primary_phase(
    plan: list[tuple[str, str, str, str, int, int]],
    plan_query_ids: list[str],
    total_pool_chunks: int,
    doc: dict[str, Any],
    out_path: Path,
    session: dict[str, Any],
    session_pass1: list[tuple[str, str]],
) -> None:
    i = 0
    breadcrumb: str | None = None
    while i < len(plan):
        qid, cid, qtext, ctext, chunk_idx, chunk_total = plan[i]
        judged_now = len(_pass1_keys(doc))
        q_pos = plan_query_ids.index(qid) + 1
        header = (
            f"Query {qid}  ·  query {q_pos} of {len(plan_query_ids)} this session  ·  "
            f"chunk {chunk_idx} of {chunk_total}  ·  {judged_now} of {total_pool_chunks} judged"
        )
        _render(header, qtext, ctext, breadcrumb)
        key = _prompt()

        if key == "q":
            return
        if key == "b":
            if i == 0:
                breadcrumb = "◀  already at the first chunk of the session"
            else:
                i -= 1
                breadcrumb = "◀  went back one — re-judge to overwrite"
            continue
        if key == "s":
            session["skipped"] += 1
            breadcrumb = f"↷  skipped {qid} {cid[:8]} (stays unjudged)"
            i += 1
            continue

        grade = int(key)
        revised = _upsert_pass1(doc, qid, cid, grade)
        if (qid, cid) not in session_pass1:
            session_pass1.append((qid, cid))
        session["primary_judged"] = len(session_pass1)
        _atomic_write(out_path, doc)
        breadcrumb = f"✓  {qid} {cid[:8]} → {grade}" + ("  (revised)" if revised else "")
        i += 1


def _consistency_phase(
    session_pass1: list[tuple[str, str]],
    pool_by_id: dict[str, dict[str, Any]],
    doc: dict[str, Any],
    out_path: Path,
    session: dict[str, Any],
    seed: int,
) -> None:
    n = len(session_pass1)
    if n == 0:
        return
    sample_size = max(1, round(0.10 * n))
    rng = random.Random(seed ^ 0x5A5A5A5A)
    sample = rng.sample(session_pass1, min(sample_size, n))
    rng.shuffle(sample)
    session["consistency"]["sample_size"] = len(sample)
    _atomic_write(out_path, doc)

    _pause(
        f"Consistency check (enh11_protocol.md §5)\n\n"
        f"  Re-judging {len(sample)} of this session's {n} chunks, unmarked.\n"
        f"  You will not be shown the grade you gave the first time.\n"
        f"  Agreement between the two passes is your error bar — it belongs in the findings."
    )

    j = 0
    breadcrumb: str | None = None
    while j < len(sample):
        qid, cid = sample[j]
        query = pool_by_id[qid]
        ctext = next(c["chunk_text"] for c in query["chunks"] if c["chunk_id"] == cid)
        header = f"Consistency re-check  ·  {j + 1} of {len(sample)}"
        _render(header, query["query"], ctext, breadcrumb)
        key = _prompt()

        if key == "q":
            return
        if key == "b":
            if j == 0:
                breadcrumb = "◀  already at the first re-check"
            else:
                j -= 1
                breadcrumb = "◀  went back one"
            continue
        if key == "s":
            breadcrumb = f"↷  skipped re-check {qid} {cid[:8]}"
            j += 1
            continue

        _append_pass2(doc, qid, cid, int(key))
        session["consistency"]["rechecked"].append({"query_id": qid, "chunk_id": cid})
        _atomic_write(out_path, doc)
        breadcrumb = f"✓  re-checked {qid} {cid[:8]}"
        j += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual ENH-11 relevance judging — docs/uat/enh11_protocol.md §6 Phase B.",
    )
    p.add_argument("--pool", type=Path, default=POOL_PATH, help=f"pool file (default {POOL_PATH})")
    p.add_argument("--out", type=Path, default=QRELS_PATH, help=f"qrels file (default {QRELS_PATH})")
    p.add_argument(
        "--queries",
        help="comma-separated query ids to judge this session, in that order "
        "(default: every query in pool order). §7 stage 2: Q1,Q4,Q7,Q10,R1-Q7",
    )
    p.add_argument(
        "--shuffle-queries",
        action="store_true",
        help="randomise query order this session (§5 anti-fatigue). Order is recorded.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for query shuffle + consistency sample (default: random, recorded in the session)",
    )
    p.add_argument(
        "--no-consistency",
        action="store_true",
        help="skip the end-of-session consistency re-check (§5) — not recommended",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not sys.stdin.isatty():
        print(
            "judge_enh11.py needs an interactive terminal — this is manual judging, "
            "there is nothing to automate. Run it directly in a shell.",
            file=sys.stderr,
        )
        return 2

    if not args.pool.exists():
        print(f"pool file not found: {args.pool}. Run build_enh11_pool.py first.", file=sys.stderr)
        return 2

    pool = _load_json(args.pool)
    pool_queries: list[dict[str, Any]] = pool["queries"]
    pool_by_id = {q["query_id"]: q for q in pool_queries}
    total_pool_chunks = sum(len(q["chunks"]) for q in pool_queries)

    doc = _load_json(args.out) if args.out.exists() else _new_doc()

    order = [q["query_id"] for q in pool_queries]
    if args.queries:
        want = [x.strip() for x in args.queries.split(",") if x.strip()]
        unknown = [x for x in want if x not in pool_by_id]
        if unknown:
            print(f"unknown query id(s): {unknown}", file=sys.stderr)
            return 2
        order = want

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    if args.shuffle_queries:
        random.Random(seed).shuffle(order)

    done = _pass1_keys(doc)
    plan, plan_query_ids = _build_plan(pool_queries, order, done)

    if not plan:
        print(
            f"Nothing left to judge for the selected queries — "
            f"{len(done)} of {total_pool_chunks} chunks already carry a grade."
        )
        return 0

    started = _now()
    session: dict[str, Any] = {
        "session_id": started,
        "started": started,
        "ended": None,
        "query_plan": plan_query_ids,
        "seed": seed,
        "primary_judged": 0,
        "skipped": 0,
        "consistency": {"sample_size": 0, "rechecked": []},
    }
    doc["sessions"].append(session)
    session_pass1: list[tuple[str, str]] = []

    interrupted = False
    try:
        _primary_phase(
            plan, plan_query_ids, total_pool_chunks, doc, args.out, session, session_pass1
        )
        if not args.no_consistency:
            _consistency_phase(session_pass1, pool_by_id, doc, args.out, session, seed)
    except KeyboardInterrupt:
        interrupted = True

    session["ended"] = _now()
    _atomic_write(args.out, doc)

    _clear()
    print(RULE)
    print(f"Session ended {session['ended']}" + ("  (interrupted — progress saved)" if interrupted else ""))
    print(RULE)
    print(f"  judged this session : {session['primary_judged']}   (skipped {session['skipped']})")
    print(f"  consistency rechecks: {len(session['consistency']['rechecked'])}")
    print(f"  total graded        : {len(_pass1_keys(doc))} of {total_pool_chunks}")
    _print_disagreement(doc)
    print(f"  saved → {args.out}")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
