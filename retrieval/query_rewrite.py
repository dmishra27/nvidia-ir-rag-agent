"""Conditional query rewriting for the retrieval front end — hypothesis D-QR.

Two deterministic, LLM-free strategies (the Anthropic key is out of credit,
and a rule-based rewriter reproduces straight from the repo, per the
`0149ca4` lesson):

1. **Legacy-terminology expansion.** A static map of CUDA vocabulary-era
   mismatches. The documented case (`docs/uat/uat_day5_retrieval.md`, "the
   most important finding of the UAT"): ``shader processor count`` retrieves
   nothing because the corpus only ever says "CUDA cores"; dense found
   ``An SM consists of: 128 CUDA cores`` and RRF discarded it. Matched
   modern terms are appended to the **dense** query; BM25 keeps the literal.

2. **camelCase identifier splitting.** ``cudaDeviceSynchronize`` also
   contributes ``cuda device synchronize`` to the dense query.

The **gate**: a query naming a CUDA API symbol, error code or struct is an
``exact_identifier`` lookup. ``rewrite_query(..., gate=True)`` leaves those
completely untouched on both sides — paraphrasing away the decisive lexical
token is the A4 failure mode. ``gate=False`` rewrites everything; that is
the ungated arm D-QR measures against.

Nothing here is wired into `agents/` or the API. Whether it should be is
what D-QR decides — see `docs/uat/round3_dqr_findings.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Terminology-era synonyms {legacy phrase: modern corpus phrase}, matched
# case-insensitively on word boundaries. Deliberately small — this is a
# hypothesis probe, not a thesaurus.
LEGACY_EXPANSIONS: dict[str, str] = {
    "shader processor": "CUDA core",
    "shader processors": "CUDA cores",
    "shading unit": "CUDA core",
    "shading units": "CUDA cores",
    "streaming processor": "CUDA core",
    "streaming processors": "CUDA cores",
    "gpu program": "kernel",
    "gpu programs": "kernels",
    "video memory": "global memory",
    "graphics memory": "global memory",
    "vram": "global memory",
}

# CUDA API symbols / error codes / structs / execution-space qualifiers.
# Any match => exact-identifier lookup, gated out of rewriting.
_IDENTIFIER_RE = re.compile(
    r"\b("
    r"cuda[A-Z]\w+"          # cudaMalloc, cudaDeviceSynchronize, cudaErrorInvalidValue
    r"|cu[A-Z][a-z]\w+"      # cuMemAlloc (driver API); cu[A-Z][a-z] avoids matching "CUDA"
    r"|cudnn\w+|cublas\w+|cufft\w+|curand\w+"
    r"|dim3|uint3|char4|float4|double2"
    r"|__[a-z]+__"           # __host__, __device__, __global__
    r")\b"
)

# camelCase word boundary: lower/digit immediately followed by upper.
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_TOKEN_RE = re.compile(r"\b[a-z]+[A-Z]\w*\b")


@dataclass(frozen=True)
class RewriteResult:
    literal_query: str
    bm25_query: str  # always == literal_query
    dense_query: str  # literal, or literal + appended expansions/splits
    rewritten: bool
    strategy: str  # gated-skip | legacy-expansion | identifier-split | legacy+split | no-op
    detail: str


def classify(query: str) -> str:
    """``exact_identifier`` if the query names a CUDA symbol, else ``conceptual``."""
    return "exact_identifier" if _IDENTIFIER_RE.search(query) else "conceptual"


def _legacy_expansions(query: str) -> list[str]:
    hits = [
        modern
        for legacy, modern in LEGACY_EXPANSIONS.items()
        if re.search(rf"\b{re.escape(legacy)}\b", query, flags=re.IGNORECASE)
    ]
    return list(dict.fromkeys(hits))


def _identifier_splits(query: str) -> list[str]:
    splits = []
    for token in _CAMEL_TOKEN_RE.findall(query):
        split = _CAMEL_SPLIT_RE.sub(" ", token).lower()
        if split != token.lower():
            splits.append(split)
    return list(dict.fromkeys(splits))


def rewrite_query(query: str, *, gate: bool = True) -> RewriteResult:
    """Rewrite ``query`` for the dense retriever; BM25 always gets the literal.

    ``gate=True`` (the conditional arm) returns the literal query untouched
    for exact-identifier lookups. ``gate=False`` (the ungated arm) applies
    both strategies to every query.
    """
    literal = query.strip()

    if gate and classify(literal) == "exact_identifier":
        return RewriteResult(
            literal, literal, literal, False, "gated-skip",
            "exact-identifier query left untouched",
        )

    expansions = _legacy_expansions(literal)
    splits = [s for s in _identifier_splits(literal) if s not in expansions]
    additions = expansions + splits

    if not additions:
        return RewriteResult(
            literal, literal, literal, False, "no-op",
            "no legacy term or camelCase identifier",
        )

    dense_query = f"{literal} {' '.join(additions)}"
    if expansions and splits:
        strategy = "legacy+split"
    elif expansions:
        strategy = "legacy-expansion"
    else:
        strategy = "identifier-split"
    detail = "; ".join(
        part
        for part in (
            f"expand: {', '.join(expansions)}" if expansions else "",
            f"split: {', '.join(splits)}" if splits else "",
        )
        if part
    )
    return RewriteResult(literal, literal, dense_query, True, strategy, detail)
