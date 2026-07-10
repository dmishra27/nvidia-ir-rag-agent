from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger()

# English stopwords covering function words that carry low information density.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "been", "being", "both", "but", "by", "can", "could",
    "did", "do", "does", "during", "each", "every", "few", "for", "from",
    "had", "has", "have", "how", "if", "in", "into", "is", "it", "its",
    "just", "may", "might", "more", "most", "no", "not", "of", "on", "only",
    "or", "other", "own", "same", "should", "so", "some", "such", "than",
    "that", "the", "then", "there", "these", "this", "those", "through",
    "to", "too", "use", "used", "using", "very", "was", "were", "what",
    "when", "where", "which", "who", "will", "with", "would",
})

QUALITY_FLAG_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class ChunkQualityResult:
    chunk_id: str
    quality_score: float
    mean_sent_len: float
    nonascii_ratio: float
    stopword_ratio: float
    flagged: bool


def score_chunk(chunk_id: str, text: str) -> ChunkQualityResult:
    """Compute a heuristic quality score for a single text chunk.

    Returns a score in [0, 1] where higher is better quality.
    Chunks below QUALITY_FLAG_THRESHOLD are flagged for review.
    """
    if not text or not text.strip():
        return ChunkQualityResult(
            chunk_id=chunk_id,
            quality_score=0.0,
            mean_sent_len=0.0,
            nonascii_ratio=1.0,
            stopword_ratio=0.0,
            flagged=True,
        )

    tokens = _tokenize(text)
    if not tokens:
        return ChunkQualityResult(
            chunk_id=chunk_id,
            quality_score=0.0,
            mean_sent_len=0.0,
            nonascii_ratio=_nonascii_ratio(text),
            stopword_ratio=0.0,
            flagged=True,
        )

    nonascii = _nonascii_ratio(text)
    stopword = _stopword_ratio(tokens)
    mean_sent = _mean_sentence_len(text)

    sent_score = _sent_len_score(mean_sent)
    nonascii_score = max(0.0, 1.0 - nonascii * 6.0)
    stopword_score = _stopword_score(stopword)

    quality = round(
        0.40 * sent_score + 0.30 * nonascii_score + 0.30 * stopword_score,
        4,
    )
    quality = max(0.0, min(1.0, quality))

    log.debug(
        "chunk_scored",
        stage="chunk_quality",
        query_id="ingestion",
        chunk_id=chunk_id,
        quality_score=quality,
        flagged=quality < QUALITY_FLAG_THRESHOLD,
    )

    return ChunkQualityResult(
        chunk_id=chunk_id,
        quality_score=quality,
        mean_sent_len=round(mean_sent, 2),
        nonascii_ratio=round(nonascii, 4),
        stopword_ratio=round(stopword, 4),
        flagged=quality < QUALITY_FLAG_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def _nonascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def _stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _STOPWORDS) / len(tokens)


def _mean_sentence_len(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return float(len(_tokenize(text)))
    lengths = [len(_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def _sent_len_score(mean_len: float) -> float:
    """Score sentence length: ideal range 8–25 tokens."""
    if mean_len < 3:
        return 0.10
    if mean_len < 8:
        return 0.50 + (mean_len - 3) * 0.08
    if mean_len <= 25:
        return 1.00
    if mean_len <= 50:
        return 1.00 - (mean_len - 25) * 0.020
    return 0.30


def _stopword_score(ratio: float) -> float:
    """Score stopword ratio: ideal range 0.20–0.55."""
    if ratio < 0.10:
        return 0.40   # likely pure code / tokenisation artefact
    if ratio < 0.20:
        return 0.70
    if ratio <= 0.55:
        return 1.00
    if ratio <= 0.75:
        return 1.00 - (ratio - 0.55) * 2.0
    return 0.20       # pure filler text
