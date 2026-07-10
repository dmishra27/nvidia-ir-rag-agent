"""Unit tests for retrieval/chunk_quality.py.

All functions are deterministic — no mocking required.
Tests are grouped by concern: edge cases, non-ASCII penalty,
clean technical text, flagging invariant, and each internal helper.
"""

from __future__ import annotations

import dataclasses

import pytest

from retrieval.chunk_quality import (
    QUALITY_FLAG_THRESHOLD,
    ChunkQualityResult,
    _mean_sentence_len,
    _nonascii_ratio,
    _sent_len_score,
    _stopword_ratio,
    _stopword_score,
    _tokenize,
    score_chunk,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable text samples
# ---------------------------------------------------------------------------

_CUDA_PARA = (
    "CUDA kernels are launched with a grid of thread blocks, where each block "
    "contains a configurable number of threads. Shared memory within a block "
    "enables low-latency communication between threads. The warp scheduler "
    "executes 32 threads simultaneously, hiding memory latency through context "
    "switching between warps. Efficient GPU programs maximise arithmetic "
    "intensity to keep the compute units busy."
)

_TENSORRT_PARA = (
    "TensorRT optimises deep learning inference by fusing layers, selecting "
    "the most efficient kernel implementations, and calibrating quantisation "
    "parameters from a representative dataset. The engine is serialised to "
    "disk and loaded at runtime, providing consistent sub-millisecond latency "
    "on NVIDIA GPUs across multiple inference requests."
)

_H100_PARA = (
    "The H100 Tensor Core GPU delivers up to 4 petaFLOPS of FP8 performance "
    "using the new Transformer Engine. NVLink 4.0 provides 900 GB/s of "
    "bandwidth between GPUs in an NVLink domain."
)

# Garbled text: heavy accented/Latin-extended chars produce high nonascii_ratio
_GARBLED_ACCENTS = (
    "CUdA \xfc\xf1iFi\xebd M\xeb M\xf6R\xff \xe0Ll\xf6WS th\xeb G\xdeU."
    " \xe0\xf1D cP\xfc t\xf6 Sh\xc0r\xc8."
)

# Heavy CJK mix with minimal ASCII tokens — exercises both nonascii and sentence-length penalties
_CJK_MIXED = "的的的的的 abc. 的的的的. de."


# ---------------------------------------------------------------------------
# ChunkQualityResult contract
# ---------------------------------------------------------------------------

class TestChunkQualityResultContract:
    def test_result_is_frozen(self) -> None:
        r = score_chunk("x", "CUDA memory.")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.quality_score = 99.0  # type: ignore[misc]

    def test_chunk_id_is_preserved_on_non_empty_text(self) -> None:
        r = score_chunk("my-specific-id", "GPU kernel launch configuration.")
        assert r.chunk_id == "my-specific-id"

    def test_chunk_id_is_preserved_on_empty_text(self) -> None:
        r = score_chunk("preserve-me", "")
        assert r.chunk_id == "preserve-me"

    def test_quality_score_always_in_unit_interval(self) -> None:
        texts = ["", "   ", "∑∫∂", _CJK_MIXED, _CUDA_PARA, "a"]
        for text in texts:
            r = score_chunk("bound", text)
            assert 0.0 <= r.quality_score <= 1.0, (
                f"quality_score={r.quality_score!r} out of [0, 1] for text={text!r}"
            )


# ---------------------------------------------------------------------------
# Empty and whitespace inputs
# ---------------------------------------------------------------------------

class TestEmptyAndWhitespace:
    def test_empty_string_scores_zero(self) -> None:
        r = score_chunk("c1", "")
        assert r.quality_score == 0.0

    def test_empty_string_is_flagged(self) -> None:
        r = score_chunk("c1", "")
        assert r.flagged is True

    def test_whitespace_only_scores_zero(self) -> None:
        r = score_chunk("c2", "   \n\t  ")
        assert r.quality_score == 0.0

    def test_whitespace_only_is_flagged(self) -> None:
        r = score_chunk("c2", "   \n\t  ")
        assert r.flagged is True

    def test_pure_nonalpha_unicode_scores_zero(self) -> None:
        """Symbols with no alphanumeric tokens produce zero quality."""
        r = score_chunk("c3", "∑∫∂≈→∞")
        assert r.quality_score == 0.0

    def test_pure_nonalpha_unicode_is_flagged(self) -> None:
        r = score_chunk("c3", "∑∫∂≈→∞")
        assert r.flagged is True


# ---------------------------------------------------------------------------
# Non-ASCII penalty
# ---------------------------------------------------------------------------

class TestNonAsciiPenalty:
    def test_clean_ascii_text_has_zero_nonascii_ratio(self) -> None:
        r = score_chunk("clean", _CUDA_PARA)
        assert r.nonascii_ratio == 0.0

    def test_garbled_accented_text_has_elevated_nonascii_ratio(self) -> None:
        r = score_chunk("garbled", _GARBLED_ACCENTS)
        assert r.nonascii_ratio > 0.20

    def test_high_nonascii_text_scores_below_clean_equivalent(self) -> None:
        """The garbled variant must score strictly lower than the clean CUDA paragraph."""
        r_clean = score_chunk("clean", _CUDA_PARA)
        r_garbled = score_chunk("garbled", _GARBLED_ACCENTS)
        assert r_garbled.quality_score < r_clean.quality_score

    def test_heavy_cjk_mix_produces_high_nonascii_ratio(self) -> None:
        r = score_chunk("cjk", _CJK_MIXED)
        assert r.nonascii_ratio > 0.40

    def test_heavy_cjk_mix_is_flagged(self) -> None:
        r = score_chunk("cjk", _CJK_MIXED)
        assert r.flagged is True

    def test_garbled_accented_text_is_flagged(self) -> None:
        r = score_chunk("garbled", _GARBLED_ACCENTS)
        assert r.flagged is True

    def test_garbled_accented_text_scores_below_0_40(self) -> None:
        r = score_chunk("garbled", _GARBLED_ACCENTS)
        assert r.quality_score < 0.40

    def test_cjk_mixed_scores_below_0_40(self) -> None:
        r = score_chunk("cjk", _CJK_MIXED)
        assert r.quality_score < 0.40


# ---------------------------------------------------------------------------
# Clean technical text
# ---------------------------------------------------------------------------

class TestCleanTechnicalText:
    def test_cuda_paragraph_scores_above_0_75(self) -> None:
        r = score_chunk("cuda", _CUDA_PARA)
        assert r.quality_score > 0.75, f"got {r.quality_score}"

    def test_tensorrt_paragraph_scores_above_0_75(self) -> None:
        r = score_chunk("trt", _TENSORRT_PARA)
        assert r.quality_score > 0.75, f"got {r.quality_score}"

    def test_h100_paragraph_scores_above_0_75(self) -> None:
        r = score_chunk("h100", _H100_PARA)
        assert r.quality_score > 0.75, f"got {r.quality_score}"

    def test_clean_text_is_not_flagged(self) -> None:
        for chunk_id, text in [("cuda", _CUDA_PARA), ("trt", _TENSORRT_PARA), ("h100", _H100_PARA)]:
            r = score_chunk(chunk_id, text)
            assert r.flagged is False, f"{chunk_id} incorrectly flagged (score={r.quality_score})"

    def test_clean_text_has_zero_nonascii_ratio(self) -> None:
        for chunk_id, text in [("cuda", _CUDA_PARA), ("trt", _TENSORRT_PARA)]:
            r = score_chunk(chunk_id, text)
            assert r.nonascii_ratio == 0.0

    def test_cuda_paragraph_mean_sentence_len_in_ideal_range(self) -> None:
        r = score_chunk("cuda", _CUDA_PARA)
        assert 8.0 <= r.mean_sent_len <= 25.0, f"mean_sent_len={r.mean_sent_len}"

    def test_short_single_sentence_technical_text_scores_above_0_75(self) -> None:
        text = "CUDA warp scheduler hides memory latency through context switching."
        r = score_chunk("sent", text)
        assert r.quality_score > 0.75, f"got {r.quality_score}"


# ---------------------------------------------------------------------------
# Flagging invariant: flagged ⟺ quality_score < QUALITY_FLAG_THRESHOLD
# ---------------------------------------------------------------------------

class TestFlaggingInvariant:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "∑∫∂≈→∞",
        _CJK_MIXED,
        _GARBLED_ACCENTS,
        "a",
        "CUDA warp scheduler hides memory latency through context switching.",
        _CUDA_PARA,
        _TENSORRT_PARA,
        _H100_PARA,
    ])
    def test_flagged_equals_score_below_threshold(self, text: str) -> None:
        r = score_chunk("t", text)
        assert r.flagged == (r.quality_score < QUALITY_FLAG_THRESHOLD), (
            f"text={text!r}: score={r.quality_score}, flagged={r.flagged}, "
            f"threshold={QUALITY_FLAG_THRESHOLD}"
        )

    def test_score_exactly_at_threshold_is_not_flagged(self) -> None:
        """'a' scores exactly 0.40 — equal to the threshold, not below it."""
        r = score_chunk("boundary", "a")
        assert r.quality_score == pytest.approx(QUALITY_FLAG_THRESHOLD)
        assert r.flagged is False

    def test_low_quality_garbled_text_is_flagged(self) -> None:
        r = score_chunk("low", _GARBLED_ACCENTS)
        assert r.quality_score < QUALITY_FLAG_THRESHOLD
        assert r.flagged is True

    def test_good_quality_text_is_not_flagged(self) -> None:
        r = score_chunk("good", _CUDA_PARA)
        assert r.quality_score > QUALITY_FLAG_THRESHOLD
        assert r.flagged is False


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases_output(self) -> None:
        assert _tokenize("CUDA") == ["cuda"]

    def test_splits_on_punctuation(self) -> None:
        tokens = _tokenize("hello, world! foo-bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_retains_underscore_in_identifiers(self) -> None:
        assert "my_var" in _tokenize("my_var = 42")

    def test_includes_digits(self) -> None:
        tokens = _tokenize("CUDA 3.0 api")
        assert tokens == ["cuda", "3", "0", "api"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _tokenize("") == []

    def test_punctuation_only_returns_empty_list(self) -> None:
        assert _tokenize(".,!?;:") == []


# ---------------------------------------------------------------------------
# _nonascii_ratio
# ---------------------------------------------------------------------------

class TestNonAsciiRatio:
    def test_pure_ascii_is_zero(self) -> None:
        assert _nonascii_ratio("Hello world") == 0.0

    def test_empty_string_is_zero(self) -> None:
        assert _nonascii_ratio("") == 0.0

    def test_all_cjk_chars_is_one(self) -> None:
        assert _nonascii_ratio("的的的") == pytest.approx(1.0)

    def test_half_nonascii(self) -> None:
        assert _nonascii_ratio("ab的的") == pytest.approx(0.5)

    def test_single_nonascii_char(self) -> None:
        ratio = _nonascii_ratio("aé")   # 1 non-ASCII out of 2 chars
        assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _stopword_ratio
# ---------------------------------------------------------------------------

class TestStopwordRatio:
    def test_all_stopwords_returns_one(self) -> None:
        assert _stopword_ratio(["the", "a", "and", "or", "but"]) == pytest.approx(1.0)

    def test_no_stopwords_returns_zero(self) -> None:
        assert _stopword_ratio(["cuda", "warp", "kernel", "tensor"]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self) -> None:
        assert _stopword_ratio([]) == 0.0

    def test_half_stopwords(self) -> None:
        # "the" and "a" are stopwords; "cuda" and "kernel" are not
        assert _stopword_ratio(["the", "a", "cuda", "kernel"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _mean_sentence_len
# ---------------------------------------------------------------------------

class TestMeanSentenceLen:
    def test_single_sentence(self) -> None:
        # tokens: cuda, threads, execute, in, parallel → 5
        assert _mean_sentence_len("CUDA threads execute in parallel.") == pytest.approx(5.0)

    def test_two_equal_sentences(self) -> None:
        # "GPU memory is fast" → 4; "CPU memory is slow" → 4; mean = 4
        assert _mean_sentence_len("GPU memory is fast. CPU memory is slow.") == pytest.approx(4.0)

    def test_empty_string_returns_zero(self) -> None:
        assert _mean_sentence_len("") == pytest.approx(0.0)

    def test_no_terminal_punctuation_treated_as_one_sentence(self) -> None:
        text = "CUDA warp memory bandwidth latency"
        length = _mean_sentence_len(text)
        assert length == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _sent_len_score
# ---------------------------------------------------------------------------

class TestSentLenScore:
    @pytest.mark.parametrize("mean_len", [8.0, 12.0, 20.0, 25.0])
    def test_ideal_range_scores_one(self, mean_len: float) -> None:
        assert _sent_len_score(mean_len) == pytest.approx(1.0)

    def test_very_short_sentences_scores_low(self) -> None:
        assert _sent_len_score(1.0) == pytest.approx(0.10)

    def test_very_long_sentences_penalised(self) -> None:
        assert _sent_len_score(60.0) == pytest.approx(0.30)

    def test_score_decreases_beyond_ideal_upper_bound(self) -> None:
        assert _sent_len_score(30.0) < _sent_len_score(25.0)
        assert _sent_len_score(50.0) < _sent_len_score(30.0)

    def test_score_at_30_tokens(self) -> None:
        # 1.0 - (30 - 25) * 0.02 = 0.90
        assert _sent_len_score(30.0) == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# _stopword_score
# ---------------------------------------------------------------------------

class TestStopwordScore:
    @pytest.mark.parametrize("ratio", [0.20, 0.35, 0.55])
    def test_ideal_range_scores_one(self, ratio: float) -> None:
        assert _stopword_score(ratio) == pytest.approx(1.0)

    def test_zero_ratio_returns_code_penalty(self) -> None:
        # ratio < 0.10 → 0.40  (likely pure code, no natural-language signal)
        assert _stopword_score(0.0) == pytest.approx(0.40)

    def test_very_high_ratio_returns_filler_penalty(self) -> None:
        # ratio > 0.75 → 0.20  (almost pure function words)
        assert _stopword_score(0.80) == pytest.approx(0.20)

    def test_high_ratio_lower_than_ideal(self) -> None:
        assert _stopword_score(0.80) < _stopword_score(0.55)

    def test_low_ratio_lower_than_ideal(self) -> None:
        assert _stopword_score(0.05) < _stopword_score(0.35)
