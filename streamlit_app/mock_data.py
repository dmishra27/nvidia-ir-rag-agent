"""Mock / historical data builders shared across streamlit_app/*.py's 5 tabs.

Per Day 12 Task 3's note ("Streamlit components only -- no live data calls,
no model loading. UI shell with mock data is acceptable"), no tab queries
Postgres, Qdrant, MLflow, or a live model. Two different honesty levels live
here, kept clearly separate:

- Where this project has *real* prior live-run numbers (Day 9's Config A/C
  benchmark, Day 11's live RAGAS + citation judge), those are baked in as
  "last known" historical values, each sourced back to its storyline/run_id
  -- not fabricated to look plausible.
- Everything else (search results, per-stage latency, error counts, drift
  snapshots) is synthetic and clearly labelled as mock, but built with this
  project's real pydantic/dataclass models (Candidate, QAState, DriftResult,
  TermShiftResult, DailyScore) rather than ad hoc dicts, so a tab's
  chart/table code doesn't change shape the day a real fetch function is
  wired in behind the same builder signature.

Every builder is deterministic (seeded RNG, fixed inputs) so
tests/streamlit_app/'s AppTest-based tests see identical output every run.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agents.qa_agent import Citation, QAState
from monitoring.drift_detector import DriftResult
from monitoring.quality_regression import DailyScore
from monitoring.term_shift_monitor import TermShift, TermShiftResult
from retrieval.candidates import Candidate


class BenchmarkConfigSummary(BaseModel):
    """Aggregate metrics for one re-ranker config -- same fields
    evaluation/benchmark_runner.py's `aggregate()` returns, plus the
    MLflow run_id they were logged under."""

    config: str
    run_id: str
    ndcg_at_10: float
    mrr: float
    prec_at_3: float
    latency_ms: float
    cost_usd: float


# ---------------------------------------------------------------------------
# Real historical results -- Day 9 benchmark, Day 11 RAGAS + citation judge
# ---------------------------------------------------------------------------

# docs/daily_progress/day_09_storyline.md §3 -- Config A/C benchmark, 15
# cached queries, MLflow experiment `reranker_benchmark`.
DAY9_BENCHMARK_SUMMARIES: list[BenchmarkConfigSummary] = [
    BenchmarkConfigSummary(
        config="config_A_ms_marco",
        run_id="a55012a4",
        ndcg_at_10=0.5333,
        mrr=0.5333,
        prec_at_3=0.2444,
        latency_ms=48.90,
        cost_usd=0.0,
    ),
    BenchmarkConfigSummary(
        config="config_C_cohere_rerank",
        run_id="c827cd71",
        ndcg_at_10=0.5280,
        mrr=0.5333,
        prec_at_3=0.2444,
        latency_ms=291.21,
        cost_usd=0.03,
    ),
]
DAY9_BENCHMARK_NUM_QUERIES = 15
DAY9_CITATION_ACCURACY = 0.7037  # 27 (claim, chunk) pairs judged, day_09_storyline.md §3

# docs/daily_progress/day_11_storyline.md §3 -- live RAGAS run, MLflow run `7d8f1005`.
DAY11_RAGAS_SCORES = {"faithfulness": 0.7616, "answer_relevancy": 0.2497}
DAY11_RAGAS_NUM_QUERIES = 10
DAY11_RAGAS_RUN_ID = "7d8f1005"

# Day 9's own scope note: NDCG threshold used by .github/workflows/ci.yml's
# eval gate, based on Config A's 0.5333.
CI_NDCG_GATE = 0.50


# ---------------------------------------------------------------------------
# search_tab.py -- synthetic query / passages / answer
# ---------------------------------------------------------------------------

_MOCK_PASSAGES = [
    Candidate(
        chunk_id="c_cuda_malloc_01",
        text="cudaError_t cudaMalloc(void** devPtr, size_t size) allocates `size` bytes of "
        "linear device memory and returns a pointer to it in *devPtr.",
        score=0.91,
        rank=1,
    ),
    Candidate(
        chunk_id="c_cuda_memcpy_02",
        text="cudaMemcpyAsync enqueues a copy on the given stream; the call may return before "
        "the copy completes, so synchronize the stream before reading the destination.",
        score=0.84,
        rank=2,
    ),
    Candidate(
        chunk_id="c_nvlink_03",
        text="NVLink 5.0 (Blackwell generation) provides up to 1.8 TB/s of bidirectional "
        "GPU-to-GPU bandwidth per link.",
        score=0.77,
        rank=3,
    ),
]


def mock_qa_state(query: str, query_id: str = "demo0001") -> QAState:
    """A fixed, synthetic QAState in the real QAState/Citation/Candidate
    shape -- same fields agents/qa_agent.py's `generate` node produces --
    regardless of the input query, since this tab makes no live retrieval or
    LLM call."""
    return QAState(
        query_id=query_id,
        query=query,
        reranked_results=_MOCK_PASSAGES,
        answer=(
            "cudaMalloc(void** devPtr, size_t size) allocates device memory and writes the "
            "pointer to *devPtr [c_cuda_malloc_01]. Copies to/from that buffer should use "
            "cudaMemcpyAsync on a stream, followed by a synchronize before the host reads the "
            "result [c_cuda_memcpy_02]."
        ),
        citations=[
            Citation(claim="cudaMalloc allocates device memory via an out-pointer.", chunk_ids=["c_cuda_malloc_01"]),
            Citation(
                claim="cudaMemcpyAsync requires a stream synchronize before reading the result.",
                chunk_ids=["c_cuda_memcpy_02"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# eval_dashboard.py -- quality regression trend (real evaluate_regression())
# ---------------------------------------------------------------------------


def mock_quality_regression_history(num_days: int = 10, seed: int = 12) -> list[DailyScore]:
    """10 days of synthetic daily RAGAS scores, seeded for determinism, that
    drift near Day 11's real faithfulness/answer_relevancy baseline and end
    in a deliberate 3-day decline -- so eval_dashboard.py can run this
    through monitoring/quality_regression.py's *real* `evaluate_regression`
    and demonstrate what the alert looks like, rather than re-implementing
    the regression logic in the UI layer."""
    rng = np.random.default_rng(seed)
    base_date = datetime.date(2026, 7, 29)
    faithfulness = DAY11_RAGAS_SCORES["faithfulness"] + rng.normal(0, 0.015, size=num_days - 3)
    faithfulness = np.clip(faithfulness, 0.0, 1.0)
    decline = faithfulness[-1] - np.array([0.03, 0.07, 0.11])
    faithfulness = np.concatenate([faithfulness, decline])

    relevancy = np.clip(DAY11_RAGAS_SCORES["answer_relevancy"] + rng.normal(0, 0.02, size=num_days), 0.0, 1.0)

    return [
        DailyScore(
            date=(base_date + datetime.timedelta(days=i)).isoformat(),
            scores={"faithfulness": round(float(faithfulness[i]), 4), "answer_relevancy": round(float(relevancy[i]), 4)},
            num_queries=20,
        )
        for i in range(num_days)
    ]


# ---------------------------------------------------------------------------
# monitoring_tab.py -- per-stage latency + error rate
# ---------------------------------------------------------------------------

STAGES = ["bm25", "dense", "rrf", "rerank", "llm"]  # api/telemetry.py's traced stages


def mock_stage_latency(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_ms = {"bm25": 12, "dense": 60, "rrf": 3, "rerank": 45, "llm": 1400}
    rows = []
    for stage in STAGES:
        samples = rng.gamma(shape=4.0, scale=base_ms[stage] / 4.0, size=500)
        rows.append({"stage": stage, "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95))})
    return pd.DataFrame(rows)


def mock_error_timeseries(num_days: int = 14, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = datetime.date.today() - datetime.timedelta(days=num_days - 1)
    requests = rng.integers(180, 260, size=num_days)
    errors = rng.poisson(2.0, size=num_days)
    dates = [base_date + datetime.timedelta(days=i) for i in range(num_days)]
    return pd.DataFrame(
        {
            "date": dates,
            "requests": requests,
            "errors": errors,
            "error_rate": np.round(errors / requests, 4),
        }
    )


# ---------------------------------------------------------------------------
# drift_tab.py -- PSI drift + term shift snapshot
# ---------------------------------------------------------------------------


def mock_drift_result() -> DriftResult:
    """A moderate-severity snapshot in the real DriftResult shape --
    monitoring/drift_detector.py's own severity thresholds decide the label,
    not a value hardcoded here."""
    return DriftResult(psi=0.164, baseline_size=500, current_size=143, is_drifted=True, severity="moderate")


def mock_term_shift_result() -> TermShiftResult:
    """A plausible drift narrative: Blackwell-generation terms rising as new
    docs are ingested, older Hopper-era terms receding -- in the real
    TermShiftResult/TermShift shape."""
    shifted = [
        TermShift(term="blackwell", baseline_freq=0.0021, current_freq=0.0187, delta=0.0166),
        TermShift(term="nvlink5", baseline_freq=0.0003, current_freq=0.0098, delta=0.0095),
        TermShift(term="gb200", baseline_freq=0.0000, current_freq=0.0071, delta=0.0071),
        TermShift(term="hopper", baseline_freq=0.0142, current_freq=0.0089, delta=-0.0053),
        TermShift(term="h100", baseline_freq=0.0118, current_freq=0.0070, delta=-0.0048),
    ]
    return TermShiftResult(
        shifted_terms=shifted,
        num_baseline_docs=340,
        num_current_docs=98,
        new_terms=["gb200", "nvlink5", "grace-blackwell"],
        dropped_terms=["v100", "pascal"],
    )
