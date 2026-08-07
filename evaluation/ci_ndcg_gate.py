"""CI quality gate: fail the pipeline if any re-ranker config's mean
NDCG@10 has regressed below `DEFAULT_THRESHOLD`.

Per AGENTS.md ("End-to-end: ... no live API calls in CI" /
"LLM quality: RAGAS threshold gates in CI, not unit tests"), this does not
re-run evaluation/benchmark_runner.py live -- that needs BM25Index,
DenseIndex (Qdrant), and a Cohere API key, none of which a GitHub Actions
runner has. Instead it reads evaluation/benchmark_baseline.json, a
committed snapshot of Day 9's real, live-measured Config A/C benchmark
results (see that file's "_note" and docs/daily_progress/day_09_storyline.md
§3), and gates on it. This makes the gate a regression check against the
last live measurement, not a fresh live run -- refreshing the snapshot
(re-running evaluation/benchmark_runner.py against real infra and
overwriting benchmark_baseline.json) is a manual, deliberate step, not
something CI does on every push.

Mirrors monitoring/quality_regression.py's shape: a pure, deterministic
core (`check_ndcg_gate`) fully unit-tested, plus a thin CLI entry point
(`main`) for `python -m evaluation.ci_ndcg_gate` in CI, which exits
non-zero on failure so the workflow step fails the job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog
from pydantic import BaseModel

log = structlog.get_logger()

BASELINE_PATH = Path("evaluation/benchmark_baseline.json")
DEFAULT_THRESHOLD = 0.50


class ConfigResult(BaseModel):
    config: str
    run_id: str
    mean_ndcg_at_10: float
    num_queries: int


class GateResult(BaseModel):
    threshold: float
    results: list[ConfigResult]
    passed: bool
    failed_configs: list[str]


def check_ndcg_gate(results: list[ConfigResult], threshold: float = DEFAULT_THRESHOLD) -> GateResult:
    """Pure core: pass iff every config's mean_ndcg_at_10 >= threshold.
    Raises on an empty `results` list -- a gate that silently passes on no
    data would be worse than no gate at all."""
    if not results:
        raise ValueError("results must be non-empty")

    failed = [r.config for r in results if r.mean_ndcg_at_10 < threshold]
    return GateResult(threshold=threshold, results=results, passed=not failed, failed_configs=failed)


def load_baseline(path: Path = BASELINE_PATH) -> list[ConfigResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ConfigResult(**c) for c in data["configs"]]


def main() -> int:
    results = load_baseline()
    gate = check_ndcg_gate(results, threshold=DEFAULT_THRESHOLD)

    for r in gate.results:
        status = "PASS" if r.mean_ndcg_at_10 >= gate.threshold else "FAIL"
        print(f"[{status}] {r.config}: mean_ndcg_at_10={r.mean_ndcg_at_10:.4f} (threshold={gate.threshold})")

    if gate.passed:
        log.info("ci_ndcg_gate_pass", stage="ci_ndcg_gate", threshold=gate.threshold)
        print(f"\nNDCG gate PASSED ({len(gate.results)} config(s) >= {gate.threshold})")
        return 0

    log.error("ci_ndcg_gate_fail", stage="ci_ndcg_gate", failed_configs=gate.failed_configs)
    print(f"\nNDCG gate FAILED: {', '.join(gate.failed_configs)} below threshold {gate.threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
