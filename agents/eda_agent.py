"""PandasAI-powered EDA agent over this project's eval/benchmark/drift
results -- Andres' Week 5 deliverable: conversational analysis over a
pandas DataFrame (an LLM writes and executes a pandas query per question),
distinct from streamlit_app/eval_dashboard.py's fixed-metric dashboard.

Three DataFrames, sourced from real project data wherever it exists:
- `benchmark`: evaluation/benchmark_runner.py's real `benchmark_results`
  Postgres table -- Day 9's Config A/C run, 45 rows, live-queried via
  SQLAlchemy (schema/models.py's `BenchmarkResults`).
- `eval`: evaluation/day9_citation_judgments.json, Day 9's real committed
  citation-judge output. schema/schema.sql's `eval_results` table has 0
  rows -- nothing in this project writes to it yet -- so the committed JSON
  artifact is the actual real eval data source here, not the (empty) DB
  table.
- `drift`: synthetic. monitoring/drift_detector.py and term_shift_monitor.py
  aren't wired to a live Airflow DAG yet (Day 11 open item still open), so
  there is no real drift history to load. Built locally with a seeded RNG
  rather than imported from streamlit_app/mock_data.py -- UI-layer code
  importing into an agent would invert this project's dependency direction,
  so this duplicates the small synthetic-builder pattern locally, matching
  monitoring/term_shift_monitor.py's own stated convention of local
  duplication over a cross-module import of a helper that isn't a shared
  interface.

pandasai ships no built-in Anthropic connector (only separate provider
packages like pandasai-openai, none of which are installed -- AGENTS.md is
Anthropic-only, no OpenAI key exists in this project), so
`AnthropicPandasAILLM` adapts `anthropic.Anthropic` to pandasai.llm.base.LLM's
interface directly. Per AGENTS.md, the client is constructor-injected into
`build_eda_agent` rather than instantiated inside it, mirroring
evaluation/citation_judge.py's convention for other batch-analysis entry
points.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Callable

import anthropic
import numpy as np
import pandas as pd
import pandasai as pai
import structlog
from pandasai.llm.base import LLM
from sqlalchemy.orm import sessionmaker

from schema.models import BenchmarkResults, get_engine, get_session_factory

log = structlog.get_logger()

MODEL = "claude-sonnet-5"
DAY9_CITATION_JUDGMENTS_PATH = Path("evaluation/day9_citation_judgments.json")


class AnthropicPandasAILLM(LLM):
    """Adapts `anthropic.Anthropic` to pandasai.llm.base.LLM's `call()`
    interface -- pandasai's PromptStudio prompts (schema description +
    conversation history + question) render to a single string via
    `instruction.to_string()`, which maps directly onto one Claude message."""

    def __init__(self, client: anthropic.Anthropic, model: str = MODEL) -> None:
        super().__init__()
        self._client = client
        self._model = model

    @property
    def type(self) -> str:
        return "anthropic"

    def call(self, instruction: Any, context: Any = None) -> str:
        prompt = instruction.to_string()
        self.last_prompt = prompt
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Data loaders -- real where real data exists, clearly-labelled synthetic
# where it doesn't (see module docstring)
# ---------------------------------------------------------------------------


def load_benchmark_df(session_factory: sessionmaker | None = None) -> pd.DataFrame:
    """Real evaluation/benchmark_runner.py results: Day 9's Config A/C
    benchmark, live-queried from Postgres' `benchmark_results` table."""
    session_factory = session_factory or get_session_factory(get_engine())
    with session_factory() as session:
        rows = session.query(BenchmarkResults).all()
        return pd.DataFrame(
            [
                {
                    "config": r.config,
                    "query_id": r.query_id,
                    "ndcg_at_10": r.ndcg_at_10,
                    "mrr": r.mrr,
                    "prec_at_3": r.prec_at_3,
                    "prec_at_5": r.prec_at_5,
                    "prec_at_10": r.prec_at_10,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                }
                for r in rows
            ]
        )


def load_eval_df(path: Path = DAY9_CITATION_JUDGMENTS_PATH) -> pd.DataFrame:
    """Day 9's real citation-judge output (query_id, claim, chunk_id,
    supported, rationale per judged citation)."""
    judgments = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(judgments)


def build_mock_drift_df(num_days: int = 14, seed: int = 5) -> pd.DataFrame:
    """Synthetic daily PSI + shifted-term-count history. Deterministic
    (seeded) so tests/agents/test_eda_agent.py sees identical output."""
    rng = np.random.default_rng(seed)
    base_date = datetime.date.today() - datetime.timedelta(days=num_days - 1)
    psi = np.clip(rng.normal(0.10, 0.05, size=num_days), 0.0, 0.4)
    num_shifted_terms = rng.integers(0, 8, size=num_days)
    return pd.DataFrame(
        {
            "date": [(base_date + datetime.timedelta(days=i)).isoformat() for i in range(num_days)],
            "psi": psi.round(4),
            "num_shifted_terms": num_shifted_terms,
        }
    )


DataFrameLoaders = dict[str, Callable[[], pd.DataFrame]]

DEFAULT_LOADERS: DataFrameLoaders = {
    "benchmark": load_benchmark_df,
    "eval": load_eval_df,
    "drift": build_mock_drift_df,
}


def load_dataframes(loaders: DataFrameLoaders = DEFAULT_LOADERS) -> dict[str, pd.DataFrame]:
    dataframes: dict[str, pd.DataFrame] = {}
    for name, loader in loaders.items():
        try:
            dataframes[name] = loader()
        except Exception as exc:
            log.error("eda_agent_load_failed", stage="eda_agent", dataframe=name, exc=str(exc))
    return dataframes


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_eda_agent(
    dataframes: dict[str, pd.DataFrame], client: anthropic.Anthropic, model: str = MODEL
) -> pai.Agent:
    llm = AnthropicPandasAILLM(client, model=model)
    pai_dataframes = [pai.DataFrame(df, _table_name=name) for name, df in dataframes.items()]
    return pai.Agent(pai_dataframes, config={"llm": llm, "save_logs": False})


def ask(agent: pai.Agent, question: str) -> Any:
    log.info("eda_agent_ask", stage="eda_agent", question=question)
    return agent.chat(question)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    dataframes = load_dataframes()
    for name, df in dataframes.items():
        print(f"{name}: {len(df)} rows, columns={list(df.columns)}")

    agent = build_eda_agent(dataframes, client=anthropic.Anthropic())
    for question in [
        "Compare mean ndcg_at_10 between config_A_ms_marco and config_C_cohere_rerank in the benchmark dataframe.",
        "What fraction of rows in the eval dataframe have supported == True?",
        "On which date was psi highest in the drift dataframe?",
    ]:
        print(f"\nQ: {question}")
        print(ask(agent, question))


if __name__ == "__main__":
    main()
