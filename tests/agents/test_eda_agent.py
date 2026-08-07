"""Unit tests for agents/eda_agent.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"),
`AnthropicPandasAILLM.call()` is tested against a MagicMock Anthropic
client, mirroring evaluation/citation_judge.py's test convention -- never a
real Claude call. `load_benchmark_df` takes an injected fake
`session_factory` (mirrors this project's constructor-injection convention,
e.g. tests/evaluation/test_benchmark_runner.py's session factory fakes), so
no real Postgres connection is opened. `pandasai.Agent.chat()`'s real
code-generation-and-execution pipeline is not exercised here (same reason
tests/evaluation/test_ragas_suite.py never calls a real `ragas.evaluate()`)
-- `build_eda_agent` is verified only up to constructing a valid `Agent`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pandasai as pai

from agents.eda_agent import (
    AnthropicPandasAILLM,
    build_eda_agent,
    build_mock_drift_df,
    load_benchmark_df,
    load_dataframes,
    load_eval_df,
)


# ---------------------------------------------------------------------------
# AnthropicPandasAILLM
# ---------------------------------------------------------------------------


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _instruction(prompt: str) -> MagicMock:
    instruction = MagicMock()
    instruction.to_string.return_value = prompt
    return instruction


def test_llm_type_is_anthropic() -> None:
    llm = AnthropicPandasAILLM(MagicMock())
    assert llm.type == "anthropic"


def test_llm_call_sends_rendered_prompt_and_returns_text() -> None:
    client = MagicMock()
    client.messages.create.return_value = _text_response("df['x'].mean()")
    llm = AnthropicPandasAILLM(client, model="claude-sonnet-5")

    result = llm.call(_instruction("Write pandas code to compute the mean of x."))

    assert result == "df['x'].mean()"
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["messages"][0]["content"] == "Write pandas code to compute the mean of x."


def test_llm_call_records_last_prompt() -> None:
    client = MagicMock()
    client.messages.create.return_value = _text_response("code")
    llm = AnthropicPandasAILLM(client)

    llm.call(_instruction("the prompt"))

    assert llm.last_prompt == "the prompt"


def test_llm_call_concatenates_multiple_text_blocks() -> None:
    client = MagicMock()
    block1, block2 = MagicMock(type="text", text="part one "), MagicMock(type="text", text="part two")
    client.messages.create.return_value = MagicMock(content=[block1, block2])
    llm = AnthropicPandasAILLM(client)

    result = llm.call(_instruction("q"))

    assert result == "part one part two"


def test_llm_call_ignores_non_text_blocks() -> None:
    client = MagicMock()
    text_block = MagicMock(type="text", text="kept")
    other_block = MagicMock(type="tool_use")
    client.messages.create.return_value = MagicMock(content=[other_block, text_block])
    llm = AnthropicPandasAILLM(client)

    result = llm.call(_instruction("q"))

    assert result == "kept"


# ---------------------------------------------------------------------------
# load_benchmark_df() -- injected fake session factory
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_load_benchmark_df_maps_orm_rows_to_dataframe() -> None:
    row = _FakeRow(
        config="config_A_ms_marco",
        query_id="bq01",
        ndcg_at_10=0.9,
        mrr=0.8,
        prec_at_3=0.7,
        prec_at_5=0.6,
        prec_at_10=0.5,
        latency_ms=42.0,
        cost_usd=0.0,
    )
    session = MagicMock()
    session.query.return_value.all.return_value = [row]
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session_factory = MagicMock(return_value=session)

    df = load_benchmark_df(session_factory=session_factory)

    assert len(df) == 1
    assert df.iloc[0]["config"] == "config_A_ms_marco"
    assert df.iloc[0]["ndcg_at_10"] == 0.9


def test_load_benchmark_df_empty_table_returns_empty_dataframe() -> None:
    session = MagicMock()
    session.query.return_value.all.return_value = []
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session_factory = MagicMock(return_value=session)

    df = load_benchmark_df(session_factory=session_factory)

    assert df.empty


# ---------------------------------------------------------------------------
# load_eval_df()
# ---------------------------------------------------------------------------


def test_load_eval_df_reads_judgments_json(tmp_path) -> None:
    judgments = [
        {"query_id": "q1", "claim": "c1", "chunk_id": "ck1", "supported": True, "rationale": "r1"},
        {"query_id": "q1", "claim": "c2", "chunk_id": "ck2", "supported": False, "rationale": "r2"},
    ]
    path = tmp_path / "judgments.json"
    path.write_text(json.dumps(judgments), encoding="utf-8")

    df = load_eval_df(path=path)

    assert len(df) == 2
    assert set(df.columns) == {"query_id", "claim", "chunk_id", "supported", "rationale"}
    assert df["supported"].sum() == 1


# ---------------------------------------------------------------------------
# build_mock_drift_df() -- pure, deterministic
# ---------------------------------------------------------------------------


def test_build_mock_drift_df_shape_and_columns() -> None:
    df = build_mock_drift_df(num_days=7)

    assert len(df) == 7
    assert list(df.columns) == ["date", "psi", "num_shifted_terms"]
    assert (df["psi"] >= 0).all()


def test_build_mock_drift_df_is_deterministic_given_same_seed() -> None:
    df1 = build_mock_drift_df(num_days=10, seed=1)
    df2 = build_mock_drift_df(num_days=10, seed=1)

    pd.testing.assert_frame_equal(df1, df2)


def test_build_mock_drift_df_different_seeds_differ() -> None:
    df1 = build_mock_drift_df(num_days=10, seed=1)
    df2 = build_mock_drift_df(num_days=10, seed=2)

    assert not df1["psi"].equals(df2["psi"])


# ---------------------------------------------------------------------------
# load_dataframes() -- injected loaders, one raising
# ---------------------------------------------------------------------------


def test_load_dataframes_calls_every_loader() -> None:
    loaders = {"a": lambda: pd.DataFrame({"x": [1]}), "b": lambda: pd.DataFrame({"y": [2]})}

    result = load_dataframes(loaders)

    assert set(result.keys()) == {"a", "b"}
    assert list(result["a"]["x"]) == [1]


def test_load_dataframes_skips_a_failing_loader_without_raising() -> None:
    def _boom() -> pd.DataFrame:
        raise RuntimeError("db unreachable")

    loaders = {"good": lambda: pd.DataFrame({"x": [1]}), "bad": _boom}

    result = load_dataframes(loaders)

    assert set(result.keys()) == {"good"}


# ---------------------------------------------------------------------------
# build_eda_agent() -- constructs a real pandasai.Agent, LLM mocked
# ---------------------------------------------------------------------------


def test_build_eda_agent_wires_one_pandasai_dataframe_per_entry() -> None:
    dataframes = {"benchmark": pd.DataFrame({"ndcg_at_10": [0.5]}), "eval": pd.DataFrame({"supported": [True]})}

    agent = build_eda_agent(dataframes, client=MagicMock())

    assert isinstance(agent, pai.Agent)
    # pandasai 3.0.0 stores per-agent state on `_state` (an AgentState),
    # not the `context` attribute some older docs/versions reference --
    # `agent.context` doesn't exist on this installed version at all.
    assert len(agent._state.dfs) == 2


def test_build_eda_agent_uses_the_injected_client() -> None:
    client = MagicMock()

    agent = build_eda_agent({"a": pd.DataFrame({"x": [1]})}, client=client, model="claude-sonnet-5")

    assert isinstance(agent._state.config.llm, AnthropicPandasAILLM)
    assert agent._state.config.llm._client is client
