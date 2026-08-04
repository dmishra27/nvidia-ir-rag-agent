"""LangGraph eval agent: evaluate.

Wraps evaluation/citation_judge.py's `judge_qa_output` + `citation_accuracy`
as a single LangGraph node so agents/orchestrator.py can chain it after
agents/qa_agent.py's `generate` node. Per SKILLS.md's build-langgraph-node
pattern, and per citation_judge.py's own convention (constructor-injected
Anthropic client, since it's a batch-evaluation entry point callable
independently of the agent), the client is injected into
`make_evaluate_node` rather than instantiated inside the node — unlike
qa_agent.py's `generate`, which instantiates it directly.
"""

from __future__ import annotations

from typing import Callable

import anthropic
import structlog
from pydantic import Field

from agents.qa_agent import QAState
from evaluation.citation_judge import CitationJudgment, citation_accuracy, judge_qa_output

log = structlog.get_logger()

MODEL = "claude-sonnet-5"


class EvalState(QAState):
    citation_judgments: list[CitationJudgment] = Field(default_factory=list)
    citation_accuracy: float | None = None


def make_evaluate_node(client: anthropic.Anthropic, model: str = MODEL) -> Callable[[EvalState], EvalState]:
    def evaluate(state: EvalState) -> EvalState:
        log.info("evaluate", query_id=state.query_id, stage="evaluate")
        if state.error:
            return state
        judgments = judge_qa_output(state, client=client, model=model)
        return state.model_copy(
            update={"citation_judgments": judgments, "citation_accuracy": citation_accuracy(judgments)}
        )

    return evaluate
