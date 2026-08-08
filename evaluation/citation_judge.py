"""Claude Sonnet LLM-as-judge for citation accuracy: (claim, chunk) -> 0/1.

Verifies the per-claim citations agents/qa_agent.py's `generate` node
produces — each Citation(claim, chunk_ids) asserts that a claim is grounded
in specific passages, but nothing in `generate` verifies the model actually
followed its own instructions. `judge_qa_output()` closes that loop:  for
every (claim, chunk_id) pair in a QAState's citations, it either judges the
pair against the real chunk text (forced `judge_citation` tool call,
mirroring `answer_with_citations`'s pattern) or, if the cited chunk_id isn't
among the retrieved passages at all, flags it unsupported directly —  a
hallucinated citation to a chunk that was never retrieved is not a judgment
call, so no LLM call is spent on it.

Per AGENTS.md, the Anthropic client is constructor-injected into
`judge_claim`/`judge_qa_output` (unlike qa_agent.py's `generate`, which
instantiates it directly) since these are batch-evaluation entry points
callable independently of the agent — tests inject a MagicMock instead of
patching a module-level import.
"""

from __future__ import annotations

import anthropic
import structlog
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam
from pydantic import BaseModel

from agents.qa_agent import QAState
from retrieval.candidates import Candidate

log = structlog.get_logger()

MODEL = "claude-sonnet-5"

JUDGE_TOOL: ToolParam = {
    "name": "judge_citation",
    "description": (
        "Judge whether a passage (chunk) actually supports a claim: does the "
        "chunk contain information that substantiates the claim?"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean",
                "description": "True if the chunk supports the claim, false otherwise.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence explaining the judgment.",
            },
        },
        "required": ["supported", "rationale"],
    },
}

_UNRETRIEVED_RATIONALE = "cited chunk_id not found in retrieved passages"


class CitationJudgment(BaseModel):
    query_id: str
    claim: str
    chunk_id: str
    supported: bool
    rationale: str


def judge_claim(
    query_id: str,
    claim: str,
    chunk_id: str,
    chunk_text: str,
    client: anthropic.Anthropic,
    model: str = MODEL,
) -> CitationJudgment | None:
    log.info("judge_claim", query_id=query_id, stage="citation_judge", chunk_id=chunk_id)
    tool_choice: ToolChoiceToolParam = {"type": "tool", "name": "judge_citation"}
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": f"Claim: {claim}\n\nChunk: {chunk_text}",
        }
    ]
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            tools=[JUDGE_TOOL],
            tool_choice=tool_choice,
            messages=messages,
        )
    except Exception as exc:
        log.error("judge_claim_failed", query_id=query_id, stage="citation_judge", exc=str(exc))
        return None

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        log.error("judge_claim_no_tool_use", query_id=query_id, stage="citation_judge")
        return None

    return CitationJudgment(
        query_id=query_id,
        claim=claim,
        chunk_id=chunk_id,
        supported=bool(tool_block.input.get("supported")),
        rationale=tool_block.input.get("rationale", ""),
    )


def judge_qa_output(state: QAState, client: anthropic.Anthropic, model: str = MODEL) -> list[CitationJudgment]:
    passages: list[Candidate] = state.reranked_results or state.fused_results
    chunk_text_by_id = {c.chunk_id: c.text for c in passages}

    judgments: list[CitationJudgment] = []
    for citation in state.citations:
        for chunk_id in citation.chunk_ids:
            chunk_text = chunk_text_by_id.get(chunk_id)
            if chunk_text is None:
                log.warning(
                    "judge_qa_output_unretrieved_citation",
                    query_id=state.query_id,
                    stage="citation_judge",
                    chunk_id=chunk_id,
                )
                judgments.append(
                    CitationJudgment(
                        query_id=state.query_id,
                        claim=citation.claim,
                        chunk_id=chunk_id,
                        supported=False,
                        rationale=_UNRETRIEVED_RATIONALE,
                    )
                )
                continue

            judgment = judge_claim(
                query_id=state.query_id,
                claim=citation.claim,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                client=client,
                model=model,
            )
            if judgment is not None:
                judgments.append(judgment)

    return judgments


def citation_accuracy(judgments: list[CitationJudgment]) -> float:
    if not judgments:
        return 0.0
    return sum(1 for j in judgments if j.supported) / len(judgments)
