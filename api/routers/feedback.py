"""POST /feedback — thumbs up/down on a search result, written to
feedback_log via the shared FeedbackLog ORM model.

Mirrors slackbot/feedback_handler.py's record_feedback write shape (same
FeedbackLog columns, same "up"/"down" reaction vocabulary) independently
rather than importing it: slackbot/ is its own consumer of this API (it
calls POST /ask over HTTP), not a shared library api/ should depend on —
schema/models.py is the shared layer both actually sit on top of, per
AGENTS.md's folder structure. source="web" (the schema's default is
"slackbot") is what lets both input paths land in the same feedback_log
table monitoring/feedback_aggregator.py already reads, with no schema
change needed to tell them apart.

chunk_id identifies which result card the click came from but isn't
persisted — feedback_log has no column for it (the schema instructed not
to change), so it's logged for observability only, not written to the row.

feedback_log.query_id is a nullable FK to query_log.query_id, but nothing
in the live /search or /ask path writes a query_log row -- so hydrating
this route straight from the client's query_id would let a real Postgres
reject the insert with a foreign-key violation for any feedback on an
actual search, an infrastructure gap, not something either UI's user did
wrong. This route looks the parent row up first and stores None instead of
a dangling reference when it isn't there, same tolerance
slackbot/feedback_handler.py already has for a query_id it couldn't
resolve at all -- not a fix for the missing query_log write, just this
endpoint refusing to be the thing that 500s because of it. Schema
unchanged, per instruction.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from api.schemas import FeedbackRequest, FeedbackResponse
from schema.models import FeedbackLog, QueryLog, get_engine, get_session_factory

log = structlog.get_logger()

router = APIRouter()


@lru_cache
def get_feedback_session_factory() -> sessionmaker[Session]:
    """Same lru_cache'd-provider shape as api/dependencies.py's retrieval
    providers — Depends()-injectable so tests override it with a fake
    session factory instead of touching a real Postgres."""
    return get_session_factory(get_engine())


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    body: FeedbackRequest,
    session_factory: sessionmaker[Session] = Depends(get_feedback_session_factory),
) -> FeedbackResponse:
    feedback_id = str(uuid.uuid4())
    query_id = body.query_id
    try:
        with session_factory() as session:
            if query_id is not None and session.get(QueryLog, query_id) is None:
                # No query_log row for this query_id -- store None rather
                # than a dangling FK reference; see the module docstring.
                query_id = None
            session.add(
                FeedbackLog(
                    feedback_id=feedback_id,
                    query_id=query_id,
                    user_id=body.user_id or "anonymous",
                    reaction=body.reaction,
                    source="web",
                )
            )
            session.commit()
    except Exception as exc:
        log.warning(
            "feedback_write_failed",
            stage="feedback",
            query_id=body.query_id,
            chunk_id=body.chunk_id,
            exc=str(exc),
        )
        raise HTTPException(status_code=500, detail="failed to record feedback") from exc

    log.info(
        "feedback_recorded",
        stage="feedback",
        query_id=query_id,
        requested_query_id=body.query_id,
        chunk_id=body.chunk_id,
        reaction=body.reaction,
        feedback_id=feedback_id,
    )
    return FeedbackResponse(feedback_id=feedback_id)
