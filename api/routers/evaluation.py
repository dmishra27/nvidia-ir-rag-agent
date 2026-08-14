"""GET /evaluation — static retrieval-evaluation report (api/static/evaluation.html).

Mirrors api/routers/ui.py exactly: static, hand-written HTML with no
template engine, read from disk per-request (not cached in a module-level
variable) so an edit is visible on the next request without restarting the
process. Its own router (not folded into ui.py) so each route stays a
one-file, one-purpose module, matching health.py/ui.py's shape.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_EVALUATION_HTML_PATH = Path(__file__).resolve().parents[1] / "static" / "evaluation.html"


@router.get("/evaluation", response_class=HTMLResponse, include_in_schema=False)
def evaluation() -> HTMLResponse:
    return HTMLResponse(content=_EVALUATION_HTML_PATH.read_text(encoding="utf-8"))
