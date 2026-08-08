"""GET / — the guided HTML search UI (api/static/index.html).

Static, hand-written HTML/CSS/JS with no template engine and no build
step: the page calls POST /search itself via fetch(), so there's no
server-side data to inject. Read from disk per-request (not cached in a
module-level variable) so an edit to the HTML is visible on the next
request without restarting the process -- this file is tiny and rarely
requested at high volume, so the extra disk read is not a real cost.

Kept as its own router (mirrors health.py's one-route shape) rather than
inlined in api/main.py, so app.py stays just the FastAPI app assembly.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(content=_INDEX_HTML_PATH.read_text(encoding="utf-8"))
