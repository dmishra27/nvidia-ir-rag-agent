"""Post-install patch for ragas==0.4.3's llms/base.py.

ragas 0.4.3 unconditionally does `from langchain_community.chat_models.vertexai
import ChatVertexAI` at import time. That submodule doesn't exist in this
project's pinned langchain-community==0.4.2 (Vertex AI's chat model
integration moved out to the standalone langchain-google-vertexai package),
so importing ragas -- and therefore evaluation/ragas_suite.py and anything
that imports it -- raises ModuleNotFoundError before a single test runs.

This project has no Vertex AI integration (AGENTS.md's re-ranker/LLM config
is Anthropic + Cohere only) and deliberately does *not* carry
langchain-google-vertexai as a dependency: it requires pyarrow>=19.0.1, which
is incompatible with pandasai==3.0.0's pyarrow<19.0.0 (agents/eda_agent.py's
real, in-use dependency) -- see requirements_notes.txt. So the fix is to
drop ragas's one broken import and its one reference, not reintroduce
langchain-google-vertexai.

Run once after `pip install -r requirements.txt` (ci.yml, Dockerfile).
Idempotent: safe to run against an already-patched file or a ragas version
that no longer has the problem.
"""

from __future__ import annotations

from pathlib import Path

import ragas.llms

IMPORT_LINE = "from langchain_community.chat_models.vertexai import ChatVertexAI\n"
IMPORT_REPLACEMENT = (
    "# ChatVertexAI import removed -- not available in current langchain-community\n"
)
USAGE_LINE = "    ChatVertexAI,\n"
USAGE_REPLACEMENT = "    # ChatVertexAI removed\n"


def main() -> None:
    path = Path(ragas.llms.__file__).parent / "base.py"
    text = path.read_text(encoding="utf-8")

    if IMPORT_LINE not in text:
        print(f"{path}: nothing to patch (already patched, or ragas version changed)")
        return

    text = text.replace(IMPORT_LINE, IMPORT_REPLACEMENT)
    text = text.replace(USAGE_LINE, USAGE_REPLACEMENT)
    path.write_text(text, encoding="utf-8")
    print(f"{path}: patched (removed ChatVertexAI import)")


if __name__ == "__main__":
    main()
