# nvidia-ir-rag-agent

Read these two files at the start of any session on this repo:

- **AGENTS.md** ? 8-layer architecture, coding standards (structlog, no print()),
  MCP server list, folder structure, retrieve-hybrid and build-langgraph-node patterns
- **SKILLS.md** ? reusable code patterns

## Host constraint

8 GB total, frequently under 500 MB free. Check RAM before any heavy step.
See DEF-23 and DEF-26 in docs/uat/correction_notice_a1.md - timeout and
connection symptoms on this host are often the memory ceiling in disguise.

## Method

Hypotheses are pre-registered before implementation. Commit the harness
before running it and the raw output before analysing it - see 0149ca4.
Relevance labels are circular (A2); use target-chunk rank, not NDCG,
until ENH-11 lands.
