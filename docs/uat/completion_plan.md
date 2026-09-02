# Completion Plan — Closing the Remaining Gaps
### nvidia-ir-rag-agent · post-capstone

**Prepared:** 22 August 2026
**Revised:** 22 August 2026 — DVC work complete (`7bc6730`); Stage 1 narrowed accordingly
**Scope:** Everything outstanding across reproducibility, defect records, rubric criteria, and documentation

---

## What this can and cannot change

The August submission is graded at `8569bfb` and is frozen. **Nothing below alters that score.**

What it does do:

- Closes the reproducibility work properly, which is what an interviewer actually probes
- Fills the one real capability gap (query rewriting)
- Makes existing work visible that currently isn't
- Leaves the project in a state where Attempt 3 (deadline 8 Sept) is a decision rather than a scramble — if you enter, all of this counts

---

## Already closed

**Data versioning — complete and verified** (`7bc6730`). All eight DVC pointers committed, remote configured on an orphan `dvc-storage` branch, and verified from a genuine fresh clone: 8/8 blobs pulled and sha256-checked with no credentials. The corpus still reads exactly **5,389 chunks** after every change — the number that had to survive, and did.

Four defects were fixed along the way: pointer files silently swallowed by a blanket `.gitignore` (DEF-19), silent partial-corpus ingestion (DEF-20), a line-ending corruption risk on binary blobs (DEF-24), and an inverted exit code on idempotent re-runs (DEF-25).

**CC-ACQ-02 — passed**, the highest-risk case in the clean-clone protocol.

Full plain-language account: `docs/explainers/data_versioning_explained.md`.

---

## Current standing

| # | Criterion | Score | Action needed |
|---|---|---|---|
| 1 | Retrieval evaluation | **2/2** | None — strongest criterion |
| 2 | RAG evaluation | 2/2? | **Verify** — Stage 3 |
| 3 | Interface | **2/2** | None |
| 4 | Ingestion pipeline | **2/2** | None |
| 5 | Monitoring | 2/2? | **Verify chart count** — Stage 3 |
| 6 | Containerization | **2/2** | None |
| 7 | Problem description | 2/2? | Confirm in Stage 5 README pass |
| 8 | RAG flow | **2/2** | None |
| 9 | Reproducibility | 1→2 | **CC-ACQ-02 passed** — corpus DVC-pinned and clean-clone verified. Rest of protocol — Stage 2 |
| 10 | Best practices | 2/3 | **Query rewriting** — Stage 4 |
| 11 | Bonus | 2/2 + 3 | Framing — Stage 5 |

---

## Stage 1 · Defect records *(20 min — do first, while fresh)*

**Narrowed since first drafting.** DEF-19 and DEF-20 were fixed during the DVC work and are recorded in `7bc6730`'s commit message. What remains is getting **DEF-23, DEF-24 and DEF-25 out of session output and into the correction notice** — they currently exist only in a Claude Code transcript and this plan, which is precisely the persistence failure that cost two extra runs on A5 and left A2/A3/A4/A6 unreproducible.

> **Prompt for Claude Code:**
>
> Update `docs/uat/correction_notice_a1.md` §6 with defects surfaced by the DVC work. DEF-19 and DEF-20 are already fixed at `7bc6730` — update their entries to record the fix and the actual root cause, then add three new ones.
>
> **DEF-19 — revise the root cause.** Current text says `data/raw/` is orphaned because the ingest doesn't read it. The deeper defect: `data/raw.dvc` was **never committed** — commit `4ebf3af`'s blanket `data/` gitignore silently swallowed the pointer file, so the DVC layer was inert from 10 July and invisible to anyone cloning. Record the fix (per-file `dvc add`, DVC-managed per-file ignores) and closure at `7bc6730`. Keep the Round 1 misdiagnosis consequence — NVLink and H100 queries recorded as scope decisions while the whitepapers sat untracked.
>
> **DEF-20 — mark fixed.** Ingest now asserts the expected document count and fails on shortfall; the three dead URLs (cuDNN, TensorRT, Thrust) removed with a comment recording the 404s. Closed at `7bc6730`.
>
> **DEF-23 — memory gate is advisory, not enforcing.** During verification, free RAM reached **17 MB** against a documented <200 MB hard stop, *after* a pre-flight check had passed. The check is single-shot; a multi-hundred-page PyMuPDF parse allocates past it mid-run. Note that `docs/uat/clean_clone_test_protocol.md` §2.1 assumes the 200 MB figure is meaningful, and it currently isn't for any parse-heavy stage. Propose — don't implement — either a mid-run check or a page-batch cap. **Open.**
>
> **DEF-24 — line-ending corruption risk on binary DVC blobs.** Windows `autocrlf` was about to corrupt a binary cache object on commit to `dvc-storage`. Caught pre-push, fixed via `.gitattributes` (`* -text`), all 8 objects byte-verified before push. Record because the failure mode is silent: `dvc pull` would have *succeeded* and returned a damaged PDF. Closed at `7bc6730`.
>
> **DEF-25 — ingest exit code inverted on idempotent re-run.** Exit status keyed off `docs_written > 0`, so a correct no-op re-run against an unchanged corpus reported failure. Now keys off `failed_doc_ids`. Closed at `7bc6730`.
>
> **Also add a verification subsection** recording that CC-ACQ-02 passes: fresh clone into a temp directory with no pre-existing cache, `dvc pull`, all 8 blobs sha256-verified identical. Note the one honest caveat — two PDFs (Best Practices, Nsight) were re-downloaded during this work and differ in bytes from the July originals while matching on page count (118, 344) and producing an unchanged 5,389 chunks. **The pin is faithful in content, not byte-identical to the originals**, which no longer exist anywhere.

## Stage 2 · Clean-clone protocol, remaining cases *(half a day)*

CC-ACQ-02 is closed. Nine cases remain in `docs/uat/clean_clone_test_protocol.md`.

### The execution problem, and how to handle it

**§1.2 is the whole point of this test**, and Claude Code working inside your repo will violate it by default — it has the codebase in context and will fill a documentation gap competently without noticing it was a gap. That destroys the finding.

Two workable approaches:

**Preferred — you run it, Claude Code assists on demand.** Clone to a temp directory yourself, follow the README literally, and stop the moment something doesn't work. Ask Claude Code only *after* recording the failure. Slower, but it's the only version that produces trustworthy findings.

**Acceptable — a fresh Claude Code session in the temp clone**, launched from that directory, with the adversarial rule stated explicitly and repeatedly. Weaker, because it can still infer from the code it can read.

> **Prompt if delegating (launch from the temp clone directory, not the working repo):**
>
> You are testing whether this repository can be reproduced by someone who has never seen it. Execute `docs/uat/clean_clone_test_protocol.md`, cases CC-ACQ-03 through CC-VER-02. CC-ACQ-02 is already closed — skip it.
>
> **The governing rule, from §1.2 — this is the entire point of the exercise.** When a step fails or a required action isn't documented: **stop, record it, and only then work around it.** Record the workaround separately as an undocumented prerequisite. Do not fix things silently. Do not infer a missing step from reading the code and proceed as if the docs had said it. A finding you smooth over is a finding destroyed.
>
> Specific things to watch for, from the project's own history:
> - The torch CPU wheel index (needed for CI and Dockerfile — is it documented for a local install?)
> - Whether `patch_ragas.py` needs explicit invocation
> - Whether the docs distinguish required services from optional ones (the full stack is not needed for retrieval, and this host has 8 GB)
> - Port conflicts, especially 8000 — CC-EXE-03 tests whether failure is loud or silent
> - Whether reproducing the Config C benchmark needs a Cohere API key, and whether that's stated
>
> Classify every finding as BLOCKER / MAJOR / MINOR / UNDOCUMENTED PREREQUISITE per §4. Produce the five deliverables in §5. Per §6, finding nothing means the test was run wrong — expect three to eight undocumented prerequisites.
>
> **Memory:** this host has hit 17 MB free during a parse. Check before CC-ING-01 and CC-ING-02; if the full stack won't fit, record that as a finding rather than trimming services silently.

### Scheduling note

CC-ING-01 and CC-ING-02 need the stack up in a fresh location while your working repo's containers may also be running. On 8 GB that will not fit. Stop the working repo's containers first, and consider a Docker Desktop restart to clear accumulated allocation.

---

## Stage 3 · Verify criteria 2 and 5 *(1 hour)*

Two rubric criteria I scored optimistically without being able to check.

> **Prompt for Claude Code:**
>
> Two verification questions against the rubric, answered from the code rather than assumption.
>
> **1 · RAG evaluation (criterion 2).** Top marks require *multiple* RAG approaches evaluated and the best one used — typically prompt variants. Check `run_day9_ragas.py`, the RAGAS suite, `run_day9_citation_judge.py`, and the QA agent. Was more than one prompt or generation configuration evaluated, or only one? Report what exists; don't build anything yet.
>
> **2 · Monitoring (criterion 5).** Top marks require user feedback collected **and** a dashboard with at least 5 charts. Feedback is unambiguous (Slackbot HITL, web feedback control at `e212c7f`). For the dashboard: count the distinct charts on the single strongest dashboard a reviewer would be pointed at — Streamlit, MLflow, or Phoenix. The criterion reads as one dashboard, not five tools. Report the count and where it lives.
>
> If either falls short, propose the smallest change that closes it.

---

## Stage 4 · Query rewriting *(half a day)*

The one genuine capability gap, worth a full rubric point — and the most interesting item here, because Family A's A4 already points at it.

**Do not bolt it on untested.** A4 found indirect evidence that the cross-encoder harms exact-identifier lookups, suggesting query-dependent handling. Rewriting is the same family of idea. Specify it as a Family D hypothesis with confirm and falsify conditions before implementing, and it becomes evidence rather than a checkbox.

> **Prompt for Claude Code:**
>
> Implement query rewriting, specified as a testable hypothesis rather than an unevaluated feature.
>
> **Context.** Read `docs/uat/round3_family_a_findings.md` A4 and `docs/uat/correction_notice_a1.md` §4.3. A4 found indirect evidence that the cross-encoder favours discursive prose over terse API reference, and that identifier lookups may be better served by different handling. The Round 2 case types already separate lexical (Case 1), semantic (Case 2), and vocabulary-gap (Case 4) queries — that grouping is the natural evaluation frame.
>
> **First, write the hypothesis** into `docs/uat/round3_hypothesis_test_plan.md` as **D-QR**, following the plan's existing format: claim, confirms-if, falsifies-if, protocol. Something in the shape of *"rewriting improves retrieval on vocabulary-gap queries (Case 4) and is neutral-to-harmful on exact-identifier queries (Case 1/Case 5)"* — refine the wording yourself, but it must be falsifiable.
>
> **Then implement minimally.** A rewriting step ahead of retrieval, applied conditionally rather than universally. Candidate strategies — pick one or two, don't build a framework:
> - Expand identifier queries with related API names
> - Generate a paraphrase for the dense retriever while BM25 keeps the literal query
> - Expand legacy or marketing terminology (Round 1's `shader processor` → `CUDA cores` is the documented case)
>
> **Then evaluate** across the 15 Round 2 queries, grouped by case type, with and without rewriting. Report per-query, not just means, and carry the small-n caveat — most case types have 2–3 queries.
>
> **Persist everything** — script and JSON output committed before analysis, per `0149ca4`.
>
> **Be honest if it doesn't help.** A rewriting step that measurably hurts, reported as such, is worth more than one that ships unevaluated. That is the standard the rest of this project's evaluation work is held to.

---

## Stage 5 · Documentation pass *(2 hours — do last)*

Everything above changes what the README should say. One pass at the end, not three.

> **Prompt for Claude Code:**
>
> Final documentation pass, folding in everything from the completion plan.
>
> **1 · Corpus section.** Five documents, 5,389 chunks, with the per-document breakdown (Programming Guide 1,670 · Runtime API 1,642 · Math API 1,232 · Nsight 561 · Best Practices 284) so a reviewer can verify their ingest matches. State that the corpus is DVC-pinned and reproducible via `dvc pull` with no credentials, and that three further documents (cuDNN, TensorRT, Thrust) were originally declared but have 404'd since Oct 2025 — named as a known gap rather than omitted. Note the three hardware PDFs as tracked-but-unused per DEF-19.
>
> **2 · Undocumented prerequisites** from Stage 2's findings — the primary output of the clean-clone test. Add a stated minimum specification: RAM, disk, ports, required external accounts (including whether a Cohere key is needed for the Config C benchmark).
>
> **3 · Bonus-point visibility.** These exist and a reviewer won't find them unaided: MCP servers including `mcp-mlflow`, the A2A protocol implementation, CI/CD with a quality gate, and the Round 3 hypothesis testing — 31 pre-registered hypotheses across six families with explicit falsification conditions, which is unusual at this level. One short section each, linking to the artifacts.
>
> **4 · Deployment honesty.** The Render deployment runs BM25-only to fit the 512 MB free tier. The footer says so (`4816d2d`); the README should too, with a pointer to the evaluation page. A reviewer who reads only the README shouldn't be surprised.
>
> **5 · Problem statement.** Confirm the README opens with the problem the project solves, not the stack it uses. Rubric criterion 7 rewards clarity about the problem.
>
> **6 · Query rewriting** from Stage 4, including its evaluation result whichever way it went.
>
> **7 · Commit the remaining documentation.** `docs/explainers/data_versioning_explained.md` (plain-language account of the DVC work) and `docs/uat/completion_plan.md` (this plan) are not yet in the repository. Stage them with the rest.
>
> **8 · Data versioning in the README.** One short paragraph: the corpus is DVC-pinned across 8 files, stored on the `dvc-storage` branch, and fetchable with `dvc pull` by anyone with the clone URL — no credentials, no cloud account. Note that the blobs live on an orphan branch, so a shallow or `--single-branch` clone still needs the remote configured. Link to the explainer for the full story.

---

## Sequence and effort

| Stage | Effort | Blocks |
|---|---|---|
| ~~0 · Data versioning~~ | ~~Done~~ | **Complete — `7bc6730`** |
| 1 · Defect records | 20 min | — |
| 2 · Clean-clone protocol | Half a day | Stage 5 |
| 3 · Verify criteria 2 & 5 | 1 hour | Possibly Stage 5 |
| 4 · Query rewriting | Half a day | Stage 5 |
| 5 · Documentation pass | 2 hours | — |

**Roughly a day and a half remaining.** Stages 1 and 3 are quick and independent. Stage 2 is the long pole and generates most of the material for Stage 5.

**Do Stage 1 first and don't defer it.** DEF-23 through DEF-25 exist only in a chat transcript right now. This project has already lost the A2/A3/A4/A6 findings to exactly that, and spent two extra runs re-establishing an A5 result that was never persisted. Twenty minutes now closes a gap that has cost hours twice before.

---

## What still won't be closed

Worth naming, so the picture stays honest:

- **A7** — hardware-blocked. Needs Colab, a rented GPU hour, or a larger machine.
- **A2, A3, A3-2, A4, A6** — analysis-only, no retained scripts. Re-running them properly is a separate piece of work, and it should wait for graded relevance judgements, since those verdicts need re-testing against a non-circular answer key regardless.
- **ENH-11 graded relevance judgements** — the documented prerequisite for trusting any further re-ranking conclusion. The largest remaining item in the project and out of scope here.
- **Families B–F** — of the plan's 32 hypotheses, **B3 and B4 were run together on 2 September 2026** (`run_b3_b4_fusion_eval.py` → `evaluation/b3_b4_fusion_eval.json` → `docs/uat/round3_b3_b4_findings.md`, both committed pre-run / pre-analysis per `0149ca4`). B3 (called "B4" in CORR-001 §3.3 — an off-by-one, corrected 31 Aug) re-specified 31 Aug around single-signal displacement measured by target-chunk rank: **confirmed binary, graded form falsified as pre-registered**. B4 (score-normalised fusion): **directional remedy, predicted stability cost, not recommended** — the aggregate-NDCG test is unevaluable until ENH-11. The remaining ~22 hypotheses across B–F are unchanged and open.

---

*Nothing in this plan alters the August submission. It closes the reproducibility story — one half of which is now done and verified — fills the one capability gap, and makes existing work visible.*
