# Clean-Clone Reproducibility Test Protocol
### nvidia-ir-rag-agent

**Reference:** CCT-NVIR-2026-001
**Version:** 1.0 — planned, not executed
**Scope:** Whether a competent stranger can reproduce this project from the repository alone
**Prepared:** 20 August 2026

---

## 1. Purpose and rationale

Every prior test cycle on this project has run against a machine that already worked. The 42-case functional sign-off exercised a deployed service and a local stack that had been incrementally configured over five weeks — dependencies installed, ports free by habit, environment variables long since exported, corpus already on disk. It proved the software behaves correctly. **It did not prove the repository is sufficient to produce that software.**

Those are different claims, and only the second one matters to a reviewer, a collaborator, or a future maintainer.

The sign-off itself identified this gap and named the likely failure modes: port 8000 occupied by an unrelated project returning plausible-but-wrong JSON to the test harness; MLflow experiments resident in a second project's Docker volume; Streamlit tests failing intermittently when that second project was slow. It concluded that **cross-project coupling was the dominant reproducibility risk**, and recorded that the repository was not reliably reproducible from a clean clone. Commit `2394ad7` (12 August) containerised the api and streamlit services and decoupled tests from live MLflow, which should have addressed much of this.

Should have. This protocol establishes whether it did.

### 1.1 The measurement principle inherited from the sign-off

Sign-Off v1.0 accepted four test cases on inference rather than measurement. v2.0 measured all four directly: three confirmed, **one failed** — page load, once actually timed, came in at 72.56 s cold against a 60 s criterion, while the warm request measured 0.186 s. The sign-off's own comment stands as this protocol's governing principle:

> *"Inference was right three times out of four, which is precisely why the fourth matters."*

Applied here: **it is not sufficient to believe the repository is reproducible.** Every step below is executed literally, in order, on a machine with no prior knowledge of this project, and the outcome is recorded whether or not it matches expectation.

### 1.2 The adversarial stance

The tester's job is not to get the system running. It is to **find out what the repository fails to say.**

This is a real behavioural discipline, not a formality. Anyone who has built the system will, on hitting a missing step, supply it automatically from memory and continue — and in doing so destroy the finding. Every such moment is precisely the defect the test exists to catch.

**Rule: when a step fails, stop. Record it. Only then apply the workaround, and record that separately as an undocumented prerequisite.**

---

## 2. Environment requirements

The test environment must be **clean** in a specific sense: no artifact of prior development on this project may be present or reachable.

| Requirement | Rationale |
|---|---|
| Fresh clone into a new directory | Not `git clean` on the existing tree — untracked-but-necessary files must be exposed |
| No other Docker projects running | The dominant risk identified by the sign-off |
| Ports 8000, 5432, 6333, 5000, 8501, 6006, 16686 all free | Verified explicitly, not assumed |
| No inherited environment variables | New shell; no `.env` copied from the working tree |
| Docker Desktop running, containers stopped | Baseline state |
| **≥ 4 GB free RAM at start** | See §2.1 |

### 2.1 The memory constraint

The host is 7.65 GB total. This project's documented operating range treats **< 200 MB free as a hard stop**, and during Family A experimentation free memory was observed at 103 MB, 134 MB, and 350 MB at various points. A nine-service compose stack starting cold, plus model loads, is the heaviest thing this machine does.

**If a memory-related failure occurs, it is recorded as a finding, not worked around silently.** A reviewer on a similar machine will hit the same wall, and the repository should say so. The absence of a stated minimum specification is itself a documentation defect.

### 2.2 A note on what a "reviewer" means here

The test persona is a competent Python engineer with Docker experience, no knowledge of this project, and no ability to ask the author questions. They have the repository, its README, and its setup guide. Nothing else.

---

## 3. Test cases

Nine cases across four sections. Each records **Pass / Fail / Blocked**, the evidence, and — critically — **any step performed that the documentation did not specify**.

---

### Section A — Acquisition

#### CC-ACQ-01 · Clone and orient

**Steps**
1. `git clone` into a new directory
2. Read the README start to finish before doing anything else
3. Locate the setup guide

**Pass criteria**
- Repository clones without authentication beyond public access
- README states, within its first screen: what problem the project solves, what the corpus is, and where setup instructions live
- A reader can state the project's purpose after one read

**Record:** time to locate setup instructions; anything materially unclear on first reading.

---

#### CC-ACQ-02 · Dataset acquisition ⚠️ **HIGHEST RISK**

The corpus is DVC-tracked (`921281c`), and `data/` is excluded from git (`4ebf3af`). Five NVIDIA PDFs produce the 5,389 chunks every downstream component depends on. **If a reviewer cannot obtain them, nothing else in this protocol can execute** — and the rubric criterion is explicit that missing data caps reproducibility at 1 point regardless of instruction quality.

**Steps**
1. Inspect `.dvc/config` for the configured remote
2. Determine whether that remote is reachable by an unauthenticated third party
3. Attempt `dvc pull` from the clean clone
4. If it fails, determine from the README alone whether the five source PDFs can be identified and obtained independently

**Pass criteria**
- Either `dvc pull` succeeds without project-specific credentials, **or**
- The README names all five source documents precisely enough to obtain them from NVIDIA directly, including any version or edition needed to reproduce the chunk count

**Fail criteria**
- Remote is a local path, or requires credentials not documented
- Sources are described only generically ("NVIDIA CUDA documentation") without enumeration

**Record:** exact remote configuration; whether each of the five PDFs is individually identifiable from documentation alone.

> **Note.** Step 4's pass condition is a genuine remedy, not a fallback. Publishing a DVC remote is one route; naming the sources precisely is another, and for a five-document public corpus it may be the better one — it removes an infrastructure dependency rather than adding one.

---

#### CC-ACQ-03 · Dependency installation

**Steps**
1. Create a fresh virtual environment
2. Install per documented instructions, verbatim
3. Record wall-clock duration and any resolution failures

**Pass criteria**
- Installation completes without manual intervention
- All versions pinned (the criterion requires *"versions for all dependencies are specified"*)
- No platform-specific step required that the documentation does not mention

**Historical note.** 8 August required six consecutive commits to make CI dependency resolution succeed: torch CPU wheel index, pyarrow bump, the dependency graph the torch fix unmasked, `mypy --strict` debt, and two attempts at patching ragas. **The torch CPU wheel index is a plausible undocumented prerequisite for a fresh local install** — it was needed for CI and the Dockerfile, and a local installer may need it too. Test whether the instructions mention it.

**Record:** duration; every manual intervention; whether `patch_ragas.py` requires explicit invocation.

---

### Section B — Ingestion

#### CC-ING-01 · Infrastructure start

**Steps**
1. Verify all seven ports free, explicitly, before starting anything
2. Start the compose stack per documentation
3. Record free RAM before, during, and after
4. Confirm each service reaches healthy

**Pass criteria**
- All services start without editing compose files or `.env`
- No port conflicts
- Documented health checks pass

**Record:** startup duration; peak memory; any service requiring a manual restart. If the documentation does not state which services are required versus optional, that is a finding — the full stack is not necessary for retrieval and a reviewer on a constrained machine needs to know that.

---

#### CC-ING-02 · Corpus ingestion

**Steps**
1. Run the ingestion pipeline per documentation
2. Verify the resulting chunk count in Postgres
3. Verify Qdrant point count

**Pass criteria**
- **Postgres contains exactly 5,389 chunks**
- **Qdrant contains 5,389 points**, 768-dimension cosine
- Ingestion completes without manual intervention

**Why the exact number matters.** 5,389 is the anchor for every evaluation figure in this project. A different count means different chunk boundaries, which means the benchmark numbers, the qrels, and every Family A finding refer to a corpus the reviewer does not have. **A near-miss is a fail**, and diagnosing it (chunker version? PDF edition? parameter default?) is more valuable than the pass would have been.

**Record:** actual counts; duration; whether Airflow is required or a direct runner suffices, and whether the documentation makes that clear.

---

#### CC-ING-03 · Index construction

**Steps**
1. Build the BM25 index
2. Confirm the dense index is populated
3. Verify a live query returns results through each path

**Pass criteria**
- BM25 index builds and persists to the expected location
- Both retrieval paths return non-empty, plausible results

**Record:** whether index build is a separate documented step or implicit in ingestion — `3c3c632` and `7b0b862` suggest BM25 index shipping has been a recurring friction point.

---

### Section C — Execution

#### CC-EXE-01 · API service

**Steps**
1. Start the API per documentation
2. Retrieve the OpenAPI spec
3. Execute `/search` and `/ask` against documented examples
4. Confirm the response echoes the resolved `reranker_mode` (DEF-03, closed at `8569bfb`)

**Pass criteria**
- Service starts on the documented port
- Both endpoints return well-formed responses
- Responses are self-describing as to retrieval mode

**Record:** cold-start duration. `2968cbe` reduced this from ~3 min to ~32 s via lazy tab imports; measure it rather than assuming, per §1.1.

---

#### CC-EXE-02 · Streamlit interface

**Steps**
1. Access the UI
2. Exercise the example queries
3. Confirm the footer states the deployed retrieval mode
4. Confirm the evaluation page renders benchmark figures
5. Confirm the corpus transparency panel names sources and coverage gaps

**Pass criteria**
- UI loads and responds
- Retrieval mode is visible without inspecting code
- Evaluation and transparency content renders

**Note.** Items 3–5 test the honesty work of 13–16 August (`79ac4f6`, `e678ebe`, `85a2c83`, `4816d2d`). Their purpose is to prevent a user over-estimating the system, so verifying they survive a clean install is verifying that the honesty is structural rather than incidental to one machine's configuration.

---

#### CC-EXE-03 · Cross-project isolation ⚠️

The specific defect the sign-off named. Tests whether `2394ad7` actually closed it.

**Steps**
1. With the stack running, start an unrelated service on port 8000
2. Re-run the API test cases
3. Observe whether failure is **loud** (refuses to start, clear error) or **silent** (harness receives plausible-but-wrong responses from the wrong service)
4. Verify MLflow experiments resolve to this project's volume, not another's
5. Confirm the test suite passes with no external MLflow reachable

**Pass criteria**
- Port conflict produces an immediate, unambiguous error
- MLflow data is project-scoped
- Tests pass without live MLflow

**Fail criteria**
- Any component silently binds to or reads from another project's resources

**This is the case most likely to yield a genuine finding.** Silent cross-binding is precisely the class of defect that survives every test run on a familiar machine and destroys the first one on a stranger's.

---

### Section D — Verification

#### CC-VER-01 · Test suite

**Steps**
1. Run the full suite per documentation
2. Record pass/fail counts and duration
3. Note any test requiring external services

**Pass criteria**
- Suite runs from a clean clone
- No unexplained failures
- Any external dependency is documented as such

**Record:** actual count against the documented count. A discrepancy is a finding in either direction.

---

#### CC-VER-02 · Evaluation reproduction

The claim under test: **can a reviewer regenerate the numbers the project reports?**

**Steps**
1. Run the retrieval benchmark
2. Compare against documented Config A / Config C figures (0.5333 / 0.5280)
3. Attempt to reproduce A5's committed artifacts using the three `run_a5_*.py` scripts

**Pass criteria**
- Benchmark reproduces documented NDCG to reasonable precision
- A5 scripts execute and regenerate their JSON outputs

**Deliberate scope limit.** Only A1 and A5 are artifact-backed. A2, A3, A3-2, A4 and A6 are recorded in `docs/uat/round3_family_a_findings.md` as analysis-only, with no scripts retained — **they are out of scope here, and their absence is not a defect of this test.** Attempting to reproduce them would fail by design, and the findings document already flags them.

**Record:** whether reproduction requires a Cohere API key, and whether the documentation says so. Config C is a hosted API; a reviewer without a key can only reproduce half this criterion, and that limitation should be stated rather than discovered.

---

## 4. Findings classification

| Class | Definition | Example |
|---|---|---|
| **BLOCKER** | Prevents reproduction entirely | Corpus unobtainable |
| **MAJOR** | Requires undocumented knowledge to resolve | Torch index URL needed but unmentioned |
| **MINOR** | Cosmetic or easily inferred | Wrong path in an example |
| **UNDOCUMENTED PREREQUISITE** | Step performed from prior knowledge, absent from docs | Stopping another project's containers |

The last class is the one this protocol exists to surface. It will be **underreported unless deliberately watched for**, because supplying a missing step from memory is close to involuntary for someone who built the system. Every entry in this class is a defect a stranger would hit.

---

## 5. Deliverables

1. **Findings register** — every case, its outcome, evidence
2. **Undocumented prerequisites list** — the primary output
3. **Minimum specification** — RAM, disk, ports, external accounts, derived from observation
4. **README/setup diff** — the specific changes the findings demand
5. **Reproducibility statement** — a defensible sentence on whether the repository is reproducible by a stranger, stated with the same directness the sign-off applied to its own results

---

## 6. What a good outcome looks like

**Not** "everything passed."

A clean-clone test that finds nothing has most likely been run by someone who supplied missing steps unconsciously. The realistic good outcome is **three to eight undocumented prerequisites, one or two majors, and a clear view of whether the corpus is obtainable** — followed by a documentation change that closes them.

The project's own history supports that expectation. The 42-case functional test found 18 defects on a system its author believed was working. This protocol tests something less exercised than the software: the instructions.

> Estimated effort: **half a day** to execute, **half a day** to remediate and re-verify §CC-ACQ-02 and any BLOCKER.

---

*Protocol status: v1.0, planned, not executed. Findings to be recorded against this document and versioned as the sign-off was — v2.0 replacing inference with measurement wherever v1.0 assumed.*
