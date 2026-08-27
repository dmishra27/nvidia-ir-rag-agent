# Clean-Clone Reproducibility Test — Findings
### nvidia-ir-rag-agent

**Reference:** CCT-NVIR-2026-001
**Version:** 2.0 — executed
**Executed:** 25 August 2026
**Commit under test:** `9d267e3` (origin/main)
**Environment:** Windows 11, 7.65 GB RAM, Docker Desktop, fresh clone to `%TEMP%\cleantest`
**Protocol:** `docs/uat/clean_clone_test_protocol.md`

---

## Executive summary

Six of nine cases executed. Three blocked by hardware constraints that are themselves a finding.

**The repository is reproducible — the corpus and retrieval results match exactly — but the documented default configuration does not work on a clean clone.** A reviewer following `setup.md` verbatim gets `{"results":[]}` from the project's own example query, with HTTP 200 and no error.

**Headline finding:** a dense-retrieval failure discards already-succeeded BM25 results. The README's own Known Limitations section predicts this; the adjacent Setup section contradicts it. The pessimistic statement is the correct one.

**Strongest positive:** ingestion reproduced **5,389 chunks** exactly, with per-document counts matching to the unit. Every evaluation figure in the project is anchored to a corpus a stranger can rebuild.

| | Result |
|---|---|
| BLOCKER | 1 |
| MAJOR | 6 |
| MINOR | 2 |
| UNDOCUMENTED PREREQUISITE | 5 |
| Retracted during testing | 4 |

> **Status as of 27 Aug 2026:** F-14 and F-15 are fixed. Three of the twelve §4 documentation changes are closed (D3, D4, D5); remaining work is D1, D2, and D6–D12.

---

## 1 · Findings register

### CC-ACQ-01 · Clone and orient — **PASS**

README is unusually strong: problem statement leads, criteria-to-evidence mapping is explicit, Known Limitations is candid. A stranger can state the project's purpose after one read and locate `setup.md` immediately.

| ID | Finding | Class |
|---|---|---|
| **F-01** | README states topics outside the five documents are *"out of scope **by construction**"*, naming NVLink, H100, TensorRT. Per DEF-19/DEF-20 these are ingestion gaps, not scope decisions — the Ampere and Hopper whitepapers are DVC-tracked but never ingested; TensorRT's URL 404s. Presents a plumbing defect as a design choice. | **MAJOR** |
| **F-02** | No mention of DVC anywhere. Reproducibility row cites pinned dependencies and the committed BM25 index but omits `dvc pull` entirely. A reviewer has no idea the corpus is version-controlled or how to fetch it. | **MAJOR** |
| **F-03** | Test count disagrees across three documents: README 520, `setup.md` 508, functional sign-off 501. <br>*(27 Aug: suite now 550 tests, was 533 at time of test; README still 520, `setup.md` still 508 — D11 open.)* | **MINOR** |

---

### CC-ACQ-02 · Dataset acquisition — **PASS** *(closed 22 Aug)*

Fresh clone, no pre-existing cache, `dvc pull` retrieved all 8 blobs, sha256-verified identical. No credentials, no cloud account. Closed at `7bc6730`; see `docs/explainers/data_versioning_explained.md`.

---

### CC-ACQ-03 · Dependency installation — **PASS, with conditions**

Install succeeded: **34.3 minutes**, **2.54 GB**, `pip check` clean, all 384 pins resolved on Python 3.11.9.

| ID | Finding | Class |
|---|---|---|
| **F-04** | Install fails on Python 3.14 — `pandasai==3.0.0` has no compatible release, and the resolver emits a wall of `Requires-Python` exclusions that never names the version as the cause. `setup.md` §0 states Python 3.11 in prose; **nothing enforces it** — no `.python-version`, no `python_requires`, no runtime check. `python -m venv` uses whatever `python` resolves to. | **BLOCKER** |
| **F-05** | Install takes 34.3 minutes. `setup.md` §3 warns the *Docker* build takes 5–15 min but says nothing about the local pip path. | **UNDOC PREREQ** |
| **F-06** | Venv consumes 2.54 GB. No disk requirement stated anywhere. | **UNDOC PREREQ** |
| **F-07** | The shell fell out of the venv **three times** during this test (new window, `deactivate`, directory change), each time silently resolving to system Python 3.14. Every resulting error looked like a project defect: `ModuleNotFoundError: No module named 'fitz'`, missing `pymupdf`, unresolvable `pandasai`, a 2-hour resolver stall. **All four were false.** Someone who built this project misdiagnosed it repeatedly in a single session. | **UNDOC PREREQ** |

**Credits:** the torch CPU wheel index is correctly documented in §4 and worked as written. `patch_ragas.py` is documented and ran cleanly. All 384 pins resolve without conflict.

---

### CC-ING-01 · Infrastructure start — **PASS, with conditions**

All nine services reached healthy after recovery. Image build 588 s (9.8 min), inside the documented 5–15 min range. API healthy in 25.3 s.

| ID | Finding | Class |
|---|---|---|
| **F-08** | **Two instances of this project cannot run concurrently.** With the working repo's containers up, `docker compose up -d` fails on qdrant/6333 (loud, aborts) and postgres/5432 (silent — the container never starts and nothing names it). `setup.md` §6 documents the port map without stating the constraint. The API was moved to 8001 to dodge a known 8000 collision; 5432, 6333, 5001, 8501 kept defaults. | **MAJOR** |
| **F-09** | **An aborted `up` leaves containers off the network, and the retry reports success without repairing it.** After the collision, `docker compose up -d` reported all nine `Healthy` while `docker network inspect` showed only six attached — postgres and qdrant absent. The API could not resolve `postgres`; `/search` failed with `[Errno -2]` after a 10 s timeout. **A full `docker compose down` was required.** The documented recovery (stop the conflicting containers, re-run `up`) is insufficient. | **MAJOR** |
| **F-10** | `pgadmin` exited 255 during the initial `up` and stayed dead **four hours** unnoticed, because `up -d` had reported success. Compose's output is a launch report, not a health assertion. | **MAJOR** |
| **F-11** | **Full stack leaves 0.46 GB free on a 7.65 GB host.** `setup.md` §0 states no RAM requirement. The project's own docs treat <200 MB as a hard stop; DEF-23 records a run reaching 17 MB. **The full stack and the ingestion pipeline do not comfortably coexist on the machine this project was built on.** | **UNDOC PREREQ** |
| **F-12** | Restarting a single service with the stack up **OOM-killed the API (exit 137)**. Completing the test required stopping airflow, phoenix, jaeger and streamlit — a deviation from the documented path, and the only way to finish on this hardware. | **MAJOR** |

---

### CC-ING-02 · Corpus ingestion — **PASS** ✅

**The strongest result in the test.**

```
docs=5  chunks=5389  coverage_pct=100.0  failed_doc_ids=[]
```

| Document | Chunks | Expected |
|---|---|---|
| CUDA C++ Programming Guide | 1,670 | 1,670 ✓ |
| CUDA Runtime API | 1,642 | 1,642 ✓ |
| CUDA Math API | 1,232 | 1,232 ✓ |
| Nsight Systems User Guide | 561 | 561 ✓ |
| CUDA C++ Best Practices Guide | 284 | 284 ✓ |
| **Total** | **5,389** | **5,389 ✓** |

No manual intervention. **A stranger cloning today rebuilds the exact corpus every evaluation figure is anchored to.** This validates the DVC pinning work end to end.

---

### CC-ING-03 · Index construction — **PARTIAL**

BM25 path verified end to end (see CC-EXE-01). Dense path untested — `populate_qdrant.py` is a documented ~86-minute step and was not run given F-11 and F-12.

| ID | Finding | Class |
|---|---|---|
| **F-13** | Qdrant's `nvidia_ir_chunks` collection does not exist after `docker compose up -d` + ingestion. It is created only by `populate_qdrant.py`. Consequence in F-14. | **MINOR** *(documented behaviour)* |

---

### CC-EXE-01 · API service — **FAIL** ❌

`/health` returns `{"status":"ok"}`. `/search` on the documented example query returns `{"results":[]}` with HTTP 200.

| ID | Finding | Class |
|---|---|---|
| **F-14** | **A dense-retrieval failure discards working BM25 results.** With the default `RERANKER_MODE=live_fast`, `retrieve` calls dense, receives `404 Collection 'nvidia_ir_chunks' doesn't exist!`, and the entire retrieval fails — including the BM25 half, which has its committed index and 5,389 chunks and answers correctly. **Proven** by setting `RERANKER_MODE=fallback`, which returns `cc6c8e53936d04e9b192a7d5` — the exact `cudaMalloc(void**, size_t)` signature chunk — at rank 1, scores `1/61`, `1/62`, `1/63`. The README's Known Limitations predicts precisely this; the Setup section's claim that *"`/search` returns real BM25 results immediately after `docker compose up -d`"* is **false on the default configuration**. **Fix: degrade to BM25-only when dense fails.** | **MAJOR** |
| **F-15** | Retrieval failure returns **HTTP 200 with an empty list** — indistinguishable from "no matches found". A reviewer concludes search doesn't work. Should be 5xx or carry an error field. | **MAJOR** |
| **F-16** | `/health` returned `{"status":"ok"}` while Postgres was unreachable (every request logging `request_log_write_failed`) and while retrieval was failing. The endpoint asserts nothing about dependencies. Same class as DEF-01/DEF-16 — instrumentation present, not wired to what it claims. | **MAJOR** |
| **F-17** | First `/search` took **281 s** on a healthy stack (cold model loads under memory pressure); 101 s on the earlier broken-network attempt. No warning of first-request cost. | **UNDOC PREREQ** |

> **F-14 · CLOSED** and **F-15 · CLOSED** (27 Aug 2026). `/search` degrades to BM25-only when dense fails — `7f3107d`; `/ask` does the same — `f832dcc`; total retrieval failure now returns HTTP 503 with an error body instead of 200 + `[]`. The shared `agents/hybrid_retrieve.py` extraction — `1c958a1` — removed the `retrieval_agent` / `qa_agent` duplication that had let F-14 survive its first fix.

**Credit:** DEF-03's `reranker_mode` echo works correctly and was decisive in diagnosing F-14.

---

### CC-EXE-02 / CC-EXE-03 / CC-VER-01 / CC-VER-02 — **BLOCKED**

Not executed. Remaining cases need the full stack, which OOM-killed the API (F-12). Per §2.1, recorded as a finding rather than worked around.

---

## 2 · Undocumented prerequisites

The protocol's primary output. Each is a step or condition a reviewer needs and the docs don't state.

1. **Python 3.11 specifically** — install fails on 3.14 with an error that never names the version (F-04)
2. **Verify the venv before every step** — `python --version`; falling out produces convincing false failures (F-07)
3. **~35 minutes for the local pip install** (F-05)
4. **~2.6 GB disk for the venv**, plus ~3 GB of Docker images (F-06)
5. **No other instance of this project running** — port collisions, one silent (F-08)
6. **`docker compose down`, not just `up`, after any failed start** (F-09)
7. **`docker compose ps` after `up`** — success output doesn't mean services are alive (F-10)
8. **~2 GB free RAM for the full stack**; more for ingestion (F-11)
9. **Don't restart individual services with the full stack up** — OOM (F-12)
10. **`RERANKER_MODE=fallback`, or run `populate_qdrant.py` first**, or `/search` returns nothing (F-14)
11. **First `/search` takes minutes**, not seconds (F-17)

---

## 3 · Minimum specification

Derived from observation, not estimate. None of this is currently documented.

| | Requirement | Evidence |
|---|---|---|
| **Python** | 3.11 exactly | Install fails on 3.14 (F-04) |
| **RAM** | 4 GB free minimum; 8 GB total is marginal | 0.46 GB free with stack up; API OOM on restart (F-11, F-12) |
| **Disk** | ~6 GB | 2.54 GB venv + ~3 GB images |
| **Ports** | 8001, 8501, 5432, 5050, 6333, 6334, 5001, 8080, 16686, 4317, 4318, 6006 | `setup.md` §6, verified |
| **Time to first query** | ~50 min | 34 min install + 10 min build + ~5 min ingest |
| **Time to full pipeline** | ~2 h 20 min | above + ~86 min embedding |
| **External accounts** | None for BM25 search. `ANTHROPIC_API_KEY` for `/ask`; `COHERE_API_KEY` for Config C benchmark | `setup.md` §2 |

---

## 4 · Recommended documentation changes

| # | Change | Closes |
|---|---|---|
| **D1** | Add a **Requirements** section: Python 3.11 exactly, 4 GB free RAM, 6 GB disk, ~50 min setup | F-04, F-05, F-06, F-11 |
| **D2** | Add a runtime guard to `run_ingest_direct.py` and `populate_qdrant.py` — fail fast with *"requires Python 3.11, found X"*. Would have prevented four false findings in this test | F-04, F-07 |
| **D3** | State that `/search` needs **either** `populate_qdrant.py` **or** `RERANKER_MODE=fallback`. Remove or qualify the "works immediately" claim | F-14 — **DONE** `7f3107d` |
| **D4** | Fix `agents/retrieval_agent.py` to degrade to BM25-only on dense failure — the actual bug behind F-14 | F-14 — **DONE** `7f3107d`, refactored `1c958a1` |
| **D5** | Return 5xx or an error field when retrieval fails, rather than 200 with an empty list | F-15 — **DONE** `7f3107d` (/search), `f832dcc` (/ask) |
| **D6** | Make `/health` assert its dependencies, or add `/health/ready` that does | F-16 |
| **D7** | Add to §3: run `docker compose ps` after `up`; use `down` before retrying a failed start | F-09, F-10 |
| **D8** | Note that only one instance can run at a time, and which ports collide | F-08 |
| **D9** | Add a **Data versioning** section — corpus DVC-pinned, `dvc pull`, no credentials; note blobs live on the `dvc-storage` orphan branch | F-02 |
| **D10** | Reword the corpus scope claim — NVLink/H100/TensorRT are ingestion gaps (DEF-19/20), not scope decisions | F-01 |
| **D11** | Reconcile the test count across README, `setup.md`, and the sign-off | F-03 |
| **D12** | Warn that the first `/search` takes minutes | F-17 |

---

## 5 · Reproducibility statement

> **The repository is reproducible in substance and unreliable in practice.**
>
> A competent stranger with only the public clone URL can obtain the exact corpus (`dvc pull`, 8 blobs sha256-verified), rebuild it to the exact chunk count (5,389, per-document counts matching to the unit), and retrieve the exact expected result (`cc6c8e53…` at rank 1 for the documented example query). **The reproducibility claim holds under test, not merely under assertion.**
>
> But they will not get there by following the documentation. The documented default configuration returns `{"results":[]}` with HTTP 200 on the project's own example query, because a dense-retrieval failure discards working BM25 results — a limitation the README states plainly in one section and contradicts in another. Reaching a working search required: knowing to use Python 3.11 exactly, a full `docker compose down` after a partial start, stopping four services to free memory, and setting an environment variable the docs never mention.
>
> **Eleven undocumented prerequisites separate "the code works" from "a stranger can run it."** None is difficult. All are invisible from the machine that built the project.

---

## 6 · Retractions

Four findings were recorded and withdrawn during testing. Recorded here because the *reason* is itself the most reproducible defect found (F-07).

| Retracted | Cause |
|---|---|
| ~~BLOCKER: ingestion fails, `No module named 'fitz'`~~ | Shell had fallen out of the venv; `fitz` imports correctly under 3.11 |
| ~~BLOCKER: 152 of 384 packages installed~~ | `pip list` run under system Python 3.14 |
| ~~MAJOR: `pymupdf` silently absent~~ | Same |
| ~~MAJOR: dependency resolution pathologically slow (2 h)~~ | `--dry-run` under 3.14 backtracking against an impossible `pandasai` constraint |

Each was plausible, each pointed at a real-looking defect, and each was wrong. **The tester built this project and misdiagnosed it four times in one session** — which is the strongest available argument for D2, a runtime version guard.

---

## 7 · Assessment against the protocol's own expectations

§6 predicted *"three to eight undocumented prerequisites, one or two majors."*

Observed: **eleven undocumented prerequisites, six majors, one blocker.** More than expected in both categories.

§6 also warned that finding nothing would mean the test was run wrong. That risk did not materialise — though F-07 shows the opposite hazard is real, and four findings had to be withdrawn on re-examination.

---

*Executed 25 August 2026 against `9d267e3`. Six of nine cases completed; four blocked by the hardware constraint recorded as F-11 and F-12.*
