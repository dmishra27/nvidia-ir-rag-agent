# DVC, Chronologically
### The whole data-versioning story in one pass, simply

*nvidia-ir-rag-agent · 10 July – 22 August 2026*

---

## First: the only concept you need

DVC splits your data into **three separate places**. Almost every confusion about DVC comes from mixing them up, so here they are once:

```
   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
   │   1. YOUR DISK      │   │   2. GIT            │   │   3. THE REMOTE     │
   │                     │   │                     │   │                     │
   │  The actual PDFs    │   │  Small text files   │   │  A copy of the PDFs │
   │  4.6 MB, 7.9 MB…    │   │  called POINTERS    │   │  stored somewhere   │
   │                     │   │  ~110 bytes each    │   │  Git isn't          │
   │  Only you have      │   │                     │   │                     │
   │  these              │   │  EVERYONE who       │   │  Anyone can fetch   │
   │                     │   │  clones gets these  │   │  from here          │
   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

A **pointer** is a tiny text file holding a fingerprint of the real file:

```
outs:
- md5: 3f2a9c...          ← fingerprint of the actual bytes
  size: 4660484
  path: cuda_c_programming_guide.pdf
```

**The rule:** a stranger clones your repo and gets box 2 only. `dvc pull` reads those pointers and fetches the matching files from box 3. If either box is missing or empty, they get nothing.

**Your project had box 1 only.** For 43 days.

---

## The chronology — only the commits that matter

Of 63 commits, seven touch this story.

| Date | Commit | What happened | Effect on DVC |
|---|---|---|---|
| 10 Jul | `1bc7f0c` | Project initialised, first `.gitignore` created | — |
| 10 Jul | `921281c` | `dvc add` run on the PDFs — *"DVC-track nvidia PDF corpus"* | Box 1 ✓ · **Pointer created but never staged into Git** |
| 10 Jul | `daa4ae9` | Ingest runner built → **5,389 chunks** | Downloads from NVIDIA URLs, ignores `data/raw/` |
| 10 Jul 20:33 | **`4ebf3af`** | **One line added to `.gitignore`: `data/`** | **Box 2 now permanently blocked** |
| 12 Jul | `4ba23cf` | Round 1 UAT — NVLink and H100 queries fail | Blamed on "corpus coverage gaps" (wrong — see below) |
| 16 Aug | `8569bfb` | **Capstone submitted** | Still boxes 2 and 3 empty |
| 22 Aug | **`7bc6730`** | **The fix** | Boxes 2 and 3 filled · verified from a clean clone |

**Two independent failures on 10 July, four commits apart.** The pointer wasn't staged at `921281c`, and then `4ebf3af` made staging it impossible. Either one alone would have broken this.

---

## What it looked like at each stage

### Stage 1 — after 10 July, and for the next six weeks

```
   YOUR DISK                    GIT                      REMOTE
   ┌──────────────┐            ┌──────────────┐         ┌──────────────┐
   │ 6 PDFs ✓     │            │  (nothing)   │         │  (nothing)   │
   │              │            │              │         │              │
   │ raw.dvc ✓    │──blocked──▶│   ✗ .gitignore         │  ✗ no remote │
   │ (pointer)    │   by       │     says     │         │    ever      │
   │              │  data/     │    "data/"   │         │  configured  │
   └──────────────┘            └──────────────┘         └──────────────┘
          │
          └── and the ingest script doesn't read this folder anyway.
              It downloads fresh from docs.nvidia.com every run.
```

**What a stranger cloning the repo got:** no PDFs, no pointers, no fingerprints — no evidence the corpus was ever versioned at all. They'd run the ingest, download whatever NVIDIA publishes *that day*, and get some chunk count. Possibly not 5,389.

**And nothing complained.** Git doesn't warn you when it ignores a file you meant to keep — ignoring files is its job.

### Stage 2 — after the fix, 22 August

```
   YOUR DISK                    GIT                      REMOTE
   ┌──────────────┐            ┌──────────────┐         ┌──────────────┐
   │ 8 PDFs ✓     │            │ 8 pointers ✓ │◀───────▶│ 8 blobs ✓    │
   │              │───────────▶│              │  linked │              │
   │ 8 pointers ✓ │  committed │ .dvc/config  │   by    │ on branch    │
   │              │            │ has remote ✓ │ finger- │ dvc-storage  │
   └──────────────┘            └──────────────┘  print  └──────────────┘
          ▲                                                     │
          └──── ingest now reads here first ───┐                │
                                               │   dvc pull ────┘
                                       (fetches and verifies)
```

**What a stranger gets now:** clone → 8 pointers → `dvc pull` → the exact PDFs, fingerprint-verified. No account, no credentials, no asking you.

**Verified for real** — cloned fresh into a temp directory with no cache, pulled, checked all 8 files. Not inspected. Tested.

---

## The one line that caused it

```diff
  *.log
  verify_phase_c*.py
+
+ data/
```

That's the entire defect. Commit message: *"chore: exclude data/ directory from git — DVC-tracked."*

**The reasoning was right.** PDFs shouldn't be in Git — that's precisely why DVC exists. Excluding them is correct practice.

**The rule was two characters too broad.** `data/` means *everything* in that folder. The PDFs, yes — intended. The pointer files, also yes — not intended, and not noticed.

This is why it survived six weeks: the commit looks correct on review. You'd approve it. The failure is invisible from the machine that made it, because on that machine everything works — the files are there, DVC tracks them, `dvc status` is clean. **It only shows up from outside**, and nobody had looked from outside.

---

## The bonus casualty

On 12 July, Round 1 testing ran nine queries. Two failed completely, including *"NVLink 4.0 bandwidth specifications."* The write-up concluded:

> *"the corpus has no GPU-architecture whitepaper with NVLink 4.0's actual bandwidth numbers. Also a coverage gap, not a ranking bug."*

**The whitepapers were on disk.** `hopper_architecture_whitepaper.pdf` and `ampere_architecture_whitepaper.pdf` were downloaded and DVC-tracked on 10 July — two days *earlier*. They were never ingested, because the ingest reads a hardcoded URL list rather than `data/raw/`.

So it wasn't a scope decision. It was a plumbing gap that looked exactly like one — and a test in the codebase now asserts NVLink and H100 are out of scope, which passes, and quietly encodes the accident as an intention.

---

## Verify any of this yourself

```powershell
# The line that caused it
git --no-pager show 4ebf3af -- .gitignore

# Proof the pointer was NEVER committed — returns nothing at all
git --no-pager log --all --oneline -- data/raw.dvc

# When the pointers first appeared
git --no-pager log --oneline -- data/raw/cuda_c_programming_guide.pdf.dvc

# The corpus survived it all
docker exec -it nvidia-ir-rag-agent-postgres-1 `
  psql -U nvidia_ir -d nvidia_ir_db -c "SELECT COUNT(*) FROM chunks;"
```

The second command is the important one. **Empty output means the file never existed in any commit, on any branch, ever** — which is the whole finding in one line.

---

## Before and after, plainly

| | Before | After |
|---|---|---|
| PDFs on your disk | ✓ | ✓ |
| Pointers in Git | ✗ *silently ignored* | ✓ 8 committed |
| Remote configured | ✗ *`.dvc/config` empty* | ✓ `dvc-storage` branch |
| Ingest uses pinned files | ✗ *downloads every run* | ✓ local first |
| A stranger can reproduce | ✗ | ✓ **tested, not assumed** |
| Corpus size | 5,389 | **5,389** *(unchanged)* |

---

## If someone asks

> *The corpus is five NVIDIA manuals, 5,389 chunks, pinned with DVC and reproducible from a clean clone with no credentials — I verified that by cloning fresh and hashing every file.*
>
> *Getting there turned up three defects. The DVC pointer files had been silently swallowed by a `.gitignore` rule, so the versioning layer was invisible to anyone cloning. No remote had ever been configured. And the ingest downloaded from live URLs instead of reading the pinned copies — so the corpus was whatever NVIDIA published that day, with no way to detect drift.*
>
> *It also corrected an earlier misdiagnosis: two queries recorded as "corpus coverage gaps" were documents that had been downloaded and tracked but never ingested.*

The transferable point: **a tool being installed is not the same as a tool working**, and the only way to tell the difference is to test it from outside the machine that set it up.

---

*Fixed at `7bc6730`. Full technical detail: `docs/explainers/data_versioning_explained.md`. Defect records: DEF-19, DEF-20, DEF-24, DEF-25.*
