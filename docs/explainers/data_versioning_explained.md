# The Data Versioning Story
### What was broken, what we found, and what it means — in plain language

*nvidia-ir-rag-agent · 22 August 2026 · **Status: complete, verified** — fix committed at `7bc6730`*

---

## Part One: The problem data versioning solves

Your project turns five NVIDIA PDF manuals into **5,389 searchable chunks of text**. Every number the project reports is anchored to that corpus. The benchmark scores, the relevance labels, all seven Family A experiments — every one of them means "against *these* 5,389 chunks."

So a question arises immediately: **if someone else runs your project, do they get the same 5,389 chunks?**

If yes, they can check your work. If no, they get *a* system that *sort of* resembles yours, and none of your reported figures apply to what they're looking at. Your evaluation becomes unverifiable — not wrong, just unconfirmable, which in research terms is nearly as bad.

That's the problem data versioning solves. Git handles this for code: `8569bfb` is exactly the code you submitted, and always will be. Git is bad at large binary files like PDFs, which is why a companion tool exists.

### What DVC actually does

**DVC (Data Version Control)** works by splitting the problem in two.

The heavy files — your PDFs — go somewhere Git isn't: cloud storage, a separate branch, wherever. In their place, DVC leaves a small text file called a **pointer**, and *that* goes in Git.

A pointer looks roughly like this:

```
outs:
- md5: 3f2a9c...
  size: 4660484
  path: cuda_c_programming_guide.pdf
```

Three facts: a fingerprint, a size, a filename. A few hundred bytes standing in for a 4.6 MB PDF.

The fingerprint is the important part. It's computed from the file's actual bytes, so **any change to the PDF produces a different fingerprint.** That's what makes the pointer a *version* rather than just a name.

Then two commands:

- **`dvc push`** — send the real files to storage
- **`dvc pull`** — read the pointers, fetch the matching files, verify each fingerprint

The result: someone clones your repository, gets pointers instead of PDFs, runs `dvc pull`, and receives the exact files you used. Verified byte for byte.

---

## Part Two: What was actually happening

DVC was set up on **10 July 2026**, the first day of the project. Commit `921281c` reads *"DVC-track nvidia PDF corpus — 5 PDFs verified clean."*

So the intent was right from day one. But **none of it worked**, for three separate reasons — and each one alone would have been enough to break it.

### Problem 1 — The pointer files were never saved

This is the one that surprised us, and it's the most serious.

A commit on the same day, `4ebf3af`, added a rule to `.gitignore` — the file that tells Git what to ignore:

```
data/
```

Meaning: *ignore everything in the `data` folder.*

That's a sensible-looking rule. The PDFs live in `data/raw/`, and PDFs shouldn't go in Git — that's the whole reason DVC exists. The rule reads like exactly the right thing to do.

**But DVC's pointer files live in that folder too.** And the rule doesn't distinguish. It ignored the PDFs, which was intended, and it silently ignored the pointers, which was not.

So `data/raw.dvc` — the file recording which PDFs the project uses and what their fingerprints are — **was never committed to Git.** Not once. It existed on your machine and nowhere else.

Consider what that means for anyone cloning your repository. They get no PDFs, which is expected. They also get **no pointers** — so there's no record that the PDFs ever existed, no fingerprints, nothing for `dvc pull` to act on. The data-versioning layer was completely invisible to the outside world.

And nothing complained. Git doesn't warn you about ignoring a file you meant to keep — ignoring files is its job. This sat silently for six weeks.

### Problem 2 — No storage was ever configured

`dvc push` needs somewhere to push *to*: a cloud bucket, a server, something.

DVC records that destination in `.dvc/config`. We checked yours:

```
(empty)
```

No storage was ever set up. So even if the pointers *had* been committed, `dvc pull` would have found them and had nowhere to fetch from.

### Problem 3 — Nothing read the versioned files anyway

The deepest one, and the easiest to miss.

Your ingestion script — the code that reads PDFs and produces the 5,389 chunks — **never looks at `data/raw/`.** It downloads fresh copies from NVIDIA's website into a temporary folder each time it runs.

So even with problems 1 and 2 fixed, the pinned PDFs would have sat there unused. The corpus would still have come from live web downloads.

### The three problems together

Think of it as a chain with three broken links:

```
   PDFs                pointers           storage            the code
   pinned by DVC  →    in Git        →    for retrieval  →   reads them
        ✓                  ✗                   ✗                 ✗
                    swallowed by        never configured    downloads from
                    .gitignore                              the web instead
```

Only the first link worked. Which is why this was hard to spot: `dvc add` had been run, the files *were* tracked locally, and everything looked fine from your machine. The failure only becomes visible from outside — and nobody had looked from outside.

---

## Part Three: Why it mattered

Here's the thing that makes this more than housekeeping.

Your ingestion downloads from URLs like `https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf`. That URL always serves **whatever NVIDIA publishes today.** It isn't pinned to a version.

NVIDIA revises these documents with every CUDA release. Your corpus came from **CUDA Runtime API v13.3.1** — that version string is embedded in your chunk text.

So picture someone running your project in six months. NVIDIA has shipped v14.0. The URL serves the new edition. Your script downloads it, chunks it, and produces… some other number. Not 5,389.

Every downstream figure then refers to a corpus that person doesn't have:

- The 0.5333 and 0.5280 benchmark scores
- All the relevance labels
- The A5 finding that two re-rankers disagree 53% of the time on the top result
- Every Family A verdict

None of it would be wrong. It would be **unverifiable**, which for a project whose central argument is *"here is the script, here is the output, run it yourself"* is close to fatal.

**And there'd be no warning.** The script would run cleanly, report success, and hand over a subtly different system. Silent failures are the expensive kind.

### One piece of luck

When we checked, all five PDFs still matched their original page counts exactly — 598, 722, 468, 118, 344. NVIDIA hadn't revised any of them in the six weeks since ingestion.

So pinning them now captured the genuine original corpus. Six months later that might not have been true, and the original would have been unrecoverable.

---

## Part Four: What we did

### Fix 1 — Made the pointers visible

Removed the blanket `data/` rule and switched to per-file tracking. DVC now writes precise ignore rules itself — one per file, covering the PDF and nothing else. The pointers are committed and visible to anyone cloning.

Eight files are tracked: the five that make up the corpus, plus three hardware documents (the A100 datasheet, and the Ampere and Hopper whitepapers) that are tracked but deliberately *not* ingested. More on those below.

### Fix 2 — Set up storage that works for strangers

The requirement was specific: someone with nothing but your public repository URL — no account, no credentials, no request to you — must be able to fetch the corpus.

The solution: a separate branch in your own repository called `dvc-storage`, holding the files. DVC pulls from it over plain HTTPS. No cloud account, nothing to expire, nothing to pay for. About 19 MB, kept off the main branch.

**A near-miss worth recording.** Windows has a habit of "helpfully" converting line endings in text files. It was about to do that to a PDF, which would have corrupted it. Caught before pushing and fixed with a `.gitattributes` rule declaring the files binary.

That failure mode is worth understanding: `dvc pull` would have **succeeded**, and handed over a damaged PDF. No error, no warning, just a file that doesn't work. Exactly the kind of silent failure this whole exercise is about preventing.

### Fix 3 — Made the code use the pinned files

The ingestion now checks `data/raw/` first and only downloads when a file is genuinely missing. The pinned copies are the source of truth; the URLs are a fallback for bootstrapping.

This is the link that makes the other two matter. Without it, the pinning would be decorative.

### Fix 4 — Made silent failures loud

Three related repairs, all of the same character:

**The corpus count is now checked.** Your script declared eight source documents; three had 404'd since at least October 2025. It caught the failures, logged a warning, and reported success with five. It now fails if the expected count isn't met, and the three dead URLs have been removed with a comment explaining why.

**File contents are now fingerprinted.** The script previously identified documents by hashing their *URL*, which cannot detect a revised file — same URL, different content, same identifier. It now records a SHA-256 of the actual bytes in the database, so drift is detectable.

**The exit code was backwards.** Success was reported based on "did we write new records?" — so a correct re-run against an unchanged corpus, which correctly writes nothing, reported *failure*. Now keyed to whether any document actually failed.

### Fix 5 — Proved it works

Not "checked the configuration." Actually tested it, from a genuinely clean state:

1. Cloned the repository fresh into a temporary directory
2. Confirmed there was no cached data — only the eight pointer files, no PDFs
3. Ran `dvc pull`
4. All eight files arrived
5. Verified each one's fingerprint against the originals — identical, including the PDF the line-ending fix had protected

That is the actual claim, demonstrated rather than assumed: **a stranger with only your public URL gets the exact corpus you used.**

### Fix 6 — Confirmed the corpus survived the surgery

The whole point of pinning is that the corpus doesn't change. After every fix was in place, the database still reads:

```
SELECT COUNT(*) FROM chunks;  →  5389
```

Unchanged. That's the number every benchmark figure, every relevance label, and every Family A verdict is anchored to, and it came through all of this untouched.

### One honest distinction about the pin

Two of the five PDFs weren't in `data/raw/` when this work started — the Best Practices Guide and the Nsight Systems User Guide — so they were downloaded fresh from NVIDIA. They're now larger than the July originals (2.3 MB and 10.9 MB).

Their **page counts match the database exactly** (118 and 344), and the chunk count is unchanged, so the text NVIDIA serves today extracts to the same content that was ingested in July. But the file *bytes* differ — NVIDIA re-renders these PDFs periodically without changing the content.

So the pin is **faithful in content, not byte-identical to the July originals.** Those originals no longer exist anywhere, so byte-identity was never available. Worth knowing the distinction if someone asks precisely what was verified: page counts and resulting chunk counts, not the original bytes.

For the three PDFs that *were* already in `data/raw/`, the pin is byte-exact — they've sat untouched since 10 July.

---

## Part Five: A misdiagnosis this uncovered

There's a bonus finding, and it's a good illustration of how a plumbing defect can disguise itself as a design decision.

Your first round of testing, back on 12 July, ran nine queries. Two failed completely, including *"NVLink 4.0 bandwidth specifications."* The write-up concluded:

> *"the corpus has no GPU-architecture whitepaper with NVLink 4.0's actual bandwidth numbers. Also a coverage gap, not a ranking bug."*

Reasonable-sounding. The corpus is CUDA programming documentation; hardware specs are out of scope.

**Except the whitepapers were sitting right there.** `hopper_architecture_whitepaper.pdf` and `ampere_architecture_whitepaper.pdf` had been downloaded and tracked on 10 July — two days *before* that test ran. They were never ingested, because the ingestion reads from a hardcoded URL list, not from the folder.

So it wasn't a scope decision. It was an ingestion gap that *looked* like one.

**And it got locked in.** A test in your codebase now asserts that NVLink, H100 and TensorRT are out of scope. That test passes, and it correctly describes current behaviour — but it encodes an accident as an intention, which makes it considerably harder to notice or reverse later.

The three hardware PDFs remain tracked and deliberately unused. Ingesting them is a real decision with a real cost: it would change the corpus, which would invalidate 5,389 and every figure attached to it. Worth doing perhaps, but deliberately, not as a side effect of a cleanup.

---

## Part Six: Before and after

| | Before | After |
|---|---|---|
| Pointer files in Git | ✗ Silently ignored for six weeks | ✓ Committed and visible |
| Storage configured | ✗ None | ✓ Public branch, no credentials |
| Ingestion uses pinned files | ✗ Downloaded from the web every run | ✓ Local first, download as fallback |
| Reproducible by a stranger | ✗ Whatever NVIDIA publishes that day | ✓ Verified fresh-clone test |
| Missing documents | ✗ Warning logged, success reported | ✓ Run fails |
| Detects a revised PDF | ✗ Identified by URL only | ✓ Content fingerprint stored |
| Corpus documented | ✗ Code claimed eight, database had five | ✓ Five, with the three gaps named |
| Corpus intact after fixes | — | ✓ Still exactly 5,389 chunks |
| Verified from a clean clone | ✗ Never attempted | ✓ 8/8 blobs, sha256-checked |

---

## Part Seven: What this is worth saying about

If someone asks how you handle data versioning, the answer is now a real one:

> *The corpus is five NVIDIA CUDA manuals producing 5,389 chunks, pinned with DVC and reproducible from a clean clone with no credentials — I verified that by cloning fresh and checking every file's hash.*
>
> *Getting there turned up three separate defects. The DVC pointer files had been silently swallowed by a `.gitignore` rule, so the versioning layer was invisible to anyone cloning the repo. No storage remote had ever been configured. And the ingestion downloaded from live URLs rather than reading the pinned copies at all — which meant the corpus was whatever NVIDIA happened to be publishing that day, with no way to detect drift.*
>
> *It also corrected an earlier misdiagnosis: two queries recorded as "corpus coverage gaps" were actually documents that had been downloaded and tracked but never ingested.*

That's a stronger answer than "I used DVC," because it demonstrates the thing that actually matters — **checking whether a mechanism works rather than assuming it does because it's installed.**

Which is, when you look at it, the same lesson the rest of this project has been circling: a tool that's present isn't a tool that's working, and the only way to know the difference is to test it from the outside.

### A small thing worth copying

The commit that fixed this (`7bc6730`) records the *reasoning*, not just the change: why per-file `dvc add` beat directory-level tracking, why three PDFs stay tracked but deliberately unused, and exactly what the verification consisted of.

That's the difference between a decision you can defend in six months and one you have to reconstruct from memory. Most commit messages say what changed. The useful ones say why the alternative was rejected.

---

*Related records: DEF-19 (pointer files ignored, Round 1 misdiagnosis), DEF-20 (silent partial-corpus ingestion), DEF-24 (line-ending corruption risk), DEF-25 (inverted exit code). Clean-clone verification: `docs/uat/clean_clone_test_protocol.md` CC-ACQ-02 — passed. Fix committed at `7bc6730`.*
