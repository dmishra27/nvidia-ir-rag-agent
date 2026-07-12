# Day 5 UAT — Retrieval Pipeline Validation Across 9 Queries

## Methodology

Three configs run against the live 5,389-chunk corpus (CUDA C++ Programming
Guide, CUDA C++ Best Practices Guide, CUDA Math API Reference, CUDA Runtime
API Reference, Nsight Systems User Guide) via `run_uat_day5.py`:

- **Config A (BM25-only)** — `retrieval/bm25_index.py`, lexical search.
- **Config B (Dense-only)** — `retrieval/dense_index.py`, e5-base-v2 cosine
  search over the Day 4 Qdrant collection.
- **Config C (RRF hybrid)** — `retrieval/rrf_fusion.py`, `k=60`, fused over
  a **top-100 candidate pool** from each of A and B.

9 queries across 3 categories (exact technical, semantic/conceptual, legacy
terminology), top-3 shown per config. Raw data:
[`docs/uat/uat_day5_raw.json`](uat_day5_raw.json).

**Caveat**: only the top-3 of each single-signal config is displayed below
for readability, but RRF fuses the full top-100 pool from each — an RRF
result can come from rank 4–100 of a signal that only shows its top-3 here.
Relevance judgments below are qualitative (read against query intent), not
computed from graded judgments — this is a UAT pass, not a metrics eval.

---

## Type 1 — Exact technical

### Q1. "NVLink 4.0 bandwidth specifications"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `97724de868d508c6796761ea` | 15.9543 | `. ‣ DRAM Write Bandwidth - dramc__write_throughput.avg.pct_of_peak_sustained_elapsed , dram__write_t` |
| A (BM25) | 2 | `aae9e31dae08f840680449cd` | 14.8923 | `[CUDA C++ Programming Guide, Release 13.3] CUDA C++ Programming Guide, Release 13.3 574 25. Lazy Loa` |
| A (BM25) | 3 | `0fc9e36da0ab43104dee747a` | 12.4626 | `CUDA C++ Best Practices Guide, Release 13.3 9.2.1. Theoretical Bandwidth Calculation Theoretical ban` |
| B (Dense) | 1 | `97724de868d508c6796761ea` | 0.8232 | `. ‣ DRAM Write Bandwidth - dramc__write_throughput.avg.pct_of_peak_sustained_elapsed , dram__write_t` |
| B (Dense) | 2 | `0fc9e36da0ab43104dee747a` | 0.8213 | `CUDA C++ Best Practices Guide, Release 13.3 9.2.1. Theoretical Bandwidth Calculation Theoretical ban` |
| B (Dense) | 3 | `05fad39c01b3325a10595ac0` | 0.8208 | `[www.nvidia.com] Profiling from the CLI www.nvidia.com User Guide v2023.3.1 | 72 Short Long Possible` |
| C (RRF) | 1 | `97724de868d508c6796761ea` | 0.0328 | `. ‣ DRAM Write Bandwidth - dramc__write_throughput.avg.pct_of_peak_sustained_elapsed , dram__write_t` |
| C (RRF) | 2 | `0fc9e36da0ab43104dee747a` | 0.0320 | `CUDA C++ Best Practices Guide, Release 13.3 9.2.1. Theoretical Bandwidth Calculation Theoretical ban` |
| C (RRF) | 3 | `ade018c96de7e57a8c3d618c` | 0.0299 | `9 Performance Metrics 25 9.1 Timing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ` |

**Read**: all three configs agree on the same top-1 — a Nsight profiling
metric ("DRAM Write Bandwidth"), not an NVLink-specific chunk. The corpus
has no GPU-architecture whitepaper with NVLink 4.0's actual bandwidth
figures, so none of the three configs can find what the query is really
asking for — this is a **corpus coverage gap**, not a ranking bug.

### Q2. "CUDA cudaMalloc function parameters"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `381cf7a1dddd75346b7446ee` | 12.1774 | `[__host__cudaError_t cudaFreeMipmappedArray] Modules CUDA Runtime API v13.3.1 Version | 128 See also` |
| A (BM25) | 2 | `cc6c8e53936d04e9b192a7d5` | 11.9900 | `[__host____device__cudaError_t cudaMalloc (void] Modules CUDA Runtime API v13.3.1 Version | 138 See ` |
| A (BM25) | 3 | `81b9c458ed8d5bbf219819b8` | 11.1421 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| B (Dense) | 1 | `7cb10cb8b14c66c9987417cb` | 0.8944 | `[void *cudaMemAllocNodeParams::dptr] Data Structures CUDA Runtime API v13.3.1 Version | 649 void *cu` |
| B (Dense) | 2 | `a0bfa55c394eb552119fbe03` | 0.8894 | `. The function is defined in the cudadevrt system library, which must be linked with a program in or` |
| B (Dense) | 3 | `81b9c458ed8d5bbf219819b8` | 0.8881 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| C (RRF) | 1 | `81b9c458ed8d5bbf219819b8` | 0.0317 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| C (RRF) | 2 | `b390a0086cd2ebb9237c24b4` | 0.0286 | `[cudaHostFn_t cudaHostNodeParams::fn] Data Structures CUDA Runtime API v13.3.1 Version | 641 cudaHos` |
| C (RRF) | 3 | `a1db062e97d26a49ab333961` | 0.0272 | `. Note: as specified by ... this function may also return error codes from previous, asynchronous launches.` |

**Read**: **BM25 rank 2** (`cc6c8e53...`) is the actual `cudaMalloc()` API
reference entry — the correct answer to this query. Dense never surfaces it
in its top-3, and **RRF loses it entirely**: the `cudaStreamAddCallback`
chunk appears at rank 3 in *both* BM25 and dense, so its combined RRF score
(`2/(60+3) ≈ 0.0317`) beats the true answer's single-list score
(`1/(60+2) ≈ 0.0161`), even though that chunk isn't about `cudaMalloc` at
all. This is RRF's **corroboration bias**: two mediocre agreeing hits can
outrank one strong single-signal hit.

### Q3. "H100 HBM2e memory capacity"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `4894443247eb04565ae790ce` | 14.1777 | `CUDA C++ Programming Guide, Release 13.3 20.8.3. Shared Memory Similar to the NVIDIA Ampere GPU arch` |
| A (BM25) | 2 | `ab8c38f5987e9d6af5172b39` | 10.8264 | `. Unlike Kepler, the driver automatically configures the shared memory capacity for each kernel to a` |
| A (BM25) | 3 | `44cacb220dc77ec95d6bb5c2` | 10.5875 | `CUDA C++ Programming Guide, Release 13.3 (continued from previous page) ∕∕carveout = cudaSharedmemCa` |
| B (Dense) | 1 | `66e68be5172026f2326e92e8` | 0.8355 | `. HBM2 memories, on the other hand, provide dedicated ECC resources, allowing overhead-free ECC prot` |
| B (Dense) | 2 | `0fc9e36da0ab43104dee747a` | 0.8184 | `CUDA C++ Best Practices Guide, Release 13.3 9.2.1. Theoretical Bandwidth Calculation Theoretical ban` |
| B (Dense) | 3 | `22e542f6af50aed6362c7385` | 0.8113 | `. Note: Requires compute capability >= 2.0. 328 Chapter 9. Double Precision Intrinsics` |
| C (RRF) | 1 | `976f1f192bb0046f1aded1f0` | 0.0289 | `Modules CUDA Runtime API v13.3.1 Version | 544 Device supports host memory registration via cudaHost` |
| C (RRF) | 2 | `9c31fd555773b828af84224d` | 0.0273 | `. The shared memory capacity can be set to 0, 8, 16, 32, 64, 100, 132 or 164 KB for devices of compu` |
| C (RRF) | 3 | `94322bcefc6f37836b7c9640` | 0.0256 | `. 20.9.2. Global Memory Global memory behaves the same way as for devices of compute capability 5.x ` |
| B (Dense), for reference | pool | (chunk above, dense rank 1) | — | HBM2/ECC chunk never appears in RRF's top-3 |

**Read**: `"H100 HBM2e memory capacity"` is a case of BM25's exact
vocabulary trap — "memory capacity" lexically matches on-chip *shared*
memory capacity chunks, not HBM2e DRAM. **Dense alone correctly finds the
HBM2 chunk at rank 1** via semantic similarity despite zero shared
terminology with "H100 HBM2e" (the corpus text says "HBM2", not "HBM2e" or
"H100"). But **RRF fails to inherit dense's win** — BM25 never ranked that
chunk highly enough (or at all) within its own pool to give it a second
vote, so it's crowded out by chunks both signals rank moderately. Clear
regression vs. dense-only.

---

## Type 2 — Semantic / conceptual

### Q4. "How does GPU memory work for parallel processing"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `2fa6d1612e31780857d91317` | 18.4653 | `CUDA C++ Best Practices Guide, Release 13.3 communicated between device memory and host memory as de` |
| A (BM25) | 2 | `490bb9d312a2fb7e58e6591f` | 17.1214 | `Chapter 3. Introduction Warning: This document has been replaced by a new CUDA Programming Guide . T` |
| A (BM25) | 3 | `6083e1f4926192af0e474f1e` | 16.7753 | `. The schematic Figure 1 shows an example distribution of chip resources for a CPU versus a GPU. Dev` |
| B (Dense) | 1 | `6083e1f4926192af0e474f1e` | 0.8610 | `. The schematic Figure 1 shows an example distribution of chip resources for a CPU versus a GPU. Dev` |
| B (Dense) | 2 | `970b7cb8843ba5bf598058ac` | 0.8548 | `. ▶ The GPU is considered active when it is running any kernel, even if that kernel does not make us` |
| B (Dense) | 3 | `e15064d6ce1016c1196ae1a2` | 0.8502 | `CUDA C++ Programming Guide, Release 13.3 into coarse sub-problems that can be solved independently i` |
| C (RRF) | 1 | `6083e1f4926192af0e474f1e` | 0.0323 | `. The schematic Figure 1 shows an example distribution of chip resources for a CPU versus a GPU. Dev` |
| C (RRF) | 2 | `490bb9d312a2fb7e58e6591f` | 0.0318 | `Chapter 3. Introduction Warning: This document has been replaced by a new CUDA Programming Guide . T` |
| C (RRF) | 3 | `970b7cb8843ba5bf598058ac` | 0.0318 | `. ▶ The GPU is considered active when it is running any kernel, even if that kernel does not make us` |

**Read**: solid query for all three. BM25 rank 3 and dense rank 1 agree on
the "CPU vs. GPU chip resources" chunk, so RRF correctly promotes it to
rank 1. RRF's top-3 is a genuine blend of BM25's and dense's best picks —
the intended hybrid-search effect.

### Q5. "best practices for optimising neural network training"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `1bc7e850882f3acb0dd8b500` | 13.1774 | `[CUDA C++ Best Practices Guide] CUDA C++ Best Practices Guide Release 13.3 NVIDIA Corporation Jun 25` |
| A (BM25) | 2 | `cc7faa789d2f0578fed00f8e` | 13.0707 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 54 10. Mem` |
| A (BM25) | 3 | `64179456d5f50e99ce7f6954` | 13.0707 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 74 12. Ins` |
| B (Dense) | 1 | `cc4b86761f1005773c609b79` | 0.8425 | `Chapter 19. Recommendations and Best Practices This chapter contains a summary of the recommendation` |
| B (Dense) | 2 | `69c2f815978cac64bb8c388b` | 0.8363 | `Chapter 2. Preface This Best Practices Guide is a manual to help developers obtain the best performa` |
| B (Dense) | 3 | `1c6c6224465f82a2f9965a09` | 0.8284 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 22 7. Gett` |
| C (RRF) | 1 | `64179456d5f50e99ce7f6954` | 0.0310 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 74 12. Ins` |
| C (RRF) | 2 | `cc7faa789d2f0578fed00f8e` | 0.0304 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 54 10. Mem` |
| C (RRF) | 3 | `6124f6366c1e67c875aa02ef` | 0.0284 | `[21 Notices] 20.1 nvcc . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .` |

**Read**: **all three configs fail** — and correctly so. The corpus is CUDA
C++ programming/best-practices/API-reference documentation, not deep
learning framework documentation; there is no neural-network-training
content to retrieve. Every result here is generic CUDA "best practices"
boilerplate or table-of-contents noise, matched only on the shared phrase
"best practices" / "optimi[s/z]ing". This is the expected, correct outcome
of an out-of-domain query against this corpus, not a retrieval defect.

### Q6. "what causes memory errors in GPU applications"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `5d026d79fdf2f55052434d82` | 13.3621 | `Chapter 15. Stream Ordered Memory Allocator Warning: This document has been replaced by a new CUDA P` |
| A (BM25) | 2 | `1294b9a04e36b4a6b1fed64b` | 12.6031 | `CUDA C++ Programming Guide, Release 13.3 Table 30 – continued from previous page Variable Values Des` |
| A (BM25) | 3 | `18f06fd220df55b621d25913` | 11.8136 | `. cudaErrorJitCompilationDisabled = 223 This indicates that the JIT compilation was disabled. The JI` |
| B (Dense) | 1 | `c83c81319bfdefae04d81df6` | 0.8663 | `. There are however special considerations as described below when the system is in SLI mode. First,` |
| B (Dense) | 2 | `68350e3f7388624f4e8594f2` | 0.8624 | `[www.nvidia.com] CUDA Trace www.nvidia.com User Guide v2023.3.1 | 182 page faults can cause overhead` |
| B (Dense) | 3 | `e0d64832cd47434e0a7e4a3f` | 0.8581 | `CUDA C++ Programming Guide, Release 13.3 24.3.2. Unified memory on Windows or devices with compute c` |
| C (RRF) | 1 | `e3d457e876f094d63cff72dd` | 0.0301 | `. This often leads to lower performance and higher peak memory utilization for applications. Essenti` |
| C (RRF) | 2 | `c83c81319bfdefae04d81df6` | 0.0297 | `. There are however special considerations as described below when the system is in SLI mode. First,` |
| C (RRF) | 3 | `e0d64832cd47434e0a7e4a3f` | 0.0251 | `CUDA C++ Programming Guide, Release 13.3 24.3.2. Unified memory on Windows or devices with compute c` |

**Read**: BM25 fails — it returns error-code tables and deprecation
warnings, none of which explain *causes* of memory errors. Dense does
better (rank 2, "page faults can cause overhead", is a genuine cause), but
its own rank 1 (SLI mode) is off-target. **RRF's rank 1 is the best single
result of any config** — a chunk about allocation patterns causing "lower
performance and higher peak memory utilization," pulled from deeper in the
pools than either single config's displayed top-3. Clear win for RRF here.

---

## Type 3 — Legacy terminology

### Q7. "shader processor count"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `9e3709105824054667991d5e` | 10.4684 | `[www.nvidia.com] Stutter Analysis www.nvidia.com User Guide v2023.3.1 | 163 9.2. Frame Health The Fr` |
| A (BM25) | 2 | `a8c35fe2813fdaec5c5356ab` | 9.7688 | `GPU Metrics www.nvidia.com User Guide v2023.3.1 | 204 Note: The percentage will always be very low a` |
| A (BM25) | 3 | `8ddd5e09211a51cec7acd9d2` | 8.9866 | `. Any read accesses from any processor to this region will create a read-only copy of at least the a` |
| B (Dense) | 1 | `35b73f3371037bf5ba8fefc0` | 0.8281 | `. 20.4. Compute Capability 5.x 20.4.1. Architecture An SM consists of: ▶ 128 CUDA cores for arithmet` |
| B (Dense) | 2 | `aed2815db4e8ab51d9c522ff` | 0.8254 | `. 28 2 FP64 cores for double-precision arithmetic operations for devices of compute capabilities 7.5` |
| B (Dense) | 3 | `c0a6fa3aecd4d794dd957899` | 0.8218 | `Post-Collection Analysis www.nvidia.com User Guide v2023.3.1 | 297 ‣ CPU Page Faults : Number of CPU` |
| C (RRF) | 1 | `a8c35fe2813fdaec5c5356ab` | 0.0306 | `GPU Metrics www.nvidia.com User Guide v2023.3.1 | 204 Note: The percentage will always be very low a` |
| C (RRF) | 2 | `ba52947371305a0ee68f2c73` | 0.0260 | `[www.nvidia.com] Post-Collection Analysis www.nvidia.com User Guide v2023.3.1 | 268 Replace "SELECT ` |
| C (RRF) | 3 | `9e3709105824054667991d5e` | 0.0229 | `[www.nvidia.com] Stutter Analysis www.nvidia.com User Guide v2023.3.1 | 163 9.2. Frame Health The Fr` |
| B (Dense), for reference | pool | `35b73f3371037bf5ba8fefc0` | — | "128 CUDA cores" chunk never appears in RRF's top-3 |

**Read**: the standout finding of this UAT. "Shader processor" is
pre-CUDA-era GPU marketing terminology for what NVIDIA now calls "CUDA
cores"; the corpus never uses "shader processor" anywhere. **BM25 has zero
useful lexical signal and returns noise. Dense alone correctly finds "An SM
consists of: 128 CUDA cores..." at rank 1** — a real semantic match across a
total vocabulary gap. **RRF loses it completely**: since BM25 never ranked
that chunk at all, it gets only a single-list score, and gets crowded out
by the "GPU Metrics" chunk that both lists rank moderately. This is the
clearest demonstration of RRF's corroboration-bias limitation on
vocabulary-mismatch queries — dense-only strictly beats hybrid here.

### Q8. "global memory coalescing techniques"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `3c8573a7f22cd37a771b4816` | 17.3520 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programmin` |
| A (BM25) | 2 | `91945270119932f8ee8d8fe0` | 12.1923 | `. Also, the compiler reports total local memory usage per kernel ( lmem ) when compiling with the --` |
| A (BM25) | 3 | `dca95bd9fefa84908469de51` | 11.7716 | `CUDA C++ Best Practices Guide, Release 13.3 For devices of compute capability 6.0 or higher, the req` |
| B (Dense) | 1 | `3c8573a7f22cd37a771b4816` | 0.8898 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programmin` |
| B (Dense) | 2 | `4949c564d4bf9dc335b73b66` | 0.8519 | `. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34 10.2.1 Coalesced ` |
| B (Dense) | 3 | `5a9b26920b6f9d08105f1f4d` | 0.8445 | `. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 264 10.34 Breakpoint Function . . . . . . ` |
| C (RRF) | 1 | `3c8573a7f22cd37a771b4816` | 0.0328 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programmin` |
| C (RRF) | 2 | `dca95bd9fefa84908469de51` | 0.0315 | `CUDA C++ Best Practices Guide, Release 13.3 For devices of compute capability 6.0 or higher, the req` |
| C (RRF) | 3 | `48ba7e1f690b446f51401f82` | 0.0297 | `. For example, if a 32-byte memory transaction is generated for each thread's 4-byte access, through` |

**Read**: both BM25 and dense independently find the exact "Coalesced
Access to Global Memory" chunk at rank 1, so RRF correctly keeps it at
rank 1 (double corroboration working as intended). Dense's own rank 2/3 are
low-value table-of-contents entries; **RRF's rank 3 pulls in a genuinely
useful chunk about 32-byte memory transactions and per-thread access
granularity** that neither single config surfaces in its displayed top-3 —
RRF's best result of the nine queries.

### Q9. "warp divergence performance impact"

| Config | Rank | chunk_id | Score | Text (first 100 chars) |
|---|---|---|---|---|
| A (BM25) | 1 | `3c9b5ecfb38e9ee9377cf4f6` | 15.1868 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different executi` |
| A (BM25) | 2 | `fb0ae459357d3e075cf717c6` | 14.4848 | `. Thus, when TL < PL , the thread will unintentionally wait for additional, more recent batches. In ` |
| A (BM25) | 3 | `3805d3ce01d885478a2e78c0` | 13.6572 | `CUDA C++ Programming Guide, Release 13.3 A warp executes one common instruction at a time, so full e` |
| B (Dense) | 1 | `011463350540da0a7eb7a0f6` | 0.8487 | `CUDA C++ Best Practices Guide, Release 13.3 This last case can be avoided by using single-precision ` |
| B (Dense) | 2 | `df5d7d80f5afd618919ad5f7` | 0.8415 | `CUDA C++ Programming Guide, Release 13.3 This code is invalid because CUDA does not guarantee that t` |
| B (Dense) | 3 | `3c9b5ecfb38e9ee9377cf4f6` | 0.8399 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different executi` |
| C (RRF) | 1 | `3c9b5ecfb38e9ee9377cf4f6` | 0.0323 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different executi` |
| C (RRF) | 2 | `fb0ae459357d3e075cf717c6` | 0.0318 | `. Thus, when TL < PL , the thread will unintentionally wait for additional, more recent batches. In ` |
| C (RRF) | 3 | `011463350540da0a7eb7a0f6` | 0.0315 | `CUDA C++ Best Practices Guide, Release 13.3 This last case can be avoided by using single-precision ` |
| — | — | `3c9b5ecfb38e9ee9377cf4f6` | — | Same chunk_id as the "warp divergence" judged-relevant chunk used in Day 4's bi-encoder eval set |

**Read**: BM25 nails it directly (rank 1 exact match on "Branching and
Divergence", rank 3 also genuinely on-topic). Dense is weaker here — its
top 2 are off-target and the correct chunk only reaches rank 3. RRF
correctly keeps the shared correct chunk at rank 1, matching BM25's
quality; not an improvement, but no regression either.

---

## UAT summary

| # | Query | Type | Winner (best top-1) | Notes |
|---|---|---|---|---|
| 1 | NVLink 4.0 bandwidth specifications | Exact | Tie (all identical, weak) | Corpus has no NVLink-4.0 spec content — partial failure, all configs |
| 2 | CUDA cudaMalloc function parameters | Exact | **BM25** | RRF loses the correct chunk to corroboration bias |
| 3 | H100 HBM2e memory capacity | Exact | **Dense** | RRF fails to inherit dense's unique win |
| 4 | How does GPU memory work for parallel processing | Semantic | **RRF** | Clean hybrid blend of BM25's and dense's best picks |
| 5 | Best practices for optimising neural network training | Semantic | None (all fail) | Out-of-domain query — corpus has no NN-training content |
| 6 | What causes memory errors in GPU applications | Semantic | **RRF** | Best single result of any config, pulled from deeper in the pool |
| 7 | Shader processor count | Legacy | **Dense** | BM25 has zero lexical signal; RRF loses dense's unique win |
| 8 | Global memory coalescing techniques | Legacy | **RRF** (ties BM25/Dense at rank 1) | RRF's rank 3 is the most valuable addition of the whole UAT |
| 9 | Warp divergence performance impact | Legacy | **BM25** (RRF ties it) | Dense alone is weak here; RRF matches BM25's quality |

### Where did RRF improve over BM25 alone?

- **Q4** (modest) — promotes the stronger conceptual chunk to rank 1.
- **Q6** (clear) — RRF's rank 1 is a real answer; BM25's rank 1 is generic error-code noise.
- **Q8** (rank-3 enrichment) — surfaces a highly relevant chunk neither single config's top-3 contained.

### Where did RRF *not* improve, or regress, vs. BM25 alone?

- **Q2** — regression: BM25 alone had the correct `cudaMalloc` chunk at rank 2; RRF buries it.
- **Q3, Q7** — RRF fails to inherit a win that was dense-only, but BM25 was already failing on both, so no regression *vs. BM25* specifically — RRF ties BM25's failure rather than fixing it.
- **Q9** — neutral; BM25 was already strong, RRF just preserves it.

### Where did RRF improve over Dense alone?

- **Q6** (moderate) — RRF's rank 1 beats dense's off-target rank 1.
- **Q8** (moderate) — replaces dense's table-of-contents noise at rank 2/3 with substantive content.
- **Q9** (strong) — promotes the correct chunk from dense's rank 3 to RRF's rank 1 via BM25 corroboration.

### Where did RRF regress vs. Dense alone?

- **Q3** (significant) — dense uniquely found the HBM2/ECC chunk; RRF loses it entirely.
- **Q7** (significant) — dense uniquely found the "128 CUDA cores" chunk; RRF loses it entirely. This is the most important finding of the UAT: **for legacy/vocabulary-mismatch queries where BM25 has no lexical signal at all, RRF can strictly underperform dense-only**, because a chunk only one signal ranks (even at rank 1) is worth less under RRF than two chunks both signals rank moderately.
- **Q2** (mild) — dense's rank 1 (memory-allocation-adjacent) is arguably more useful than RRF's rank 1 (unrelated callback chunk).

### Queries where all configs failed

- **Q5** ("neural network training") — genuine corpus-coverage gap; the
  corpus is CUDA systems documentation, not ML-framework documentation.
  Expected, correct behavior, not a defect.
- **Q1** ("NVLink 4.0 bandwidth") — partial failure; all three configs
  converge on the same topically-adjacent-but-not-on-point chunk because
  the corpus has no GPU-architecture whitepaper with NVLink 4.0's actual
  bandwidth numbers. Also a coverage gap, not a ranking bug.

## Key findings

1. **RRF's corroboration bias is real and query-dependent.** On 3 of 9
   queries (Q2, Q3, Q7) RRF lost a correct answer that a single signal had
   found, because `k=60` scoring rewards two-signal agreement over
   one-signal confidence. This is most damaging on **legacy-terminology
   queries (Q7)**, where BM25 structurally cannot contribute any signal —
   dense-only should arguably be preferred, or weighted higher, when BM25's
   candidate pool has near-zero score variance (a proxy for "no lexical
   signal").
2. **RRF is a clear net positive on semantic/conceptual queries** (Q4, Q6)
   where BM25 and dense partially agree but each has blind spots — this is
   RRF's intended use case and it performs as expected there.
3. **On exact-match queries where BM25 already has the answer** (Q9,
   partially Q8), RRF neither helps nor hurts materially — the true
   positive is well-corroborated and survives fusion.
4. **Two of nine queries (Q1, Q5) failed across all three configs due to
   corpus coverage gaps**, not retrieval-algorithm defects — worth keeping
   in mind when interpreting future NDCG/MRR numbers against a broader
   query set.
5. **Recommendation for Layer 3b**: a cross-encoder re-ranker reading full
   chunk text (not just rank position) should recognize "shader processor"
   ≈ "CUDA cores" the way dense embeddings do, and should not be subject to
   RRF's corroboration bias since it scores each candidate independently —
   Q7 and Q3 are good regression-test candidates once re-ranking lands.
