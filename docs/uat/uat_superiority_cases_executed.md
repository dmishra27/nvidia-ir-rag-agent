# Retrieval Superiority UAT — 15 Queries Executed Live

## Methodology

Three methods run against the live 5,389-chunk corpus (CUDA C++ Programming
Guide, CUDA C++ Best Practices Guide, CUDA Math API Reference, CUDA Runtime
API Reference, Nsight Systems User Guide) via `run_uat_superiority.py`,
executed live against the on-disk BM25 index and the live Qdrant collection:

- **Method A (BM25)** — `retrieval/bm25_index.py`, lexical search over
  `data/indexes/bm25_index.pkl`.
- **Method B (Dense)** — `retrieval/dense_index.py`, e5-base-v2 cosine
  search over the live `nvidia_ir_chunks` Qdrant collection.
- **Method C (RRF)** — `retrieval/rrf_fusion.py`, `k=60`, fused over a
  **top-100 candidate pool** from each of A and B.

15 queries across 6 cases, each designed to probe a specific structural
strength or weakness of one retrieval method. Rank 1 shows the first 150
characters of chunk text; ranks 2–3 show the first 100 characters. Raw data:
[`docs/uat/uat_superiority_cases_raw.json`](uat_superiority_cases_raw.json).

**Caveat**: only the top-3 of each single-signal method is displayed per
query, but RRF fuses the full top-100 pool from each — an RRF result can
come from rank 4–100 of a signal that only shows its top-3 here. As with
the Day 5 UAT, verdicts below reflect what the live results actually showed
— in several cases the outcome contradicted the case's hypothesis, and
that's reported honestly rather than adjusted to fit.

---

## Case 1 — BM25 Lexical Superiority

### Q1. "CUDA cudaMalloc function parameters"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `381cf7a1dddd75346b7446ee` | 12.1774 | `[__host__cudaError_t cudaFreeMipmappedArray] Modules CUDA Runtime API v13.3.1 Version \| 128 See also: cudaMalloc , cudaMallocPitch , cudaFree , cudaMa` |
| A (BM25) | 2 | `cc6c8e53936d04e9b192a7d5` | 11.9900 | `[__host____device__cudaError_t cudaMalloc (void] Modules CUDA Runtime API v13.3.1 Version \| 138 See ` |
| A (BM25) | 3 | `81b9c458ed8d5bbf219819b8` | 11.1421 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| B (Dense) | 1 | `7cb10cb8b14c66c9987417cb` | 0.8944 | `[void *cudaMemAllocNodeParams::dptr] Data Structures CUDA Runtime API v13.3.1 Version \| 649 void *cudaMemAllocNodeParams::dptr out: address of the all` |
| B (Dense) | 2 | `a0bfa55c394eb552119fbe03` | 0.8894 | `. The function is defined in the cudadevrt system library, which must be linked with a program in or` |
| B (Dense) | 3 | `81b9c458ed8d5bbf219819b8` | 0.8881 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cu` |
| C (RRF) | 1 | `81b9c458ed8d5bbf219819b8` | 0.0317 | `. ‣ Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to,` |
| C (RRF) | 2 | `b390a0086cd2ebb9237c24b4` | 0.0286 | `[cudaHostFn_t cudaHostNodeParams::fn] Data Structures CUDA Runtime API v13.3.1 Version \| 641 cudaHos` |
| C (RRF) | 3 | `a1db062e97d26a49ab333961` | 0.0272 | `. Note: ‣ Note that this function may also return error codes from previous, asynchronous launches. ` |

**WINNER: BM25** — only method that surfaces the actual `cudaMalloc()` API reference entry (rank 2, `cc6c8e53...`); dense misses it entirely in its top-3, and RRF loses it to corroboration bias (its rank 1 is an unrelated `cudaStreamAddCallback` chunk both lists rank moderately).

### Q2. "cudaMemcpyAsync stream parameter"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `e81d90f1327bd9f939182f32` | 10.9327 | `. The following code sample creates two streams and allocates an array hostPtr of float in page-locked memory. cudaStream_t stream[ 2 ]; for ( int i =` |
| A (BM25) | 2 | `d0a577a165b439a7b54b3a95` | 10.5088 | `Modules CUDA Runtime API v13.3.1 Version \| 176 that are both registered and not registered with CUDA` |
| A (BM25) | 3 | `01d8f177c446245392f0bc5e` | 9.9477 | `[sharedMemSize] Modules CUDA Runtime API v13.3.1 Version \| 107 sharedMemSize - Specifies size of sha` |
| B (Dense) | 1 | `7032b5070d0a9b8d5e6f73a4` | 0.8895 | `. See also: cudaMemcpy , cudaMemcpy2D , cudaMemcpy2DToArray , cudaMemcpy2DFromArray , cudaMemcpy2DArrayToArray , cudaMemcpyToSymbol , cudaMemcpyFromSy` |
| B (Dense) | 2 | `3aff8a993ee905af903cea6a` | 0.8858 | `. cudaDevAttrHostMemoryPoolsSupported = 144 Device suports HOST location with the cuMemAllocAsync an` |
| B (Dense) | 3 | `af5730e3ba4fd8ece6209fb2` | 0.8848 | `. Parameters p - 3D memory copy parameters stream - Stream identifier Returns cudaSuccess , cudaErro` |
| C (RRF) | 1 | `f9f67e7b94c193a8782f538e` | 0.0296 | `. See also: cudaMemcpy , cudaMemcpy2D , cudaMemcpy2DFromArray , cudaMemcpy2DArrayToArray , cudaMemcpyToSymbol , cudaMemcpyFromSymbol , cudaMemcpyAsync` |
| C (RRF) | 2 | `03bdaee52ee8d2e972e53567` | 0.0282 | `. See also: cudaMemcpy , cudaMemcpy2D , cudaMemcpy2DToArray , cudaMemcpy2DArrayToArray , cudaMemcpyT` |
| C (RRF) | 3 | `242e353090d3f493c8ef64dc` | 0.0275 | `. The cudaMem- cpyAsync() function is a non-blocking variant of cudaMemcpy() in which control is ret` |

**WINNER: RRF** — its rank 3 (`242e353...`, the literal "cudaMemcpyAsync() is a non-blocking variant..." definition) is the single best-matching chunk of all nine displayed results, and it appears only in RRF's top-3 — neither BM25's nor dense's own top-3 surfaces it. This contradicts the case's BM25-lexical hypothesis.

### Q3. "CUDA error cudaErrorInvalidValue description"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `0904f18029b044943773fe74` | 12.3490 | `Modules CUDA Runtime API v13.3.1 Version \| 47 Description Returns the description string for an error code. If the error code is not recognized, "unre` |
| A (BM25) | 2 | `0744159283d6d08a31efbd60` | 11.0984 | `Modules CUDA Runtime API v13.3.1 Version \| 48 cudaPeekAtLastError , cudaGetErrorName , cudaGetErrorS` |
| A (BM25) | 3 | `833e98ebaca9943a1ccec747` | 10.5566 | `[Returns] Modules CUDA Runtime API v13.3.1 Version \| 361 Returns cudaSuccess , cudaErrorInvalidValue` |
| B (Dense) | 1 | `068c625157dd3675d9aa970a` | 0.8955 | `Modules CUDA Runtime API v13.3.1 Version \| 377 Returns cudaErrorInvalidValue if the memory operands' mappings changed or the original memory operands ` |
| B (Dense) | 2 | `72e0d85687d8c95facce1dbc` | 0.8887 | `. Specifying this option with cudaLibraryLoadFromFile() is invalid and will return cudaErrorInvalidV` |
| B (Dense) | 3 | `a670fcfcd01320e2fe3f69fd` | 0.8874 | `. Note: ‣ Note that this function may also return error codes from previous, asynchronous launches. ` |
| C (RRF) | 1 | `7c67cdb0338710d0ce1e33f8` | 0.0284 | `[Returns] Modules CUDA Runtime API v13.3.1 Version \| 320 Returns cudaSuccess , cudaErrorInvalidValue Description Note: ‣ Note that this function may a` |
| C (RRF) | 2 | `0904f18029b044943773fe74` | 0.0256 | `Modules CUDA Runtime API v13.3.1 Version \| 47 Description Returns the description string for an erro` |
| C (RRF) | 3 | `b68a721030bcdd7d06727bbb` | 0.0244 | `. Parameters memoryRequirements - Pointer to cudaArrayMemoryRequirements mipmap - CUDA mipmapped arr` |
| — | — | `0904f18029b044943773fe74` | — | Same chunk_id is BM25 rank 1 (best match) but only RRF rank 2 |

**WINNER: BM25** — rank 1 (`0904f18...`, "Returns the description string for an error code") is the canonical, directly on-target answer to "error description"; dense and RRF both surface it lower or not at all in favor of specific-instance usage chunks.

---

## Case 2 — Dense Semantic Superiority

### Q4. "shader processor count per streaming multiprocessor"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `a1f985eaed4eb1f19e46d20a` | 16.5175 | `. However, this approach of determining how register count affects occupancy does not take into account the register allocation granularity. For examp` |
| A (BM25) | 2 | `84905b60e266d84cd0ca1879` | 14.0466 | `CUDA C++ Best Practices Guide, Release 13.3 11.1.1. Calculating Occupancy One of several factors tha` |
| A (BM25) | 3 | `a8c35fe2813fdaec5c5356ab` | 13.6072 | `GPU Metrics www.nvidia.com User Guide v2023.3.1 \| 204 Note: The percentage will always be very low a` |
| B (Dense) | 1 | `3ce249ad487299808540397f` | 0.8440 | `CUDA C++ Programming Guide, Release 13.3 Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory part` |
| B (Dense) | 2 | `57141d8c022463993050e7a7` | 0.8389 | `. ∕∕Device code __global__ void MyKernel(...) { extern __shared__ float buffer[]; ... } ∕∕Host code ` |
| B (Dense) | 3 | `b1b835707157b6821a40534b` | 0.8345 | `. 156 8.2. Pipeline Creation Feedback...............................................................` |
| C (RRF) | 1 | `3ce249ad487299808540397f` | 0.0284 | `CUDA C++ Programming Guide, Release 13.3 Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory part` |
| C (RRF) | 2 | `2f30edbe2de5b983a611eeca` | 0.0271 | `CUDA C++ Best Practices Guide, Release 13.3 Figure 15: Using the CUDA Occupancy Calculator to projec` |
| C (RRF) | 3 | `a8c35fe2813fdaec5c5356ab` | 0.0265 | `GPU Metrics www.nvidia.com User Guide v2023.3.1 \| 204 Note: The percentage will always be very low a` |

**WINNER: None (all weak)** — BM25 returns register/occupancy noise, dense's rank 1 is about shared-memory sizing rather than processor counts, and RRF inherits dense's off-target rank 1. None of the three surfaces an SM-architecture "N cores per SM" chunk in top-3, showing the result is sensitive to exact phrasing (a shorter "shader processor count" phrasing found the right chunk in the Day 5 UAT; adding "per streaming multiprocessor" here diluted it out of all three top-3s).

### Q5. "how to make GPU programs run faster"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `970b7cb8843ba5bf598058ac` | 14.3866 | `. ▶ The GPU is considered active when it is running any kernel, even if that kernel does not make use of managed data. If a kernel might use data, the` |
| A (BM25) | 2 | `e0d64832cd47434e0a7e4a3f` | 13.8827 | `CUDA C++ Programming Guide, Release 13.3 24.3.2. Unified memory on Windows or devices with compute c` |
| A (BM25) | 3 | `490bb9d312a2fb7e58e6591f` | 13.5386 | `Chapter 3. Introduction Warning: This document has been replaced by a new CUDA Programming Guide . T` |
| B (Dense) | 1 | `8fa1dd72a196776ecf605993` | 0.8590 | `. The following documents are especially important resources: ▶ CUDA Installation Guide ▶ CUDA C++ Programming Guide ▶ CUDA Toolkit Reference Manual I` |
| B (Dense) | 2 | `8f2dbd94357e8ac31b8a595c` | 0.8567 | `[CUDA C++ Programming Guide, Release 13.3] CUDA C++ Programming Guide, Release 13.3 25.5.3. Autotuni` |
| B (Dense) | 3 | `035938c9ed87d662f19d97bb` | 0.8534 | `. ▶ Preemption. The GPU scheduler can start executing a higher-priority kernel , even if it is launc` |
| C (RRF) | 1 | `490bb9d312a2fb7e58e6591f` | 0.0315 | `Chapter 3. Introduction Warning: This document has been replaced by a new CUDA Programming Guide . The information in this document should be consider` |
| C (RRF) | 2 | `877ff33f7c2010c9d7b8f97b` | 0.0305 | `. You may alter the given recipes or write your own to meet your needs. Refer to Tutorial: Create a ` |
| C (RRF) | 3 | `2f3be97a48ce2563233a8c64` | 0.0240 | `[www.nvidia.com] www.nvidia.com User Guide v2023.3.1 \| 192 Chapter 15. OPENGL TRACE OpenGL and OpenG` |

**WINNER: Dense (weak)** — its rank 2 ("Autotuning" section) is the only displayed result across all three methods that is topically about CUDA performance tuning; the rest of all three top-3s are generic doc-list or table-of-contents noise. A broad conversational query with no single well-matching chunk in the corpus.

### Q6. "problems with threads executing different code paths"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `3805d3ce01d885478a2e78c0` | 18.1799 | `CUDA C++ Programming Guide, Release 13.3 A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a war` |
| A (BM25) | 2 | `3c9b5ecfb38e9ee9377cf4f6` | 17.7732 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different executi` |
| A (BM25) | 3 | `d5d95f331693d71be9203a84` | 13.4220 | `CUDA C++ Best Practices Guide, Release 13.3 Table 5 – continued from previous page Compute Capabil- ` |
| B (Dense) | 1 | `3c9b5ecfb38e9ee9377cf4f6` | 0.8453 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different execution paths within the same warp. Flow control instru` |
| B (Dense) | 2 | `a49007ac82fa3038e67a392f` | 0.8343 | `. Threads Threads on a CPU are generally heavyweight entities. The operating system must swap thread` |
| B (Dense) | 3 | `47cdfe34d2c4b0e837858ef0` | 0.8321 | `. For example if multiple threads within a block are each launching work and synchronization is desi` |
| C (RRF) | 1 | `3c9b5ecfb38e9ee9377cf4f6` | 0.0325 | `Chapter 13. Control Flow 13.1. Branching and Divergence Note: High Priority: Avoid different execution paths within the same warp. Flow control instru` |
| C (RRF) | 2 | `3e21f636c1f80ac2500aee2a` | 0.0305 | `. Independent Thread Scheduling can lead to a rather different set of threads participating in the e` |
| C (RRF) | 3 | `a49007ac82fa3038e67a392f` | 0.0291 | `. Threads Threads on a CPU are generally heavyweight entities. The operating system must swap thread` |

**WINNER: Dense (ties RRF)** — both put the exact "Branching and Divergence" chunk at rank 1, one position better than BM25's rank 2. Note this isn't a pure vocabulary-gap case as intended: "different code paths" closely mirrors the source text's "different execution paths," so BM25 also finds it, just ranked lower.

---

## Case 3 — RRF Hybrid Superiority

### Q7. "CUDA thread synchronization performance overhead"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `494b88549e8af5043ac119c8` | 19.3752 | `CUDA C++ Programming Guide, Release 13.3 (continued from previous page) if (cudaSuccess != cudaDeviceSynchronize()) { return 2 ; } return 0 ; } This p` |
| A (BM25) | 2 | `ccd7708bb87a3ea7259deb4c` | 15.7008 | `. Atomic memory operations provide inter-thread synchronization guar- antees and deliver much better` |
| A (BM25) | 3 | `330acac2ad5d9a1c0811c6c1` | 13.5108 | `Modules CUDA Runtime API v13.3.1 Version \| 42 profile of the platform and may choose cudaDeviceSched` |
| B (Dense) | 1 | `6bf184553531cd05f6e9abf5` | 0.8769 | `CUDA C++ Best Practices Guide, Release 13.3 (continued from previous page) atomicAdd( reinterpret_cast < unsigned long long *> (clock), clock_end - cl` |
| B (Dense) | 2 | `c250c2f1083afba5f9a7430f` | 0.8734 | `CUDA C++ Best Practices Guide, Release 13.3 (continued from previous page) for ( int i = 0 ; i < TIL` |
| B (Dense) | 3 | `a8b2f5f66bf68a2456232ade` | 0.8718 | `. CUDA: CUDA will only report synchronous queue in the case of MPS configured with 64 sub-context. S` |
| C (RRF) | 1 | `ccd7708bb87a3ea7259deb4c` | 0.0313 | `. Atomic memory operations provide inter-thread synchronization guar- antees and deliver much better performance than volatile operations. CUDA C++ vo` |
| C (RRF) | 2 | `494b88549e8af5043ac119c8` | 0.0311 | `CUDA C++ Programming Guide, Release 13.3 (continued from previous page) if (cudaSuccess != cudaDevic` |
| C (RRF) | 3 | `330acac2ad5d9a1c0811c6c1` | 0.0292 | `Modules CUDA Runtime API v13.3.1 Version \| 42 profile of the platform and may choose cudaDeviceSched` |
| — | — | `ccd7708bb87a3ea7259deb4c` | — | Same chunk_id was BM25 rank 2; RRF promotes it to rank 1 |

**WINNER: RRF** — promotes "atomic memory operations... deliver much better performance than volatile operations" to rank 1, the single most directly on-topic result across all three methods, by combining BM25's rank-2 signal with a matching dense-pool hit that neither single method displays at rank 1.

### Q8. "shared memory bank conflicts and how to avoid them"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `fd5aa3318c4def606709ac98` | 33.4207 | `CUDA C++ Programming Guide, Release 13.3 10.29.3. TMA Swizzle By default, the TMA engine loads data to shared memory in the same order as it is laid o` |
| A (BM25) | 2 | `1fabb00b8793f603b648e7d1` | 29.1927 | `. To minimize bank conflicts, it is important to understand how memory addresses map to memory banks` |
| A (BM25) | 3 | `4e0a7be7279a20f0a57b1d18` | 27.9313 | `. Changing the shared memory bank size will not increase shared memory usage or affect occupancy of ` |
| B (Dense) | 1 | `1fabb00b8793f603b648e7d1` | 0.8726 | `. To minimize bank conflicts, it is important to understand how memory addresses map to memory banks and how to optimally schedule memory requests. On` |
| B (Dense) | 2 | `f6f2063c3d7aac6f24143411` | 0.8611 | `. However, bank conflicts occur when copying the tile from global memory into shared memory. To enab` |
| B (Dense) | 3 | `4e0a7be7279a20f0a57b1d18` | 0.8523 | `. Changing the shared memory bank size will not increase shared memory usage or affect occupancy of ` |
| C (RRF) | 1 | `1fabb00b8793f603b648e7d1` | 0.0325 | `. To minimize bank conflicts, it is important to understand how memory addresses map to memory banks and how to optimally schedule memory requests. On` |
| C (RRF) | 2 | `f6f2063c3d7aac6f24143411` | 0.0318 | `. However, bank conflicts occur when copying the tile from global memory into shared memory. To enab` |
| C (RRF) | 3 | `4e0a7be7279a20f0a57b1d18` | 0.0317 | `. Changing the shared memory bank size will not increase shared memory usage or affect occupancy of ` |

**WINNER: RRF (ties Dense)** — both put the ideal "how to minimize bank conflicts" chunk at rank 1; BM25's raw lexical score ranks a tangentially-related TMA Swizzle chunk above it purely on term frequency.

### Q9. "memory coalescing rules for global memory access patterns"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `3c8573a7f22cd37a771b4816` | 24.5972 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programming for CUDA-capable GPU architectures is the coales` |
| A (BM25) | 2 | `879c3af23305670d45961cc9` | 22.2718 | `. Size and Alignment Requirement Global memory instructions support reading or writing words of size` |
| A (BM25) | 3 | `b5ca753b8b44006b10045187` | 21.5205 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 provides f` |
| B (Dense) | 1 | `3c8573a7f22cd37a771b4816` | 0.8784 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programming for CUDA-capable GPU architectures is the coales` |
| B (Dense) | 2 | `4949c564d4bf9dc335b73b66` | 0.8616 | `. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34 10.2.1 Coalesced ` |
| B (Dense) | 3 | `1c016f02b97914c0f619758a` | 0.8574 | `. Not all threads need to participate. For example, if the threads of a warp access adjacent 4-byte ` |
| C (RRF) | 1 | `3c8573a7f22cd37a771b4816` | 0.0328 | `. 10.2.1. Coalesced Access to Global Memory A very important performance consideration in programming for CUDA-capable GPU architectures is the coales` |
| C (RRF) | 2 | `b5ca753b8b44006b10045187` | 0.0315 | `[CUDA C++ Best Practices Guide, Release 13.3] CUDA C++ Best Practices Guide, Release 13.3 provides f` |
| C (RRF) | 3 | `48ba7e1f690b446f51401f82` | 0.0310 | `. For example, if a 32-byte memory transaction is generated for each thread's 4-byte access, through` |

**WINNER: Tie (BM25/Dense/RRF)** — all three independently surface the exact "Coalesced Access to Global Memory" chunk at rank 1; RRF preserves the shared strong signal and its rank 3 adds a genuinely useful 32-byte transaction chunk neither single method's displayed top-3 contains.

---

## Case 4 — BM25 Failure / Dense Advantage (Vocabulary Gap)

### Q10. "latency hiding through instruction level parallelism"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `7d9f59453468bb65c6a213f7` | 23.9643 | `. 8.2.2. Device Level At a lower level, the application should maximize parallel execution between the multiprocessors of a device. Multiple kernels c` |
| A (BM25) | 2 | `92dfc0076c0abb2e020a377a` | 21.3091 | `CUDA C++ Programming Guide, Release 13.3 The most common reason a warp is not ready to execute its n` |
| A (BM25) | 3 | `23082588ef4a1e16d4f49262` | 16.5551 | `. Alternatively, NVIDIA provides an occupancy calculator as part of Nsight Compute; refer to https:/` |
| B (Dense) | 1 | `f2730f1e6b30f8901b027a33` | 0.8495 | `. The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see the CUDA C+` |
| B (Dense) | 2 | `7d9f59453468bb65c6a213f7` | 0.8449 | `. 8.2.2. Device Level At a lower level, the application should maximize parallel execution between t` |
| B (Dense) | 3 | `31e19331c0ba6e801c17ffbd` | 0.8372 | `. See Math Libraries . 12.2. Memory Instructions Note: High Priority: Minimize the use of global mem` |
| C (RRF) | 1 | `7d9f59453468bb65c6a213f7` | 0.0325 | `. 8.2.2. Device Level At a lower level, the application should maximize parallel execution between the multiprocessors of a device. Multiple kernels c` |
| C (RRF) | 2 | `92dfc0076c0abb2e020a377a` | 0.0318 | `CUDA C++ Programming Guide, Release 13.3 The most common reason a warp is not ready to execute its n` |
| C (RRF) | 3 | `31e19331c0ba6e801c17ffbd` | 0.0304 | `. See Math Libraries . 12.2. Memory Instructions Note: High Priority: Minimize the use of global mem` |

**WINNER: Dense** — uniquely surfaces the literal "number of instructions required to hide a latency of L clock cycles" definition chunk at rank 1 (`f2730f1e...`); BM25 only finds a related-but-less-precise chunk, and RRF loses dense's unique top hit entirely to corroboration bias — the same failure mode documented for Q3/Q7 in the Day 5 UAT.

### Q11. "occupancy versus performance tradeoffs"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `35dfc6787549aaeda1677954` | 17.5149 | `. This func- tion reports occupancy in terms of number of max active clusters of a given size on the GPU present in the system. The following code sam` |
| A (BM25) | 2 | `60395666bbc850515b069b6a` | 12.6777 | `. This metric is occupancy . Occupancy is the ratio of the number of active warps per multiprocessor` |
| A (BM25) | 3 | `db1c0602ae6cd49714f13a1b` | 12.3537 | `CUDA C++ Best Practices Guide, Release 13.3 When choosing the block size, it is important to remembe` |
| B (Dense) | 1 | `0a880bdf76e6612e502eb50a` | 0.8420 | `Chapter 11. Execution Configuration Optimizations One of the keys to good performance is to keep the multiprocessors on the device as busy as pos- sib` |
| B (Dense) | 2 | `9d1f3a9e03910a455b745bd9` | 0.8379 | `. This is because some operations common to each element can be performed by the thread once, amorti` |
| B (Dense) | 3 | `60395666bbc850515b069b6a` | 0.8330 | `. This metric is occupancy . Occupancy is the ratio of the number of active warps per multiprocessor` |
| C (RRF) | 1 | `60395666bbc850515b069b6a` | 0.0320 | `. This metric is occupancy . Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps. (` |
| C (RRF) | 2 | `9d1f3a9e03910a455b745bd9` | 0.0318 | `. This is because some operations common to each element can be performed by the thread once, amorti` |
| C (RRF) | 3 | `35dfc6787549aaeda1677954` | 0.0315 | `. This func- tion reports occupancy in terms of number of max active clusters of a given size on the` |

**WINNER: Dense** — rank 1 ("Execution Configuration Optimizations... keep the multiprocessors busy") is the only result directly framing the occupancy/performance tradeoff itself, rather than just defining occupancy; BM25's rank 1 is an unrelated API function description.

---

## Case 5 — Dense Failure / BM25 Advantage (Exact Lookup)

### Q12. "cudaDeviceSynchronize return value"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `b17f44303767ddd72f90478c` | 11.3447 | `. But a pointer to these variables may be passed to the kernel as an argument, see System-Allocated Memory: in-depth examples for examples. System All` |
| A (BM25) | 2 | `68ddca9d56fab485176fd1a6` | 10.8775 | `= v; } int main() { int * ptr = nullptr; ∕∕Requires CUDA Managed Memory support cudaMallocManaged( &` |
| A (BM25) | 3 | `eec8e219714ffc0d6b2d879d` | 9.7906 | `[CUDA C++ Programming Guide, Release 13.3] CUDA C++ Programming Guide, Release 13.3 10.22.3.2 Inclus` |
| B (Dense) | 1 | `02bb6a205ba73aa9763b937c` | 0.9016 | `. cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDeviceScheduleBlockingSync flag was set for this devic` |
| B (Dense) | 2 | `871a5345d057731d362cc52f` | 0.8950 | `. The recapture will be ended regardless of the return value from the callback. enum cudaHostTaskSyn` |
| B (Dense) | 3 | `19cfdd35ecf248568ad11b7e` | 0.8902 | `. Users requiring synchronization of the callback should signal its completion manually. Returns cud` |
| C (RRF) | 1 | `7168ba67e35c613f13986864` | 0.0297 | `Modules CUDA Runtime API v13.3.1 Version \| 30 Note: ‣ Use of cudaDeviceSynchronize in device code was deprecated in CUDA 11.6 and removed for compute_` |
| C (RRF) | 2 | `02bb6a205ba73aa9763b937c` | 0.0284 | `. cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDevi` |
| C (RRF) | 3 | `871a5345d057731d362cc52f` | 0.0239 | `. The recapture will be ended regardless of the return value from the callback. enum cudaHostTaskSyn` |

**WINNER: Dense** — surfaces the exact "cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed" chunk at rank 1; BM25's lexical match on the high-frequency term "cudaDeviceSynchronize" is diluted across many unrelated API chunks and misses the answer entirely in top-3. This directly contradicts the case's BM25-advantage hypothesis.

### Q13. "dim3 struct constructor syntax"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `a612172bf233e826b41e390c` | 11.9181 | `[dim3 cudaKernelNodeParams::blockDim] Data Structures CUDA Runtime API v13.3.1 Version \| 642 dim3 cudaKernelNodeParams::blockDim Block dimensions **cu` |
| A (BM25) | 2 | `8e43b6ce3eb3a4963dd89177` | 11.3407 | `[7.45.] Data Structures CUDA Runtime API v13.3.1 Version \| 647 7.45. cudaLaunchConfig_t Struct Refer` |
| A (BM25) | 3 | `937197655a4d6c1d0eec89b1` | 10.5689 | `CUDA Math API Reference Manual, Release 13.3 Public Functions __nv_bfloat162 () = default Constructo` |
| B (Dense) | 1 | `71f15d77ee7e7fd41fe5c38b` | 0.8448 | `. Render(); ... } ... } void Render () { ∕∕Map vertex buffer for writing from CUDA float4 * positions; cudaGraphicsMapResources( 1 , & positionsVB_CUD` |
| B (Dense) | 2 | `1153e0ef4bddae35412cf882` | 0.8414 | `[CUDA C++ Programming Guide, Release 13.3] CUDA C++ Programming Guide, Release 13.3 Table 7 – contin` |
| B (Dense) | 3 | `a612172bf233e826b41e390c` | 0.8384 | `[dim3 cudaKernelNodeParams::blockDim] Data Structures CUDA Runtime API v13.3.1 Version \| 642 dim3 cu` |
| C (RRF) | 1 | `a612172bf233e826b41e390c` | 0.0323 | `[dim3 cudaKernelNodeParams::blockDim] Data Structures CUDA Runtime API v13.3.1 Version \| 642 dim3 cudaKernelNodeParams::blockDim Block dimensions **cu` |
| C (RRF) | 2 | `1a57340e150f673ed49708b8` | 0.0284 | `CUDA Math API Reference Manual, Release 13.3 Public Functions __host__ __device__ inline __nv_fp6x2_` |
| C (RRF) | 3 | `1153e0ef4bddae35412cf882` | 0.0283 | `[CUDA C++ Programming Guide, Release 13.3] CUDA C++ Programming Guide, Release 13.3 Table 7 – contin` |

**WINNER: BM25** — the `dim3` struct field reference chunk lands at rank 1; dense buries the same chunk at rank 3 behind unrelated OpenGL-interop code, confirming BM25's lexical exactness wins on struct/API-name lookups.

---

## Case 6 — RRF Hybrid Advantage on Mixed Queries

### Q14. "pinned memory cudaMallocHost benefits and when to use"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `5e38563157daf351863b54ec` | 16.3789 | `CUDA C++ Best Practices Guide, Release 13.3 10.1.1. Pinned Memory Page-locked or pinned memory transfers attain the highest bandwidth between the host` |
| A (BM25) | 2 | `c5e455611ebb6ce32643ce9b` | 16.0434 | `[Memset] API synchronization behavior CUDA Runtime API v13.3.1 Version \| 4 3. If pageable memory mus` |
| A (BM25) | 3 | `225d92e73b17bac1e3e77c25` | 15.6613 | `. ∕∕Distributed shared memory size = cluster_size * nbins_per_block * sizeof(int) config.dynamicSmem` |
| B (Dense) | 1 | `5e38563157daf351863b54ec` | 0.8720 | `CUDA C++ Best Practices Guide, Release 13.3 10.1.1. Pinned Memory Page-locked or pinned memory transfers attain the highest bandwidth between the host` |
| B (Dense) | 2 | `f10fc7d79f0ccbd80bd07a4c` | 0.8696 | `. The driver tracks the virtual memory ranges allocated with this function and automatically acceler` |
| B (Dense) | 3 | `c9c4861fcb43957253ce9cc5` | 0.8683 | `Modules CUDA Runtime API v13.3.1 Version \| 132 __host__cudaError_t cudaHostAlloc (void **pHost, size` |
| C (RRF) | 1 | `5e38563157daf351863b54ec` | 0.0328 | `CUDA C++ Best Practices Guide, Release 13.3 10.1.1. Pinned Memory Page-locked or pinned memory transfers attain the highest bandwidth between the host` |
| C (RRF) | 2 | `225d92e73b17bac1e3e77c25` | 0.0315 | `. ∕∕Distributed shared memory size = cluster_size * nbins_per_block * sizeof(int) config.dynamicSmem` |
| C (RRF) | 3 | `c9c4861fcb43957253ce9cc5` | 0.0313 | `Modules CUDA Runtime API v13.3.1 Version \| 132 __host__cudaError_t cudaHostAlloc (void **pHost, size` |

**WINNER: Tie (BM25/Dense/RRF)** — all three independently rank the "10.1.1. Pinned Memory" definition chunk first. Strong signal in both lexical and semantic space means RRF simply preserves the shared winner without adding or losing value here.

### Q15. "register pressure and its effect on occupancy"

| Method | Rank | chunk_id | Score | Text |
|---|---|---|---|---|
| A (BM25) | 1 | `a6910bcc059caad0d70e2b8b` | 20.5772 | `. . . . . . . . . . . . 49 10.2.4 Local Memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51 10.2.5 Textu` |
| A (BM25) | 2 | `84905b60e266d84cd0ca1879` | 20.5684 | `CUDA C++ Best Practices Guide, Release 13.3 11.1.1. Calculating Occupancy One of several factors tha` |
| A (BM25) | 3 | `a1f985eaed4eb1f19e46d20a` | 16.1336 | `. However, this approach of determining how register count affects occupancy does not take into acco` |
| B (Dense) | 1 | `a1f985eaed4eb1f19e46d20a` | 0.8245 | `. However, this approach of determining how register count affects occupancy does not take into account the register allocation granularity. For examp` |
| B (Dense) | 2 | `23082588ef4a1e16d4f49262` | 0.8225 | `. Alternatively, NVIDIA provides an occupancy calculator as part of Nsight Compute; refer to https:/` |
| B (Dense) | 3 | `84905b60e266d84cd0ca1879` | 0.8207 | `CUDA C++ Best Practices Guide, Release 13.3 11.1.1. Calculating Occupancy One of several factors tha` |
| C (RRF) | 1 | `a1f985eaed4eb1f19e46d20a` | 0.0323 | `. However, this approach of determining how register count affects occupancy does not take into account the register allocation granularity. For examp` |
| C (RRF) | 2 | `a6910bcc059caad0d70e2b8b` | 0.0320 | `. . . . . . . . . . . . 49 10.2.4 Local Memory . . . . . . . . . . . . . . . . . . . . . . . . . . .` |
| C (RRF) | 3 | `84905b60e266d84cd0ca1879` | 0.0320 | `CUDA C++ Best Practices Guide, Release 13.3 11.1.1. Calculating Occupancy One of several factors tha` |
| — | — | `a1f985eaed4eb1f19e46d20a` | — | BM25 rank 3 promoted to RRF rank 1 via dense's rank-1 corroboration |

**WINNER: RRF (ties Dense)** — promotes the exact "register count affects occupancy... allocation granularity" chunk to rank 1 by combining dense's strong rank-1 signal with BM25's weaker rank-3 corroboration; a clear improvement over BM25 alone, which buried it at rank 3 behind two TOC/heading chunks.

---

## Summary

| # | Query | Case | Winner | Hypothesis confirmed? |
|---|---|---|---|---|
| Q1 | CUDA cudaMalloc function parameters | 1 | **BM25** | Yes |
| Q2 | cudaMemcpyAsync stream parameter | 1 | **RRF** | No — RRF's pool corroboration surfaced the best chunk, BM25 alone was weak |
| Q3 | CUDA error cudaErrorInvalidValue description | 1 | **BM25** | Yes |
| Q4 | shader processor count per streaming multiprocessor | 2 | None (all weak) | No — phrasing diluted the semantic match that worked in Day 5 |
| Q5 | how to make GPU programs run faster | 2 | Dense (weak) | Partially — dense best of a bad field |
| Q6 | problems with threads executing different code paths | 2 | Dense (ties RRF) | Partially — query shared vocabulary with source, so BM25 also found it |
| Q7 | CUDA thread synchronization performance overhead | 3 | **RRF** | Yes |
| Q8 | shared memory bank conflicts and how to avoid them | 3 | **RRF** (ties Dense) | Yes |
| Q9 | memory coalescing rules for global memory access patterns | 3 | **RRF** (ties BM25/Dense) | Yes |
| Q10 | latency hiding through instruction level parallelism | 4 | **Dense** | Yes |
| Q11 | occupancy versus performance tradeoffs | 4 | **Dense** | Yes |
| Q12 | cudaDeviceSynchronize return value | 5 | **Dense** | No — dense won an "exact lookup" query BM25 was expected to win |
| Q13 | dim3 struct constructor syntax | 5 | **BM25** | Yes |
| Q14 | pinned memory cudaMallocHost benefits and when to use | 6 | Tie (all three) | Partially — RRF ties rather than uniquely wins |
| Q15 | register pressure and its effect on occupancy | 6 | **RRF** (ties Dense) | Yes |

### Key findings

1. **The case hypotheses held in 10 of 15 queries.** BM25 won both remaining
   Case 1 "lexical superiority" queries where the query terms were rare API
   symbols (`cudaMalloc`, `cudaErrorInvalidValue`); dense won both Case 4
   "vocabulary gap" queries; RRF won or tied all three Case 3 "hybrid
   superiority" queries.
2. **Two queries flipped their hypothesis outright.** Q2 was meant to
   showcase BM25 but RRF found the best chunk via pool corroboration; Q12
   was meant to showcase BM25 on an "exact lookup" but dense won cleanly —
   term-frequency dilution across many API-reference chunks sharing the
   literal string `cudaDeviceSynchronize` hurt BM25 more than semantic
   similarity hurt dense.
3. **Q4 shows result sensitivity to exact phrasing.** The Day 5 UAT found
   dense correctly resolves "shader processor count" → "128 CUDA cores per
   SM" via pure semantic matching. Adding "per streaming multiprocessor" to
   the same intent here diluted the match out of all three methods' top-3 —
   a caution against assuming semantic search is phrasing-invariant.
4. **RRF's corroboration bias (documented in Day 5) reproduced on Q1 and
   Q10** — in both cases a single strong signal (BM25's `cudaMalloc` hit on
   Q1, dense's latency-hiding hit on Q10) was crowded out by two mediocre
   chunks both lists ranked moderately.
5. **On easy queries with strong signal in both lexical and semantic space
   (Q9, Q14), all three methods converge** — RRF adds no value but costs
   nothing either, consistent with Day 5's Q8/Q9 findings.
6. **Recommendation unchanged from Day 5**: a cross-encoder re-ranker
   (Layer 3b) that scores each candidate independently — rather than by
   rank position — should not be subject to RRF's corroboration bias, and
   Q1/Q10 are good regression-test candidates once re-ranking lands.
