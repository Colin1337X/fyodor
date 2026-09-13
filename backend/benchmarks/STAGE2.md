# Second runtime optimization pass — September 10, 2026

This continues the [first residency milestone](README.md). Fyodor remains its
own implementation: no ggml/llama.cpp library, copied engine, or required CUDA
math-library dependency was introduced. Tests and measurements below use the
same TinyLlama Q4_K_M file, Ryzen 5 9600X, RTX 5060 Ti 16 GiB, and Release C17
build as the earlier work. Fresh baselines were recorded because desktop load
and clocks differ between sessions.

## Retained changes

* **CUDA F32 GEMM:** a 64×64 output tile with 4×4 per-thread register tiles
  doubles shared-operand reuse over the previous 32×32 tile. Accumulation stays
  F32 and traverses K in the same order. Shapes below 32 rows/tokens retain the
  small tile. All seven supported storage formats use both implementations.
  Kernels live in `CUDA/gemm.cu`, separately from decode kernels.
* **CUDA storage conversion:** hardware F16-to-F32 conversion replaces software
  exponent decoding in normal kernels. Byte assembly preserves alignment safety
  for packed GGUF scales. Every finite F16 and BF16 encoding is checked against
  the scalar decoder; reference mode retains software conversion.
* **CUDA embedding gather:** prefill gathers a bounded packet of up to 512 token
  IDs in one launch. The packet is a 2048-byte kernel parameter, below the legacy
  4 KiB parameter budget. Decode retains the small single-token kernel. There is
  no new activation upload or per-layer synchronization.
* **CPU packed prefill:** pack the prompt once per projection into K-major,
  SIMD-padded storage. Two decoded weight rows share each activation vector;
  four partial sums per row limit rounding depth. Scratch belongs to the CPU
  context and grows only when necessary. Short batches retain row-dot dispatch.
* **CPU attention:** a checked host-F32 attention descriptor extends the backend
  vtable. AVX2/AVX-512 kernels process contiguous head lanes and retain value
  accumulation in registers. Independent query/head jobs share the existing
  worker pool, with private score scratch per worker. Causal/GQA/sliding-window
  and shared-KV offsets are preserved. Bidirectional multimodal spans and MTP
  borrowed prefixes retain their existing reference window semantics.
* **Upload ordering fix:** expanded resize/reuse tests exposed a real race.
  Pageable `cuMemcpyHtoD` may return before DMA completes, while the nonblocking
  execution stream does not implicitly wait for that copy. Initialization and
  standalone host-buffer dispatch now finish the legacy copy stream before
  consuming uploads. Resident token execution adds no such fence. See NVIDIA's
  [copy synchronization contract](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).

## Controls and accounting

Defaults select the new paths when available. These controls allow isolated
comparisons; they do not mutate model weights or training arithmetic:

| Control | Behavior |
|---|---|
| `NYA_CUDA_GEMM_TILE=32` | Force the older F32 prefill tile; default uses 64 for sufficiently large shapes |
| `NYA_CUDA_REFERENCE=1` | Existing reference matvec/attention, software half conversion, small GEMM, FMA disabled |
| `NYA_CPU_GEMM=dot` | Retain the previous row-decode + token-dot prefill path |
| `NYA_CPU_ATTENTION=reference` | Decline optimized attention so the transformer uses its scalar implementation |
| `NYA_CPU_ISA=scalar\|avx2\|avx512` | Existing safe ISA selection; unavailable ISA falls back |
| `NYA_CPU_PROFILE=1` | Print coarse wall-clock prefill total, projections, and attention/RoPE milliseconds |
| `NYA_CUDA_PROFILE=1` | Existing CUDA event profile; disables graphs and adds diagnostic overhead |

CPU extra scratch is bounded by the largest accepted projection/window so far:
two decoded rows per worker, `columns × round_up(batch,16)` packed F32 inputs,
and one score row per worker. Arithmetic is checked before allocation. CPU
projection scratch and attention jobs are externally serialized by the compute
context contract. All scratch is freed at context destruction.

CUDA allocated weight/KV/activation bytes are unchanged. The larger tile uses
per-block shared memory, not a model-wide dequantized copy. pp512 kernel-node
count dropped from 2532 to 999 across three repetitions (844→333 per prompt).
Throughput from the gather change alone was within measurement variation; this
is a dispatch-count reduction, not a claimed large speedup. pp still has one
final-logit download/fence per repetition, and tg has one per token. Token IDs
remain kernel parameters and are not counted as buffer uploads.

## Measured results

The preserved pre-pass executable and final executable were rerun consecutively,
with six threads, CPU batch 32, CUDA batch 512, three repetitions and one warmup.
These paired results are the main before/after comparison, rather than choosing
the fastest historical baseline. Units are tokens/s, mean ± sample standard
deviation. [Commands, binary hashes and records](stage2-paired-metadata.json).

| Backend | Before pp512 | After pp512 | Before tg128 | After tg128 |
|---|---:|---:|---:|---:|
| CPU | 75.10 ± 0.97 | 169.68 ± 0.53 | 48.36 ± 0.57 | 51.40 ± 0.53 |
| CUDA | 693.17 ± 1.31 | 1365.38 ± 4.60 | 168.05 ± 0.27 | 204.18 ± 0.55 |

Paired improvement: CPU prefill **2.26×**, CPU decode **1.063×**; CUDA prefill
**1.97×**, CUDA decode **1.215×**. CPU prompt latency fell 6818.25→3017.49 ms,
and CUDA prompt latency 738.63→374.99 ms. Decode latency fell 20.681→19.457
ms/token on CPU and 5.951→4.898 ms/token on CUDA. Summed measured elapsed time
is retained in each JSON. The raw records are
[CPU before](stage2-paired-before-cpu.json), [CPU after](stage2-paired-after-cpu.json),
[CUDA before](stage2-paired-before-cuda.json), [CUDA after](stage2-paired-after-cuda.json).

Intermediate measurements are not perfectly controlled ablations: CPU prefill
varied 119–170 tokens/s between intermediate and final runs, and whole-desktop
GPU occupancy changed during the session. The final paired rerun confirms the
overall improvement but does not assign every percentage point to one kernel.

### Independent llama-bench comparison

Official llama.cpp b10809 (`5266f24da`), same model and hardware, serial runs.
The F32-versus-F16 KV and synthetic-input caveats below apply.

| Engine | pp512 | tg128 | tg128 after 1024-token prefix |
|---|---:|---:|---:|
| Fyodor CPU | 165.10 ± 3.06 | 50.13 ± 0.63 | not measured |
| llama-bench CPU | 539.26 ± 14.02 | 68.29 ± 1.41 | not measured |
| Fyodor CUDA | 1374.38 ± 3.00 | 205.48 ± 0.32 | 139.42 ± 0.42 |
| llama-bench CUDA | 10157.38 ± 570.75 | 289.81 ± 0.82 | 283.89 ± 4.15 |

Raw outputs, commands, model hash and resource samples:
[CPU](stage2-final-cpu/metadata.json), [CUDA](stage2-final-cuda/metadata.json),
[long context](stage2-final-context1024/metadata.json). This leaves approximately
3.27×/1.36× CPU and 7.39×/1.41× CUDA prefill/decode throughput gaps; long-context
CUDA remains 2.04× below the reference. These are measured gaps, not parity claims.

| Resource in the comparison run | CPU | CUDA short | CUDA 1024-prefix |
|---|---:|---:|---:|
| Sampled peak process working set, MiB | 641.12 | 774.33 | 773.51 |
| Resident weights, bytes | n/a | 667078656 | 667078656 |
| Resident KV, bytes | n/a | 23068672 | 51904512 |
| Resident activation/scratch, bytes | n/a | 73531136 | 115474176 |
| Sampled whole-GPU memory peak, MiB | 809 | 1849 | 1915 |

Whole-GPU samples include other applications; they are not Fyodor allocation
totals. CUDA utilization samples reached 99%, with initialization included in
the sample window. CPU device zeros in benchmark JSON mean unavailable/not
resident; they do not report CPU RAM. The paired baseline did not sample process
RAM, so a measured before/after RAM delta is unavailable. Explicit CUDA weight,
KV and scratch byte accounting is unchanged before/after.

## Rejected experiments

Numerical thresholds were not relaxed to admit a faster kernel.

* A three-product TF32 tensor-core prototype appeared fast, but real TinyLlama
  logits failed the existing scaled-error bound. `stage2-tf32-cuda.json` is a
  **rejected numerical result**, not an accepted speedup.
* Six-product TF32 with short accumulation chains and compensated totals passed
  TinyLlama and Gemma checks, but measured only 592.08 pp512 / 152.71 tg128,
  slower than the baseline. See `stage2-tf32-corrected-cuda.json` and
  `stage2-tf32-logits.log`. It is not included in production source or selectable
  by an environment flag. A shorter three-product corrected variant passed the
  short check but was not promoted to a production feature.
* Two warps per decode row passed checks but measured 203.33 tg128, effectively
  identical to one warp (203.29). Four warps failed the TinyLlama scaled-error
  bound (0.00105589 > 0.001) and were not benchmarked. Neither variant is shipped.

The expanded quantized tests preserve the original 17-row/three-token checks.
New cancellation-heavy rows additionally use an independent double-precision
dot oracle with a `1e-6 × (1 + sum(abs(products)))` forward-error bound, tighter
than the worst-case 1024-term F32 reduction bound. Full-model scaled-error
thresholds remain unchanged; large errors are not hidden by this kernel test.

## Reproduction

`stage2-before-cpu.json` and `stage2-before-cuda.json` are fresh pre-change
baselines. Intermediate accepted measurements are `stage2-tile64-cuda.json`,
`stage2-half-cuda.json`, `stage2-gather-cuda.json`, `stage2-blocked-cpu.json`,
`stage2-blocked-unrolled-cpu.json`, and `stage2-attention-cpu.json`.

Use [compare.py](compare.py) to run the final engines serially and capture
commands, model hash, environment, JSON, process RAM and whole-device telemetry.
CPU batch/ubatch is 32, CUDA is 512, six CPU threads, three measured repetitions
and one Fyodor warmup. Timing excludes load/tokenization/sampling/HTTP. llama-bench
b10809 uses F16 KV because it rejects F32 KV, whereas Fyodor uses F32; synthetic
token sequences and warmup semantics also differ. These are same-hardware,
same-model comparisons, not exact configuration parity.

## Verification scope

The expanded tests cover 65-row matrices with batches 3/8/17/32/65, odd F32 K
tails, both CUDA tiles, all seven quant/storage formats, exhaustive finite
F16/BF16 conversion, growing/shrinking scratch, output canaries, and repeated
contexts. Independent double-precision attention tests cover GQA, causal and
sliding windows, odd head widths, scratch reuse, nonfinite failure/recovery,
invalid dimensions and checked overflow rejection. Existing training/autograd,
checkpoint/export, Gemma shared-KV, multimodal, MTP/speculative, GPU fallback,
allocation-failure and API tests are preserved.

Real-model logit evidence is retained in `stage2-final-logits-cpu.log` and
`stage2-final-logits-cuda.log`: TinyLlama prefixes 65/64/63, supplied Gemma 12B
prefixes 8/7/6, and synthetic 1024/1023/1022-token cache checks. These numerical
checks do not constitute model-quality evaluation.

Maximum real-model absolute/scaled errors: TinyLlama CPU 0.0000434/0.0000366,
CUDA 0.0000467/0.0000275; Gemma 12B CPU 0.000544/0.000422, CUDA
0.001143/0.000813. All remain within the unchanged real-model scaled bound of
0.001. The independent small Gemma fixtures retain their stricter bounds.
The 1024-token synthetic cache check stayed below 1.8e-7 absolute error.

| Executed configuration | Result |
|---|---:|
| GCC 16.2 C17 CPU | 21/21 passed |
| GCC 16.2 C17 CUDA + Vulkan | 34/34 passed |
| LLVM-MinGW ASan + UBSan CPU | 21/21 passed |
| Standalone GCC C23 + ONNX Runtime | 22/22 passed |
| Fresh source physically without `/CUDA`, CPU | 21/21 passed |
| Fresh source physically without `/CUDA`, independent Vulkan | 24/24 passed |
| Live HTTP/process checks | 36/36 passed |
| Vite production build and Tauri Windows MSI/NSIS bundles | Passed |

See [validation records](stage2-validation/). No sanitizer findings occurred in
these exercised paths. This is not proof of absence of every UB. Linux/macOS,
MSVC inference, ARM and additional GPUs were not executed here.

After batched gather was added, the complete CUDA/Vulkan suite was rerun and
TinyLlama 65/64/63 plus the 1024/1023/1022-token synthetic cache were rechecked:
[final gather logits](stage2-gather-logits-cuda.log), unchanged numerical results.
The packaged backend binary is checked against the tested build by SHA-256.

## Files changed

* CUDA: `CUDA/CMakeLists.txt`, `cuda.c`, `matvec.cu`, new `gemm.cu`,
  `resident.cu`, `resident.inc`.
* Backend contract: `backend/include/compute.h`, `compute_backend.h`,
  `backend/core/compute.c`; the Vulkan vtable in `backend/vulkan/vulkan.c`
  explicitly declines the new host attention operation.
* CPU execution: `backend/core/cpu_backend.c`, `cpu_kernels.c`, `cpu_kernels.h`,
  `cpu_simd.inc`, `cpu_simd_undef.inc`, `llm_cpu.c`, `llm_cpu_prefill.inc`.
* Tests/build: `backend/CMakeLists.txt`, `backend/tests/test_compute.c`,
  `test_quant_kernels.c`, new `test_attention.c`.
* Documentation/evidence: `backend/README.md`, `backend/benchmarks/README.md`,
  this report, paired/comparison JSON, profiles, logits and validation logs.
* Generated desktop artifacts: refreshed backend/trainer resources and Windows
  MSI/NSIS bundles. Frontend application source and native training math were
  not changed in this pass.

[Source hashes](stage2-validation/source-sha256.json) identify the tested code.

## Remaining bottlenecks and the next architectural step

The final [CPU profile](stage2-profile-final-cpu.log) recorded 3156 ms prefill:
1987 ms projections (63%), 358 ms attention/RoPE (11%), and 811 ms other work
(norm, activation, embedding and orchestration). These are coarse wall-clock
diagnostics. The earlier attention share was about 41%.

Separate CUDA event profiles attribute 326.10 of 373.18 ms (87%) of
[pp512 kernel time](stage2-profile-pp-cuda.log) to projections. For
[tg128](stage2-profile-tg-cuda.log), projections take 520.52 of 707.04 ms (74%).
Event instrumentation disables graphs and changes launch behavior; its summed
times must not be substituted for normal end-to-end latency. There are no
ordinary layer-to-layer host/device transfers left to eliminate in dense CUDA.

Ranked next work, by expected benefit for these measured workloads:

1. **CUDA operand packing and a numerically validated tensor-core GEMM path.**
   Matrix work and the 7.39× prefill gap dominate. The tested TF32 correction
   scheme is too expensive; a new tiled operand representation, accumulation
   strategy and memory budget are needed. An optional dynamically loaded tuned
   library could coexist with the validated custom F32 fallback.
2. **CPU cache-blocked packed weight panels / quantized activation kernels.**
   Current packing shares activations between two rows but still repeatedly
   traverses large K spans. A multi-level K/M/N packing schedule, or an explicit
   quantized activation representation, is the next substantial matrix step.
   Scalar norms and activation functions also account for part of the remaining
   non-matrix cost.
3. **Stable device request metadata and persistent decode graphs.** Graphs still
   recapture/update token/position-dependent arguments per token. Device metadata
   and indexed KV writes would permit one captured graph to replay unchanged;
   device sampling could then avoid a complete logit download.
4. **Tiled online-softmax attention and optional F16 KV.** This needs a different
   cache/attention layout and explicit precision policy. Long-context decode is
   still about 2× below the F16-KV reference.

Resident routed experts/PLE/multimodal/MTP lowering, resident Vulkan, deliberate
per-layer offload, continuous batching and GPU training remain separate work.
Training/autograd is preserved on its independent reference path. No unsupported
operation is silently treated as executed, and `/CUDA` remains removable.
