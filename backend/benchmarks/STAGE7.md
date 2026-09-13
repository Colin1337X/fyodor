# Query-tiled F32 prefill attention

September 13, 2026. Fyodor remains slower than llama.cpp. This checkpoint improves
prefill and memory use without changing KV precision, training, model formats,
HTTP behavior or the removable CUDA/CUTLASS boundary.

## Accepted architecture change

`CUDA/resident.cu` now groups eight prefill queries per block and stages 32 keys
and values in shared memory. K is transposed with a 33-float pitch: cooperative
global loads remain contiguous and the transpose avoids shared-memory bank
conflicts. Each query warp computes 32 scores independently, one per lane, using
four F32 partial sums. Q is broadcast from shared memory. F32 online softmax
rescales the running numerator and denominator when a tile raises the maximum.
It does not materialize the prompt's score matrix in global memory.

The kernel supports 64- and 128-wide heads with at least eight queries. It uses
18.25/36.5 KiB of shared memory, respectively, without per-operation allocation.
Every query applies its own causal and sliding-window mask. Entirely masked
tiles skip their warp update, avoiding `-infinity - -infinity`. Inactive tail
queries still participate in block barriers. Existing prefixes and shared-KV
aliases use the same persistent cache indexing as decode.

Decode, other widths and explicit reference mode keep their previous attention
kernels. When every layer qualifies, global score scratch holds at most seven
fallback queries; mixed-width models and reference mode retain full batch
scratch. Checked memory accounting uses this smaller requirement before weights
are admitted. The native, cuBLAS and optional CUTLASS matrix paths all use the
same attention dispatch. Benchmark JSON reports `tiled_attention_calls`.

No new library, layer transfer, synchronization or host sampling operation was
introduced. F32 cache semantics and all numerical thresholds remain unchanged.
This is native SIMT query tiling, not a tensor-core attention implementation.

## Measured comparison

RTX 5060 Ti 16 GiB, Ryzen 5 9600X, six exposed logical processors, Windows,
driver 610.88. TinyLlama Q4_K_M SHA-256:
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
Both Fyodor versions use cuBLAS F32 prefill and native decode. The saved stage-6
executable is the before baseline. Each workload runs in forward/reverse engine
order, with nine repetitions and three warmups per run. Numbers below combine
18 repetitions; standard deviations include the difference between run means.

| Workload | Before tokens/s | After tokens/s | llama.cpp tokens/s |
|---|---:|---:|---:|
| pp512 | 2,931.03 ± 29.49 | **3,268.38 ± 31.48** | 11,578.58 ± 166.49 |
| tg128, empty prefix | 206.28 ± 0.97 | 206.54 ± 0.63 | 291.88 ± 1.95 |
| tg128, 1024-token prefix | 140.27 ± 0.23 | 140.50 ± 0.37 | 212.52 ± 2.79 |

Prefill improves **11.5%**; mean prompt latency falls **174.70 → 156.67 ms**.
Decode differences are small enough that this checkpoint claims no decode gain.
Short decode remains approximately 4.84 ms/token and long decode 7.12 ms/token.
llama.cpp remains about **3.54× faster in prefill, 1.41× in short decode and
1.51× in long decode**.

| Device allocation | Before | After |
|---|---:|---:|
| Weights | 636.18 MiB | 636.18 MiB |
| Short-context KV | 22.00 MiB | 22.00 MiB |
| Short-context scratch, including optional matrix scratch | 118.13 MiB | **86.56 MiB** |
| Long-context scratch, including optional matrix scratch | 158.13 MiB | **87.11 MiB** |

Prefill performs one final logits download/sync per repetition; decode performs
one per token. Nine pp512 repetitions dispatch 198 tiled attention kernels.
Decode still replays 1152 graphs for nine tg128 repetitions, without recapture.
Kernel launch counts and transfer counts are unchanged.

Separate forward/reverse runs with the same repetition/warmup counts also
confirmed native prefill **1361.90 → 1431.80 tokens/s (+5.1%)** and CUTLASS
prefill **2174.46 → 2364.23 (+8.7%)**. Their raw records are in
[stage7-providers](stage7-providers/metadata.json); they are not pooled with the
cuBLAS comparison because matrix implementation is a separate configuration.

Paired CUDA-event profiles over four pp512 workloads measured cuBLAS attention
**184.23 → 109.87 ms (-40.4%)**, while matrix time stayed **498.70 → 499.93 ms**.
Matrices now account for about **80%** of that stream profile, attention **18%**.
Native/CUTLASS attention likewise fell 176.49 → 113.67 / 181.44 → 110.79 ms.
Profiling disables graphs and adds timing overhead; these category intervals
can contain submission gaps and are not substituted for benchmark wall time.
Removing the remaining attention cost alone would still leave a large gap.

The independent [llama C API harness](COMPARISON_CAPI.md) uses release b10809,
commit `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, matching token IDs, F32 K/V,
full CUDA offload, six CPU threads and batch 512, with flash attention disabled.
The long workload has identical valid tokens but llama rounds its 1152-token
requested capacity to 1280; Fyodor allocates 1152. Internal kernels differ.
Model loading, tokenization and untimed prefix setup are excluded.

Nonessential competing GPU apps were stopped under the user's authorization.
Windows desktop components and Codex remained active; this is not headless
isolation. No build, test or other Fyodor job ran beside the timed comparisons.
Raw JSON, commands, hashes and GPU telemetry are in [stage7-final](stage7-final/metadata.json).

## Rejected experiments

* Four independent Q4_K/Q6_K decode accumulators passed the existing quantized
  oracle but reduced short decode **206.35 → 193.31 tokens/s** and long decode
  **140.28 → 135.95**. The original matvec implementation was restored.
  Evidence: [stage7-ilp](stage7-ilp/metadata.json).
* Accumulating four value slices together in the original attention kernel
  provided no useful prefill gain and reduced long decode **140.18 → 135.96**.
  That implementation was also restored before adding the separate prefill
  kernel. Evidence: [stage7-attention](stage7-attention/metadata.json).
* The first query-tiled prototype still reduced every key across a warp. Its
  2957.55 tokens/s probe and 186.71 ms aggregate attention profile did not show
  a meaningful gain. Transposing K and assigning independent scores to lanes
  produced the accepted improvement. Probe/profile measurements are separate
  from the final matched benchmark and are not used as acceptance numbers.

## Validation and changed files

* GCC `-O3`, C17 CUDA + Vulkan: **46/46 passed**.
* GCC `-O3`, C23 CUDA + Vulkan: **46/46 passed**.
* C17 CPU: **22/22**; C23 + ONNX Runtime: **23/23**.
* CPU ASan + UBSan: **22/22**.
* CUTLASS physically absent in a fresh CUDA/Vulkan source tree: **41/41**.
* CUDA physically absent in fresh source trees: **22/22 CPU**, **25/25 Vulkan**.
* Real TinyLlama prefixes 65/64/63: worst scaled logit error **0.00005922**.
  Real Gemma 4 12B prefixes 33/32/31: worst scaled error **0.00044673**.
  The real-model threshold remains **0.001**; synthetic and independent golden
  thresholds are unchanged.
* New seeded graph cases cover widths 32/64/128/130/256, windows 0/1/17,
  shared KV, query/tile tails and fallback. The 1025-token shared-cache case
  crosses two full prefill chunks and checks **3051** graph replays across
  resets and mixed prefill/decode, including intermediate logits/KV snapshots.
* NVIDIA Compute Sanitizer 2025.4.1: full native-pipeline **memcheck, initcheck
  and synccheck each report zero errors**; attention-filtered **racecheck
  reports zero hazards/errors/warnings**. These runs use the seeded shape suite
  with native matrices, not vendor-library kernels or all possible models.
* HTTP/API: **36 passed**. Frontend unit/packaging: **8 passed**. Browser:
  **73 passed**. Windows MSI/NSIS bundles rebuilt; packaged backend hash matches
  the tested executable. Both optional matrix libraries and CUTLASS license remain
  bundled. Installers were built, not installed.

An initial attention-only `initcheck` run incorrectly excluded producer kernels,
so initialized buffers appeared unwritten. The final run instruments the entire
pipeline and reports zero errors. The invalid filtered run is retained as evidence,
not counted as a passing check. The validation helper now avoids that filter for
memory, initialization and synchronization checking. Tool usage follows
[NVIDIA's Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Build/test logs, raw profiles, source/artifact hashes, helper scripts and rejected
experiment patches are preserved in [stage7-validation](stage7-validation/metadata.json).
No numerical thresholds were loosened. The Q4_K/Q6_K experiment's separate NVCC
register report changed Q4_K from 39 to 40 registers and left Q6_K at 40, with
no spills. This does not establish an occupancy explanation for its regression;
exact NVRTC/JIT instruction profiling remains follow-up work.

Production changes are in `CUDA/resident.cu`, `CUDA/resident.inc`,
`backend/include/compute.h`, and `backend/core/bench_main.c`. Tests add head-width,
window, shared-cache, batch-tail and long-prefix coverage in
`backend/tests/test_inference.c` and `backend/CMakeLists.txt`. The frontend's
benchmark importer validates the new optional counter. `backend/README.md` and
this report document dispatch and memory policy. CUTLASS C++ and its C ABI are
unchanged; `/CUDA/cutlass` and all of `/CUDA` remain removable.

## Remaining work

1. Precision-corrected matrix kernels/direct quantized tensor-core operands:
   matrix multiplication remains the largest prefill cost and performance gap.
2. Profile/tune quantized decode data movement and projection fusion. The rejected
   accumulator experiment shows that extra instruction overlap alone is not
   sufficient; preserve register occupancy and measure whole-token latency.
3. Extend tiled attention to larger heads and evaluate tile sizes per device.
   The current selection is validated on this GPU, not a universal optimal tile.
4. CPU packed matrix kernels and a resident Vulkan graph remain larger independent
   performance steps. Hybrid layer offload, continuous batching and GPU training
   are still incomplete and are not claimed by this checkpoint.
