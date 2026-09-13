# Persistent decode graph checkpoint — September 12, 2026

Fyodor remains an independent C engine. The primary same-hardware llama.cpp
performance objective is **not achieved**. This checkpoint removes repeated
decode graph construction while preserving the reference/autograd paths.

## Architecture change

Previously every decoded token captured hundreds of operations, updated an
executable graph and launched it. Now each resident session captures once.
A two-word device packet carries token ID and position. One stream-ordered
kernel updates that packet before each graph replay; gather, RoPE and attention
read it on device. All ordinary layer operations remain on the same CUDA stream.

K/V projections use fixed temporary addresses so graph pointers never encode a
position. Key RoPE commits rotated keys and normalized values to the indexed
cache in the same launch. Shared-KV consumers use their producer's cache and
omit the store. No reduction or quantization arithmetic changes. The initial
prototype used a separate cache-store kernel per layer; measurement motivated
fusing those stores into RoPE before promotion.

Prefill retains batched GEMM, independently of decode. A graph remains valid
across reset, changed token IDs, a shorter prefix and interleaved prefill/decode.
Host validation still checks token IDs, prefix bounds and fault injection before
submission. Failed execution preserves the existing whole-prefix CPU replay.
Each session owns its graph, metadata, temporary K/V and KV cache; teardown
synchronizes before freeing them. CUDA types remain private to `/CUDA`.

The additional owned scratch is `2 * maximum_KV_width * sizeof(float) + 8`
bytes: 2,056 bytes for TinyLlama. No new host buffer transfers occur between
layers. JSON `graph_captures` and `graph_replays` count actual graph work;
`kernel_launches` includes executed graph nodes and the metadata kernel.
After warmup, five tg128 repetitions report **zero captures, 640 replays** and
640 final-logit downloads/fences. `NYA_CUDA_GRAPHS=0` retains uncaptured execution
for diagnosis; event profiling also disables graphs.

## Measurements

The exploratory unfused prototype measured 185.67 ±0.47 tg128 versus preserved
stage 3 runs of 183.91 ±0.91 and 181.11 ±5.70. This small, noisy difference is
not a clean acceptance claim. Other user GPU applications remained present.
Raw exploratory files are `stage4-before.json`, `stage4-after.json` and
`stage4-before-repeat.json`. The fused implementation is measured separately;
do not attribute these prototype numbers to it.

Final fused measurements use TinyLlama Q4_K_M on RTX 5060 Ti 16 GiB, six CPU
workers, batch 512, F32 KV, five repetitions and two warmups. Both executables
use identical token IDs and timed work. Mean ± sample standard deviation,
tokens/second:

| Reverse-order pair | pp512 | tg128, empty prefix | tg128, 1024-token prefix |
|---|---:|---:|---:|
| Preserved stage 3, first | 2700.30 ±5.92 | 185.16 ±0.86 | 121.75 ±2.26 |
| Persistent graph, first | 2675.47 ±15.66 | 187.87 ±0.68 | 123.13 ±2.89 |
| Persistent graph, reverse | 2676.41 ±7.31 | 187.95 ±0.46 | 123.74 ±1.63 |
| Preserved stage 3, reverse | 2671.27 ±17.57 | 183.35 ±1.35 | 111.03 ±32.10 |

The short-decode improvement is modest, approximately 1.5–2.5% across these
observations. Prefill arithmetic is unchanged; no prefill gain is claimed.
The high variance in the last long-context baseline prevents a reliable
long-context speedup claim. User applications remained on the GPU; recorded
whole-device memory was about 7 GiB before/after these runs. These are not clean
acceptance measurements. Nothing is discarded to improve the reported ratio.

The first fused run measured prompt latency 191.37 ms, short decode 5.323 ms/token
and long decode 8.125 ms/token. Weights remain 667,078,656 bytes; KV is 23,068,672
bytes for short runs and 51,904,512 bytes with the long prefix. Scratch is
123,864,840 / 165,807,880 bytes respectively, including optional BLAS storage.
See [commands, hashes, telemetry and raw results](stage4-final-paired/metadata.json).

### Same-hardware llama.cpp context

A separate consecutive run with b10809 (commit `5266f24da`) produced:

| Engine | pp512 | tg128 |
|---|---:|---:|
| Fyodor | 2712.02 ±92.69 | 173.97 ±29.13 |
| llama-bench | 9627.71 ±509.53 | 269.78 ±1.27 |

Both use the same file, batch 512, six threads and CUDA. **Fyodor KV is F32;
llama KV is F16**, and their synthetic token sequences/warmups differ. External
GPU activity remains a confounder, particularly for this Fyodor decode run.
This is context, not an equivalent-workload acceptance result or a win.
[Raw comparison, process RAM and whole-device samples](stage4-llama-contended/metadata.json)
preserve all results.

### Separate operation profiles

Single pp512 and tg128 diagnostic runs with `NYA_CUDA_PROFILE=1`, no warmup,
report these CUDA event durations in milliseconds:

| Operation | pp512 | tg128 |
|---|---:|---:|
| Matrices (including expansion/BLAS during prefill) | 138.274 | 603.579 |
| RMSNorm | 0.844 | 79.620 |
| RoPE | 0.519 | 56.125 |
| Vector/activation operations | 2.824 | 87.380 |
| Attention | 47.586 | 45.352 |
| Embedding gather | 0.022 | 0.537 |

Matrices account for approximately 73% of measured prefill event time and 69%
of decode; prefill attention adds about 25%. Removing host graph construction
does not remove those GPU costs. Decode norm/vector/RoPE launches are secondary
fusion targets. Event profiling disables graphs and can include submission gaps;
these figures are diagnostic intervals, not hardware instruction attribution or
normal graph-replay wall time. Other GPU activity was not suppressed. Raw
[prefill](stage4-profile-prefill.log) and [decode](stage4-profile-decode.log)
records remain beside their benchmark JSON.

## Validation

| Configuration/check | Result |
|---|---|
| GCC C17 CUDA + Vulkan | 39/39 passed |
| GCC C17 CPU | 21/21 passed |
| GCC C23 + ONNX Runtime | 22/22 passed |
| ASan + UBSan diagnostic CPU | 21/21 passed |
| CUDA physically absent, fresh CPU source | 21/21 passed |
| CUDA physically absent, fresh Vulkan source | 24/24 passed |
| Live HTTP/API compatibility | 36 checks passed |
| Frontend unit / isolated Edge browser checks | 6 / 73 passed |
| Vite production / Windows MSI and NSIS | built successfully |

Two new tests compare every logit through changed tokens, reset and mixed
prefill/decode in fast and reference CUDA modes, and require one capture rather
than accepting CPU fallback as graph coverage. Gemma fixtures require four
replays of one graph while retaining their independent float64 goldens and
3e-5 absolute bound. The long-context fixture checked **3,060 replays** across
three 1024-token passes; maximum scaled error was 1.27e-7. TinyLlama decode
65/64/63-token prefixes passed with maximum scaled error 4.28e-5.
Gemma 12B Q4_K_M decode at 33/32/31-token prefixes passed with maximum scaled
error 0.0003794 and maximum absolute error 0.0004139, inside the unchanged
real-model scaled bound of 0.001.

The test suites retain training/autograd finite differences, optimizer and
checkpoint behavior, quantized parity, speculative decoding, API bounds,
allocation failure, model memory rejection and CPU replay. GPU training is not
introduced. No F16 KV, ROCm or MLX support is implied by this change.

[Validation logs and source hashes](stage4-validation/source-hashes.json) are
retained beside the benchmark records. The packaged sidecar hash matches the
CUDA build. Native process smoke found a live Fyodor window, one backend child
and no console child. Native visual inspection remains unverified because the
Computer tool could not initialize; this is a process check, not native UI QA.

## Files changed in this continuation

* `CUDA/resident.inc`, `CUDA/resident.cu`: request metadata, one-time graph
  capture, fixed K/V projection buffers, fused RoPE/cache stores and accounting.
* `CUDA/cuda.c`: remove the now-unused graph-update driver dependency.
* `backend/include/compute.h`, `backend/core/bench_main.c`: graph counters.
* `backend/tests/test_inference.c`, `backend/tests/test_architectures.c`,
  `backend/CMakeLists.txt`: reset/interleaving/reference/shared-KV graph checks.
* Backend documentation and raw benchmark/validation records; refreshed frontend
  QA evidence and packaged sidecar. The larger preceding UI/BLAS change is
  documented in [stage 3](STAGE3.md).

## Remaining bottlenecks and next work

1. **Native precision-preserving tensor-core prefill and quantized matrix
   packing.** The largest throughput gap remains projection work; optional
   F32 BLAS still expands quantized matrices and uses F32 arithmetic. The native
   tiled fallback remains substantially slower. Reduced precision must pass
   existing logits bounds before promotion.
2. **Explicit F16 KV with independently checked conversion/indexing.** This
   reduces long-context traffic and enables matching the available llama-bench
   configuration. Existing F32/F16 comparison tables do not prove equivalence.
3. **Decode kernel fusion and quantized activation dot dispatch.** Graph
   recapture is removed; GPU projection, attention and small-op execution remain.
   Profile these separately before selecting another kernel change.
4. **CPU weight repacking/wider GEMM, then Vulkan graph residency.** CPU SIMD
   and worker pools remain; Vulkan still uses synchronous matvec assistance.
5. **Native async generation/streaming and further product separation.** The
   redesigned frontend streams remote text; native generation is still blocking.

Other applications are not stopped or reconfigured to improve benchmark numbers.
Clean performance acceptance and an equivalent-workload win remain open.
