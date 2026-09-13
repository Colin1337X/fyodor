# Removable CUTLASS and matched F32 measurements

September 12, 2026. The goal of beating llama.cpp remains open. This checkpoint
adds a useful optional prefill path and removes a comparison precision mismatch;
it does not establish llama.cpp-class throughput.

## Architecture and selection

`CUDA/cutlass` owns all CUTLASS C++ and exports a small versioned C ABI. The
portable runtime remains GCC C17/C23; a standalone NVCC/host-C++ build produces
an optional shared library. No C++ header or CUDA platform type reaches public
Fyodor headers. The complete directory can be removed before building. Builds
without CUTLASS headers/compiler remain valid C/NVRTC builds, and deleting all
of `CUDA` still permits independent CPU/Vulkan builds.

The bridge uses pinned CUTLASS v4.5.1, commit
`2e602843e75100d0e03934efb386b3e1e35d7907`, with a 3xTF32 GEMM and F32
accumulation/output. Fyodor's own decoder expands one bounded matrix chunk on
device for F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K storage. CUTLASS consumes that chunk
and resident activations on the existing CUDA stream. Weights, F32 KV, residuals,
Q/K/V and scratch remain resident. No layer host transfer, request allocation or
explicit synchronization is introduced by the bridge. Decode uses native kernels.

The initial 128x64x16 tile was slower than cuBLAS. Three further tile candidates
passed the quantized oracle and synthetic full-inference checks before timing.
The best measured candidate was 64x64x32; [raw tuning](stage6-tile-tuning/metadata.json)
is retained. cuBLAS remains the automatic first choice. When it is unavailable,
an available CUTLASS bridge accelerates eligible prefill. `NYA_CUDA_CUTLASS=0`
disables it; `=1` forces it for comparison. Missing/incompatible libraries retain
native CUDA. Unsupported shapes retain existing paths; execution failures report
an error and use complete-prefix CPU recovery. Reference math always bypasses it.

The existing optional scratch budget/accounting is reused, including the current
4 MiB workspace reservation. CUTLASS's selected kernel itself needs no workspace.
Vendor/driver overhead is separate from the planner's explicit allocations.

## Numerical evidence

No tolerance was relaxed. CUTLASS's corrected TF32 arithmetic is distinct from
the previously rejected single-TF32 and F16 KV experiments.

| Check, final 64x64x32 tile | Maximum scaled logits error | Existing bound |
|---|---:|---:|
| TinyLlama Q4_K_M, prefixes 65/64/63 | 0.0000606370 | 0.001 |
| Gemma 12B Q4_K_M, prefixes 33/32/31 | 0.000367090 | 0.001 |
| Synthetic long cache, prefixes 1024/1023/1022 | 0.000000189365 | 0.0005 |

Tests require real CUTLASS calls when that path is requested. They cover all seven
weight formats, aligned/tail shapes, 1 MiB chunking with output leading dimensions,
independent double dot-product oracles, reset, 3,060 graph replays and bounded
cache snapshots. Existing Gemma independent goldens, training finite differences,
checkpoint/export, speculative decoding and fallback tests remain intact.

## Same-hardware comparison

The new [C API comparison harness](COMPARISON_CAPI.md) uses upstream b10809
(`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`) with F32 K/V. Its C API accepts F32
although the official llama-bench argument whitelist does not. This is separate
benchmark tooling, not a Fyodor runtime dependency or a renamed llama-bench.

Hardware: RTX 5060 Ti 16 GiB, Ryzen 5 9600X (six exposed logical processors),
driver 610.88, Windows. All runs use the identical TinyLlama Q4_K_M file (SHA256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`), token IDs,
F32 KV, requested capacity, 512-token batch, six CPU threads, nine repetitions
and three full-workload warmups. Upstream flash attention is disabled. pp returns
last-position logits; tg returns host logits for every token. Loading/tokenizing,
sampling, cache reset and long-context prefix preparation are untimed.

With the user's authorization, competing applications including Roblox, Powder,
Steam, browsers and Spotify were stopped. Windows compositor and Codex remained
active, with approximately 9–11% reported idle GPU activity. This is reduced
contention, **not headless isolation**. No builds, tests, packaging or other Fyodor
jobs ran concurrently with the comparisons. Engines were run in forward and
reverse order. Internal arithmetic and allocator alignment remain engine-specific.
Short-context allocation is 512 tokens in both engines. For the 1,024-token
prefix plus 128 decode steps, Fyodor allocates 1,152 cache positions and upstream
rounds the same request to 1,280; both evaluate the same valid token positions.
This allocator difference is reported, not treated as identical VRAM usage.

The table averages the two nine-repetition runs. Per-run standard deviations,
latency, transfers, allocations, commands, executable/library hashes and telemetry
are in [the raw evidence](stage6-quiet/metadata.json).

| Runtime / matrix path | pp512 tokens/s | tg128 tokens/s | tg128 after prefix 1024 |
|---|---:|---:|---:|
| Fyodor native CUDA baseline | 1,346.96 | 204.05 | 139.47 |
| Fyodor + CUTLASS | 2,159.83 | 203.25 | 139.27 |
| Fyodor + cuBLAS F32 | 2,907.51 | 203.97 | 139.41 |
| llama.cpp C API, F32 KV | 11,453.91 | 286.46 | 210.79 |

CUTLASS improves the native prefill fallback by **60.3%** in this workload. It
does not improve decode and remains slower than cuBLAS. The best Fyodor prefill
path is still about 3.94 times slower than upstream; short decode is about 1.40
times slower and long-context decode about 1.51 times slower. Earlier contended
runs are preserved separately and are not compared to these as code speedups.

For short-context Fyodor runs, weights occupy 636.18 MiB, F32 KV 22 MiB and
native scratch 70.13 MiB. Optional matrices add 48 MiB to scratch (118.13 MiB
total), not another model copy. Each pp run has one final logits download/sync;
each tg step has one. All three Fyodor paths use the same native decode graph.

After the benchmark, fresh CUDA-event profiles attributed about **71%** of
cuBLAS prefill stream time to matrix operations and **27%** to attention.
CUTLASS prefill was approximately **79%** matrices / **19%** attention. The
separate decode profile was approximately **71%** matrices, **8%** norms,
**6%** RoPE, **10%** vector operations and **5%** attention. These coarse event
intervals include submission gaps, disable executable graphs and add overhead;
their timings are not normal inference latency. They support matrix/attention
work as the next targets without pretending every millisecond is kernel math.

## Validation and product checkpoint

* GCC C17 CUDA + Vulkan: **45/45 passed**, with CUTLASS, auto/disabled/missing-library
  cases, cuBLAS and native/reference paths.
* GCC C23 CUDA + Vulkan: **45/45 passed** in a fresh build with the same optional bridge.
* GCC C17 CPU: **22/22 passed**; GCC C23 + ONNX Runtime: **23/23 passed**.
* CPU ASan + UBSan: **22/22 passed**. GPU compute-sanitizer was not run.
* CUTLASS physically absent in a fresh CUDA/Vulkan source tree: **40/40 passed**.
* All CUDA physically absent: **22/22 CPU**, **25/25 Vulkan** passed.
* Real-model and long-cache results above passed without threshold changes.
* Live HTTP/API: **36 passed**. Frontend unit/packaging: **8 passed**; browser:
  **73 passed**. The final parser-only metadata guard was rechecked by unit tests.
* Windows MSI and NSIS bundles built, carrying the current C sidecar, optional
  CUTLASS DLL and upstream license. Optional DLL packaging/stale removal is tested.

The benchmark view now identifies native/cuBLAS/CUTLASS execution and reports
weight, KV and scratch allocations without counting library scratch twice.
No API or Tauri/backend protocol was changed.
Build/test transcripts, profiles, artifact hashes and source hashes are preserved
in [the validation record](stage6-validation/metadata.json). The record also
retains intermediate failed build attempts; the accepted build/test logs are
named `stage6-core-final-build.log` and `stage6-tests-accepted.log`.

## Changed areas and next targets

`CUDA/cutlass/{CMakeLists.txt,bridge.h,host.h,loader.inc,gemm.cu,README.md}` owns
the optional adapter. `CUDA/{CMakeLists.txt,cuda.c,blas.inc,resident.inc}` integrates
selection and existing scratch. Compute stats and `bench_main.c` expose actual
CUTLASS dispatch counts. Inference/quantized tests and the new
`VerifyWithoutCutlass.cmake` prove execution and removability. Benchmark tooling,
READMEs, frontend benchmark rendering and sidecar preparation complete this pass.

Ranked next work, with gains to be measured rather than promised:

1. **Higher-throughput, precision-corrected tensor-core prefill.** Current F32
   cuBLAS and 3xTF32 paths leave the largest gap. Evaluate BF16 decomposition or
   a native quantized CUTLASS operand path against the same numerical oracles;
   single low-precision products are not an acceptable substitute.
2. **Tiled prefill attention and KV reuse across queries.** The current kernel
   repeats key/value reads per query. Online-softmax tiling could reduce traffic
   and scratch while preserving F32 cache semantics and Gemma shared KV.
3. **Decode quantized kernels.** Improve instruction overlap/vector loads and
   projection fusion, then profile graph execution under the reduced background
   load. Decode memory traffic and ordinary transformer kernels are unchanged here.
4. **CPU packed weights / batch kernels**, followed by resident Vulkan lowering.
   Preserve scalar/autograd reference execution and SIMD dispatch validation.
5. **Smaller runtime costs:** cache CUTLASS launch setup and remove its unused
   workspace reservation when cuBLAS is absent. These are lower priority than
   matrix throughput and attention.

Resident MoE/MTP/PLE/multimodal/Vulkan graphs, hybrid layer offload, continuous
batching and GPU training remain separate incomplete work. F16 KV remains rejected
pending an accuracy solution. This checkpoint does not claim those features.
