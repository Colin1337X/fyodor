# Matrix dispatch and product checkpoint — September 11, 2026

The performance goal remains **open**. This pass advances Fyodor's own runtime;
it does not embed, wrap or replace it with llama.cpp/ggml. No result in this
document establishes an equivalent-workload win over llama.cpp.

## Source audit and priorities

Read-only source checkouts under ignored `.tools` pin the references:

* llama.cpp benchmark release **b10809**, commit
  `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`.
* Current llama.cpp, commit `16378d93f94012d4228c8c7683adce3f286aee5d`.
* Cherenkov, commit `446b631e5294f2b36535453e15b3557e21e80c83`.

The relevant differences are concrete:

| Area | Reference implementation | Fyodor at start of this pass |
|---|---|---|
| CUDA quantized prefill | Shape/device dispatch selects MMVQ or MMQ. MMQ packs activations into Q8_1 blocks with padding, then uses specialized integer/MMA tiles. | Repeated F32 tiled multiplication with quantized weights decoded inside each tile. Previous profile attributes about 87% of prefill kernel time to matrices. |
| Floating matrices | MMVF/MMF/cuBLAS dispatch depends on storage, shape, architecture and layout. | Native 32/64 output tiles for every batched projection. |
| Decode | Quantized activation dot kernels, architecture-specific tuning and graph fusion. | Resident compressed weights and F32 activations, one warp per output row; graph recapture/update per token. |
| Scheduling | Backend-owned buffers, asynchronous copies, graph splits and explicit dependencies. | Whole-graph CUDA admission, persistent KV/scratch and one final readback; no ordinary per-layer PCIe transfers. |
| CPU | Repacked backend buffers, quantized activation dots and reusable thread pools. | SIMD dequant/dot and packed F32 prefill using the existing worker pool. |
| UI | llama.cpp separates sidebar, draft/form state and response rendering; Cherenkov groups editor/control/provider panels and streams incrementally. | Dense monolithic renderer, hard-coded dark colors, full-view refreshes and no remote streaming. |

Source anchors:
[CUDA dispatch](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/ggml-cuda.cu),
[MMQ selection/packing](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/mmq.cu),
[current MMVF/MMF family](https://github.com/ggml-org/llama.cpp/blob/16378d93f94012d4228c8c7683adce3f286aee5d/ggml/src/ggml-cuda/mmf.cuh),
[backend scheduling](https://github.com/ggml-org/llama.cpp/blob/16378d93f94012d4228c8c7683adce3f286aee5d/ggml/src/ggml-backend.cpp),
[llama.cpp UI](https://github.com/ggml-org/llama.cpp/tree/16378d93f94012d4228c8c7683adce3f286aee5d/tools/ui),
[Cherenkov](https://github.com/Colin1337X/cherenkov/tree/446b631e5294f2b36535453e15b3557e21e80c83).
These are architecture references, not imported engine or UI implementations.

## Retained runtime changes

Optional cuBLAS provides an intermediate F32 matrix path inside the existing
resident graph. It does not tokenize, interpret GGUF, execute attention, manage
KV, or generate tokens. All those remain Fyodor code. A small private C ABI
loader lives in `/CUDA`; SDK cuBLAS headers and link libraries are unnecessary.
The host project remains GCC C17/C23 and device code remains NVRTC-only.
The previous device templates were replaced by force-inlined functions with
constant format arguments; portable headers expose no vendor types.

For batch >=32, admitted plans decode one matrix/chunk into reused GPU F32
scratch and run GEMM on the existing nonblocking stream. F32 source matrices
need no expansion. Row-major weights/input/output map to BLAS transpose/leading
dimensions without activation copies. A 64 MiB expansion ceiling and a separate
4 MiB workspace bound owned optional scratch; the actual allocation is sized
from transformer projections, excluding the vocabulary embedding/output which
are not batched in this graph. Larger matrices split along output rows, keeping
each dot product intact. No allocation, upload, download or synchronization is
introduced between layers.

Optional-scratch OOM or admission shortage retains native CUDA GEMM. Missing
libraries retain native CUDA. Execution errors are reported and propagate to
the existing complete-prefix CPU replay path; partially computed GPU outputs
are never returned as successful results. The custom CUDA path remains fully
usable after deleting optional cuBLAS DLLs, and CPU/Vulkan builds remain usable
after removing `/CUDA` itself.

F32 BLAS math is explicitly selected; reduced-precision TF32 mode is not enabled.
An experimental BF16x9 run was slower than the F32 trial and was removed.
Previous TF32 prototypes remain rejected experiments, not supported features.
The native/autograd scalar reference and all numerical thresholds are unchanged.
See [NVIDIA's math-mode and GEMM contracts](https://docs.nvidia.com/cuda/cublas/index.html).

## Controls and measurement

* `NYA_CUDA_BLAS=0`: native CUDA matrices even if libraries are present.
* `NYA_CUDA_BLAS_LIBRARY`: explicit optional library file.
* `NYA_CUDA_BLAS_SCRATCH_MIB=1..64`: expansion ceiling; additional workspace is 4 MiB.
* `NYA_CUDA_PROFILE=1`: existing event profile, including BLAS matrix duration;
  graph capture is disabled while profiling, so rates are diagnostic.

JSON records the selected prefill implementation and F32 KV explicitly.
`external_matmul_bytes` is a subset of reported scratch, not another allocation
to add. `kernel_launches` counts explicit Fyodor kernels;
`external_matmul_calls` separately counts opaque BLAS calls. Vendor-internal
kernel counts and allocations require a vendor profiler/device telemetry.
Early `stage3-blas-cuda.json` and `stage3-paired-blas.json` predate that counter
split and count each BLAS dispatch as one additional logical launch.

The initial three-repetition baseline was 1344.48 ±2.49 pp512 and 200.53 ±0.45
tg128. A later paired native/BLAS run gave 1778.23 ±6.05 versus 3773.15 ±28.60
pp512, but decode varied 259.16 ±2.28 versus 219.04 ±24.46. The large change in
the **unchanged native path** and decode variance demonstrate that different
desktop runs cannot be treated as controlled speedups. No decode gain is claimed.
Raw evidence remains in this directory, including rejected experiments.

The final reverse-order set was **contended**: KoboldCpp and Roblox were observed
on the same GPU, with approximately 8 GiB of whole-device VRAM in use. These
applications were left alone. All runs below use TinyLlama Q4_K_M, batch 512,
six CPU workers, RTX 5060 Ti 16 GiB, driver 610.88. Values are mean ± sample
standard deviation in tokens/second. No clean performance acceptance is claimed.

| Paired run (7 repetitions, 3 warmups) | pp512 | tg128 |
|---|---:|---:|
| Preserved stage 2, first | 806.71 ±17.62 | 117.51 ±0.88 |
| Native stage 3, first | 802.24 ±19.51 | 116.71 ±2.98 |
| Optional F32 BLAS, first | 1721.66 ±36.58 | 116.73 ±1.79 |
| Optional F32 BLAS, reverse | 1737.44 ±33.45 | 117.20 ±1.24 |
| Native stage 3, reverse | 809.12 ±8.47 | 116.38 ±2.63 |
| Preserved stage 2, reverse | 793.06 ±20.66 | 116.28 ±0.64 |

The observed prefill ratio is about 2.15× under this contention; decode remains
unchanged. [Raw paired records](stage3-final-paired/metadata.json) include exact
commands and binary hashes. The optional path owns 48 MiB additional scratch
for this model, included in total scratch of 123,862,784 bytes. Compressed weights
occupy 667,078,656 bytes and KV occupies 23,068,672 bytes for the pp512 run.

| Separate contended comparison | Fyodor F32 KV | llama-bench b10809 F16 KV |
|---|---:|---:|
| pp512 | 1745.26 ±46.60 | 6113.44 ±347.55 |
| tg128, empty prefix | 117.13 ±0.67 | 163.43 ±3.96 |
| tg128, 1024-token prefix | 83.81 ±0.09 | 160.18 ±1.58 |

These are **not equivalent-KV results** and cannot prove the primary objective.
The raw [short-context](stage3-llama-contended/metadata.json) and
[long-context](stage3-context-contended/metadata.json) records preserve both
engines, RAM samples and whole-device telemetry. Fyodor prompt latency was
293.55 ms, short-context decode 8.54 ms/token, and long-context decode
11.93 ms/token. With graph capture disabled, the combined diagnostic profile
reported matrices 1076.85 ms, norm 158.27 ms, RoPE 142.43 ms, vector ops
211.18 ms, attention 154.59 ms and gather 0.58 ms. This aggregates pp512 and
tg128; it is not an isolated prefill or decode breakdown.

## Validation at this checkpoint

| Configuration/check | Result |
|---|---|
| GCC C17 CUDA + Vulkan | 37/37 passed |
| GCC C17 CPU | 21/21 passed |
| GCC C23 + ONNX Runtime | 22/22 passed |
| ASan + UBSan diagnostic CPU build | 21/21 passed |
| Fresh source with CUDA physically absent, CPU | 21/21 passed |
| Fresh source with CUDA physically absent, Vulkan | 24/24 passed |
| Live HTTP compatibility/API checks | 36 passed |
| Frontend Node tests / browser checks | 6 / 73 passed |
| Vite production / Windows MSI and NSIS | built successfully |

Real-model CPU/CUDA comparisons passed at the unchanged 0.001 scaled-logit
bound: TinyLlama maximum scaled error 0.0000579, Gemma 12B 0.0004468; a synthetic
1024-token context reached 0.000000164. Prefix reset/reuse is included. Training,
finite differences, checkpoint/export, quantized kernels and fallback cases are
part of the suites above. Native process smoke verified a live Fyodor window,
one backend sidecar and no console child. **Native visual inspection remains
unverified** because the Computer tool failed to initialize its runtime twice;
browser checks do not substitute for native visual inspection.

## Product changes

The frontend retains Tauri/Vite/vanilla ES modules. Shared semantic palettes
replace the hard-coded dark style. Six palettes plus system, compact spacing,
grouped/collapsible navigation, Models and measured-JSON Benchmarks views are
implemented. Message rendering safely supports fenced code; long chats render
a bounded initial window. Remote text streaming has bounded SSE parsing,
cancellation and per-frame incremental updates. Native HTTP generation remains
non-streaming and is labeled accordingly. Training polling no longer destroys
the focused configuration form; telemetry uses actual step/loss lines. Log
updates preserve the toolbar and allow pausing automatic scrolling.

Initial browser evidence: **73 checks pass** across six themes, eight screens,
390px layouts, persistence, native generation/error recovery and remote
streaming/Stop/edit. Six Node tests cover hostile text, invalid benchmark data,
split UTF-8/SSE, truncated streams and size limits. See
[frontend validation](../../frontend/qa/stage3/results.json).

## Remaining work, ranked

1. **Native tensor-core prefill with an explicit precision contract.** Close the
   remaining matrix gap without depending on BLAS availability. Investigate
   weight/activation packing, reuse and accumulation error independently; the
   earlier TF32 failures do not justify relaxing the logits tests.
2. **F16 KV as an explicit execution resource.** Implement independent cache
   precision parity tests and matched llama-bench KV settings. Current F32 versus
   F16 comparisons are context only and cannot prove the main acceptance goal.
3. **Stable decode graphs/device request metadata and fused small ops.** The
   subsequent [stage 4](STAGE4.md) removes per-token recapture. Further fusion
   still requires separate host-submission and GPU operation profiling.
4. **CPU weight repacking and wider blocked GEMM.** Packed activations are already
   present; repeatedly decoded weight rows and F32 arithmetic remain bottlenecks.
5. **Vulkan resident graph lowering.** It remains an independent synchronous
   matvec provider; CUDA improvements must not be presented as Vulkan gains.
6. **Further product separation and native streaming.** Native asynchronous
   cancellation/streaming, benchmark launching, dataset/RAG workflows and fuller
   settings remain real work, not disabled mock screens.

ROCm/MLX hardware execution is not implemented or benchmarked. This host cannot
verify AMD/Apple kernels. Their absence is a normal build; no fake providers or
performance claims are included. Native training remains CPU/autograd and is
preserved independently of optimized inference.
