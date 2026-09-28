# Partitioned CUDA decode attention — September 25–26, 2026

This pass starts from `aa298f6` on `codex/native-training-execution`. The broad
performance/training goal remains open. Fyodor still owns model loading,
execution, KV, attention, generation, and training; no external runtime was
introduced.

## Profile and source study

The [fresh diagnostic profile](inference-20260925/baseline/metadata.json)
confirmed matrix work dominates prefill. The long-context diagnostic attributed
1,495 ms to attention versus 3,161 ms to matrices. These totals include warmup
and untimed 1,024-token prefix preparation; profiling disables graphs and can
include host submission gaps. They are not isolated decode-kernel timings.

The existing decode kernel assigns eight warps in one block to each head.
TinyLlama has 32 query heads, leaving too few independent blocks while each
block traverses a long context. Read-only study of llama.cpp b10809
(`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`) followed
[`flash_attn_combine_results` and partition dispatch](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/fattn-common.cuh).
The relevant technique is to compute independent partial softmax results,
then rescale each numerator and denominator to a common maximum.

This is original Fyodor code applying that technique to its existing F32
kernel. No upstream source was copied. The upstream MIT license was read;
source SHA-256 is `3e46bc5ca40bbcf6a28d8f8055da5ceac70ad0bcae017959957448757a3c4053`,
license SHA-256 `bcd8ec749126d45cb06737d0690295d73df4b6e7e194205bcf91190368f27285`.
It does not import upstream tensors, scheduling, FP16 KV, or Flash Attention.

## Implementation and limits

For one-token execution at position 256 or later, eligible 64/128-wide heads
divide their populated context into four nonempty contiguous ranges. Each
block retains the existing F32 QK dot product and computes an unnormalized
value sum, local maximum, and denominator. A second F32 kernel combines the
four results. Sliding windows of at most 256 tokens, unsupported widths,
reference mode, and batched prefill retain their existing paths.

No precision mode or numerical tolerance changed. Partitioning changes the
order of value/softmax reductions, so bitwise equality is not claimed. KV and
activations remain F32, compressed weights remain resident, and there are no
extra host transfers, per-layer allocations, or host synchronizations.

One persistent buffer holds the largest eligible layer's
`4 * head_count * (head_width + 2)` floats: **33 KiB** for TinyLlama. This is
included in scratch accounting and admission. Each eligible layer adds one
combine-kernel launch to long decode. A session retains at most two executable
graphs, selected by the short/long threshold. Reset and prefill preserve both
graphs; request metadata supplies current token and position. Both are freed
on cleanup. Driver-owned graph memory is outside the explicit tensor totals.

`NYA_CUDA_SPLIT_ATTENTION=0` selects the original unpartitioned algorithm at
every length. `NYA_CUDA_GRAPHS=0` still executes the new kernels without graphs.
Deleting optional CUTLASS or all CUDA remains supported. Host code and public
interfaces remain C; device code stays in the detachable CUDA directory.

## Measurement discipline

Windows 11 26200, Ryzen 5 9600X with six exposed cores, RTX 5060 Ti 16 GiB,
driver 610.88; GCC 16.2 Release C17, `-O3`. The model is TinyLlama Q4_K_M,
SHA-256 `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
The preserved pre-change executable is compared with the final executable,
using native decode and optional cuBLAS F32 prefill, CUTLASS disabled.

Both orders run serially: before/after/upstream, then upstream/after/before
where upstream is included. Each process has three full warmups and nine
measured repetitions. Pool both orders (18 observations per engine/shape),
including every noisy sample. KV is F32, batch/ubatch 512, six CPU threads,
identical deterministic token IDs, and one host-logit read per decode token.
Prefix preparation, loading, tokenization, and sampling are untimed. Upstream
uses the pinned C API harness with Flash Attention off, not `llama-bench`.
Intermediate arithmetic and allocator alignment remain engine-specific.

The first acceptance attempt was stopped by its idle gate after KoboldCpp
appeared and whole-GPU memory rose above 12 GiB. The user authorized stopping
competing GPU work; that process was stopped. Those results are **excluded**,
not selectively pooled. A first WDDM process-guard implementation also rejected
the desktop before running any samples. A subsequent attempt caught a newly
appearing Thorium GPU process and stopped; Thorium was then closed under the
same authorization. All attempts and the initial exploratory
probe are retained with [explicit exclusion notes](inference-20260925/PROBE_NOTES.md).

The final run requires two consecutive 7–12% whole-GPU idle readings before
and after each process. GPU Engine counters identified DWM and ChatGPT as the
active desktop clients. Other existing graphics clients remain open, so this
is an interactive-desktop comparison. The sampler rejects another pure CUDA
compute process or a GPU identity absent from its initial desktop snapshot.
No builds, tests, packaging, or other agent benchmarks run concurrently.

Results and resource observations are recorded in
[the full comparison](inference-20260925/accepted-clean3/metadata.json).

Mean tokens/s ± pooled sample standard deviation, 18 observations per cell:

| Untimed prefix | Timed workload | Before | After | Change | llama C API |
|---:|---|---:|---:|---:|---:|
| 0 | pp512 | 3576.61 ± 42.40 | 3530.96 ± 89.39 | -1.28% | 11344.65 ± 606.11 |
| 0 | tg128 | 206.47 ± 1.86 | 205.97 ± 2.30 | -0.24% | 292.09 ± 6.11 |
| 256 | tg128 | 184.54 ± 0.77 | 201.83 ± 1.13 | +9.37% | — |
| 512 | tg128 | 166.36 ± 0.77 | 195.59 ± 1.13 | +17.57% | — |
| 1024 | tg128 | 140.96 ± 0.82 | 183.96 ± 1.16 | +30.50% | 216.74 ± 1.48 |
| 1536 | tg128 | 121.44 ± 0.50 | 172.51 ± 0.88 | +42.05% | — |

The retained benefit is long-context decode. Short decode and prefill did not
improve; their small observed declines and the noisier prefill measurements are
retained. These data do not prove zero overhead. Results cover this model and
GPU, not all supported models/devices. llama.cpp remains about 3.21× faster at
prefill, 1.42× at short decode, and 1.18× at the 1,024-token prefix.

At prefix 1,024, explicit weight bytes remain 667,078,656 and KV bytes remain
51,904,512. Scratch increases from 91,342,600 to 91,376,392 bytes (+33 KiB).
Across nine measured 128-token sequences, both versions have 1,152 downloads,
1,152 synchronizations, 1,152 graph replays, and zero buffer uploads/captures.
Kernel dispatches increase from 384,768 to 410,112 (22 combines per token).
Process working-set/private-commit and whole-device memory/clock/power/thermal
samples are retained separately; they include startup and are not model-only
allocation totals. [Resource summary](inference-20260925/resource-summary.json).

The [post-change diagnostic profile](inference-20260925/reprofile/metadata.json)
records attention 1,496.410 → 864.601 ms for the long workload, while matrix
work remains 3,118.106 → 3,000.478 ms. These event totals include warmup and
prefix preparation and cannot be substituted for the normal graph throughput
above. Prefill attention stays about 107–109 ms and matrix work about 451–456 ms;
matrix kernels remain the main next inference target.

## Correctness and survivability

The [15-stage validation matrix](inference-20260925/matrix/validation.json)
passed on the final implementation:

| Configuration | Passed | Unavailable hardware skips |
|---|---:|---:|
| CPU C17 | 47 | 4 |
| CUDA/Vulkan C17 | 74 | 4 |
| CUDA/Vulkan C23 | 74 | 4 |
| ONNX enabled | 48 | 4 |
| Clang ASan/UBSan | 47 | 4 |
| optcpp physically absent | 47 | 4 |
| CUDA physically absent | 47 | 4 |
| CUTLASS physically absent | 69 | 4 |

Frontend unit tests and production build also pass. Four skips per native
configuration are real ROCm/MLX hardware checks; mocked-provider tests pass.
No AMD/Apple hardware, POSIX, or MSVC-native-backend result is claimed.

Expanded CUDA attention tests cover widths 32/64/128/130/256, odd tails,
multi-chunk prefill, 1,025-token capacity, GQA, shared KV, and windows 1/17/257/513.
They compare complete logits and KV snapshots with the scalar CPU reference,
cross the 256-token threshold repeatedly through reset and prefill interleaving,
and require at most two reusable captures. The same shape suite runs with
partitioning disabled and with graphs disabled. Existing fallback, allocation
failure, inference/export, and training/recovery checks remain in the matrix.

[Compute Sanitizer](inference-20260925/cuda-sanitizers.json) memcheck, racecheck,
initcheck, and synccheck each pass with zero errors (racecheck: zero warnings),
including both graph variants and 795 checked decode calls across reset/prefill.

Real TinyLlama forced-decode checks retain the existing scaled-error bound
`abs(candidate-reference)/(1+abs(reference)) <= 0.001`:

* [Scalar double-reduction reference at 273/272/271 tokens](inference-20260925/real-273.log):
  maximum scaled error **4.04925e-5**.
* [Validated native CPU kernels at 1,153/1,152/1,151 tokens](inference-20260925/real-1153-cpu-kernels.log):
  maximum scaled error **1.29978e-5**. This separate comparison uses the new
  opt-in `NYA_TEST_CPU_OPTIMIZED` test control; default oracle tests remain scalar.

The native CPU comparison makes longer real-model checks practical; it does
not replace the scalar reference or change acceptance bounds.

An additional [late-failure check](inference-20260925/late-failure.json) forces
failure at token 269 after executing both short and long decode. Complete
273/272/271-token results match full scalar CPU replay **exactly**, with graphs
enabled and disabled.

## Desktop package

The NSIS installer was rebuilt and extracted without installing it. Every
bundled backend resource matches its prepared source by SHA-256; the desktop
binary differs only by Tauri's expected bundle-type marker. Its backend is
identical to the validated C17 binary. The extracted trainer passes the native
CLI workflow suite. The extracted backend passes authenticated loopback HTTP
load, CUDA selection, repeated long-prompt greedy generation, runtime inspection,
and unload; both generations return identical text. No interactive UI check is
claimed. [Package evidence](inference-20260925/desktop-package.json).

`frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe`:
406,256,270 bytes; SHA-256
`7184320764b588ce3c3d5a28dedafedf8904e2cbc4a7df6c58d67ba2b677469a`.
The prior installer and pre-change benchmark/backend binaries remain under
ignored `.tools`. Prepared backend/DLL artifacts follow the existing ignore
policy; the existing tracked trainer artifact is refreshed by packaging.

[The final evidence audit](inference-20260925/audit.json) verifies source hashes,
all 15 validation stages, four GPU sanitizers, 24 accepted benchmark processes,
real-model error bounds, late-failure replay, and installer/backend identity.
