# F32 fusion and cache diagnostics — September 12, 2026

The primary same-hardware llama.cpp objective remains open. This pass tested and
rejected an F16 KV implementation, retained F32 execution, and combined residual
addition with FFN RMSNorm without changing precision or reduction order.

## Rejected F16 cache experiment

The prototype owned two-byte CUDA K/V caches, selected precision per session,
and used independent integer ties-to-even rounding for CPU reference storage.
It supported batched prefill, graph replay, reset and CPU prefix recovery. The
conversion test exhaustively covered finite binary16 encodings and adjacent
rounding boundaries. Forty-four synthetic/backend tests passed, but that was
insufficient evidence for real-model correctness:

| Real-model F16 comparison | Maximum scaled error | Existing bound | Result |
|---|---:|---:|---|
| TinyLlama Q4_K_M, prefix 65 | 0.00203030 | 0.001 | failed |
| Gemma 12B Q4_K_M, prefix 33 | 0.01732450 | 0.001 | failed |

Cache inspection localized the divergence. At layer zero, a CPU key value of
`-2.2431640625` and GPU value of `-2.24316692352` round to `-2.2421875` and
`-2.244140625` respectively. Only two of 2,304 first-layer key entries differed
in the half-cache trace, but later layers amplified those perturbations. A
four-layer diagnostic reached 0.02029 scaled logit error at prefix eight.

The CPU converter passed exhaustive arithmetic-oracle checks; CUDA used explicit
round-to-nearest-even conversion. This shows why correct isolated conversion
and small fixtures did not establish acceptable complete-model parity. No
threshold was relaxed. **The F16 session option, kernels and benchmark switch
were removed from the normal build.** Its source patch, raw failed runs and
boundary traces remain under [rejected experiment](stage5-rejected/README.md).
Its measured half-size allocation and approximately unchanged decode throughput
are not promoted performance results. Current KV remains F32; comparisons with
llama-bench's F16 KV still carry that mismatch. A separate, newly validated
upstream C API harness removes it: the b10809 C API accepts F32 KV even though
the official llama-bench argument parser excludes it.

## F32 execution change

In each dense LLaMA/Gemma layer, attention output is added to the residual stream
immediately before FFN RMSNorm. For single-token states of at most 256 elements,
`nya_add_norm` performs both in one block: it stores the rounded F32 residual sum, uses the original warp/block
sum reduction and applies the same epsilon/weight normalization. It preserves
the residual needed by the later FFN addition. The normal fast/reference math
selection is unchanged; no half conversion enters inference.

The initial all-shape prototype removed 22 launches per TinyLlama token, but
paired measurements did not show a throughput benefit. It reduced vector-grid
parallelism by doing wider residual addition inside the normalization block.
The final dispatcher retains the original path for wider states and batched
prefill. Only small decode states select fusion; weights, KV and scratch sizes
remain unchanged. The two-layer synthetic fixture removes two launches/token.

## Explicit cache inspection

`nya_llm_session_read_kv` and the provider-level `nya_compute_plan_read_kv` copy a
bounded valid cache slice into two caller-owned F32 arrays. Model layer binding
resolves shared KV; callers do not receive device pointers. Wrong layer, shape,
position or prefix bounds fail before writing. Reset makes the session prefix
unreadable. CUDA reports driver failures and accounts for explicit diagnostic
readbacks; these calls are never inserted into normal inference or benchmarks.
CPU fallback reads its own complete reference cache.

`NYA_TEST_CACHE_TRACE=1` enables layer-by-layer cache diagnostics in the inference
test executable. `NYA_TEST_LAYERS=N` selects a borrowed prefix-of-layers model view
for localization without mutating the loaded model. Neither is a production
generation setting. `nya_llm_round_f16` is retained for diagnostics only, with
an exhaustive test; it does not enable F16 inference or change autograd.

## Validation

| Configuration/check | Result |
|---|---|
| GCC C17 CUDA + Vulkan | 40/40 passed |
| GCC C17 CPU | 22/22 passed |
| GCC C23 + ONNX Runtime | 23/23 passed |
| ASan + UBSan diagnostic CPU | 22/22 passed |
| CUDA physically absent, fresh CPU source | 22/22 passed |
| CUDA physically absent, fresh Vulkan source | 25/25 passed |
| Live HTTP/API | 36 passed |
| Frontend unit / isolated browser | 6 / 73 passed |

The sanitizer build first caught an implicit float-to-bool conversion warning
in diagnostic counting; an explicit comparison fixed it. Existing warnings stay
errors. TinyLlama F32 prefixes 65/64/63 passed with maximum scaled error 5.79e-5.
The 1024-token fixture passed 3,060 checked graph replays and explicit cache
snapshots across reset/interleaved prefill, with maximum scaled error 1.27e-7.
Independent Gemma goldens retain their 3e-5 bound; training/finite-difference,
checkpoint/export, quantized kernels, fallback and speculative tests remain.
Real Gemma 12B F32 prefixes 33/32/31 also passed, with maximum scaled error
0.000446722 and maximum absolute error 0.000514984.

## Matched F32 comparison

The new `llama_capi_bench.c` is independent comparison tooling, never a Fyodor
runtime dependency. It links the official b10809 shared libraries, uses the
same deterministic token IDs, full-workload warmups, F32 K/V, requested context,
512-token batch and six CPU threads. Prefill computes only final-position
logits; decode synchronizes and returns CPU-visible logits every token. Neither
engine times model loading, tokenization, sampling or untimed prefix preparation.
Upstream flash attention is disabled for this F32 comparison. Internal arithmetic
and allocator alignment remain engine-specific.

RTX 5060 Ti 16 GiB, Ryzen 5 9600X, driver 610.88; nine repetitions after three
warmups. A/B denotes forward/reversed engine ordering. Other user GPU workloads
remained active, so these are contended measurements, not clean acceptance:

| Engine | pp512 A / B, tokens/s | tg128 A / B, tokens/s | tg128 at prefix 1024 A / B |
|---|---:|---:|---:|
| Stage 4 Fyodor | 2750.61 / 2728.57 | 193.18 / 191.64 | 172.22 / 172.96 |
| Stage 5 Fyodor | 2742.52 / 2747.42 | 190.91 / 190.38 | 172.24 / 171.54 |
| llama.cpp b10809 C API | 10847.79 / 10847.31 | 271.54 / 271.34 | 256.68 / 239.62 |

Raw standard deviations, latencies, memory accounting, calls and telemetry:
[short context](stage5-matched-short/metadata.json),
[long context](stage5-matched-long/metadata.json).
This establishes a substantial remaining gap, not a win. It also confirms that
the restricted small-state fusion did not improve TinyLlama. The earlier
all-shape prototype results are retained in `stage5-final-paired`; they were
the reason for restricting dispatch. Small synthetic decode measurements are
noisy: an initial 512-capacity run regressed, while a subsequent 128-capacity
50-repetition run measured 9,653.65 before versus 10,023.06 after (standard
deviations 245.90 / 266.13). Do not extrapolate that small-graph result to models.

## Files and next work

* `CUDA/resident.cu`, `CUDA/resident.inc`: fused F32 operation and bounded KV readback.
* Compute interfaces/providers and `llm_cpu.c`: optional snapshot dispatch and session bounds.
* `llm_quant.c`, `test_kv_precision.c`: independent diagnostic rounding and exhaustive checks.
* `test_inference.c`, backend CMake: cache parity, reset, canaries, invalid bounds and tracing.
* Documentation, rejected experiment and benchmark records; refreshed packaged sidecar.

The largest remaining throughput targets remain native precision-preserving
tensor-core matrix kernels and prefill attention. Full F16 cache adoption needs
a stronger end-to-end accuracy solution; it cannot be justified by increasing
the current test tolerance. CPU weight repacking, Vulkan graph residency and
native streaming remain later work. No ROCm/MLX execution or GPU training is
claimed by this checkpoint.
