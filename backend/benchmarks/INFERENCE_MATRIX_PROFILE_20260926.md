# CUDA matrix shape profile — September 26, 2026

Following the rejected [two-chain decode experiment](INFERENCE_DOT_PROBE_20260926.md),
`NYA_CUDA_PROFILE=1` now breaks matrix events down by storage type, dimensions,
batch size and execution path. Device kernels and arithmetic are unchanged.
The bounded table holds 63 distinct shapes and an explicit overflow bucket.
Its optional allocation occurs only during profiling; failure retains aggregate
timings. Early initialization failures and normal teardown release the table.
The ordinary path does not allocate profiling storage.

These are original C diagnostics behind the existing detachable CUDA backend.
No dependency, public ABI, tensor representation or precision mode changes.
Vendor events include expansion and all chunks of the matrix operation. Native
entries time a kernel launch. Graphs remain disabled, and event intervals may
contain host submission gaps. This is diagnostic attribution, not throughput
acceptance or isolated hardware-counter evidence.

## Findings

Starting revision `0c5290080c3a5691930018586f34442f7271b4dc`; GCC 16.2 Release
C17, RTX 5060 Ti 16 GiB, Windows 11, driver 610.88, Ryzen 5 9600X. Same TinyLlama
Q4_K_M and F32 KV as [Stage 9](STAGE9.md), CUDA batch 512, six CPU threads,
cuBLAS F32 prefill and native decode, CUTLASS disabled. Three sequences per
workload include one warmup and two measured repetitions. The profile includes
warmup and the long workload's untimed prefix preparation. Existing desktop
clients remain open. No tests or builds overlap these diagnostic runs.

| Workload | Matrix event total | Q4_K 5632×2048 gate/up | Share of matrix time |
|---|---:|---:|---:|
| pp512 | 343.034 ms | 171.668 ms | 50.04% |
| tg128, empty prefix | 1584.291 ms | 653.960 ms | 41.28% |
| tg128, prefix 1024 | 2262.487 ms | 1004.091 ms (prefill + decode) | 44.38% |

The gate/up pair is the largest matrix target. The output vocabulary projection
is smaller: 60.535 ms in short decode. These findings motivate investigating
shared input reads and dispatch fusion for gate/up, while preserving F32 math.
They do not prove whether memory traffic, instructions, or launch overhead is
the limiting hardware resource.

[Raw profiles and commands](inference-matrix-profile-20260926/baseline/metadata.json)
and [shape summary](inference-matrix-profile-20260926/summary.json) retain all
dimensions, counts and times. `summarize.py` validates exact expected dispatch
counts: short decode has 59,520 matrix calls (384 tokens × 155 projections);
prefill has 462 vendor calls and three final-logit projections; long context
has 924 vendor calls plus 59,523 decode/output calls. Profile sums are checked
against the operation-class aggregate, which also counts request metadata
kernels as matrix work.

## Validation and remaining issue

C17 profiling checks pass resident execution, reference arithmetic, vendor
prefill, allocation failure and prefix replay. C23 profiling checks also pass.
Normal-mode CUDA tests cover attention shapes, graphs, kernels, fallback and
generation: all 25 selected tests pass across the initial run and the corrected
CUTLASS dependency rerun. Real TinyLlama forced decode at 33/32/31 tokens with
profiling enabled passes the unchanged 0.001 scaled-error bound against the
validated native CPU kernels; maximum error is 6.28670e-6.

Two setup failures are retained. The first profile run left the graph test
expecting captures despite profiling disabling graphs; explicitly setting
`NYA_CUDA_GRAPHS=0` makes its expected behavior match the profile. The normal
suite initially omitted the optional CUTLASS DLL path; its two inference checks
skipped, and the vendor chunk test instead exercised native fallback. Supplying
the configured bridge makes all three pass.

That accidental fallback run also exposed a separate numerical case needing
investigation: the native BF16 GEMM with K=8192 returned 2076.98633 where the
analytic fixture expects exactly 2077. This profiler changes no device math,
and successful vendor checks do not validate that native result. Preserve the
failure and address it before adopting the proposed gate/up optimization.

This diagnostic-only change does not claim another performance win, a fresh
desktop package, or a complete portability/sanitizer rerun. Stage 9 retains
the prior acceptance matrix. The exact commands and transcripts for this
change are under `inference-matrix-profile-20260926/`.
