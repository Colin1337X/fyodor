# Paired query/key RoPE trial - October 16, 2026

The active target is at least 125% of pinned llama.cpp throughput on both
short (prefix 0) and long (prefix 1024) decode. Latest accepted scores before
this trial are 92.4% / 113.5%; no trial timing is an accepted score yet.

Diagnostic profiling at 426cb9a reports matrix 1002.342 ms, normalization
117.122 ms, RoPE 74.384 ms, attention 88.183 ms and gather 1.324 ms. Profiling
disables graphs and includes host submission gaps. These totals guide the
experiment and do not predict whole-runtime gains.

The original Fyodor candidate puts query and key rotations into one launch,
retaining the separate kernels' trigonometry, ordered arithmetic and cache
writes. It applies to single-token, non-Gemma, non-reference layers owning
their KV state. Shared-KV consumers and batched execution use existing paths.
NYA_CUDA_ROPE_PAIR=0 disables it. No extra allocation is needed.

The six pair-only diagnostic cases match Q/K and full cache buffers exactly,
including guards, positions 0 through 1023, request-supplied positions and
partial blocks. Isolated graph latency improves 70-96%; this is not a
whole-runtime result and the diagnostics lack continuous contamination checks.

The shared-trigonometry alternative failed exact comparison at width 64,
32 query / 4 KV heads, position 257 with request metadata. Its failed run is
preserved in initial/failure.json; it is excluded from production. The cause
is not established. No tolerance is changed to accept that alternative.

C17 and C23 regression passes each passed 25/25. Final builds include the
profiling category and oversized-index dispatch guard. The CPU logits oracle
passes at unchanged tolerances. Fusion on/off checks 795 real-model graph
decodes across resets/prefix preparation with exact logits and KV, saving
17,490 launches.

The initial synthetic harness failed repeatability under racecheck (including
the separate baseline itself). Diagnostic byte differences localized a later
failure to all 2048 elements of the value-cache destination. Pageable cache
reset uploads were not explicitly ordered against the nonblocking kernel
stream. Adding a context synchronization after reset fixes that harness race.
All six production-kernel cases and memcheck, initcheck, racecheck, synccheck
then pass; failed runs and diagnostics are retained. Earlier diagnostic timing
and the shared-trigonometry failure require reevaluation with this corrected
harness. No production arithmetic or tolerance was changed in this repair.

The first final-build attempt lacked the bundled make directory in PATH;
the logged retry with the proper tool path succeeds in both language modes.
All twelve paired baseline activities and eight matched reference activities
pass idle and contamination gates. No competing workload required termination.
Baseline pooled gains: short 345.483 +/- 17.801 to 354.879 +/- 1.085 tok/s
(+2.72%); prefix 512 327.180 +/- 3.801 to 332.112 +/- 2.626 (+1.51%);
prefix 1024 304.838 +/- 2.055 to 306.606 +/- 3.594 (+0.58%). Both orders
are positive; the reverse long gain is only 0.01%. Noise limits precision,
and no broad larger-model improvement is established.

Matched pinned llama.cpp refresh (ten samples per engine/context):

| Prefix | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| 0 | 358.024 +/- 0.697 | 384.224 +/- 0.752 | 93.2% |
| 1024 | 308.638 +/- 0.293 | 270.011 +/- 1.252 | 114.3% |

TinyLlama Q4_K_M, F32 KV, full GPU offload, llama.cpp b10809 commit
5266f24da75dc449bd56cbed7addb9c8e4a6a73e, reference flash attention off,
six CPU threads, maximum batch 512, 128 decoded tokens, two warmups and five
repetitions per process, reversed engine order. Both return host logits every
token; model loading and prefix preparation are untimed. RTX 5060 Ti 16 GB,
driver 617.42, Ryzen 5 9600X, Windows interactive desktop. Three consecutive
idle readings must be <=30% GPU and <=15W; unknown/competing GPU jobs reject
acceptance. Resource/process samples and all timed observations are retained.

Accepted as a small launch-saving improvement with exact real-model output.
The new scores remain below 125% for both workloads; decode work continues.

The synchronized shared-trigonometry diagnostic still fails at position 257:
419 differing float words (examples differ by one bit) across rotated outputs
and cached keys. Its mismatch.json preserves every differing word. This is
separate from the reset race; the precise compiler/arithmetic cause is not yet
established. That variant remains excluded from production.
