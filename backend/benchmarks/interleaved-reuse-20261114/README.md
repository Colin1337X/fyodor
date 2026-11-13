# Interleaved activation input for batch-four weight reuse

Original Fyodor kernel in optcpp/matvec/interleaved_reuse.cu. A conversion
kernel transposes four row-major input vectors into aligned float4 elements.
The dot kernel loads these four values together and retains each output's
existing weight arithmetic, accumulation order and warp reduction order.
The conversion is included in every timed graph invocation. No weights are
requantized or expanded.

## Correctness and memory

All 17 raw cases passed byte-exact comparison against native weight reuse,
including all 11 profiled Gemma shapes and six small cases. Conversion output
is checked byte-for-byte against an independent CPU transpose, with guards
on both sides; matvec output guards and finite checks also pass. Block scales
vary, including negative and non-binary F16 values. Small cases pass the
existing independent float64 bound. Compute Sanitizer reports zero errors
across the same cases in memcheck.log.

## Microbenchmark results

These are synthetic repeated-weight graph diagnostics on the RTX 5060 Ti,
not end-to-end decode scores. Each row contains ten measured samples, five
in each variant order, with three warmup graph launches per variant/order.
Graphs contain ten conversions plus matvecs for the candidate, or ten native
matvecs for control; each measurement replays its graph ten times. Raw means,
sample standard deviations, environments and source hashes are retained.

| Type | Columns | Rows | Native ms | Convert + interleaved ms | Speed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 3840 | 512 | 0.014832 | 0.010310 | 143.860 |
| 12 | 3840 | 2048 | 0.034965 | 0.024316 | 143.798 |
| 12 | 3840 | 4096 | 0.056456 | 0.038318 | 147.335 |
| 12 | 3840 | 8192 | 0.107283 | 0.070992 | 151.120 |
| 12 | 3840 | 15360 | 0.201020 | 0.131379 | 153.007 |
| 12 | 4096 | 3840 | 0.056522 | 0.038053 | 148.535 |
| 12 | 8192 | 3840 | 0.118702 | 0.075731 | 156.741 |
| 12 | 15360 | 3840 | 0.225607 | 0.155057 | 145.500 |
| 14 | 3840 | 2048 | 0.035813 | 0.029737 | 120.433 |
| 14 | 3840 | 262144 | 4.002343 | 2.649346 | 151.069 |
| 14 | 15360 | 3840 | 0.259958 | 0.186868 | 139.113 |

## Opt-in resident integration

Configure NYA_ENABLE_INTERLEAVED_REUSE=ON (requires present optcpp source),
then set NYA_CUDA_INTERLEAVED_REUSE=1 before session creation. Both default
off. Only resident batch-four Q4_K/Q6_K projections use this path. Vendor
BLAS preference, other batch sizes/types, reference mode, missing kernels
and physically absent optcpp retain existing dispatch. The generic assisted
matvec interface is unchanged.

A plan reserves 4*maximum_projection_input_width+3 floats when its admitted
batch can reach four. The three padding floats permit 16-byte alignment. This
storage is counted before weight admission, shrinks to zero when the planner
reduces maximum batch below four, and belongs to the existing plan allocation.
Conversion and projection run on the same stream. Failure uses the existing
resident failure/replay lifecycle. Dispatch traces count completed projection
submissions; matrix event profiling includes both conversion and dot kernels.

C17 and C23 each pass 17 verification/fixture tests. The four real Gemma
checks (prefix 257, verification count 2/4, linear/ring KV) pass full-logit
comparison with maximum scaled error 1.69166745e-5, below the unchanged 0.0005
bound. Both batch-four cases report 656 interleaved projection submissions.
Reset, rewind/replay, output guards and rejected verification are exercised by
the harness. Physical removal with the new build option explicitly ON passes
82 tests (four expected absent-hardware skips) plus the CPU smoke workflow.
The integrated Gemma ring-KV memory check also passes with zero errors at
prefix 7, including full-logit comparisons, reset/replay and confirmed
interleaved projection submissions. Raw-kernel memory checking is complete. End-to-end paired generation timings are recorded below. No
llama.cpp score is promoted: accepted ratios remain 85.61025294% short and
106.8707255% at 8K, below the requested 125% on both.

## Paired generation result

All 48 calls pass exact output parity. Same revision a5dbe9c, same executable,
matched settings except NYA_CUDA_INTERLEAVED_REUSE; three rounds per mode and
context, first excluded, mode order reversed in round one. Only two warmed
observations per cell; feature-off process precedes feature-on. Both historical
desktop gates reject acceptance. Complete resource samples are preserved.

Warmed decode-loop means +/- sample standard deviations (milliseconds):

| Context / mode | Disabled | Enabled | Enabled speed % |
| --- | ---: | ---: | ---: |
| short / ordinary | 756.339 +/- 0.305 | 758.281 +/- 0.315 | 99.7440 |
| short / mtp-serial2 | 827.464 +/- 0.851 | 831.156 +/- 0.516 | 99.5558 |
| short / mtp-batch2 | 788.193 +/- 1.300 | 789.384 +/- 2.499 | 99.8491 |
| short / mtp-batch4 | 770.812 +/- 1.180 | 672.805 +/- 0.122 | 114.5670 |
| long / ordinary | 934.924 +/- 1.597 | 935.908 +/- 1.063 | 99.8948 |
| long / mtp-serial2 | 1043.359 +/- 0.119 | 1042.886 +/- 1.409 | 100.0454 |
| long / mtp-batch2 | 1110.640 +/- 0.843 | 1111.854 +/- 1.987 | 99.8909 |
| long / mtp-batch4 | 1042.134 +/- 1.738 | 947.073 +/- 0.178 | 110.0373 |

The feature saves 98.007 ms short and 95.061 ms at 8K per batch-four generation.
Other modes vary by at most about 0.45%; they do not execute the new path.
Batch-four MTP is 112.7044% of same-run ordinary Fyodor speed at short context
and 98.8211% at 8K. Thus it now helps this short generation workload but does
not yet improve its long-context counterpart.

The contract is 32 output tokens including the initial prefill-logit sample.
Decode-loop wall time includes assistant, sampling and transfers; prompt
ingestion, setup and teardown are excluded. Diagnostic B512 ordinary prefill
is patched only in the harness. This differs from the accepted llama.cpp
fixed-forward contract. No extrapolated llama.cpp percentage is promoted.
Accepted Gemma ratios remain 85.61025294% short and 106.8707255% at 8K.
