# Batch-two interleaved activation experiment

Original Fyodor prototype in optcpp/matvec/interleaved_two.cu. It converts two
activation vectors to float2 input while retaining native Q4_K/Q6_K weight
arithmetic and each output accumulation/reduction order. Q6_K is integrated
behind default-off build `NYA_ENABLE_INTERLEAVED_TWO` and runtime
`NYA_CUDA_INTERLEAVED_TWO` switches. Q4_K remains a raw experiment only.

All 17 raw correctness cases pass exact comparison, finite/output guards and
independent CPU transpose/guards. All 11 actual Gemma projection shapes and
six small cases are included, with varied signed/non-binary F16 block scales.
Small cases also pass the unchanged independent float64 bound. The dedicated
batch-two Compute Sanitizer run passes all 17 cases with zero memory errors.
Eight real Gemma cases pass at prefix 257: batch two/four, linear/ring KV,
and batch-four interleaving disabled/enabled. Maximum scaled logit error is
1.69166745e-5 against the unchanged 0.0005 bound. Batch-two cases each trace
88 new Q6 dispatches. Reset, rewind, replay and rejected verification checks
also pass. `validation.json` retains source/binary hashes and case outcomes.

C17 and C23 each pass 37 targeted checks. The first C17 run exposed the
test harness's existing requirement for a batch limit greater than one; its
log is retained. An explicit serial test mode now checks the batch-one
fallback, without weakening other tests. Combined-option tests check that
scratch is shared instead of allocated twice. Physical-removal validation
passes 82 tests (four expected ROCm/MLX skips) and CPU training/export/inference
with the new option requested but optional source physically absent.

Graph microbenchmarks include the conversion in every candidate invocation.
Five samples in each variant order, three warmup graphs, 100 projections per
sample, synthetic reused weights. These are diagnostic timings, not accepted
decode benchmarks or predictions of whole-generation speed.

| Type | Columns | Rows | Native ms | Convert + interleaved ms | Speed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 3840 | 512 | 0.006820 | 0.006999 | 97.434 |
| 12 | 3840 | 2048 | 0.019741 | 0.020023 | 98.589 |
| 12 | 3840 | 4096 | 0.035067 | 0.035183 | 99.671 |
| 12 | 3840 | 8192 | 0.066017 | 0.065790 | 100.344 |
| 12 | 3840 | 15360 | 0.124000 | 0.122595 | 101.146 |
| 12 | 4096 | 3840 | 0.035098 | 0.035168 | 99.802 |
| 12 | 8192 | 3840 | 0.068441 | 0.068297 | 100.210 |
| 12 | 15360 | 3840 | 0.138173 | 0.138644 | 99.661 |
| 14 | 3840 | 2048 | 0.024108 | 0.019704 | 122.349 |
| 14 | 3840 | 262144 | 2.744699 | 2.211233 | 124.125 |
| 14 | 15360 | 3840 | 0.179095 | 0.139706 | 128.194 |

Q4 does not justify integration: gains are small and some shapes regress. Q6
improves 22.349-28.194% in this sweep and merits further validation as a
format-specific candidate. No blanket batch-two dispatch change is warranted.
The correctness workload was also observed by a failed PowerShell monitoring
experiment; this was not a timing run. Monitor limitations are documented in
gpu-monitor-20261114. All measured results, including regressions, are retained.

## Memory validation

On source revision 3eb163f, CUDA Compute Sanitizer 13.1.118 ran
`check_memcheck.py` using `--tool memcheck --error-exitcode 99
--target-processes all`. The process exited zero and reported zero errors;
`memcheck.log` and `memcheck-oracle.json` retain the output and source hash.
The harness repeats the complete correctness sweep, writing a separate
oracle artifact to preserve the earlier result. This is correctness evidence,
not a performance run. No numerical tolerance was changed.

Immediately before validation, five one-second dynamic PDH samples matched
the RTX 5060 Ti adapter LUID and showed no positive engine activity. The
device also reported 0% utilization, 32 C and 3 W. No active competitor was
identified, so no application was terminated. This brief idle check does not
establish clean coverage for a future timing run or change accepted scores.

Integrated Gemma ring-KV Compute Sanitizer validation also passes with zero
errors, both interleaving options enabled, batched vocabulary projection on,
prefix 7 and two verification tokens across both reset cycles. The maximum
scaled error is 1.66424333e-5. See integrated-memcheck.json and its log.
Whole-generation paired diagnostic measurements are recorded below; no accepted score changes.

## Paired full-generation diagnostic

Same binary and revision 71e5876; only NYA_CUDA_INTERLEAVED_TWO differs.
Batch-four interleaving and prior assistant/verification optimizations remain
enabled in both arms. All 48 generation calls pass exact greedy-output parity.
Three rounds per context/mode; first excluded, middle round reversed; two
warmed observations per cell. Values are mean +/- sample standard deviation
of QPC decode-loop milliseconds for 32 outputs (including the first token
sampled from prefill logits). Setup and prefill are excluded from this clock.

| Context | Mode | Off ms | On ms | Speed % of off |
| --- | --- | ---: | ---: | ---: |
| short | ordinary | 756.150 +/- 0.130 | 757.654 +/- 0.005 | 99.802 |
| short | mtp-serial2 | 827.254 +/- 1.186 | 828.840 +/- 0.805 | 99.809 |
| short | mtp-batch2 | 787.406 +/- 0.945 | 778.192 +/- 2.222 | 101.184 |
| short | mtp-batch4 | 672.754 +/- 0.042 | 673.607 +/- 1.116 | 99.873 |
| long | ordinary | 935.582 +/- 0.741 | 934.866 +/- 0.764 | 100.077 |
| long | mtp-serial2 | 1046.008 +/- 2.105 | 1045.950 +/- 3.258 | 100.006 |
| long | mtp-batch2 | 1113.587 +/- 0.210 | 1097.418 +/- 1.112 | 101.473 |
| long | mtp-batch4 | 949.405 +/- 2.469 | 948.074 +/- 0.674 | 100.140 |

Batch-two mode saves 9.213 ms short and 16.169 ms long in this sweep.
Batch-four remains the faster speculative mode on this workload. The option
stays off by default: this is a modest measured diagnostic improvement, not a
125% llama.cpp result. Both conservative environment gates failed. These
results also use a diagnostic matched-B512 prefill path and a generation
contract different from the accepted fixed-forward benchmark. The accepted
85.6103% short and 106.8707% long llama.cpp ratios remain unchanged.
