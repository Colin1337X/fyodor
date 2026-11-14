# Isolated model Q6 paired-row dispatch

Extends the validated Q4 paired-row/QKV reuse experiment with optional
`NYA_CUDA_Q6_PAIR_ROWS=1` dispatch. The native Q6_K batch-four kernel is
selected only for columns/rows 3840/2048, 8192/3840 and 15360/3840, with exact
native stride. Missing kernel, disabled flag, other types/batches/shapes retain
the existing dispatch. Vendor BLAS precedence is preserved. No new scratch,
packing format or transfer is introduced. The Q6 counter measures captures or
direct calls, not graph replays.

Build admission checks prior full-model Q4 short memcheck and long generation,
Q6 independent numerical proofs, all three sanitizers, and both improving
isolated timing sweeps. C17 and C23 O3 builds passed. Actual extracted C
selection predicates passed 3780 cases per standard, including missing kernel,
disabled selection, unsupported types/batches/shapes and incorrect strides.

The lifecycle runner requires a positive well-formed Q6 execution marker in
addition to existing Q4/QKV/graph/readback/reset/rewind and numerical checks.
Full-model validation is pending. Kernel improvements are diagnostic and do
not establish model-level throughput. Production source/archive is unchanged.
Accepted llama.cpp scores remain 85.6103% short / 106.8707% at 8K.

C17 and C23 short lifecycle checks passed both resets at prefix 257 with
four verified tokens, 44 Q6 capture/direct calls each, and maximum scaled
logit error 0.0000219546455 (unchanged bound 0.0005). Reset/rewind/readback
and graph reuse checks passed; all 836 hashes per result verified. The 8K
lifecycle check is pending. The generation runner now requires three positive,
well-formed Q6 markers as well as the existing twelve-case parity inventory,
Q4/QKV and graph-reuse checks. Its missing-long-lifecycle refusal was exercised
without launching a competing GPU workload. Model timings remain pending.
