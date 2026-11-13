# Batched FDPP numerical alignment experiment

Original Fyodor extension of optcpp/attention/flash_decoding_pp_warp.cu.
The optional flash_decoding_pp_batch.cu kernel assigns each query its own
partial rows, query vector and output. The per-query arithmetic, partition
bounds, retry decision and stable fallback remain the serial algorithm.
The combine kernel uses its query grid coordinate in both the normal output
and the existing stable attention fallback. No external source was copied.

This is not runtime-integrated, enabled by CMake, or a performance claim.
It is the attention half of the reduction-order investigation recorded in
../wide-batch-split-20261114. Production decode settings remain unchanged.

Run `python backend/benchmarks/fdpp-batch-20261114/check.py` from the repository
root with PYTHONDONTWRITEBYTECODE=1. Existing result JSON is not overwritten.
The 50 cases cover widths 64/128/256, batches 2/4/32/256, offsets 0/17/224,
full and sliding windows, GQA, and an actual 16-query-head/8-KV-head shape.
They include empty partitions, forced retries and a late retry after earlier
keys have contributed. Every case compares all partials and final outputs
byte-for-byte against separate serial warp-kernel launches. Output, partial
and score buffers have checked prefix/suffix guards. Sampled first/last queries
and heads are also checked against an independent Python float64 stable
softmax oracle at the unchanged 0.0005 bound.

All 50 cases pass. Maximum sampled oracle scaled error is
1.37795345115e-6. This does not imply whole-model parity or speed improvement.
CUDA Compute Sanitizer 13.1.118 memcheck also passes all 50 cases with zero
reported errors and exit code zero. The source hash and numerical results
match the unsanitized run. Invocation: compute-sanitizer --tool memcheck
--error-exitcode 99 --target-processes all <python> check.py oracle-memcheck.json.
Integration and real-model validation remain outstanding.
