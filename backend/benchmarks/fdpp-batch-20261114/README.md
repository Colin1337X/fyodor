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

## Isolated model pipeline

aligned_prefix.py and aligned_transform.py build a separate native-C library
copy for the fixed Gemma model and preserve the production archive. The
experiment retains batch-512 admission, uses the previously tested four-query
weight-reuse projection kernels, and allocates per-query attention partials.
Prefix chunks stop at token 256 so short FDPP and long split attention use
the same arithmetic as optimized serial decode. The new batched FDPP handles
256-wide heads before that boundary; the existing batched split kernel handles
256/512-wide heads afterward. The original single-token dispatch is retained.
This fixed-model experimental patch is not a general runtime feature and does
not yet provide the required optional-feature admission/fallback checks.

The 2048-token run passes with zero scaled and absolute logit error and all
144 sampled K/V vectors byte-identical. It asserts admitted batch 512 and
records 40 batched FDPP calls plus 192 batched split-attention calls. Unlike
the earlier ordered-prefix control, serial decode keeps wide split enabled
and FDPP set to short. C17/C23 strict compilation passes. Raw patches,
settings, build commands and hashes are retained in aligned-prefix-ctx2048.*.
The matching 8192-token run also passes with zero logit error and all 144 KV
samples byte-identical. It retains admitted batch 512, records 40 batched FDPP
calls and 768 batched split-attention calls, and leaves the production archive
unchanged. This establishes prefix parity in the isolated pipeline at both
tested lengths while retaining optimized serial attention.

verify_aligned.py retains the existing test_verify.c assertions and 0.0005
bound, specializing only the verification token count. The prefix-257,
linear-KV, two-token case passes both resets with zero scaled error, including
rewind, invalid-input rejection and replacement-token replay. The original
prefix-8192, linear-KV, two-token failure case is now running against this
candidate. Other lifecycle cases and integrated memory checking remain open.

Before production integration, admission must cover optional dependencies,
model/precision eligibility, checked partial-buffer sizing, and fallback.
The experimental patch currently splits only prefill at token 256; verification
batches crossing that boundary need equivalent handling. It also assumes a
long enough plan to reserve batched split partials; small-capacity plans must
be explicitly gated or budgeted. Existing interleaved verification projections
and selective verification-attention dispatch should be preserved when adding
the prefix path, rather than inheriting the experiment's blanket dispatch.
No production integration or performance acceptance is claimed here.
