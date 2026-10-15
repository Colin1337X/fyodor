# Prefill investigation closed on October 12

The user redirected optimization exclusively to decode until throughput exceeds
llama.cpp. No prefill runtime change is retained from this investigation.

Diagnostic profiles confirm current cuBLAS beats the existing CUTLASS bridge.
At prompt 512, aggregate matrix event times across one warmup plus two measured
sequences were 275.617/292.475 ms for cuBLAS, 441.616/444.140 ms for CUTLASS,
and 827.588/798.094 ms for native kernels. Reversed order, explicit dispatch
counts and hashes are recorded. These are diagnostic desktop event intervals,
including submission gaps; they do not change the accepted scorecard.

The prior rejected six-product BF16 adapter used instruction shape 16x8x16.
The pinned upstream F32-to-BF16 specialization uses 16x8x8. Testing that shape
removed the large error, but accumulating corrections into the running total
still failed the existing Q4_K oracle. A separate correction subtotal passed
1024-column cases but failed 8192-column cases. Computing both leading and
correction subtotals before adding the previous accumulator passed both the
chunked 1024-column and extended 8192-column suites without tolerance changes.
The initial plain oracle exit 0 is NOT BF16 validation: its 65-row shapes are
ineligible for CUTLASS. The aligned NYA_TEST_CUTLASS=1 checks are authoritative.

The corrected candidate remains rejected: at prompt 512 its matrix totals were
897.800/900.171 ms versus cuBLAS 274.989/292.089 ms, and at prompt 128 it took
278.914/293.578 ms versus cuBLAS 119.424/120.399 ms. No full-model numerical or
performance acceptance is claimed. The baseline CUTLASS bridge passed the same
extended long oracle. Production files were restored, with the exact candidate
patch and three adapter variants retained here for reproducibility.

Upstream source inspected read-only: NVIDIA CUTLASS v4.5.1,
2e602843e75100d0e03934efb386b3e1e35d7907 (BSD-3-Clause),
include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h and mma_tensor_op.h.
The adapter is a revision of Fyodor's previous experiment; no upstream source
was copied. Existing third-party notices remain in CUDA/cutlass.

Reapply rejected-bf16-k8.patch and copy bf16-k8-subtotal.h.out to
CUDA/cutlass/bf16_six_product.h to reproduce the final experiment. The archived
build command uses local toolkit/header paths. The first system CMake attempt
failed because its installed module directory is missing; the bundled CMake
built successfully. The final experimental DLL SHA256 is
59c8ec4835155f5e56c796e067a35a09b861fafbc60d2e80024b220a648d4f23.

Current accepted scores remain 30.6% prefill, 69.6% short decode, 89.8% long
decode versus pinned llama.cpp, and 71.0% for small eager PyTorch training.
Further work is decode only until the user's greater-than-100% target is met.
