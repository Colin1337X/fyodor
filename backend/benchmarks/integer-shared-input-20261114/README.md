# Shared activation tile for Q4 batch-four decode

The joint-decode profile identifies Q4 feed-forward expansion as its largest
matrix group. The current native-layout kernel loads the same float4 activation
tile independently for eight row warps in a block. This original Fyodor
experiment stages the 256-column tile once in 4 KiB shared memory, then reuses
it across those warps. Quantized weight layout, arithmetic, accumulation and
warp reduction order are unchanged. Two block barriers protect every tile;
inactive tail-row threads participate in both before returning.

This trades repeated activation load instructions for shared-memory accesses
and barriers. It may regress: caching already handles the global reads, and
barrier/register costs may outweigh reuse. No speedup is claimed.

The optional kernel is isolated in `optcpp/matvec/integer_shared_input.cu`.
NVRTC compute_120 compilation passed without a CUDA context or GPU execution.
Production dispatch and existing validated binaries remain unchanged.

`check_kernel.py` prepares 207 byte-exact cases against both the serial native
oracle and the current metadata kernel, including 1/7/9-row tails and finite
half-coefficient edge cases. It refuses to create a GPU context until the
currently running long-generation proof is complete and the process gate
passes. That pending-proof refusal was exercised.

After raw parity, `run_sanitizer.py memcheck|racecheck|synccheck` retains separate
logs and requires identical case inventories, zero errors (and zero race
warnings), matching sources and harnesses. GPU correctness, synchronization,
timing and model integration remain pending. No accepted score is changed.
