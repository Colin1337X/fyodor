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

Offline SM120 assembly uses 55 registers, 4096 bytes of shared memory, and no
stack or spill traffic for the candidate. The current metadata candidate's
separate offline census used 66 registers. This is an allocation result only:
runtime JIT allocation, achieved occupancy and throughput remain unmeasured.
`assemble.py` checks that its PTX matches the compile proof before assembly.

`check_kernel.py` prepares 207 byte-exact cases against both the serial native
oracle and the current metadata kernel, including 1/7/9-row tails and finite
half-coefficient edge cases. It refuses to create a GPU context until the
currently running long-generation proof is complete and the process gate
passes. That pending-proof refusal was exercised.

After raw parity, `run_sanitizer.py memcheck|racecheck|synccheck` retains separate
logs and requires identical case inventories, zero errors (and zero race
warnings), matching sources and harnesses. GPU correctness, synchronization,
timing and model integration remain pending. No accepted score is changed.

`probe.py` is prepared to compare the metadata and shared-input kernels over
eight expansion/contraction shapes in forward/reverse order. Each event sample
contains 100 graph-captured operations, including the same activation packing
for both kernels, after 30 warmup operations per order. It checks output bytes
after each timed group and retains all samples and standard deviations. All
three sanitizer proofs are mandatory before its CUDA import. The missing-raw-
proof refusal was exercised. Repeated weight banks and sparse environment
snapshots make this a kernel diagnostic, not an accepted model throughput test.
