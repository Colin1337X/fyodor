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
passes. That pending-proof refusal was exercised. After long-generation
validation completed, all 207 raw cases passed byte-exact parity and guards.
The same 207 cases passed Compute Sanitizer memcheck with zero errors and an
identical case inventory. Racecheck and synccheck also passed all 207 cases
with zero errors and zero race warnings. All 57 sanitizer input/evidence hash
checks independently verified.

After raw parity, `run_sanitizer.py memcheck|racecheck|synccheck` retains separate
logs and requires identical case inventories, zero errors (and zero race
warnings), matching sources and harnesses. GPU correctness, synchronization,
checks completed as above; model integration is rejected by timing below.
No accepted score is changed.

`probe.py` is prepared to compare the metadata and shared-input kernels over
eight expansion/contraction shapes in forward/reverse order. Each event sample
contains 100 graph-captured operations, including the same activation packing
for both kernels, after 30 warmup operations per order. It checks output bytes
after each timed group and retains all samples and standard deviations. All
three sanitizer proofs are mandatory before its CUDA import. The missing-raw-
proof refusal was exercised. Repeated weight banks and sparse environment
snapshots make this a kernel diagnostic, not an accepted model throughput test.

## Rejected timing result

The shared-input kernel was slower on all eight shapes, by 18.24% to 35.73%
mean event latency. The dominant 3840-column/15360-row expansion rose from
0.121746623 to 0.154958079 ms (27.28%); the 15360-column/3840-row contraction
rose from 0.137922720 to 0.169754175 ms (23.08%). Every timed output matched.
Forward/reverse groups agree on the direction of the regression.

The lower offline register count did not compensate for this implementation's
shared-memory and synchronization overhead. The precise cost split is not
profiled here. Retain the current metadata kernel; do not integrate this
candidate or claim a model speed improvement from it.

## Upload-ordering corrected confirmation

After a separate normalization probe exposed missing upload ordering,
`probe_upload_sync.py` added a context fence after buffer initialization and
before the nonblocking graph stream is used. The fence is outside timing.
This completed eight shapes with ABBA BAAB order, 1000 warmup operations per
order, 40 observations per implementation/shape and exact timed-output checks.
Use this corrected evidence rather than the historical unfenced probe.
Positive changes mean slower than the existing metadata kernel.

| Columns | Rows | Corrected latency change |
| ---: | ---: | ---: |
| 3840 | 512 | +36.944% |
| 3840 | 2048 | +30.382% |
| 3840 | 4096 | +30.604% |
| 3840 | 8192 | +30.318% |
| 3840 | 15360 | +26.133% |
| 4096 | 3840 | +29.993% |
| 8192 | 3840 | +18.138% |
| 15360 | 3840 | +22.702% |

The production-integration decision is unchanged: no reliable improvement
for the dominant expansion workload was established. This remains a synthetic
repeated-weight diagnostic without continuous contamination acceptance; it
does not update accepted llama.cpp scores. Historical evidence is preserved.
