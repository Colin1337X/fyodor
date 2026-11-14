# Partially unrolled Q4 metadata experiment

The joint decode profile identified batch-four Q4 feed-forward expansion as
the largest matrix group. The current metadata kernel fully unrolls eight
groups and uses 66 registers in offline SM120 assembly. A shared-input tile
regressed every tested shape despite reducing registers; it remains rejected.

This original Fyodor variant changes only the group loop unroll factor from
eight to two, keeping the native 144-byte Q4_K layout, F32 activation input,
floating-point expressions, accumulation sequence and warp reduction order.
The hypothesis is that shorter live ranges and reduced instruction footprint
may offset loop overhead. Register counts alone cannot establish a speedup.

The optional source is isolated in `optcpp/matvec/integer_metadata_unroll2.cu`.
There is no production dispatch. Compilation and offline assembly run without
a GPU context while unrelated correctness checks continue. Exact raw parity,
sanitizers, multiple matrix shapes, real-model validation and controlled
performance comparisons are required before considering integration.

Offline NVRTC compilation and SM120 assembly succeeded: 48 registers, zero
stack/spill/shared bytes, compared with the current variant's 66 registers.
This is compiler allocation evidence only; correctness and speed remain
unproven. The raw harness retains 207 cases (11 matrix shapes and 196 finite
half-coefficient pairs), exact comparison against the serial and current
metadata kernels, guard checks and the original small-shape float64 bound.
GPU execution is gated behind completion of the boundary sanitizer matrix.

The raw oracle snapshots all compiled sources, its harness and all eight
boundary terminal records before GPU creation, verifies they remain unchanged
after execution, and carries those hashes into sanitizer and timing gates.
The probe labels the candidate `unroll2`. The pending-boundary gate was tested
and stopped before creating a GPU context; no raw or timing result exists yet.

Before creating its GPU context, the raw harness also calls the independent
boundary matrix auditor, checking all 336 ordered cases and the underlying
evidence hashes. Those verified hashes join the raw run's before/after input
snapshot. An incomplete C23 matrix currently prevents GPU execution.

## Rejected after correctness and timing

The 207-case raw sweep passed byte-exact comparisons against both existing
kernels, output guards and the unchanged small-shape float64 bound. All 242
raw input hashes verified. Memcheck, racecheck and synccheck each repeated
all 207 cases with zero reported errors; their 57 input/evidence hash
checks independently verified.

The synthetic metadata/unroll2/unroll2/metadata probe includes activation
packing, ten observations per kernel and shape, and byte-exact timed output
checks. The candidate regressed all eight shapes:

| Columns | Rows | Metadata ms | Unroll2 ms | Latency increase |
| ---: | ---: | ---: | ---: | ---: |
| 3840 | 512 | 0.006963 | 0.012752 | 83.15% |
| 3840 | 2048 | 0.020927 | 0.031181 | 49.00% |
| 3840 | 4096 | 0.035233 | 0.054559 | 54.85% |
| 3840 | 8192 | 0.066126 | 0.100013 | 51.24% |
| 3840 | 15360 | 0.127804 | 0.205522 | 60.81% |
| 4096 | 3840 | 0.034542 | 0.055030 | 59.31% |
| 8192 | 3840 | 0.071044 | 0.106747 | 50.25% |
| 15360 | 3840 | 0.144146 | 0.247270 | 71.54% |

These are repeated weight-bank diagnostics, not clean end-to-end decode
scores. The candidate is rejected for model integration. Its lower register
allocation did not compensate for partial unrolling; loop/addressing overhead
and reduced compiler scheduling freedom are hypotheses, not measured causal
attributions. No real-model or throughput improvement is claimed, and the
existing metadata kernel and production dispatch remain unchanged.

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
| 3840 | 512 | +69.519% |
| 3840 | 2048 | +56.394% |
| 3840 | 4096 | +56.838% |
| 3840 | 8192 | +58.554% |
| 3840 | 15360 | +50.772% |
| 4096 | 3840 | +56.863% |
| 8192 | 3840 | +47.505% |
| 15360 | 3840 | +72.733% |

The production-integration decision is unchanged: no reliable improvement
for the dominant expansion workload was established. This remains a synthetic
repeated-weight diagnostic without continuous contamination acceptance; it
does not update accepted llama.cpp scores. Historical evidence is preserved.
