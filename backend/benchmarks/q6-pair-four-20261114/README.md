# Q6 paired rows with four interleaved inputs

Original isolated candidate in `optcpp/matvec/q6_pair_rows_four.cu`. Each warp
computes two adjacent rows and shares a float4 input load across them. Native
Q6_K blocks, signed-zero scale-product handling, per-output accumulation order
and warp reduction order follow the existing batch-four Q6 integer kernel.
An odd final row uses the preceding valid row as an unobserved dummy input.

This differs from the rejected single-input experiment in
`gemma4-q6-rows-20261113`; that rejection remains valid and motivates caution.
The decode profile identifies batch-four Q6 down projections as material cost.

NVRTC compilation for SM120 passed, without creating a GPU context. Compilation
does not establish numerical correctness or speed. Required next checks are
compiled resource inspection, exact baseline equivalence plus independent
numerical bounds, tails/guards, sanitizers, then forward/reverse timing. No
production dispatch or accepted performance score changed. The source remains
optional under root optcpp and is not required by portable C builds.

SM120 offline assembly reports 80 registers, zero stack/spill bytes and zero
barriers for the candidate. The first numerical harness attempt reached all
comparisons but failed provenance serialization because an input array shadowed
the hash dictionary. Its source and partial output are retained as failed
evidence; it is not counted as a pass. The corrected harness rerun passed
367 cases with byte-exact baseline agreement and output/interleave guards.
This includes odd-row tails, partial block groups, several block counts and
model projection shapes. 359 single-block cases additionally passed an
independent Python float64 bound; maximum bound fraction was 0.01663237.
Large shapes currently have byte-equivalence checks only, not an independent
double oracle. All 42 compile/assembly/oracle hash checks verified. Sanitizers,
large-shape independent bounds and timing remain pending.

The independent host-double oracle now covers every output of all 367 GPU
cases, including large model projection shapes. Before GPU use, C17 and C23
O3 strict-warning builds each passed 352 finite fixtures against Python fsum,
plus nonfinite and invalid-argument rejection checks. The full GPU rerun passed
the unchanged absolute bound `2e-6 * sum(abs(double products)) + 1e-6`, with
maximum bound fraction 0.01663237, byte-exact baseline outputs, guards and
unchanged weights/input buffers. All 34 host-control/full-oracle hash checks
verified. Memory/race/synchronization sanitizers and paired timing are pending.

The full 367-case independent-bound and byte-parity sweep passed under
Compute Sanitizer memcheck, racecheck and synccheck. Each exited zero with
a unique zero-error/hazard summary. All 198 input/output/proof hash checks
verified. Instrumented elapsed times (11.26/27.57/8.47 seconds) are correctness
run durations, not performance results. The validator also rejected missing
or reordered cases, absent bounds, NaN/exceeded bounds, absent byte parity,
absent guards and wrong batch metadata in CPU negative checks. Unsupported
launch rejection tests and non-random activation patterns remain additional
coverage work; no real-model integration or speed claim has been made.

Two isolated timing sweeps favor the candidate on all three tested shapes.
Each captures ten interleave-plus-matvec operations per graph, warms 1000
operations per order, and retains 40 samples per kernel/shape (100 operations
per sample). Pageable uploads are synchronized before nonblocking graph use.
The second sweep reverses shape and starting-kernel order. Every timed output
passes the independent bound and is byte-exact across variants. Probe and
oracle hashes verified for both results.

| Columns x rows | First latency change | Reversed latency change |
| --- | ---: | ---: |
| 3840 x 2048 | -25.4512% | -25.5794% |
| 8192 x 3840 | -20.2769% | -20.1733% |
| 15360 x 3840 | -8.8645% | -9.0494% |

These synthetic repeated-weight-bank timings include interleaving and retain
environment snapshots, but lack continuous activity acceptance. They justify
further isolated model testing, not a production promotion or end-to-end win.
Accepted llama.cpp scores remain 85.6103% short / 106.8707% at 8K.
