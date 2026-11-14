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
