# Decode matrix diagnostics

These original Fyodor device sources are unselected experiments. They require
the native `nya_half_aligned` helper and are compiled only by diagnostic
scripts, not by CMake or runtime dispatch. Removing this directory has no build
or execution effect. They are not a complete Flash-Decoding++ flat-GEMM engine.

`packed_gate.cu` retains the ordered gate/up arithmetic while trying a packed
32-bit payload load with warp shuffles, or earlier scalar payload loads.
`shared_input_gate.cu` stages the input vector cooperatively before any warp
returns for an out-of-range row; callers supply columns*sizeof(float) shared
memory. Neither changes precision or weight format.

Packed loads and shared staging regressed tested workloads. Early scalar loads
show small diagnostic gains but have not earned runtime integration. Exact
output, resource and timing evidence is in
backend/benchmarks/decode-packed-load-20261026. No end-to-end gain is claimed.

`unroll_gate.cu` rechecks outer-loop factors with monitored rotating weights;
keep the existing factor eight. `prepacked_scales_gate.cu` expands Q4 blocks
from 144 to 192 bytes with precomputed float scales/minima. It matches tested
outputs but loses about 9% on rotating weights and is rejected. Evidence:
backend/benchmarks/decode-unroll-20261026 and decode-prepacked-scales-20261026.
These are diagnostic-only sources, never runtime dispatch options.
