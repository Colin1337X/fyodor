# Decode matrix diagnostics

These original Fyodor device sources are experiments. They require
the native `nya_half_aligned` helper. All except the fixed-width and integer scale-product sources are
compiled only by diagnostic scripts. Removing this directory keeps generic
native dispatch available. They are not a complete Flash-Decoding++ flat-GEMM engine.

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

`warp_pair_gate.cu` assigns separate warps to gate/up and combines through a
block barrier; it is exact on diagnostic cases but loses 13-14% on rotating
model-sized weights. `transposed_payload_gate.cu` preserves 144-byte blocks,
transposes payload bytes for per-lane word loads, and avoids shuffles; it loses
0.8-1.5% on rotating weights. Neither is integrated. See the matching
backend/benchmarks/decode-warp-pair-20261026 and decode-transposed-payload-20261026
evidence directories. These results do not support more compiler/load-hint
tuning without a new mechanism or hardware profiling evidence.

`fixed_width_gate.cu` retains six constant-width variants for diagnostics.
Only the 2048-column candidate merits a runtime trial (+2.93% with rotating
weights, +0.44% hot); wider cases regress, with local-memory use at width2304.
The follow-up optional runtime dispatch selects only width2048, with matching
1152-byte Q4 strides. Enable CMake NYA_ENABLE_FIXED_GATE=ON (default OFF), then
NYA_CUDA_FIXED_GATE=1 before creating a CUDA context. Missing source/symbol or
other widths use generic kernels. All other widths remain diagnostic-only.
Evidence: backend/benchmarks/decode-fixed-width-20261027 and
decode-fixed-runtime-20261027.

`fixed_width_dot.cu` tests the same constant-dimension idea in native Q4/Q6
row dots. All18 cases match exactly, but model-sized cases show no consistent
gain. This extension is rejected and never CMake-compiled. Evidence:
backend/benchmarks/decode-dot-fixed-20261027, including the updated runtime profile.

`integer_scale_gate.cu` combines the Q4 integer scale and nibble before float
conversion. The finite-half coefficient check is exhaustive and tested GPU
outputs are byte-exact. Clean rotating-weight diagnostics improve about 7%
against fixed-width baselines at widths 2048/4096; width 2304 regresses.
The width-2048 runtime trial uses CMake NYA_ENABLE_INTEGER_GATE=ON (default
OFF) and NYA_CUDA_INTEGER_GATE=1. It takes priority over fixed-gate selection;
missing source/symbol leaves fixed (if requested) or generic dispatch in place.
Other widths remain diagnostic-only. Real-model validation is tracked separately
in decode-integer-runtime-20261027. Kernel evidence: backend/benchmarks/decode-integer-scale-20261027.

`integer_scale_dot.cu` tests the same integer-product technique in generic Q4
row dots. Nine shapes plus finite-scale stress cases match byte-for-byte;
rotating weights gain 9.5-10.4% on three shapes. The optional runtime trial
uses NYA_ENABLE_INTEGER_DOT=ON and NYA_CUDA_INTEGER_DOT=1 for Q4 dot/add/QKV,
with native fallback and reference/training excluded. Validation evidence is in
decode-integer-dot-runtime-20261027. Kernel evidence:
backend/benchmarks/decode-integer-dot-20261027.

`integer_scale_q6.cu` tests signed Q6 scale products, preserving zero sign.
The exhaustive finite-half coefficient check and GPU output/guard comparisons
pass. Rotating weights gain 2.6-4.7% on larger tested shapes, while small shapes
can regress. Optional runtime trial uses NYA_ENABLE_INTEGER_Q6=ON and
NYA_CUDA_INTEGER_Q6=1 for dot/add with rows and columns >=2048; smaller shapes
use native arithmetic and QKV remains unchanged. Validation evidence:
backend/benchmarks/decode-q6-runtime-20261028.
Evidence: backend/benchmarks/decode-q6-integer-20261028.

`byte_scale_gate.cu` expands Q4 scales/minima to bytes in148-byte blocks; it
regresses and is rejected. `byte_scale_gate_aligned.cu` uses160-byte blocks,
32-byte aligned payload and word metadata loads. It matches all tested outputs
and gains about13% on two rotating model-sized shapes, with11.1% extra weight
storage. The148-byte layout remains diagnostic-only. Aligned width2048 has an optional
runtime integration: NYA_ENABLE_BYTE_GATE=ON and NYA_CUDA_BYTE_GATE=1. C owns
plan-local packing and budget/failure fallback; native weights are retained,
so memory cost is a full additional packed copy. Runtime evidence:
backend/benchmarks/decode-byte-runtime-20261028. Kernel evidence:
backend/benchmarks/decode-byte-scales-20261028 and
decode-byte-scales-aligned-20261028.
