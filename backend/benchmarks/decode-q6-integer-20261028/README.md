# Q6 signed integer scale-product diagnostic

Original Fyodor implementation, base f0530ce, diagnostic only in root optcpp.
No runtime/build changes or upstream code copied. Q6 remains a meaningful cost
in the new accepted profile: dot-add131.557ms and vocabulary dot54.439ms;
QKV with Q6 V53.513ms includes Q4 Q/K work. Total matrix919.341ms. This ungated
profile disables graphs and includes submission gaps; it identifies candidates,
not an end-to-end performance comparison.

## Arithmetic and correctness

Replace `(d * float(sc)) * float(q)` with `d * signed_product(sc,q)` for Q6.
Scale is [-128,127], quantized value [-32,31], so their integer product fits
safely in int and its magnitude is at most4096. Finite half coefficients have
at most11 significant bits; this product fits exactly in F32. Integer conversion
alone loses the sign of a zero product. The candidate ORs the XOR of operand
sign bits into the converted float's sign bit, preserving signed-zero behavior
without changing nonzero results. Accumulation and warp reduction stay ordered.

The C17 exhaustive test checks1,040,187,392 combinations of all finite half bit
patterns, signed8-bit scales and signed6-bit values. Every coefficient result
matches bit-for-bit, including signed zero. Compile coefficient_check.c using
GCC -std=c17 -O3 -ffp-contract=off and run. NaN/Inf half coefficients are excluded.
This proof alone does not establish GPU contraction or real-model correctness.

GPU tests compare nine shapes with random metadata/payload and fixed finite
half coefficients, nine more cycling14 half patterns (both signs, zero,
subnormal, normal and maximum finite), and every rotating weight bank. All
outputs and guards match byte-for-byte. Q4 arithmetic in the copied helper
remains unchanged and is not selected by the diagnostic wrapper.

## Measurements

RTX5060Ti SM120, driver617.42; NVRTC compute_120 C++11 fmad=true and
NYA_CUDA_FAST_MATH=1. Ten observations per variant, five in each opposite order,
500 graph-replayed kernels/sample after warmups. `fixed` is the legacy label
for the generic-dimension candidate, not width specialization. Rotating cases
use20,8,8 banks totaling68,812,800 /75,694,080 /295,895,040 bytes. All five
monitored activities pass idle/competitor gates; no competitor killed and no
build or coefficient proof overlapped timing. Raw samples and sample standard
deviations are retained. These synthetic kernel numbers are not decode scores.

| Case | Columns x rows | Reference ms | Candidate ms | Throughput gain |
|---|---:|---:|---:|---:|
| 14-hot | 256 x 9 | 0.000906 | 0.000922 | -1.77% |
| 14-hot | 768 x 65 | 0.001471 | 0.001511 | -2.67% |
| 14-hot | 2048 x 256 | 0.002888 | 0.002878 | +0.34% |
| 14-hot | 2048 x 2048 | 0.010218 | 0.009774 | +4.54% |
| 14-hot | 5632 x 2048 | 0.026450 | 0.025144 | +5.20% |
| 14-hot | 2048 x 32000 | 0.142725 | 0.139118 | +2.59% |
| 14-extended | 2304 x 257 | 0.003141 | 0.003129 | +0.38% |
| 14-extended | 4096 x 11008 | 0.099194 | 0.096902 | +2.37% |
| 14-extended | 8192 x 8192 | 0.154884 | 0.150747 | +2.74% |
| 14-rotating | 2048 x 2048 | 0.011545 | 0.011028 | +4.68% |
| 14-rotating | 5632 x 2048 | 0.030733 | 0.029593 | +3.85% |
| 14-rotating | 4096 x 11008 | 0.100422 | 0.097903 | +2.57% |

## Decision

Proceed to a shape-gated runtime trial for larger Q6 projections; small shapes
can regress. Validate fused residual-add and real-model outputs before accepting
anything, and avoid applying the candidate indiscriminately to small QKV V.
The dominant Q4 gate/up work remains a separate larger optimization opportunity.
Current accepted end-to-end scores stay101.88% short /120.66% long versus llama.cpp;
the requested >=125% on both is unmet. No training, prefill or portability gain
is claimed. Removing this diagnostic source has no runtime effect.

Reproduce serially using run.py, run_rotating.py and run_scales.py with fresh
output directories. The wrappers select type14 explicitly. Source hashes are
preserved; probe docstrings were clarified after testing without changing device
source. Production dispatch and builds remain untouched.
