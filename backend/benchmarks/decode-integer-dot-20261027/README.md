# Q4 row-dot integer scale-product diagnostic

Original Fyodor candidate, source base f19a551. No runtime dispatch changes.
Apply the finite-half integer-product technique from the accepted gate/up
optimization to the generic Q4 row-dot helper. Only `(d * float(sc)) * float(q)`
becomes `d * float(sc*q)`; Q6 and other types remain unchanged. The helper copy
in root optcpp is diagnostic-only. There is no dimension specialization here.

The finite-half coefficient proof and exhaustive 65,011,712-case check are in
decode-integer-scale-20261027. This diagnostic independently checks actual GPU
output, including contraction, with random payload/metadata and nine shapes.
Nine further cases cycle 14 half bit patterns covering signed zero, subnormal,
normal, ordinary and max finite coefficients of both signs. All byte comparisons
and guards pass. Rotating-weight checks compare every bank independently.
NaN/Inf coefficients, Q6 signed products, and fused dot-add/QKV dispatch are not
validated by these tests; runtime integration needs its own real-model checks.

## Profile and timing

The accepted integer-gate/selective-FDPP path profile reports matrix955.765ms,
norm128.542, rope43.299, attention86.962, gather1.328. Q4 dot-add takes127.039ms
at columns2048 and151.671ms at columns5632; Q4 QKV takes72.996ms. This supports
investigating row dots. Profiling disables graphs and includes submission gaps;
this profile was not environment-gated and is not performance acceptance.

Kernel trials: RTX5060Ti SM120 driver617.42, NVRTC compute_120 C++11 fmad=true,
NYA_CUDA_FAST_MATH=1. Ten samples per variant, five each in reverse orders,
500 graph-replayed kernels/sample after warmups. Key `fixed` means integer
candidate (legacy harness label), not fixed-width specialization. Hot trials
reuse one matrix. Rotating trials use >=64MiB weights (29,11,8 banks). All five
monitored activities pass idle and competitor gates, with no kills needed in
this diagnostic. No other builds or GPU tests overlapped timing. Aggregate
standard deviations and individual samples are retained in JSON.

| Case | Columns x rows | Reference ms | Candidate ms | Throughput gain |
|---|---:|---:|---:|---:|
| 12-hot | 256 x 9 | 0.000945 | 0.000940 | +0.51% |
| 12-hot | 768 x 65 | 0.001564 | 0.001524 | +2.66% |
| 12-hot | 2048 x 256 | 0.003146 | 0.003048 | +3.23% |
| 12-hot | 2048 x 2048 | 0.010354 | 0.009272 | +11.67% |
| 12-hot | 5632 x 2048 | 0.026915 | 0.024042 | +11.95% |
| 12-hot | 2048 x 32000 | 0.132837 | 0.121612 | +9.23% |
| 12-extended | 2304 x 257 | 0.003427 | 0.003298 | +3.90% |
| 12-extended | 4096 x 11008 | 0.089036 | 0.079082 | +12.59% |
| 12-extended | 8192 x 8192 | 0.137423 | 0.125665 | +9.36% |
| 12-rotating | 2048 x 2048 | 0.011329 | 0.010350 | +9.47% |
| 12-rotating | 5632 x 2048 | 0.029337 | 0.026586 | +10.35% |
| 12-rotating | 4096 x 11008 | 0.092118 | 0.083911 | +9.78% |

## Decision

Proceed to a runtime trial of the Q4 integer product, including fused dot-add
and QKV callers, unchanged CPU-oracle tolerances, C17/C23, removable-component
builds, and clean matched decode measurements. These diagnostics support that
next step but do not establish a whole-model gain. Accepted whole-model scores
remain 98.97% short / 117.99% long versus llama.cpp; 125% remains unmet.

Run run.py, run_rotating.py, run_scales.py serially. They require fresh output
directories; preserve prior results before rerunning. No production sources,
precision modes, training behavior or build requirements changed here.
