# Fixed dimensions for row dots, October 27, 2026

Production stays at 3ddc208. The accepted fixed-width gate/up option prompted a
fresh profile with both that option and selective FDPP enabled. Matrix time is
981.250 ms of 1246.372 recorded ms (78.7%); gate/up388.056 ms, residual projection
paths409.857 ms combined, QKV129.836 ms and output54.501 ms. Profiling disables
graphs and includes submission gaps; these are attribution diagnostics, not
throughput acceptance or a valid before/after timing comparison. Exact command,
environment, executable hash and output are retained.

This original Fyodor probe calls the existing force-inlined row-dot helper
with constant columns/stride for seven widths. Weight types12/14 are Q4_K/Q6_K;
launch geometry, arithmetic/reduction order, precision and weight storage remain
unchanged. Device wrappers live under root optcpp/matvec/fixed_width_dot.cu and
are never compiled by CMake or selected by runtime dispatch. No upstream copied
source, no extra build dependency. This is not a full FDPP flat-GEMM engine.

| Format | Rows x columns | Candidate throughput change |
| --- | --- | ---: |
| Q4_K | 9 x 256 | +9.15% |
| Q4_K | 65 x 768 | +0.85% |
| Q4_K | 256 x 2048 | +0.07% |
| Q4_K | 2048 x 2048 | -0.46% |
| Q4_K | 2048 x 5632 | -0.50% |
| Q4_K | 32000 x 2048 | -0.48% |
| Q4_K | 257 x 2304 | -0.11% |
| Q4_K | 11008 x 4096 | -0.19% |
| Q4_K | 8192 x 8192 | -0.58% |
| Q6_K | 9 x 256 | +0.67% |
| Q6_K | 65 x 768 | +2.87% |
| Q6_K | 256 x 2048 | -0.36% |
| Q6_K | 2048 x 2048 | +0.27% |
| Q6_K | 2048 x 5632 | -0.67% |
| Q6_K | 32000 x 2048 | -0.36% |
| Q6_K | 257 x 2304 | -1.07% |
| Q6_K | 11008 x 4096 | -0.25% |
| Q6_K | 8192 x 8192 | +0.01% |

Reject generalized fixed-dimension row dots: model-sized cases are mostly tied
or slower, unlike the previously accepted gate/up specialization. The tiny Q4
case improves but does not justify new dispatch for the measured bottleneck.
Q4 registers remain36 except width256 (38), Q6 resource counts are recorded;
all kernels report zero local-memory bytes. No full-model gain is inferred.

All18 format/shape combinations match native output/guard bytes, including
partial row blocks and odd quant-block counts. Fixed finite half scales,
seeded random metadata/payload and bounded input vectors are diagnostic coverage,
not a real-model/exceptional-value correctness proof. Since the experiment is
rejected, no runtime integration, new lifecycle suite or removal build is needed.

All four activities pass continuous competitor monitoring and the unchanged
idle gate; no competing process needed termination. Hot-weight CUDA graphs,
50 launches per graph, three warmup replays, five samples of ten replays in each
order (ten per cell). Every timing sample, mean, sample standard deviation,
compiled source hash, NVRTC options and kernel resources are retained. NVRTC
compute_120/C++11/fmad, RTX5060Ti driver617.42. Process snapshot trailing whitespace
alone was normalized. Run run.py with fresh output directories to reproduce.

Latest matched whole-model performance stays **96.33% short / 115.36% long** of
llama.cpp using selective FDPP plus fixed gate2048; target125% on both unmet.
The next investigation should change the operation/dataflow rather than extend
this dimension specialization to further generic row-dot wrappers.
