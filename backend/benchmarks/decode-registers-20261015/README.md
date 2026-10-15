# Decode compiler launch-bound probe - October 15, 2026

Original Fyodor diagnostic; production remains unchanged. The preceding goal
turn published aligned decode scale loads. Fresh profiles of that committed
runtime put fused gate/up at 436.811 of 1144.948 short matrix milliseconds
(38.2%). Profiles disable graphs and include host submission gaps; long logs
also include untimed prefix preparation, not decode-only matrix costs.

The probe adds `__launch_bounds__(256,N)` to copies of the fused kernel for
N=2/4/6/8. Key 1 is the unmodified compiler policy. All launches retain 256
threads, eight rows per block, existing loop unrolling and arithmetic order.
All 24 candidate/shape combinations match the baseline byte-for-byte including
output guards; odd loop lengths and partial row blocks are included.

Each graph contains 50 copies, with three warmup replays followed by five
samples of ten replays, then reversed candidate order. Every sample is kept.
These hot-weight diagnostics have snapshots but no continuous process monitor
or desktop idle gate. They do not update whole-model performance scores.

Throughput gains relative to the default (positive is faster):

| Rows x columns | Bound 2 | Bound 4 | Bound 6 | Bound 8 |
| --- | ---: | ---: | ---: | ---: |
| 9 x 256 | 12.68% | 6.79% | -0.78% | -0.39% |
| 5632 x 2048 | 0.18% | 2.14% | -1.02% | -0.06% |
| 11008 x 4096 | 3.87% | 9.18% | -21.58% | -23.12% |
| 65 x 768 | 15.39% | 24.29% | -2.46% | -4.58% |
| 257 x 2304 | 56.76% | 49.33% | 0.55% | 4.92% |
| 28672 x 8192 | 4.92% | 10.14% | -26.61% | -29.27% |

Bound four is a candidate for runtime testing: modest TinyLlama-shaped gains
and larger gains on wider projections. Bounds six/eight regress strongly on
large shapes and are rejected. Driver resources show 40 registers by default,
86 for bound two, 64 for bound four, and 40 for bounds six/eight, all with zero
local-memory bytes. The device reports 36 multiprocessors, 1536 resident threads
and 65536 registers per multiprocessor. Launch bounds are compiler constraints,
not proof of achieved occupancy: eight simultaneous 256-thread blocks would
exceed this device's thread limit. More requested blocks did not yield better
performance. No runtime hint has been accepted yet; the four-block candidate
needs production correctness checks and monitored full-model timing.

Accepted decode stays **84.1% short / 105.5% long** against pinned llama.cpp.
Both profiling workloads and both probes exited zero. No production source
was edited. `probe.py NEW_OUTPUT` runs initial shapes; append `extended` for
larger/odd-loop shapes. Output directories must not exist. The archived baseline,
source hashes, kernel attributes, device limits and all samples are included.
Scripts use bundled NVRTC and the CUDA driver without PyTorch.
