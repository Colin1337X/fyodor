# Decode cached F32 normalization trial - October 15, 2026

The last profile placed normalization at 116.880 GPU ms versus 1111.907 matrix
ms. Event profiling disables graphs and includes submission gaps; it is a
bottleneck guide, not end-to-end acceptance timing.

General register caching regressed important widths because dynamic width and
format checks remain. Exact-width F32 variants preserve each thread's ordered
sum, shared/warp reduction and final multiplication order while retaining the
input values in registers. Initial isolated speedups were about 16%, 11%, 8%
at widths 2048/4096/8192. An isolation at width 2048 distinguishes cached fixed
F32 (~16.6%) from uncached fixed F32 (~-2.1%) and uncached dynamic F32 (~5.6%).
These are hot-weight diagnostic graphs, snapshots only, not acceptance scores.

Candidate runtime dispatch is limited to single-row, single-token decode,
F32 norm weights or no weights, the three exact widths and non-reference mode.
All other shapes, formats, batching and reference execution keep the generic
kernel. No buffer is added. The candidate passed correctness but was rejected after monitored full-runtime
timing; accepted score remains 86.8% / 107.5%.

C17 and C23 each pass 25/25 configured CUDA regression checks. The production
kernel harness checks all three widths against the original kernel byte-for-
byte, including output canaries. It exercises random weighted inputs plus
zero, alternating and wide-dynamic-range values with present/absent weights
and separate/in-place output (39 cases). This is the actual production source,
not just the earlier generated prototype. No local-memory spill is reported
by the driver. Real-model optimized-CPU logits pass at existing tolerances;
this does not resolve the pre-existing stricter CPU/GPU KV-snapshot discrepancy.

Real-model graph checks pass for 795 decodes. Initial memcheck/initcheck passes
included the timing loops. Initial racecheck reached the 240-second timeout
before completing those repetitions; its verified remaining Python child was
terminated before any further GPU work. The timeout is preserved, not counted
as a pass. The bounded sanitizer mode keeps all correctness cases and captured
graph execution, reducing timing repetitions to one. Memcheck, initcheck,
racecheck and synccheck all pass that mode with zero errors/hazards. Sanitized
timings are never used as performance evidence. See bounded-sanitizer-exits.json.

## Full-runtime result: do not promote

All 12 benchmark activities passed independent monitoring and both idle gates.
No competing workload required termination during the timing run. Ten timed
samples per cell, complete forward/reverse engine order, none excluded:

| Prefix | Original tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 329.307 +/- 1.377 | 326.300 +/- 13.123 | -0.91% |
| 512 | 310.828 +/- 0.863 | 312.009 +/- 0.771 | +0.38% |
| 1024 | 288.978 +/- 0.707 | 289.975 +/- 1.233 | +0.34% |

The noisy short candidate measurements do not establish a consistent regression,
but also do not establish a gain. The small middle/long gains do not justify
three specialized kernels when the priority short workload has no demonstrated
benefit. At width 2048, saving roughly 0.33 microseconds per normalization only
saves about 15 microseconds across 45 calls per token: a small fraction of the
roughly three-millisecond token latency. This explains why a large isolated
percentage has little end-to-end effect. Keep the generic implementation.

Both production files were restored byte-for-byte and C17/C23 rebuilt. The
candidate patch, exact candidate source and pre-change source remain archived.
Standalone production-kernel checks load the archived candidate, so they remain
reproducible after rollback. Runtime validation/compare scripts require applying
the archived candidate patch and rebuilding first; do not run them on an
unmodified baseline and interpret that as a candidate comparison. The baseline
executable hash and all monitored outputs are in clean/metadata.json.

No llama.cpp reference refresh was run for this rejected candidate. Scorecard
stays at the last accepted matched measurement, 86.8% short / 107.5% long.
