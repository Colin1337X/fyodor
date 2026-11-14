# RMSNorm plus narrow-batch layout prototype

The refreshed short decode profile reports normalization as the largest
non-matrix category (45-48 ms in joint mode). Existing batch-two/four matrix
paths additionally launch an interleave conversion after normalization; that
conversion is counted with matrix work. This experiment targets the extra
conversion launch/read, not all normalization cost.

The original optional CUDA kernel preserves the existing RMSNorm reduction,
weight expression and row-major output, and writes the same value directly to
a second interleaved buffer. Batch two/four only. No production dispatch.
Strided stores and extra output traffic may outweigh the eliminated launch;
only paired measurements can justify integrating it.

Host integration must bound buffer arithmetic, preserve existing fallback
outputs and tie interleaved validity to the exact input, width and batch.
Reset, rewind, failed operations, other normalization outputs and reused
scratch storage must not expose stale layouts. Existing adjacent gate/up
reuse alone is not sufficient for this broader lifetime.

Pending: exact comparison against separate RMSNorm plus interleave, guarded
outputs, finite-value edge cases, in-place row-major behavior, both batches,
sanitizers, paired timing and full-model parity before any integration.

NVRTC compilation and offline SM120 assembly passed: 40 registers, 48 shared
bytes, no stack or spills. No GPU execution or numerical/speed claim yet.

## Completed raw and sanitizer validation

All 1248 cases matched both output layouts byte-for-byte: 13 widths, batches
two/four, F32/F16/BF16/absent weights, in-place and separate row output,
random/zero/extreme finite inputs and two epsilons. Three unsupported batch
checks left guards unchanged. The 655 raw input hashes independently verified.
Memcheck, racecheck and synccheck repeated the full inventory without errors;
1980 sanitizer input/evidence hash checks independently verified.

## Timing harness failure and correction

The initial unfenced probe failed at width 15360/batch two. A repeat retained
`timing-mismatch.json`: row-major output matched exactly, but the separate
baseline's interleaved buffer was entirely zero. That rejects the unfenced
timing attempt; it is not evidence of incorrect fused arithmetic.

`probe_upload_sync.py` fences all default-stream uploads before capture/use
on the nonblocking stream, and independently checks every timed layout
against a CPU permutation of the row-major output. All fourteen shapes then
passed. The fence is outside timed intervals. The canonical `probe.py` now
uses this corrected ordering; the original is preserved as `probe_unfenced.py`
for failure provenance, not accepted measurement.

NVIDIA documents that pageable host-to-device transfer calls can return after
staging while DMA remains incomplete:
https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html
The observed baseline corruption and its resolution after the explicit fence
are consistent with this missing cross-stream dependency.

Corrected synthetic diagnostics use ABBA BAAB order, 40 observations per
implementation/shape and 1000 warmup operations per order. Positive means slower.

| Width | Batch | Separate ms | Fused ms | Latency change |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 0.002076 | 0.001458 | -29.75% |
| 256 | 4 | 0.002078 | 0.001458 | -29.86% |
| 512 | 2 | 0.002505 | 0.001865 | -25.52% |
| 512 | 4 | 0.002482 | 0.001871 | -24.62% |
| 1024 | 2 | 0.002698 | 0.002075 | -23.08% |
| 1024 | 4 | 0.002874 | 0.002277 | -20.75% |
| 2048 | 2 | 0.003780 | 0.002917 | -22.84% |
| 2048 | 4 | 0.004747 | 0.004099 | -13.64% |
| 3840 | 2 | 0.005805 | 0.004980 | -14.21% |
| 3840 | 4 | 0.004964 | 0.005476 | +10.31% |
| 4096 | 2 | 0.004320 | 0.004486 | +3.85% |
| 4096 | 4 | 0.005238 | 0.005107 | -2.50% |
| 15360 | 2 | 0.011284 | 0.012829 | +13.69% |
| 15360 | 4 | 0.011626 | 0.013935 | +19.86% |

The important Gemma width-3840/batch-four case regressed 10.31%; wider inputs
also regress. The implementation is not promoted to model integration. Short
widths show a possible benefit but require separate admission and real-model
validation. No clean end-to-end score changed. Prior graph probes that use
the same unfenced upload/nonblocking-stream pattern need ordering review and
corrected reruns before their timing decisions are relied upon. Historical
evidence remains preserved; none of these synthetic probes is an accepted
llama.cpp score.
