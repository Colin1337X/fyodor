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
