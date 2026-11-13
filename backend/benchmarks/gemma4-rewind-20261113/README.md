# Explicit verification rollback, 2026-11-13

Original native C implementation; no new optional dependencies. The optional
provider callback truncates the valid KV prefix without reconstructing logits.
Unsupported resident providers fail before mutation. CUDA validates retained
window rows before changing the prefix, keeps physical high-water after nonzero
rewinds, and rejects forward movement. Session truncation hides stale logits and
suffix KV. Zero permits a fresh prefix. Failure-invalidated sessions still require
reset. No-op rewind preserves current logits.

The extended `test_verify.c` checks invalid forward rewind, no-op, discarded
suffix reads, single-token replay, a different replacement token versus an
independently replayed serial prefix, zero reset, failed-session rejection, and
evicted ring-prefix rejection. It retains the fixed scaled logit tolerance of
0.0005, finite-value and masked-infinity checks. CUDA verification batches use
8 tokens; the ring fixture uses a 3-token sliding window. Shared KV and graphs
on/off are exercised. This does not establish correctness for every proposal
window or supply resident assistant hidden/KV binding.

Real-model commands, environment, executable hash, and exit codes are recorded
in model-results.json. The local Gemma 4 12B Q4_K_M model is the same accepted
benchmark model (SHA256
59656d7494d6376ca97e9e20b64ea2e16cd97f12ec6d47bfccba91cb785b5134).
Prefixes 7 and 1020 exercise short context and crossing the 1024 sliding window.
These are correctness runs, not timing evidence. CPU builds ran concurrently
with part of correctness testing; no throughput is accepted from this run.

Physical removal: 82 registered tests passed, with four expected unavailable
ROCm/MLX skips, plus train/export/infer. Clang ASan/UBSan: CPU verification and
empty-prefix cases plus trained fixture passed. See retained logs.

The verified Powder Toy game process (PID 33276, exact executable path and
version metadata checked) was stopped during correctness testing under user
authorization. OS, development, remote-access, browser and download processes
were retained. This is not evidence of a clean timed benchmark environment.

Accepted decode ratios remain 85.6103% short / 106.8707% at 8K. No performance
claim: generation acceptance does not yet use the batch/rewind interface.

C17 and C23 each pass all eight verification cases plus their trained fixture.
Real Gemma prefix 7 and 1020 both pass twice; all-position maxima are
0.000383745074 and 0.000067389802 respectively. Replay and changed-token
assertions use the same fixed tolerance and pass. CUDA memcheck reports zero
errors (`memcheck-ring.log`). The initial `memcheck.log` records a harness
invocation with an incorrect fixture directory; the target exited before any
CUDA API call. The corrected command uses
`build-cuda-c23/test_models/pretraining.trained.gguf`, `--tool memcheck
--error-exitcode 99`, and environment `NYA_COMPUTE=cuda`, `NYA_CUDA_BATCH=8`,
`NYA_CUDA_RING_KV=1`, `NYA_TEST_VERIFY_RING=1`, `NYA_TEST_VERIFY_RESIDENT=1`.
