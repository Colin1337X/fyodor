# MTP resident-target bridge, 2026-11-13

Original native C host integration. NYA_MTP_RESIDENT_TARGET=1 keeps a supported
CUDA target plan resident while the existing assistant uses host activations and
assisted matrix operations. NYA_SPECULATIVE_BATCH=1 additionally batches target
verification through the existing acceptance/rewind machinery. Both default off.
NYA_MTP_TRACE=1 reports the selected path after prefill. This is a hybrid bridge,
not a fully resident assistant. No speed acceptance is claimed.

## Contract

The optional C provider callbacks expose a normalized final-hidden read and an
attention operation against the target plan's retained KV. Hidden reads normalize
the current raw final layer result; they reject missing, failed, or rewound hidden
state. Batch-last offsets and decode graph replays are tracked explicitly. Attention
consumes already normalized/rotated queries with scale 1, checks source layer/head
geometry, prefix, ring retention, and finite inputs/outputs, and reuses existing
plan scratch. It never copies the full cache to the host. Calls must be serialized
with target execution, and the target plan/model must remain alive.

One target hidden vector is downloaded per MTP round. Each assistant layer uploads
one query vector and downloads one attention result. Target raw hidden and valid KV
remain untouched by those scratch operations. The assistant freezes the target
prefix before the base token advances it, preserving the existing MTP semantics.
At least two positions of provider batch capacity are required so that the base
write cannot evict the frozen sliding window. Unsupported plans use the existing
host path; ordinary target execution failures can still replay the prefix on CPU.
A bridge operation failure aborts the request rather than reading stale host KV.

The same probability acceptance/residual correction is used. On rejection, target
verification rewinds to the corrected input before executing it. Counts include
all computed target positions, including discarded speculative work. Fully resident
assistant projections/hidden state and host-transfer elimination remain future work.

## Evidence

- Direct CUDA test compares hidden state against an independent CPU session and
  borrowed attention against a float64 softmax/value oracle over CPU KV. Checks
  cover initial state, 12-token batched prefill, 3-token window/ring eviction,
  older retained prefixes, invalid shapes/prefixes, raw-hidden preservation,
  rewind invalidation, corrected replay, cached graph replay and reset. Existing
  scaled tolerance .0005 is retained. Two attention calls have exactly two uploads
  and two downloads, with unchanged KV/scratch byte counts.
- MTP architecture fixture: greedy windows 1..8 plus 16 seeded sampling comparisons
  against fully host target/assistant execution. Batched acceptance counts, text,
  generated counts and stop reason agree. Explicit prefill failure exercises host
  fallback. C17/C23 selected suites pass; final C17 fallback suite also passes.
- CUDA memcheck: direct borrow test and complete MTP architecture test both report
  zero errors. CPU ASan/UBSan architecture tests pass with the opt-in settings.
- Physical non-C removal: all 82 registered CPU tests pass, four expected unavailable
  ROCm/MLX skips, plus train/export/infer. This copy predates the final trace placement
  and additional fallback test; those changes are separately exercised on CPU ASan.
- Supplied Gemma target and mtp-gemma-4-12B-it assistant: 12-token greedy generation
  matches ordinary target generation at prompt lengths 18 and 1058 with batching.
  Proposal acceptance is 7/8 and 8/8 respectively. Trace confirms resident borrowed KV
  and batched verification. This covers the 1024 sliding-window boundary. Serial
  bridge verification also matches at 18 and 274 prompt tokens (7/8 and 6/8 accepted).
  The initial gemma.log is a four-token smoke run before tracing was added.

model.c reuses the prior model-registry oracle; compile with GCC C17 -O3
-Wall -Wextra -Werror, include backend/include and backend/core, link
build-cuda/libnya-engine.a, bcrypt and C:/Windows/System32/vulkan-1.dll.
Arguments: none for short; long for 274 prompt tokens; window for 1058.
Commands, environment and hashes for the batched runs are in model-results.json.
The real-model harness predates test-only sampled/fallback expansion and final
trace relocation; core bridge computation is unchanged. CPU compilation overlapped
some correctness runs, so their elapsed time must not be promoted as a benchmark.

Accepted llama.cpp decode scores remain 85.6103% short / 106.8707% at 8K. The 125%
short-and-long target remains unmet. Next: measure complete MTP work and remove
assistant transfer/projection bottlenecks while preserving these numerical checks.

Final review added separate linear-cache dispatch and full-prefix attention-oracle
coverage. Both ring and linear MTP/borrow suites plus fallback pass in C17/C23
(six tests each including the fixture); linear CUDA memcheck also reports zero
errors. See linear-ring*.log and memcheck-linear.log. Earlier real-model runs
used ring KV and are unaffected by this dispatch correction. The physical-removal
copy also predates this CUDA-only dispatch correction; its CPU results remain
applicable to the portable path.
