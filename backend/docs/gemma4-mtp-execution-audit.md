# Gemma 4 MTP execution audit, 2026-11-13

The default MTP path retains coherent host target/assistant state. An opt-in hybrid
path now keeps the target resident and evaluates assistant attention against its KV
through optional C provider callbacks. It is not a fully resident assistant.

## Current execution

`NYA_MTP_RESIDENT_TARGET=1` retains a supported CUDA target with batch capacity>1.
The assistant downloads one target hidden vector each round, uses its existing
host/assisted projection path, and transfers query/result vectors for attention.
The full target KV cache stays on device. The frozen prefix, target-scaled token
embedding, recurrent assistant projection and final-normalized hidden semantics
remain unchanged. Source layers are the last non-sharing full/sliding target layers
with matching KV-head count and width. Calls are serialized within the request.

`NYA_SPECULATIVE_BATCH=1` now applies to this MTP bridge as well as standalone
drafts. All-position target logits retain before/after-token alignment; rejection
uses the explicit retained-prefix rewind before the corrected token. Target step
counts include discarded speculative positions. The optional small-batch row-dot
kernel removes the previously measured narrow-GEMM regression on TinyLlama, but
neither that diagnostic nor acceptance counts establish Gemma MTP throughput.

Absent callbacks or insufficient batch capacity select the default host path.
A target execution failure may replay its prefix on CPU; a bridge read/attention
failure aborts rather than reading an unpopulated host cache. Hidden reads reject
rewound state; borrowed attention rejects evicted rows and invalid geometry.

## Evidence and remaining work

See [bridge validation](../benchmarks/mtp-resident-target-20261113/README.md) for
CPU hidden/float64 attention oracles, graph/reset/rewind checks, greedy and seeded
MTP fixture parity, injected fallback, sanitizers, physical removal and real-model
12-token greedy parity at 18/274/1058 prompt tokens. Real batched proposal counts
are 7/8 and 8/8 at 18 and 1058 respectively; these small probes are not a sustained
acceptance-rate or quality evaluation.

Remaining: resident assistant projection and recurrent state; eliminating remaining
host transfers; broader sampled real-model evaluation; profiling and clean matched
end-to-end MTP versus ordinary target and llama.cpp decode at short and long context.
The assistant CUDA planner still declines standalone assistant plans. Per-position
vocabulary projection/readback in verification also remains serial.

Accepted Gemma decode scores remain 85.61% short / 106.87% at 8K. Neither the 125%
performance requirement nor the broader native runtime/training goal is complete.
