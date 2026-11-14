# Batched graph request prototype

This is the first implementation step from the batched verification graph audit
in `../interleave-pair-20261114/batched-graphs-audit.md`. It is original Fyodor
code, not a transplant of ggml's graph runtime. The upstream audit identified
stable topology and changing data as separate replay requirements.

`optcpp/graph/batch_request.cu` adds a stream-ordered five-word device request
and an embedding gather for exactly two or four verification tokens. Word zero
remains token zero and word one remains the absolute position, preserving the
current RoPE/attention request layout. Words two through four hold the remaining
tokens. The updater takes a bounded by-value packet and must run outside capture
on the same stream before every replay. Gather reads the device packet, so its
captured arguments no longer contain stale token IDs. Existing weight decoding,
F32 multiplication and token-major output layout are preserved.

## Completed evidence

- `check_host.py` executes the actual kernel statements, removing CUDA entry
  attributes and supplying sequential launch coordinates and an F32 weight
  decoder. C17 and C23 at O3 with strict warnings each pass 120 changing-token
  cases across batch two/four, widths 1/31/256/257/3840, and positions around
  256 and 8192 plus the unsigned-position upper boundary. Request/output guards,
  excess launch threads, invalid batch widths and zero embedding width pass.
- `compile_only.py` compiles the prototype with the existing CUDA weight decoder
  through NVRTC compute_120 and verifies both PTX entry points. No CUDA context,
  driver execution or GPU work is used. Source hashes are recorded.

These checks establish host indexing and device compilation only. They do not
establish asynchronous ordering, real quantized gather parity or graph replay.
There is no production dispatch, runtime dependency or claimed speedup.
Physical removal of the prototype leaves the production build unchanged.

## Integration still required

The current resident plan reserves only two request words. It must allocate five
before using this updater. Linear KV scratch must also reserve the admitted batch,
and batch width needs its own cached graph and output-bank path. Keep existing
single-token `graph_index` semantics separate from the new cache index: several
attention branches interpret that value as a single-token split topology.

Position-dependent dispatch also changes at 256 (wide batch) and 8192 (batch-four
full-attention split), not just at the latter boundary. Cache keys must distinguish
all admitted topologies, and replay bookkeeping must preserve state and counters.
All request-bearing ordinary/wide attention paths examined add a query offset;
the tiled prefill entry lacks the request argument and is outside this prototype's
two/four-token scope.

The isolated changing-token gather checks below are complete. Before enabling
full-model batched capture: reset/rewind/replacement and topology-boundary tests,
allocation/capture/launch failure cleanup, full-model Compute Sanitizer/parity
and controlled end-to-end timing. This prototype alone does not implement those
host-side lifecycle changes.

Accepted decode performance remains 85.61025294% short and 106.8707255% at 8K
relative to llama.cpp; the 125% target on both is unmet.

## Device replay check

`check_replay.py` captures the new gather once per dtype/width/batch and updates
the device request outside capture for each of twelve replays. The 192-case
inventory covers F32/F16/Q4_K/Q6_K, widths 256/3840 and batches two/four. Each
replay compares bytes against the existing uncaptured batched gather, checks
finite outputs, request/output guards and unchanged weights. The host oracle
above supplies an independent F32 indexing check; the GPU comparison deliberately
shares the production quantized weight decoder and does not independently prove
that decoder. Both paths run on the same explicit nonblocking stream, with
upload synchronization before capture and readback synchronization after replay.

Before importing CUDA bindings, the runner requires successful full-model metadata
sanitizer evidence and source hashes, the host and compilation proofs, and a GPU
process inventory check. Its refusal while the sanitizer remains live was
exercised without creating a CUDA context. Six CPU prerequisite tests cover
valid, missing, failed, nonzero-exit and changed-input evidence, plus refusal of
optimized Python before CUDA import so parity assertions cannot be disabled.
`--memcheck` selects a separate result filename for running under Compute
Sanitizer; that filename alone is not proof of a successful sanitizer summary.

`run_memcheck.py` now wraps that mode after ordinary replay validation succeeds.
It requires all 192 cases in their exact order, the expected changing tokens and
positions, parity/guards, a zero process exit and exactly one zero-error summary.
It also checks unchanged inputs and identical raw/sanitized compilation options
and source hashes. Seven CPU validator tests pass, including missing/duplicate
cases, incorrect replay fields, cleanup failure and invalid sanitizer summaries.
After the metadata full-model sanitizer and monitored run finished, ordinary
execution and Compute Sanitizer each passed all 192 replays, using one capture
per dtype/width/batch and twelve changing-token launches of each executable.
Sanitizer exited zero with exactly one zero-error summary. All 126 raw-run hashes,
128 sanitizer-run input hashes and the sanitizer output hashes verified after
completion. The test confirms the isolated request/gather stream ordering and
parity/guards for the tested cases. It does not establish full-model graph
lifecycle correctness or a decode speed improvement.
