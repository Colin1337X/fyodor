# Batched verification graph feasibility audit

Source inspection on 2026-11-14 identifies host launch overhead as a possible
follow-up to the measured matrix bottleneck, not a proven speed improvement.
No graph behavior or production code changes are made by this audit.

Current `CUDA/resident.inc` admits graphs only for `count == 1 && logits`.
Its two cached topologies separate ordinary and split attention at position 256.
A device request packet receives token and position before replay. Verification
at count two/four instead runs the full host dispatch path and reads its bank.

A safe batched implementation must address all of the following:

1. `nya_gather_batch` in `CUDA/resident.cu` receives a by-value 512-token packet.
   Capturing its current launch freezes token IDs. Use stream-ordered device
   request tokens or explicitly update graph-node parameters before every replay.
   The currently loaded driver interface does not expose kernel-node updates.
2. The scratch `counts` array reserves `maximum_kv * batch` for ring KV but only
   `maximum_kv` for linear KV. Enabling `active_request` redirects key/value
   projections to this scratch. Batched capture must size linear scratch for its
   admitted batch or explicitly decline the unsupported allocation configuration.
3. The batch-four full-attention split condition changes at position 8192. Graph
   topology must include this distinction, batch width and relevant attention
   settings. Rewind across a boundary must choose the matching graph.
4. Capture must end before any host synchronization or logit download. Current
   batched output projection jumps directly to `batch_done`; the single-token
   capture completion block is not on that control-flow path.
5. Replay must preserve `valid`, `hidden_valid`, `hidden_offset`, KV high-water,
   stats and error poisoning, with independent graph ownership/destruction.
6. Verify all attention and RoPE kernels use request position plus query offset
   for the admitted batch. A single-token request implementation is insufficient
   evidence for every batched backend path.

Required evidence before admission: changing-token replay parity, two/four-token
batches, linear/ring KV, reset/rewind/replacement, 255/256 and 8191/8192 boundary
crossings, allocation/capture/launch failure cleanup, Compute Sanitizer and
paired decode timing. Start behind an opt-in capability with normal batched
execution as fallback. Do not infer benefit solely from reduced host call count.
