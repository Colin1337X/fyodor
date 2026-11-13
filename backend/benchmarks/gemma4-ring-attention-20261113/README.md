# Native CUDA ring-addressed attention foundation — 2026-11-13

Status: kernel addressing validated; runtime allocation/dispatch is **not yet
integrated**. Existing linear attention wrappers pass a compile-time zero ring
capacity and retain their original indexing. New ring and ring-split entry points
reuse the native F32 partition algorithm with `t % ring_capacity` only at K/V
loads. Query positions, RoPE positions, causal/window masks, scores and reduction
order remain absolute and unchanged. No new optional C++ dependency or public ABI.

`check_kernel.py` compares the ring entries bit-for-bit against linear native
attention for 23 cases: widths 64/128/256/512, initial fill, window boundary,
multiple wraps through position 8192, batches 4/8 straddling wrap, host/device
request positions, and unsplit/four-partition decode. Both use the same finite
synthetic F32 query and logical K/V data. Output guard canaries match and all
outputs are finite. Negative controls with a window-only ring fail equivalence
for batched queries, detecting the expected overwritten-key error.

All four CUDA sanitizers pass: memcheck, racecheck, synccheck, initcheck (error
exit code 99). Logs and exact commands are retained. The original CUDA source
chain compiles in NVRTC and C17/C23 host builds pass. This comparison proves
storage-address equivalence to the existing arithmetic, not independent
higher-precision accuracy or real-model ring correctness. No speed timing is claimed.

`model_layout.c` reads the actual Gemma4 12B model and records every layer's head
width, KV head count, window and KV source. At capacity 8194 and maximum batch
512, linear F32 KV occupies 5,638,520,832 bytes. A proposed per-layer ring layout
would use 1,274,478,592 bytes (77.3969% less). This is a calculation from metadata,
not an allocation observed in inference. It does not imply that all optional
weight copies fit after scratch, desktop usage and memory reserves are considered.

## Required runtime integration

- Allocate per-owner cache offsets/capacities. Shared-KV consumers require the
  maximum retention among all consumers; any unbounded consumer requires full
  prefix retention. The tested model has independent owners.
- Retain at least `window + maximum_batch - 1` slots, capped at full capacity,
  when all projected K/V rows for a batch are written before attention. A ring
  of only `window` rows loses keys needed by early queries in that batch.
- Project batched K/V into scratch, apply absolute-position RoPE, then scatter
  into wrapped storage. Existing batched projections assume contiguous output.
  Single-token graph replay must scatter using the current device request.
- Keep attention score capacity separate from ring capacity. The current scores
  remain indexed by absolute positions; reducing that allocation needs a separate
  indexing change. Do not substitute ring capacity for score stride.
- Track retained ranges through overwrite and reset. Deep rewind must replay a
  valid prefix or explicitly use the existing fallback; it must never read
  overwritten slots. Position-zero reset must invalidate prior retained state.
- KV snapshot diagnostics must reconstruct logical order across wrap and reject
  evicted ranges explicitly. The current diagnostic API assumes a linear prefix.
- Integrate or explicitly dispatch appropriate ring-aware FDPP/portable/tiled
  paths; their current contiguous indexing cannot consume wrapped storage.
- Validate full-model logits, cache snapshots, prefill-to-decode, graph replay,
  multiple wraps, reset/rewind, shared KV, memory failure, and realistic batches.
  Only then enable opt-in runtime allocation and run matched short/long timing.

Source/model/binary hashes and build command for the layout audit are in
`manifest.json`. Accepted llama.cpp/PyTorch scores remain unchanged.
