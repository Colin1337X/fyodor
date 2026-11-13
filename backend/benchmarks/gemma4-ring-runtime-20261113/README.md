# Gemma 4 CUDA ring KV runtime integration

`NYA_CUDA_RING_KV=1` opts into bounded sliding-window storage. Default remains
linear. Host layout/allocation/retention code is C; CUDA kernels stay under CUDA,
and optional Flash-Decoding++ source stays under optcpp. No additional dependency.

Each owning layer retains min(sequence capacity, maximum consumer window +
maximum prefill batch - 1) rows. Any full-attention consumer forces full storage.
Shared consumers reuse the owning offset and must have matching KV dimensions.
Memory-budget batch reduction recomputes both cache slots and batched K/V scratch.
Score indexing stays absolute. Ring layers project into contiguous batch scratch,
then scatter during absolute-position RoPE. All allocation precedes graph capture.

Ring layers dispatch native ordinary/four-way attention. The short decode
graph may retain existing FDPP/portable dispatch when ring slots are at least
256 and a separate long graph guarantees positions below 256; these early rows
are still physically contiguous. Full-storage layers retain existing selection.
Ring prefill does not use the existing linear tiled-prefill kernel. This may affect speed; this experiment does not establish a speedup.
The maximum write position tracks eviction across rewinds; position-zero execution
starts a new prefix. Rewinds needing evicted rows return failure before submitting
work. Existing caller recovery remains responsible for replay. Diagnostic KV reads
reject evicted rows and reconstruct retained ranges across wrap in sequence order.

## Actual model results

RTX 5060 Ti (SM120), Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf,
model SHA256 59656d7494d6376ca97e9e20b64ea2e16cd97f12ec6d47bfccba91cb785b5134.
GCC C17, maximum batch 512, F32 KV, native attention, Gemma QKV enabled.
`results.json` records the complete NYA environment and executable hash.

| Capacity | Linear KV bytes | Ring KV bytes | Final logits |
| --- | ---: | ---: | --- |
| 3 | 2,064,384 | 2,064,384 | exact |
| 2,049 | 1,409,974,272 | 1,073,119,232 | exact |
| 8,194 | 5,638,520,832 | 1,274,478,592 | exact |

At 8K the actual KV allocation is 77.3969% smaller. Batched scatter adds 8,372,224
scratch bytes at batch 512; this is not a total-VRAM percentage. Both runs use
7,365,558,464 weight bytes in the isolated layout comparison.

`model_oracle.c` compares all finite logits and suppression masks, the last 1,024
KV rows (including a wrapped read), rejection of evicted rows, and graph reuse
following reset to positions 0..2. `rewind_oracle.c` compares valid rewinds and
reset directly through the plan API, rejects a rewind to position 1024 after a
2049-token prefix, and then proves the same plan remains usable.

`test_cuda_kv_layout.c` covers shared owners, different consumer windows, full
consumers, invalid layouts, size overflow, exhaustive small-capacity batch/window
nonaliasing, and retention boundaries. Registered as `cuda-host-kv-layout`, it
runs without a CUDA compiler or GPU (C17, C23, and ASan/UBSan pass).

The first model attempt could not initialize CUDA: two optional FDPP fallback
calls lacked the ring helper's new final zero argument. `initial-results.json`,
`initial-model-3.log`, and `compile-failure.log` retain that failure. Both calls
are fixed; later full-model results above require resident execution explicitly.

These runs are correctness/allocation diagnostics with other desktop GPU activity
present. They are not accepted performance measurements. Existing llama.cpp
percentages and the unmet 125% short/long target remain unchanged.

Reproduce from the repository root: compile the harness with the matching
`build-cuda/libnya-engine.a`, then run `python backend/benchmarks/gemma4-ring-runtime-20261113/run.py`.
`run_rewind.py` runs the separately compiled rewind oracle; `run_optimized.py`
uses the C23 binary with FDPP short, byte-packed dot, Gemma gate, and partial gate
packing enabled. Inspect each result file rather than assuming optional paths ran.

The first optimized reset comparison failed exact equality because the ring path
used native attention while the linear path used short FDPP. The initial optimized
log/result are retained. Preserving the safe early contiguous dispatch fixed the
2K reset comparison without changing any numerical tolerance.

Direct-plan rewind checks passed. Physical removal of non-C components passed all
80 tests (four expected unavailable ROCm/MLX hardware skips), including training,
export/reload and TUI tests. Raw output: `build-cpu/ring-removal.log`.

C23 optimized Gemma results: 2K and 8K final logits, wrapped snapshots, and reset
all match bit-for-bit. At 8K, packed gate/up weights increase from 811,008,000 to
3,538,944,000 extra bytes (11 to all 48 layers), with total weight allocations
8,176,566,464 versus 10,904,502,464 bytes. The two decode steps save 148 launches.
This is dispatch/allocation evidence, not a throughput result.

Both registered CUDA ring suites passed across nine synthetic shapes with graphs
on and off, including shared KV, window 1/17, widths 32/64/128/130/256, and split
attention. Their native CPU comparison tolerances are unchanged. The test also
asserts actual KV allocation, so silently selecting linear storage cannot pass.
`manifest.json` records source and freshly verified model hashes. GPU activity is
captured in `gpu-environment.txt` for diagnostic provenance, not timing acceptance.

CUDA Compute Sanitizer memcheck passed the complete nine-shape graph suite with
zero reported errors (`memcheck.log`, `memcheck-results.json`). This run includes
the explicit ring-allocation assertion. Kernel-level racecheck/initcheck/synccheck
coverage is retained in the earlier ring read/write experiments; those tools were
not rerun on this full-model host integration.
