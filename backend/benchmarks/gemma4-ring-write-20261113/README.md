# Wrapped KV writes with absolute-position RoPE — 2026-11-13

Status: native CUDA primitive validated; request allocation and runtime dispatch
remain unfinished. The new `nya_rope_ring` entry reuses the existing RoPE
arithmetic and scatters rotated keys and unmodified values into a ring. Rotation
uses the absolute position; only the storage index wraps. Batched scratch indices
are converted to row-local cache indices, so the batch token offset is not added
twice. The existing `nya_rope` ABI calls the same helper with constant zero ring
capacity. Current production prefill still supplies null cache pointers and
current production decode still has batch one.

Caller requirements: even supported head width, batch <= ring capacity, valid
position/batch bounds, nonoverlapping scratch/cache allocations, and cache buffers
large enough for every slot. Host admission and retained-range tracking are still
required; a kernel launch is not a substitute for validating those conditions.
No C++ dependency, public ABI, allocation policy or runtime option was added.

`check_kernel.py` verifies 24 cases across widths 64/128/256/512, KV heads 1/2/8,
batches 1/7/8/512, both Gemma and interleaved RoPE, host and device request positions,
initial fill and wrap boundaries through position 8190. Rotated scratch matches
the linear kernel bit-for-bit. Full cache buffers match a host scatter of those
rotated results and original values, including every untouched slot and leading/
trailing guard values. Source values remain unchanged and rotated outputs finite.

`check_graph.py` captures one ring-write graph and reuses it at positions
0, 1, 6, 7, 8, 14, 63, 0, 6 with seven slots. Stream-ordered request updates
control the positions; the deliberately wrong captured host position is ignored.
All rotated scratch/cache results are exact, including preserved untouched slots.
Returning to position zero tests address reuse, not complete session-reset
semantics: retained-range validity is still a host integration requirement.

Both harnesses pass CUDA memcheck, racecheck, synccheck and initcheck with
error-exitcode 99 (eight successful sanitizer runs). Commands and raw logs are
retained. GCC C17 and C23 host builds pass. The initial kernel harness completed
its assertions but failed while writing its final source hash because an offset
variable shadowed the source bytes; `oracle-exit.json` retains that failure.
`oracle-fixed.json` and sanitizer logs contain the corrected successful results.

This completes the kernel write dependency identified in the preceding ring
attention audit. Next work is per-owner cache allocation/offsets, batched scratch
sizing, dispatch integration, retained-range/rewind handling, wrapped snapshots,
and compatible optional attention paths. Full-model lifecycle/parity and matched
performance validation remain required. No runtime KV saving or speed gain is
claimed; accepted llama.cpp and PyTorch scores remain unchanged.
