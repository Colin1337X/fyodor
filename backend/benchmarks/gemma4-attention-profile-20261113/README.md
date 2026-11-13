# Per-layer decode attention profile, 2026-11-13

Extends the scoped CUDA event profiler with layer index, head width, query/KV heads, window, kernel count, and time. Split attention includes the combine kernel. No change to normal execution. Allocation failure retains aggregate profiling. Entries aggregate all selected batch sizes; use scope selection to isolate decode.

Gemma 4 12B Q4_K_M, F32 KV, 8192-token prefix, 16 decode tokens, no warmup, graphs disabled by profiling. `run.py` preserves exact command, environment, binary/source hashes, raw output, and desktop activity. `summary.json` groups the 48 layers: full-attention width 512/KV1 accounts for 192.286/261.929 ms in linear KV and 194.911/247.729 ms in ring KV. Forty sliding layers account for the remainder. Every layer reports 32 kernel events (attention and combine across sixteen tokens). Layer sums match aggregate attention within printed rounding.

Timing brackets can include host submission gaps. These are diagnostic event totals, not normal throughput or proof of a specific hardware bottleneck. No accepted performance scores change.

`scope_oracle.c` and `check_scope.py` pass in C17/C23: exact logits across resets, threshold exclusions, invalid-option fallback, and exact per-layer event counts for the fixture. There are no events for fully excluded scopes.
