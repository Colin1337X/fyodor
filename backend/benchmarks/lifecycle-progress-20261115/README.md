# Live lifecycle progress diagnostics

The original 8K filtered memcheck was stopped at the user request on 2026-11-15. Its child exit code is nonzero and candidate execution is absent. A zero-error sanitizer summary on this interrupted run does not constitute a pass. The original result remains rejected and retained under model-attention-query-cache-20261115/model-cached-memcheck-c23-8192.

Use `python -B backend/benchmarks/lifecycle-progress-20261115/run_v2.py c23 8192 NewLabel` for an explicitly started diagnostic. Add `--memcheck` for the same filtered candidate-kernel sanitizer scope. The long run has NOT been restarted. NewLabel must be unique; outputs are never overwritten. Build with build_v2.py only in a fresh build directory.

The separately named optimized C17/C23 executables report:
- Flushed operation begin/end markers mapped to the original lifecycle source lines.
- Prefill and batch start/completion: completed tokens, total tokens, percentage, elapsed monotonic seconds, and average tokens/second for that prefill call.
- Every prefill layer begin/completion, with position, layer number/count, and elapsed batch time. Completion follows GPU stream synchronization. Layer progress is not a count of completed tokens.
- Serial token completion at position 1 and every 32 tokens: completed position, session capacity, time for that token, and instantaneous tokens/second. This rate is for the sampled token, not an average for all preceding tokens.
- A 10-second runner heartbeat even when the child emits nothing: total elapsed time, PID, last progress event, and age of that event. A heartbeat means the supervisor is responsive, not that the GPU made progress.

Logs contain no prompts, token IDs, logits, or model weights. Each phase has its own progress scope; percentages are not an overall completion estimate. No ETA is fabricated. Live stdout/stderr are streamed to the terminal and saved separately. Redirect the runner output if persistent heartbeat records are desired, as the retained checks do.

These are diagnostic binaries. Added layer synchronizations alter scheduling, so their rates must not be promoted into benchmark scores. Original arithmetic and graph checks remain; original kernels and production files are unchanged. This Windows diagnostic timer uses QueryPerformanceCounter.

Real Gemma checks at prefix 257 pass original reset/rewind/numerical checks under both C17 and C23. Both include multiple prefix batches, layer completion, serial position 32, and 10-second heartbeats. Build and run manifests retain hashes. The initial SmokeC17 run passed model checks but its wrapper incorrectly expected four batched prefill summaries: there are three, while the replacement reference pass uses serial execution. That rejected result and initial runner remain retained; v2 fixes the inventory and adds serial progress. No new 8K candidate validation or sanitizer pass is claimed.
