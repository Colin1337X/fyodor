# Read-only diagnostic progress summary

`python -B backend/benchmarks/model-compile-time-memcheck-20261115/progress_summary.py backend/benchmarks/model-compile-time-memcheck-20261115/LongC23/stderr`

This reader reports completed tokens for the active prefill separately from layers in its current batch. A layer's `total=512` is a batch size, not a replacement for the prefill's `total=8192`; completing a layer does not count its tokens as completed. The last completed-batch rate remains explicitly diagnostic. Sampled serial token rates are labeled instrumented, and session capacity is not treated as the amount of requested work.

No overall completion percentage or ETA is calculated. A log cannot establish process liveness, so the summary always reports process_liveness=not_checked; use the original execution-session handle or an authoritative process check separately. The snapshot hash identifies only the bytes read, not a stable completed log or safety/performance acceptance artifact.

Incomplete trailing records are ignored because the writer may still be emitting them. Malformed or inconsistent records are counted. No model data or log files are modified. Existing runners, numerical checks, benchmark gates and the active GPU process are untouched.

Tests cover distinct prefill/batch scopes, layer transitions, partial writes, invalid records and sampled-token semantics. Replay checks against completed C17 prefix-5, C23 prefix-5 and C23 prefix-257 logs find exactly three completed prefills each and no malformed records. This is a progress interpreter, not a replacement for the strict long-context validator.
