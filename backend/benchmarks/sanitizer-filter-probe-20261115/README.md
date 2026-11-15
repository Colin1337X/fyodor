# Sanitizer filter selectivity probe

Prepared only; GPU execution has not yet been performed. Do not run concurrently with the model diagnostic.

Two named PTX kernels each intentionally write four bytes at offset 16 into a four-byte allocation in their own disposable CUDA process. This is an invalid access fixture, not a production kernel. It uses no model data. Compare memcheck with no filter, regex=nya_attention_cached_query, and a deliberately nonmatching filter. The selected fixture must produce a precise invalid-write report and error exit; the excluded fixture must produce no instrumented invalid-write report. An excluded fixture completing does not make its code memory-safe.

Use the same installed sanitizer as the model run. Preserve commands, exit status, stdout, stderr, executable version, and source hashes. Also check the unrelated kernel without a filter to prove its invalid access is detected when instrumentation is enabled. Do not infer filter behavior merely from CLI help or a zero-error summary. No performance or candidate-correctness acceptance follows from this fixture.
