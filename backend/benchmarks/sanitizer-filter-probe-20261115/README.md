# Sanitizer filter selectivity probe

Executed after the model diagnostic exited and the GPU returned to idle. All five filter-selectivity cases passed. Do not run concurrently with another GPU evaluation.

Two named PTX kernels each intentionally write four bytes at offset 16 into a four-byte allocation in their own disposable CUDA process. This is an invalid access fixture, not a production kernel. It uses no model data. Compare memcheck with no filter, regex=nya_attention_cached_query, and a deliberately nonmatching filter. The selected fixture must produce a precise invalid-write report and error exit; the excluded fixture must produce no instrumented invalid-write report. An excluded fixture completing does not make its code memory-safe.

Use the same installed sanitizer as the model run. Preserve commands, exit status, stdout, stderr, executable version, and source hashes. Also check the unrelated kernel without a filter to prove its invalid access is detected when instrumentation is enabled. Do not infer filter behavior merely from CLI help or a zero-error summary. No performance or candidate-correctness acceptance follows from this fixture.

The matching kernel and unfiltered unrelated kernel each produced an invalid-write report and exit 99. The excluded, deliberately nonmatching, and substring-excluded cases each exited zero with a zero-error summary. This demonstrates selection in the isolated fixture, not absence of all overhead in a model workload. It does not explain the roughly 250-fold model slowdown. Raw logs retain additional error reporting/cleanup details for the deliberately invalid cases. Both kernels also assembled offline with CUDA 13.1 ptxas for sm_120.
