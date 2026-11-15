# Model sanitizer overhead isolation

Four sequential C23 runs used the same diagnostic executable, Gemma Q4_K_M model, environment, prefix 5, and unchanged numerical bound. No GPU jobs overlapped; GPU was idle before and after the sequence. These are one repetition per condition, with no reverse-order trial, and are not accepted throughput benchmarks.

| Condition | Whole lifecycle seconds | Ratio to baseline |
|---|---:|---:|
| No sanitizer | 8.1177404 | 1.00 |
| Memcheck, deliberately no matching kernel | 168.4620308 | 20.75 |
| Memcheck, cached-query name filter | 168.6806534 | 20.78 |
| Memcheck, no matching kernel, stream-ordered race tracking off | 168.7842972 | 20.79 |

All four original numerical/reset/rewind and progress inventories pass. All three sanitizer commands exit zero with a unique zero-error summary. All retained input hashes validate. run_v3.py adds an explicit filter control; run_v4.py adds an explicit stream-race control. Both reject reduced-check options without --memcheck before starting a child or creating evidence. Existing runners and their hashed artifacts remain unchanged. summarize_overhead.py verifies input evidence and generates overhead-comparison.json without overwriting existing results.

Substantial overhead persists with a deliberately nonmatching filter and with stream-ordered allocation-race tracking disabled. Selecting cached-query kernels is therefore not necessary to reproduce this slowdown, and disabling this race-tracking option does not materially remove it in this small experiment. The isolated fixture separately established name-filter selectivity. These observations do not identify the remaining mechanism; do not claim a specific instrumentation, JIT, driver or hardware bug.

Prefix 5 is a fallback context and does not exercise the long cached-query candidate. No-match and reduced-check commands provide diagnostic controls, not candidate memory-safety evidence. None satisfies the still-failed interrupted 8K candidate gate. Diagnostic binaries add layer synchronizations and must not supply benchmark scores. The different whole-run ratios at prefix 5 and prefix 257 include different work amounts and fixed costs; do not extrapolate them to production decode.

The installed tool's --help documents --track-stream-ordered-races with default all and value no. Current upstream documentation was also consulted: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html . No binary-patching environment override or newer-version-only compiler instrumentation was applied.

Accepted short/long decode scores remain 85.61025294% / 106.8707255% of llama.cpp; the target remains at least 125% for both. All four jobs are terminal. No long GPU validation was restarted.
