# Gemma wide-value decode timing, 2026-11-13

These measurements are diagnostics, not accepted performance scores. The unchanged desktop GPU acceptance gate rejected the current environment before timing (`environment.json`). Retained identities include browsers, download software, OS utilities, remote-access software and the current development application. Previously stopped game launchers/music processes were not running at preflight; no additional processes were killed in this pass. The runner never changes acceptance gates.

Compare ring KV with the optional wide value kernel disabled/enabled against the matched llama.cpp C API reference. Only `NYA_CUDA_WIDE_VALUE` changes between native variants. Both use ring KV, byte-dot, Gemma QKV, partial packed Gemma gate/up, wide split and short-context FDPP. The new kernel is inactive on the short graph, so short timing is also a negative control for environment/order noise.

Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M, F32 KV, full GPU, 0/8192 prefix, 32 timed decode tokens, three repetitions and one warmup in each forward/reverse process. Same deterministic token IDs and host logits per token; reference flash attention off, batch 512 and six threads. Prefix ingestion is outside decode timing. No other GPU jobs or compilation are run concurrently. A one-second monitor records GPU identities, power, clocks, utilization, temperature, memory and CPU activity. TUI source edits during timing do not modify the benchmark executable.

`raw/metadata.json` records exact commands, environment and binary/model/library hashes. `summarize.py` pools six observations per cell with sample standard deviation and preserves order means, allocations and launch counts. Kernel sanitizers and exact model parity were established in `../gemma4-wide-value-20261113/` before this run. Reproduce with `run.py` from the repository root after building `build-cuda/fyodor-bench.exe` with `NYA_ENABLE_WIDE_VALUE=ON`; the runner requires a fresh raw directory to prevent overwrites. Run `summarize.py` after all processes finish.

## Completed results

Six observations per cell, mean tokens/s +/- pooled sample standard deviation.

| Prefix | Prior ring tokens/s | Wide tokens/s | llama.cpp tokens/s | Wide / llama | Wide change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 40.05112 +/- 0.32531 | 39.09101 +/- 1.44688 | 46.28506 +/- 0.57514 | 84.4571% | -2.3972% |
| 8192 | 24.27951 +/- 0.39680 | 31.60897 +/- 1.00413 | 22.06691 +/- 0.21621 | 143.2415% | +30.1878% |

Memory allocations and kernel launch counts match between the two native variants in both orders. At 8K the change replaces kernel work without changing topology or cache/weight allocation. Both orders improve materially, but contamination and short-control variability preclude acceptance. The short result is not evidence of a kernel regression: the candidate does not run in that topology. Keep the feature opt-in. Accepted Gemma scores remain 85.61% short / 106.87% at 8K. The 125% target for both workloads remains unmet.

## Native C TUI checkpoint

The same checkpoint adds session-local command recall to the monitor: Up/Down, draft and UTF-8 preservation, bounded sixteen-entry history, consecutive duplicate collapse, no execution until Enter, and Escape cancellation. No shell, persistent history file, or new runtime language/dependency. Six C17 TUI tests, three selected C23 monitor tests, and the state/render test under Clang ASan/UBSan pass; logs are retained here. The console lifecycle test covers the existing resource workspace; new recall behavior is covered by the monitor state test.
