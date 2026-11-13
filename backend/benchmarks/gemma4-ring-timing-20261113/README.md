# Post-cleanup Gemma ring KV decode diagnostics

These results are **not accepted performance scores**. Before timing, identified
Steam/EA/Epic launcher processes and Spotify processes were stopped under the
user's standing authorization. Names and executable paths were checked; the exact
process list is retained in `stopped-processes.json`. Development, OS,
remote-access, download and unidentified applications were preserved.

The existing unmodified acceptance gate then rejected remaining desktop GPU
identities (`gate/environment.json`). Diagnostics below run separately and never
change or bypass that gate's acceptance decision. They cannot establish the 125%
llama.cpp target or replace accepted scorecard numbers.

`run.py` measures current linear KV, ring KV, and the existing llama.cpp C API
benchmark, first in forward order and then reverse order. Both native variants
use the same Gemma QKV, wide split, short FDPP, byte-dot and fused/partial Gemma
gate options; only `NYA_CUDA_RING_KV` changes. This isolates ring KV's combined
allocation/dispatch effect, including the optional packed weights that fit after
KV shrinks. The linear variant is not the older accepted configuration.

Workloads: Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M, F32 KV, full GPU,
zero or 8192 prefix tokens, 32 timed decode tokens, three repetitions and one
warmup per process. Deterministic token IDs and host logits after each decode are
matched; llama FA is off, batch 512, six threads. Prefix ingestion is excluded
from reported decode timing. All runs are serial. A one-second monitor records
GPU process identities, utilization, memory, power, clocks and CPU activity.

`raw/metadata.json` retains commands, options, build revision, source monitor,
model, binary and library hashes. `summarize.py` pools six observations per
variant/context with sample standard deviation and retains separate order means,
KV/weight/scratch allocations and launch counts. Correctness and actual ring
storage were validated separately in `../gemma4-ring-runtime-20261113/`.

Reproduce from repository root after building `build-cuda/fyodor-bench.exe`:
`python backend/benchmarks/gemma4-ring-timing-20261113/run.py` (requires a fresh
`raw` output directory); then run `summarize.py`. This runner always labels its
outputs diagnostic and never awards acceptance.

## Completed diagnostic results

Six observations per cell; mean tokens/s +/- pooled sample standard deviation.

| Prefix | Linear KV | Ring KV | llama.cpp | Ring / llama | Ring change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 39.93324 +/- 0.26161 | 40.03027 +/- 0.09079 | 46.58917 +/- 0.16478 | 85.9218% | +0.2430% |
| 8192 | 23.35682 +/- 0.24687 | 24.47060 +/- 0.36571 | 21.70119 +/- 0.21816 | 112.7615% | +4.7686% |

At short context the cache remains linear-sized and kernel counts are identical;
the 0.24% difference is not a useful optimization claim. At 8K, ring storage
reduces KV from 5,659,164,672 to 1,275,461,632 bytes and permits all 48 gate/up
pairs to be packed instead of 11. Total weight storage rises from 8,176,566,464
to 10,904,502,464 bytes; scratch rises from 454,210,056 to 462,582,280 bytes.
These allocations are identical across the two orders. Timed launch counts fall
from 92,064 to 84,960 across 96 decode steps (74 fewer per step).

The modest 4.77% long-context gain warrants further decode-only profiling; it
does not justify default promotion from these contaminated measurements. The
accepted scores remain 85.6103% short and 106.8707% at 8K. Neither diagnostic
ratio reaches 125%, and no accepted score or target status has changed.
