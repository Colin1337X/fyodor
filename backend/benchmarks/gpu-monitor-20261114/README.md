# Adapter-attributed GPU monitoring diagnostics

No acceptance gate is changed. No old timing is promoted. These collectors
produce evidence; they do not establish clean-run acceptance by themselves.

## PowerShell attempt: rejected for continuous process discovery

capture.ps1 writes flushed JSONL with every target-adapter counter (including
zero/invalid status), positive other-adapter counters, observer CPU/QPC data,
errors and a terminal record. It refuses overwrite and supports a stop file.
Nine offline checks pass in selftest-fixed/result.json. Initial selftest/
artifacts and selftest-rejected.ps1 retain a test-harness variable-scope error.

The live test starts the CUDA workload after collection begins. All workload
correctness checks passed, but its PID never appeared among target counters.
The target instance count remained 378. live.jsonl/live-validation.json retain
the failure. Do not infer idle GPU or clean execution from this collector.
The fixed instance inventory is consistent with wildcard expansion at query
startup; this is an inference from observed behavior, not an OS implementation
audit. The PowerShell collector is unsuitable for the proposed continuous gate.

## PDH wildcard-array replacement

pdh_counter.py uses PdhAddEnglishCounterW and PdhGetFormattedCounterArrayW,
retaining the wildcard handle and enumerating its formatted instance array on
each collection. Buffer sizing retries are bounded; invalid statuses and
nonfinite values remain explicit. capture_pdh.py flushes each sample, includes
collection latency and observer CPU time, stops cooperatively, and records
provider/close errors with a nonzero exit. PID/LUID matching is case-normalized.

validate_pdh.py starts the monitor, waits for two samples, then starts a known
CUDA process with a ten-second zero-weight projection workload. Nine positive
counter observations match that PID on the fresh CUDA-identified RTX adapter
within the child's recorded execution interval. Workload and monitor exit zero;
stop is acknowledged and the terminal record is successful. pdh-live.jsonl,
pdh-identity.json, pdh-validation.json and pdh-workload.log are retained.
This demonstrates dynamic discovery on this Windows host; other systems/locales,
short-lived competitors, all error cases and production monitoring are not yet
validated. This workload is not a performance benchmark.

An acceptance policy still needs end-to-end matching of measured intervals, sample coverage,
gap/invalid-data rejection, PID lifetime checks, CPU/thermal conditions and
controlled cross-engine comparisons. Missing counters and first/new-instance
samples are not proof of idleness. Do not sum per-engine percentages. System
process activity must be interpreted conservatively, not silently allowlisted.

`coverage.py` now audits a measured QPC interval against recorded child PID
lifetimes. It requires matching PDH adapter identity, samples bracketing the
interval, monotonic timestamps, bounded sampling gaps, valid target counters,
and successful cooperative termination. Positive engines must overlap a known
workload lifetime; engine percentages are never summed. Fourteen offline tests
cover complete evidence and rejection cases (`coverage-tests.log`).

The earlier live attribution trace does **not** pass this stronger audit:
`pdh-coverage.json` retains invalid target counters and unattributed activity.
The latter comes from PID 4 (System); this is not a reason to terminate it.
The invalid counters are retained for PID 6368 with status 0xc0000bba. Dynamic
discovery worked, but that alone did not prove uncontaminated execution.
The auditor always returns `accepted_benchmark: false`, even for a clear sampled
interval. It cannot exclude missing short-lived competitors or PID reuse, and
does not check CPU activity, thermal state or equivalent workloads. No previous
timing is promoted by this change.

## Native decode intervals

NYA_GENERATION_PROFILE=1 now prints a separate `Generation interval` record
alongside the unchanged historical duration line. Both use the identical start
and end clock reads; output occurs after the timed interval. The clock is QPC
on Windows and CLOCK_MONOTONIC on other supported platforms. Invalid/nonfinite
or zero-duration measurements are explicitly marked invalid. This remains the
generation-loop contract (including the first prefill-logit sample), not the
fixed-forward llama.cpp benchmark contract. No prompt text is logged.

The mtp-cost runner now records the child PID, spawn bounds and observed-exit
time in the observer's QPC clock on Windows. Those are process lifetime bounds,
not exact creation/exit instants. Together with the native decode endpoints,
they supply inputs for the existing sampled-coverage audit. They do not change
the environment gate or automatically start/validate the PDH collector.

decode_intervals.py checks sequential duration/interval pairs, identity,
clock, finite positive intervals, ordering and matching elapsed time (allowing
only decimal-format rounding). Seven offline tests reject missing/duplicate
records, mismatched calls/clocks/durations, nonfinite or reversed timestamps,
and overlap. Ordinary native correctness assertions and speculative-generation
assertions pass with profiling off/on under strict C17 and C23 CPU execution.
The v2 validation records 4340 positive C17 intervals and 4325 positive C23
intervals, all bracketed by the Python parent's QPC observations. It explicitly
rejects 19/34 one-token synthetic calls that completed within a single clock
tick. Such calls cannot become throughput samples. Profiling-off emits no
intervals. These CPU runs occurred while GPU sanitizer work was active and are
correctness tests only.

The initial validation is preserved: the parser rejected zero-duration native
records even though their first implementation marked them valid. The runtime
now marks equal endpoints invalid; the validator explicitly checks that those
synthetic cases are rejected, instead of discarding their evidence silently.
decode-interval-validation-v2.json and the associated stdout/stderr files retain
the final observations. Incomplete decode-interval-validation.json is not a pass.
No measured performance ratio changes, and production monitor/benchmark
orchestration, CPU/thermal checks and equivalent llama.cpp reruns remain open.

## Generation-to-monitor evidence join

audit_generation.py joins a completed mtp-cost timing directory to a PDH capture.
It requires unique short/long contexts, successful child exits, QPC lifetime
bounds, all twelve expected cases in forward/reverse/forward order, output
parity, 32 generated tokens, and exactly one matching native interval per case.
It rejects timestamps outside their child lifetime, overlapping cases, mismatched
prompt counts, missing endpoints and ambiguous PID reuse. It then applies
coverage.py to each exact decode interval. Input files are hashed in the result;
invalid/incomplete evidence produces a nonzero exit and retained failure JSON.
Even complete sampled coverage leaves accepted_benchmark false. The historical
environment gate result is retained separately and is never overridden.

Eleven synthetic contract tests pass, including correct interval pairing,
competitor activity during decode, missing monitor termination, incorrect case
order, parity failure and unbounded child lifetimes. A counter interval ending
before decode is outside this sampled decode-coverage check; possible thermal
effects remain a separate acceptance concern. The initial test log retains a
test-fixture error that placed the alleged overlapping competitor before decode.
The corrected suite also tests that distinction explicitly.

The CLI rejects the existing b512-baseline timing metadata because it lacks
child lifetime evidence (`historical-generation-rejected.json`). The supplied
old PDH capture is not contemporaneous with that timing run and must not be
used to certify it. No historical result is upgraded. Live benchmark collector
orchestration and a contemporaneous joined run are still outstanding.

Primary API documentation:
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhaddenglishcounterw

The former documents formatted wildcard arrays and the two-call sizing pattern;
the latter documents language-neutral counter registration. The collector is
original Python ctypes tooling, not copied Microsoft sample source. CUDA device
identity comes from the existing gpu-activity-20261114/cuda_identity.py helper.
