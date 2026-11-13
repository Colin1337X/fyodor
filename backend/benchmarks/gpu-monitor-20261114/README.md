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

An acceptance policy still needs explicit measured intervals, sample coverage,
gap/invalid-data rejection, PID lifetime checks, CPU/thermal conditions and
controlled cross-engine comparisons. Missing counters and first/new-instance
samples are not proof of idleness. Do not sum per-engine percentages. System
process activity must be interpreted conservatively, not silently allowlisted.

Primary API documentation:
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhaddenglishcounterw

The former documents formatted wildcard arrays and the two-call sizing pattern;
the latter documents language-neutral counter registration. The collector is
original Python ctypes tooling, not copied Microsoft sample source. CUDA device
identity comes from the existing gpu-activity-20261114/cuda_identity.py helper.
