# Raw/formatted PDH failure diagnostic

This independent observer retains raw values and formatted-value bit patterns
from the same PDH query after each collection. It stops on a collection or
formatted-array API failure, retaining both API call statuses and the raw array
from that collection when available. It does not convert raw counters to rates,
repair invalid samples, attribute System/PID 4 activity, or accept performance.
The benchmark collector and coverage rules are unchanged.

Run from the engineering checkout:

    python backend/benchmarks/pdh-raw-diagnostic-20261114/capture.py --samples 900 --label UNIQUE

The output directory must not exist. Each sample is flushed. The final manifest
records source/log hashes and input immutability. The initial SanitizerA run is
an observer diagnostic during the active Q6 full-model sanitizer, not a clean
inference timing workload. It starts no competing GPU kernel or application.
Raw provider failures remain explicit in the log even when formatting succeeds.

`abi.c` checks the Windows header layout against the Python ctypes declarations.
GCC C17 O3 strict-warning compilation and the comparison passed: raw structure
size 40, timestamp/first/second/multi offsets 4/16/24/32, array item size 48,
raw-value offset 8. `abi.json` retains the checked source and executable hashes.
The first compiler invocation lacked the toolchain bin directory in PATH and
failed to find the assembler; adding that directory resolved the build.

Original diagnostic code based on documented APIs and structures, not copied
sample implementation:

- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetrawcounterarrayw
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_raw_counter
- https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_raw_counter_item_w

The two-call sizing pattern is used, with bounded re-sizing on MORE_DATA.
The diagnostic does not read failed array buffers as valid measurements.

`analyze.py COMPLETED_FOLDER TARGET_LUID` verifies a completed capture's hashes
and sample inventory before retaining raw-delta/formatted agreement, out-of-range
values, instance churn and API/status failures. It refuses live/incomplete logs.
Four manual CPU checks covered a 150% raw/format match, invalid status exclusion,
API failure with disappearing instance, and malformed adapter identity. Its
incomplete-capture refusal was also exercised. Agreement alone is not acceptance.

## Completed SanitizerA observation

All 900 samples completed with unchanged inputs, no API failures and no invalid
raw/formatted statuses. The adapter had 64,728 consecutive matched counters;
the largest difference between formatted values and 100*delta(first)/delta(second)
was 2.842170943040401e-14. There were 299 readings above 100%, maximum
145.20836230841672%. No instance-set changes occurred during this capture.
This reproduces the out-of-range symptom in raw data but not the earlier
formatted-array API failure. It proves neither actual utilization nor causal
attribution of System activity; acceptance policy remains unchanged.

The 102,416,913-byte raw log is retained in Git as `SanitizerA/samples.jsonl.gz`.
`archive.json` records both hashes and sizes; lossless decompression was checked
byte-for-byte before committing. Decompress beside the archive as `samples.jsonl`
to rerun the analyzer or verify the original manifest. The uncompressed live log
remains locally available. The analysis and capture manifests retain original
source and raw-log hashes; no samples were omitted from the compressed artifact.
