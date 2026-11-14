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

## Native-generation reproduction

`capture_native.py --samples 900 --label NativeA` is the same raw/formatted
collector with accurate native-workload metadata. It runs independently beside
`model-q6-pair-20261114/run_raw_diagnostic_generation.py long c17`. That workload
retains the validated generation runner's prerequisites, output parity, graph
reuse, execution markers and source hashes, and writes a separate evidence
folder. The observer starts first and records QPC timestamps; native generation
also records QPC intervals. The observer stops on the first API failure or its
sample limit; native generation may continue afterward. Neither process nor its
artifacts certify coverage or performance. This arrangement is for capturing
raw values on the collection where formatted data fails, not replacing the
benchmark collector or retrying failed measurements into acceptance.

Both scripts passed syntax compilation. NativeA and its native-generation
workload have now completed; results follow below. Existing accepted benchmark
rules and prior manifests are unchanged. The observer launches no GPU kernels.

## NativeA completed counter capture

All 900 samples completed without an API-level failure and with unchanged
inputs. This did not reproduce the formatted-array API failure. Analysis found
64,350 valid consecutive target-counter pairs, maximum raw-delta/formatted
difference 1.4210854715202004e-14, and 424 readings above 100% (maximum
121.51024201281837%). There were three instance-set changes and 36 invalid
formatted statuses; every raw status was valid.

The 36 invalid formatted entries were all newly appearing target-engine
instances: 18 at sample index 19 and 18 at index 21, with formatted status
0xc0000bba and raw status 0. The index-19 set disappeared at index 20; the
index-21 set belongs to the recorded native workload PID 12144. The other PID
21852 is not causally identified. This observation is consistent with missing
prior rate samples for new instances, but does not explain the earlier global
formatted-array API failure or identify System/PID 4 activity. Coverage rules
are unchanged. `status-churn.json` retains the individual entries and changes.

The full 118,841,104-byte raw log is retained losslessly as NativeA/samples.jsonl.gz
(3,965,869 bytes); archive.json records both hashes and sizes, and byte-for-byte
decompression was verified. Generation continued after the bounded capture;
the capture cannot certify full-workload coverage or performance.

## Completed native-generation workload

The independent C17 long-context workload exited successfully and passed all
twelve ordered generation cases with 8205-token prompts and 32 output tokens.
Q6 paired-row execution markers were [44,44,44], and six graph plans recorded
reuse. All 849 recorded input/output hashes verified after completion. Evidence
is retained under model-q6-pair-20261114/raw-diagnostic-generation-long-c17.
The bounded observer ended before generation; this is correctness and counter
diagnostic evidence, not full-workload coverage or an accepted timing result.

## Engine and clock audit

`clock_audit.py` verifies the completed capture manifest and sample inventory
before comparing raw denominator deltas, FILETIME timestamp deltas and QPC
collection-start deltas. It retains per-engine positive/out-of-range counts;
it does not change the collector or coverage policy. Four manual offline checks
passed: 125% preservation with matching clocks, invalid-status exclusion,
wrong-adapter exclusion, and nonmonotonic-clock rejection.

All 424 NativeA readings above 100% belong to the owned native process PID 12144,
CUDA engine 10. All 299 SanitizerA readings above 100% belong to the owned
instrumented process PID 5364, also CUDA engine 10. System/PID 4 copy engine 13
had no readings above 100%: NativeA had 17 positive samples, maximum 7.65145%;
SanitizerA had 505, maximum 8.48787%. These counts cover each entire bounded
capture, not just timed decode intervals.

For matched valid target counters, denominator deltas exactly match raw
FILETIME timestamp deltas. Maximum relative disagreement with QPC intervals
was 0.13970% for NativeA and 0.12957% for SanitizerA. This evidence does not
support a large collection-clock denominator error as the explanation for
121.51%/145.21% CUDA readings. GPU running-time accounting and System causal
attribution remain unresolved. No accepted performance scores change.

Microsoft documents that PDH_FMT_NOCAP100 preserves values exceeding 100%;
that flag is already used. This generic PDH behavior alone does not establish
that these particular GPU engine values are valid utilization measurements:
https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcountervalue

## Formatted-first diagnostic

capture_formatted_first.py reverses the same-query read order: formatted values
are read before raw values, including a raw read after a formatted API failure.
This tests whether reading raw counters first affects failure reproduction.
It does not modify the benchmark collector or relax coverage acceptance.

FormattedFirstA completed 120 idle samples without API failures or invalid
statuses. All 6426 valid target-counter pairs agreed exactly between formatted
values and raw deltas. No inference workload overlapped this capture, so it
cannot establish loaded coverage or show that changing read order fixes it.
The raw log is preserved losslessly as samples.jsonl.gz with archive hashes.

FormattedFirstLoadedA is a separate 900-sample capture started before the
model-q4-quad-20261115/run_formatted_first_generation.py short c17 workload.
The generation runner retains its numerical, execution-marker, graph-reuse and
input-hash checks. Both artifacts explicitly reject timing acceptance. They
are independent processes; the capture may end before generation completes.
Loaded results must be inspected after completion before drawing conclusions.
