# Windows per-process GPU activity diagnostic

`capture.ps1 -OutputPath NEW_FILE.json -Samples 5` records the Windows GPU Engine
counter at one-second intervals. This read-only helper does not stop processes,
change the benchmark gate, or establish performance acceptance. It refuses to
overwrite existing evidence. It records positive engine values without a noise
threshold, counts valid instances and retains invalid instance statuses. PID
names are best-effort lookups and may be unavailable if a process exits.

Keep instance names, including adapter LUID, physical adapter and engine type.
Percentages belong to individual engines; do not sum them into whole-GPU load.
All adapters are included. No LUID-to-NVIDIA-device mapping is implemented.
Missing/invalid instances do not demonstrate inactivity, and the first sample
may not span a full interval. This script currently uses the English counter
path; unavailable/localized counters fail visibly rather than implying idle.

`sanitizer-active.json` contains five samples during the full-Gemma memcheck.
Each sample had 705 valid counters and zero invalid statuses. The verification
process (PID 38812) reported 93.58-98.08% on its CUDA engine. Remaining observed
activity belonged to System/Explorer, with the largest individual engine sample
below 0.3%. This identifies the active workload during this short observation;
it does not certify a clean benchmark interval or prove sanitizer completion.
The existing conservative benchmark acceptance rules remain unchanged.

Windows counter API references: [formatted wildcard counter arrays](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw)
and [displaying performance data](https://learn.microsoft.com/en-us/windows/win32/perfctrs/displaying-performance-data).
The helper is original benchmark tooling, not a runtime dependency.
