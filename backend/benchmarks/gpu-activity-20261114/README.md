# Windows per-process GPU activity diagnostic

`capture.ps1 -OutputPath NEW_FILE.json -Samples 5` records the Windows GPU Engine
counter at one-second intervals. This read-only helper does not stop processes,
change the benchmark gate, or establish performance acceptance. It refuses to
overwrite existing evidence. It records positive engine values without a noise
threshold, counts valid instances and retains invalid instance statuses. PID
names are best-effort lookups and may be unavailable if a process exits.

Keep instance names, including adapter LUID, physical adapter and engine type.
Percentages belong to individual engines; do not sum them into whole-GPU load.
All adapters are included; capture.ps1 itself does not filter by NVIDIA device.
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

`python cuda_identity.py NEW_FILE.json` independently reads CUDA device name,
PCI bus ID, eight-byte Windows LUID and node mask through the driver. It creates
no CUDA context and performs no GPU allocation or kernel launch. The recorded
cuda-identity.json maps `luid_0x00000000_0x00013020` to the RTX 5060 Ti at
`0000:01:00.0`, matching the active verification CUDA engine in the captured
samples. The Explorer 3D engine uses a different LUID. Match within the same
boot/session; ordinal numbers are not durable IDs, and multi-node adapters need
their node masks retained. API failures are explicit, not treated as idle.
See NVIDIA's [device-management API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).

Windows counter API references: [formatted wildcard counter arrays](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw)
and [displaying performance data](https://learn.microsoft.com/en-us/windows/win32/perfctrs/displaying-performance-data).
The helper is original benchmark tooling, not a runtime dependency.
