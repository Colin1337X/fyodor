# Integrated MTP decode profile

Profiles revision 1b20e5c after narrow-batch weight reuse, unsplit wide-value
verification attention, and greedy first-proposal early rejection. All eight
generation calls passed exact ordinary-output parity: four modes at each of
25 and 8205 prompt tokens, with 32 generated tokens. Build commands, settings,
core/library/executable hashes and successful exits are in profile.json.

This is a diagnostic, not an accepted llama.cpp comparison. CUDA profiling
disables graphs; there is one observation per mode and no continuous environment
acceptance monitor. Do not use stdout wall times as decode throughput: they also
include setup and prompt ingestion. The standalone core patch enables matched
B512 ordinary prompt ingestion without changing production prefill.

| Context/mode | Matrix GPU ms | Target attention GPU ms |
| --- | ---: | ---: |
| Short ordinary | 690.685 | 29.278 |
| Short serial MTP | 690.757 | 28.884 |
| Short batch 2 | 655.787 | 24.379 |
| Short batch 4 | 635.031 | 18.632 |
| 8K ordinary | 715.985 | 182.008 |
| 8K serial MTP | 716.721 | 181.598 |
| 8K batch 2 | 670.460 | 295.705 |
| 8K batch 4 | 645.979 | 229.049 |

Compared with mtp-target-profile-20261113, 8K batch-2 target attention falls
from 537.334 to 295.705 ms, and batch-4 from 392.034 to 229.049 ms. These separate
profiles locate remaining costs; paired normal-generation experiments in
wide-batch-20261113 and mtp-early-reject-20261113 provide the timing evidence.
The combined changes also reduce verification work: ordinary and serial modes
process 31 post-prefill target inputs, batch 2 processes 34, and batch 4 processes
40. Accepted proposals remain 15 and 20 respectively. Keep rejected work in
comparisons rather than normalizing it away.

Profile thresholds 25/8205 exclude target prefill. Aggregate attention includes
borrowed helper operations while the target profile is active; per-layer target
attention excludes them. The first borrowed helper after below-threshold prefill
is omitted. At 8K the aggregate-minus-target attention durations are approximately
69.8 ms for window 2 and 102.9 ms for window 4. Borrowed wide-value attention is
explicitly disabled here, making its existing opt-in implementation a candidate
for a separately validated full-generation comparison.

Matrix work remains substantial at both contexts, including serial output-head
projection within verification. MTP remains slower than ordinary generation in
the existing paired timings. Accepted Gemma decode ratios remain **85.6103%**
short and **106.8707%** at 8K, below the **125%** target on both workloads.

After profiling, NVIDIA-SMI reported 0% GPU utilization, 736 MiB allocated,
3.44 W and P8. A separate brief Windows GPU Engine counter sample showed only
0.108% Explorer activity above the inspection filter of 0.1%. This is an idle
spot check, not evidence that the profile or earlier throughput runs were clean.
No additional applications were terminated during this inspection; retained
browser, download, networking, OS and development processes were preserved.
