# Current joint-base decode kernel profile

run.py profiles the unchanged, full-model-sanitizer-validated joint-base binary.
It verifies build/evidence hashes and external model identities, rejects competing
GPU workloads, and retains all twelve ordinary/serial2/batch2/joint3 generation
cases in forward/reverse/forward order. All short-context outputs matched.
The 25-position threshold excludes target prefill batches from kernel totals.

Profiling disables CUDA graphs. There is no continuous contamination acceptance
for this diagnostic, so neither wall times nor kernel totals update accepted
llama.cpp scores. Borrowed helper operations are included while target profiling
is active; per-layer attention totals describe target attention separately.

Joint-base matrix totals were 460.060, 460.576 and 461.499 GPU ms, approximately
85% of the recorded aggregate kernel time. Target attention was about 14.9 ms.
In the last joint round the largest matrix group was Q4 batch-four interleaved
weight reuse, 15360 rows by 3840 columns: 132.781 ms across 1728 kernel calls.
Next were serial fused Q4 gate/up at 46.410 ms, batch-four Q6 down projection
(3840 by 15360) at 42.319 ms, Q4 down projection at 38.994 ms, and batch-four Q6
output projection at 23.429 ms. Counts include the implementation's split kernels;
they are not counts of accepted generated tokens.

This identifies batch-four Q4 feed-forward expansion as the next optimization
target. Accepted Gemma decode scores remain 85.61025294% short and 106.8707255%
at 8K versus llama.cpp. The long-context profile has not been run here.

## Optional Q4 integer profile

`run.py short integer` checks the existing integer kernel's full-model oracle
and raw-kernel sanitizer evidence before enabling it in the same joint-base
binary. `short-integer/result.json` records twelve matching generation cases,
three positive selection counts (2556 each), and verified input/evidence hashes.
This combination has not received a full-model sanitizer run with integer mode
active; the joint-base full-model sanitizer used integer mode disabled.

The three joint matrix totals were 448.876, 448.857 and 449.847 GPU ms versus
460.060, 460.576 and 461.499 in the earlier baseline. The dominant Q4 expansion
group measured 126.852, 126.755 and 127.264 ms versus 132.482, 132.635 and
132.781 ms. Mean recorded matrix time decreased about 2.5%, and that expansion
group about 4.3%. These are separate, graph-disabled diagnostic runs, without
continuous contamination acceptance or interleaved baseline/candidate ordering.
They guide further investigation and do not establish a throughput improvement.
The optional kernel remains disabled by default; accepted llama.cpp percentages
remain unchanged.
