# Projection profile after assisted FFN residency

Diagnostic CUDA-event profile of the current target pipeline, with weight reuse,
wide target batches, borrowed wide attention, output projection batching,
assisted FFN residency and greedy early rejection enabled. Uses the same
one-round four-mode harness as mtp-profile-20261114. Graphs are disabled by CUDA
profiling. These event samples are for bottleneck attribution, not throughput
or an accepted llama.cpp score. The separate B512 prompt-ingestion patch affects
only this diagnostic executable. Source, library and executable hashes, commands,
settings, exit codes and full output-parity checks are recorded by run.py.

Both contexts completed, all eight generation calls matched ordinary greedy
output exactly, and the category summary passed. ranked-shapes.json retains
per-shape call counts and milliseconds sorted within each context/mode.

| Context / mode | Matrix GPU ms | Target-layer attention GPU ms |
| --- | ---: | ---: |
| Short ordinary | 690.749 | 29.641 |
| Short serial MTP | 690.718 | 29.683 |
| Short batch 2 | 641.882 | 24.537 |
| Short batch 4 | 604.123 | 18.447 |
| 8K ordinary | 716.483 | 182.005 |
| 8K serial MTP | 717.259 | 181.921 |
| 8K batch 2 | 655.836 | 296.175 |
| 8K batch 4 | 614.869 | 229.131 |

The ordinary short run spends 286.627 ms (41.5% of matrix time) in fused
15360-by-3840 Q4 gate/up, 195.015 ms across Q4/Q6 3840-by-15360 down
projections, and 64.951 ms in the Q6 262144-by-3840 vocabulary head.
Batch-4 verification's largest individual matrix bucket is Q4 gate/up-shaped
weight reuse: 143.260 ms across 672 calls, followed by 111.026 ms of fused
single-token gate/up. These measurements motivate the packed-batch and
reuse-launch experiments alongside this directory.

Aggregate attention GPU time includes borrowed assistant attention when it is
inside the profiling scope; the table uses only per-target-layer events. The
profile is a single observation per mode, disables graphs, and adds event
synchronization. It does not establish the same costs under normal graph
execution and is not a throughput comparison. Accepted Gemma/llama.cpp scores
remain 85.61025294% short and 106.8707255% at 8K; 125% on both remains unmet.
