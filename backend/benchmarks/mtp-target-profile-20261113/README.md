# MTP decode GPU profile after utility cleanup

The existing generation harness is restricted to one round of ordinary,
serial-window2, batch-window2 and batch-window4 calls. run.py links it with the
standalone candidate core object and current engine library. It uses a separate
executable after the weight-reuse timing pair has completed.

Nine verified noncritical utility clients were stopped after the paired timing
and before profiling; process-cleanup.json records identities and timestamps.
GPU utilization then reported 0% and memory use 749 MiB. Editors, terminals,
browsers, downloads, OS services and development tools remain running. This is
not a clean-room throughput measurement and does not promote accepted scores.

Set NYA_CUDA_PROFILE=1 and NYA_CUDA_PROFILE_FROM=25 for the short prompt or
8205 for the long prompt, plus the recorded B512 generation options and
NYA_CUDA_SMALL_BATCH_REUSE=1. Keep NYA_CUDA_MTP_WIDE_VALUE=0 initially to isolate
the current target-verification cost. Profiling disables graphs, so wall times
from this run cannot replace normal throughput or llama.cpp scores.

Interpretation matters: target-plan aggregate GPU totals also include bridge
hidden/borrowed-attention helpers while its profile_active flag is set. Borrowed
attention has no layer attribution (profile_layer=0), unlike target attention.
The first bridge call after below-threshold prefill is not recorded. Use per-layer
attention totals and matrix path/batch rows to assess verification; do not label
the aggregate as exclusively target-transformer work. Assistant wall-time logs
include its host operations and transfers, rather than just CUDA event time.

Check for small-batch-weight-reuse matrix entries to establish actual resident
kernel dispatch. Investigate the count>1 basic-attention path before extending
wide-value attention to batches; split buffers currently hold decode-sized
partials and cannot safely be reused across query batches unchanged.

## Results

Both contexts passed exact output/token/stop parity for all four modes. Matrix
profile rows confirm resident `small-batch-weight-reuse` dispatch at batches 2/4.
summary.json separates per-layer target attention from aggregate bridge helpers.

| Context/mode | Matrix GPU ms | Target attention GPU ms |
| --- | ---: | ---: |
| Short ordinary | 690.833 | 29.502 |
| Short batch 2 | 722.436 | 26.642 |
| Short batch 4 | 688.519 | 19.714 |
| 8K ordinary | 716.773 | 181.699 |
| 8K serial MTP | 716.948 | 181.586 |
| 8K batch 2 | 737.058 | 537.334 |
| 8K batch 4 | 698.945 | 392.034 |

These are accumulated event durations for one diagnostic generation, not
per-token throughput. Ordinary processes 31 post-prefill transformer inputs;
batch 2 processes 38 and batch 4 processes 44, including rejected proposals.
Do not normalize away that extra verification work. The aggregate 8K attention
category also contains roughly 77 ms of borrowed helpers for window 2 and
114 ms for window 4; it is not exclusively target-layer attention.

The next candidate is unsplit wide-value attention for narrow verification
batches. Existing scratch reserves at least the fewer-than-eight query fallback,
but the split partial/output layout remains single-query. Restrict a candidate
to measured batches 2/4, preserve explicit positions and ring addressing, then
validate linear/ring KV against native and float64 results before timing it.
Short-context matrix work and assistant overhead remain separate bottlenecks.
Accepted Gemma/llama.cpp ratios remain 85.6103% short and 106.8707% at 8K.
