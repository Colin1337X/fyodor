# Decode Q4_K metadata word loads - October 14, 2026

Original Fyodor implementation. The fused gate/up decode kernel now loads
its 12 packed scale/minimum bytes as three aligned 32-bit words per projection.
Constant shifts extract the same eight scales and minima. Weight quantization,
F32 activations, arithmetic order, reduction, SiLU and graph dispatch stay the
same. CUDA allocations are individually aligned; Q4_K rows and blocks are
multiples of 144 bytes, so metadata at offset four is word-aligned. The runtime
retains its unfused control and all existing format/reference fallbacks.

## Diagnostic experiment

`probe.py` compiles the archived baseline plus metadata-only, quant-value-only,
and combined wider-load variants. The baseline is archived so rerunning the
probe does not accidentally use the later production implementation. Cases
cover 9x256 (partial block), 5632x2048 (TinyLlama) and 11008x4096. All outputs
including guards match the baseline exactly. Graphs contain 50 repeated kernels;
three warmup replays precede five samples of ten replays, repeated in reverse
variant order. All ten samples are in `run/results.json` (milliseconds).

| Rows x columns | Baseline us | Metadata words us | Quant words us | Both us |
| --- | ---: | ---: | ---: | ---: |
| 9 x 256 | 1.399 | 1.440 | 1.523 | 1.471 |
| 5632 x 2048 | 54.649 | 45.198 | 53.489 | 48.443 |
| 11008 x 4096 | 232.733 | 197.015 | 245.238 | 237.007 |

Metadata-only lowers kernel time by 17.3% and 15.3% on the two model-sized
shapes; the tiny case regresses by 0.041 microseconds. Wider quant-value loads
are inconsistent and are not selected. These repeated hot-weight diagnostic
measurements have process snapshots, not continuous monitoring, and cannot
establish whole-model throughput. The initial implementation used generated
byte-extraction expressions; the final implementation simplifies them to four
explicit scale/minimum expressions, followed by fresh correctness validation.

## Validation

The direct-driver production oracle passes 21 edge shapes with exact output
and guards against the original two dot kernels plus SiLU. Final memcheck,
initcheck, racecheck and synccheck pass with zero errors/hazards. Real TinyLlama
checks pass exact fused/unfused logits and KV across 795 checked decodes,
including reset and interleaved prefix execution, retaining two graphs. This
run uses batched prefix preparation in both controls. The independent CPU
logit oracle also passes prefixes 273/272/271 at its unchanged tolerance.
The previously documented stricter CPU KV discrepancy remains unresolved;
GPU-to-GPU exactness does not replace the independent CPU oracle.

`validation-exits.json`, the individual logs, and `kernel-check-final.json`
record these results. `provenance.json` records the baseline commit and source
hashes. Build and configured C17/C23 suite exits are recorded separately.

## Full decode timing

`compare.py` compares the saved e5048e7 executable with the rebuilt candidate,
using the same model, F32 KV, six CPU threads, two warmups, five repetitions,
and 128 generated tokens at prefixes 0/512/1024. Two reversed-order process
pairs per context retain every sample. The independent monitor and three
consecutive <=30% utilization / <=15 W idle samples gate each activity.
Prefix preparation is untimed; no prefill optimization is included.
`summarize.py` checks all activity gates, replay/transfer/launch counts and
pools all ten samples per cell before accepting a result.

The configured C17 and C23 suites each passed 25/25 tests, including the
resident training graph smoke test because the device source is shared.
The before benchmark binary hash matches the accepted October 14 reference
refresh exactly. No builds or tests overlap the acceptance benchmarks.


The twelve paired activities passed monitoring and both idle gates. All ten
samples per cell are pooled with sample standard deviation:

| Prefix | Before tok/s | After tok/s | Throughput gain |
| --- | ---: | ---: | ---: |
| 0 | 278.455 +/- 1.044 | 291.688 +/- 0.504 | 4.75% |
| 512 | 261.701 +/- 8.812 | 277.033 +/- 0.663 | 5.86% |
| 1,024 | 248.659 +/- 0.314 | 259.255 +/- 0.584 | 4.26% |

The prefix-512 baseline is noisier; all of its samples remain included. Each
cell retains the same kernel count, 640 graph replays, zero timed uploads and
640 host-logit downloads. The change introduces no extra device allocations,
packing, host transfers or graph nodes. A separate paired pinned llama.cpp
refresh follows, using the unchanged decode-only contract from the preceding
scorecard. These comparisons remain restricted to this hardware and model.


## Accepted reference scores

All eight final reference activities passed the idle and process-monitor gates.
Means pool ten timed samples per engine/context; +/- denotes sample standard
deviation. The noisier short Fyodor run remains included.

| Decode | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefix 0, 128 tokens | 290.597 +/- 8.786 | 384.650 +/- 0.885 | **75.5%** |
| Prefix 1,024, 128 tokens | 259.695 +/- 0.583 | 270.418 +/- 0.912 | **96.0%** |

100% denotes throughput parity, not engineering completion. Neither workload
has reached the user's >100% target. These are monitored interactive-desktop
measurements, not headless results. No contaminating workload needed stopping
in this experiment. No GPU tests or builds ran alongside these measurements.

Pinned llama.cpp b10809 (5266f24da75dc449bd56cbed7addb9c8e4a6a73e), TinyLlama
Q4_K_M, F32 KV, full GPU offload, reference flash attention off, six CPU
threads, maximum batch 512, decode-only `-p 0`, two warmups and five repetitions
in each of two reversed-order process pairs. Both engines return host logits
every token. RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X, Windows.
Short allocation capacity limits effective batch to 128. Prefix preparation is
untimed. This does not establish parity for other models, hardware or the
latest/fastest llama.cpp configuration. Prefill and training scores are unchanged.

`reference-20261014/metadata.json` records model, source, binary, dependency
hashes, environment and all commands. `summary.json` is the validated pooled
result. The archived diagnostic only establishes the selection rationale;
acceptance rests on correctness gates and the full-model paired runs above.
