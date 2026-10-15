# Q4_K decode dot metadata - October 14, 2026

Original Fyodor implementation, extending the validated fused gate/up metadata
word-loading technique to the remaining Q4_K dot projections. Three aligned
32-bit loads replace twelve byte fields. Constant shifts reconstruct identical
scales/minima; quant values, arithmetic order, F32 accumulation and reduction
are unchanged. CUDA tensor allocations are individually aligned, and the
144-byte Q4_K blocks and row strides preserve four-byte alignment. Reference
arithmetic and other formats keep their existing implementations. No allocation,
packing, upload, synchronization, graph policy or prefill kernel is added.

The prior goal turn made progress by validating and publishing the fused
metadata change. Fresh `before-profile/` measurements identify the remaining
Q4_K dot projections as 595.45 of 1279.76 short-context matrix milliseconds
(46.5%). Profiling disables graphs and includes submission gaps, so these
figures identify targets rather than accepted throughput.

## Diagnostic and correctness method

`probe.py` compiles the archived baseline and the candidate row-dot helper,
then compares raw output including guards. Graphs contain 50 copies of the
kernel, with three warmup replays followed by five samples of ten replays.
Forward and reverse execution order retains ten samples per shape/kernel.
All five shapes matched exactly. `graphs/results.json` contains all samples.
These hot-weight diagnostic timings have process snapshots, not continuous
monitoring, and cannot update whole-model performance scores.

| Rows x columns | Baseline microseconds | Candidate microseconds |
| --- | ---: | ---: |
| 9 x 256 | 1.021 | 0.957 |
| 256 x 2048 | 3.784 | 3.259 |
| 2048 x 2048 | 13.107 | 10.492 |
| 2048 x 5632 | 34.384 | 27.235 |
| 11008 x 4096 | 117.396 | 90.101 |

`check_kernel.py` separately compiles the production source and an archived
unmodified baseline dot helper. Twenty-one edge shapes use random Q4_K bytes,
finite scales and F32 inputs, covering 1/7/8/9/63/64/65 rows and 256/2048/8192
columns; every output and guard matches byte-for-byte. This checks the changed
dot kernel directly, independent of the gate/up implementation. GPU sanitizers
instrument the production `nya_dot_12` kernel in this harness.

The independent CPU real-model logit check retains its established tolerance.
The separate exact graph test compares fused/unfused execution within the new
runtime; it checks graph integration, not old/new whole-model equivalence.
The previously documented stricter CPU/GPU KV snapshot discrepancy remains
unresolved. No tolerance was relaxed.

## Whole-model measurement

`compare.py` compares the saved 59807c6 binary with the rebuilt candidate,
using TinyLlama Q4_K_M, F32 KV, six CPU threads, two warmups, five repetitions,
128 generated tokens and prefixes 0/512/1024. Two reversed-order process pairs
per context pool all ten samples. Each activity has independent GPU-process
monitoring and three consecutive <=30% utilization / <=15 W idle samples
before and after. Builds and correctness tests finish before timing starts.
Prefix preparation is untimed; no prefill optimization is performed.

`reference.py` repeats the matched pinned llama.cpp comparison with the same
decode-only contract as the preceding scorecard. `summarize.py` validates
process gates, exits, graph/transfer/launch counts and pools all samples.

Validation completed with all four sanitizer tools reporting zero errors or
hazards. C17 and C23 each passed 25/25 configured tests, including resident
training graph smoke coverage for the shared device source. Timing resumed
on October 15; the directory retains its experiment start date.

## Accepted before/after comparison

All twelve monitored activities passed their before/after idle gates, process
checks and exits. Kernel counts, 640 graph replays, zero timed uploads and
640 host-logit downloads remain unchanged. All ten samples per cell are pooled;
+/- is sample standard deviation. The prefix-512 baseline was noisier and its
samples remain included.

| Prefix | Before tok/s | After tok/s | Throughput gain |
| --- | ---: | ---: | ---: |
| 0 | 293.047 +/- 0.909 | 313.002 +/- 0.760 | 6.81% |
| 512 | 274.409 +/- 9.911 | 294.953 +/- 0.711 | 7.49% |
| 1,024 | 259.601 +/- 0.619 | 274.448 +/- 0.609 | 5.72% |

## October 15 matched llama.cpp refresh

All eight monitored activities passed idle gates and process checks. Means
pool all ten timed samples per cell; +/- denotes sample standard deviation.

| Decode | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefix 0, 128 tokens | 312.648 +/- 0.279 | 384.357 +/- 1.541 | **81.3%** |
| Prefix 1,024, 128 tokens | 274.193 +/- 3.158 | 270.289 +/- 0.294 | **101.4%** |

The long-context mean narrowly beats the pinned reference on this workload.
One Fyodor process had greater variance (4.385 tok/s sample deviation); its
samples remain included. This is not a universal or hardware-independent win.
Short-context decode remains below parity and decode remains the priority.

Pinned llama.cpp b10809, commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e;
TinyLlama Q4_K_M, F32 KV, full GPU offload, reference flash attention off,
six CPU threads, maximum batch 512, decode-only `-p 0`, two warmups and five
repetitions in two reversed-order process pairs. Both return host logits each
token. Short allocation capacity limits effective batch to 128. Prefix
preparation is untimed. RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X, Windows.
These are monitored interactive-desktop measurements, not headless results.
No competing GPU activity required termination in this acceptance run.

`reference-20261015/metadata.json` records source/binary/model/dependency hashes,
commands and environment. `summary.json` pools the validated evidence. Prefill
and training scores remain historical and were not remeasured or optimized.
