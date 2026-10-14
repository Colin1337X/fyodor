# Gate/up loop unrolling runtime trial - October 15, 2026

Original Fyodor experiment, following the committed kernel diagnostic at
`decode-next-profile-20261015`. Only `#pragma unroll 8` is added before the
fused Q4_K gate/up kernel's outer quantization-block loop. Arithmetic order,
reductions, formats, memory layout, launch geometry and graph policy remain
unchanged. No prefill or training optimization is included.

The prior goal turn established a 2.9% isolated TinyLlama-shaped kernel gain,
not a whole-model win. This trial checks the actual runtime before deciding
whether to retain it. `candidate.patch` preserves the complete runtime edit.

## Validation and protocol

The direct-driver oracle compares production fusion against two dot kernels
plus SiLU, byte-for-byte including output guards. Thirty-five shapes cover
1/7/8/9/63/64/65 rows and 256/768/2048/2304/8192 columns; non-multiple-of-eight
block counts exercise compiler-generated remainder handling. Independent CPU
real-model logits retain the existing tolerance; exact fused/unfused GPU
comparison checks graph reuse, reset and interleaved prefix execution.
The pre-existing stricter CPU/GPU KV discrepancy is not resolved by this trial.
Four GPU sanitizers instrument the production fused kernel in the edge oracle.

`compare.py` uses the saved 60a9258 executable and the candidate with identical
TinyLlama Q4_K_M / F32 KV / six-thread settings, 128 generated tokens, prefixes
0/512/1024, two warmups, five repetitions and two reversed-order process pairs.
All ten samples per cell are retained. Independent GPU monitoring and three
consecutive <=30% utilization / <=15 W idle samples gate every activity. All
validation finishes before performance runs. Prefix preparation is untimed.
These are interactive-desktop measurements, not headless results.


## Paired decode results

All twelve activities passed their monitoring and idle gates. Means pool all
ten timed samples per cell; +/- denotes sample standard deviation. The final
long-context control run was noisier; no samples were discarded. Each cell
preserves kernel counts, 640 graph replays, zero timed uploads and 640 returned
host-logit downloads. The baseline executable hash matches the previously
accepted reference binary.

| Prefix | Before tok/s | After tok/s | Mean throughput gain |
| --- | ---: | ---: | ---: |
| 0 | 311.693 +/- 0.402 | 318.392 +/- 0.990 | 2.15% |
| 512 | 294.652 +/- 0.768 | 300.131 +/- 0.611 | 1.86% |
| 1,024 | 272.759 +/- 6.539 | 279.602 +/- 0.584 | 2.51% |

The candidate passed 35 exact kernel edge cases, independent CPU logits,
exact fused/unfused graph checks and all four GPU sanitizer tools with zero
errors/hazards. `validation-exits.json` records their exits. `validate_suites.py`
runs the configured C17/C23 regression suites after performance sampling has
finished, preventing overlap with acceptance timing.

C17 and C23 each passed all 25 configured regression tests, including the
resident training graph smoke test for the shared device compilation unit.
Both builds and both suites exited zero; see `suite-exits.json`.


## Accepted matched reference refresh

All eight final activities passed process monitoring and before/after idle
gates. Each value pools ten timed samples; +/- is sample standard deviation.

| Decode | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefix 0, 128 tokens | 318.182 +/- 0.421 | 384.079 +/- 0.802 | **82.8%** |
| Prefix 1,024, 128 tokens | 279.805 +/- 0.741 | 270.245 +/- 0.393 | **103.5%** |

Retain the unrolling directive: correctness and safety gates pass and full
runtime pairs show improvement. Long decode beats the pinned reference on
this measured workload; short decode is still below parity. No universal win
across models, hardware or llama.cpp configurations is claimed.

Reference is llama.cpp b10809, commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e,
TinyLlama Q4_K_M, F32 KV, full GPU offload, reference flash attention off,
six CPU threads, maximum batch 512, decode-only `-p 0`, two warmups and five
repetitions per process, two reversed-order process pairs per context. Both
engines return host logits every token. Short capacity limits effective batch
to 128; long-prefix preparation is untimed. RTX 5060 Ti 16 GB, driver 617.42,
Ryzen 5 9600X, Windows. No competing GPU workload needed termination during
this acceptance run. Prefill and training scores remain historical.

`reference-20261015/metadata.json` records source, executable, model and library
hashes, commands and environment. `summary.json` validates and pools the full
evidence. Kernel-specific checks and full suites are documented above. The
shared CUDA source rebuilt successfully in both C17 and C23 configurations.
