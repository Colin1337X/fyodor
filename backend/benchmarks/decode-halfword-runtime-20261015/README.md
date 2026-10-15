# Aligned FP16 scale loads: runtime trial - October 15, 2026

Original Fyodor implementation following `decode-q6-loads-20261015`. The prior
goal turn measured modest candidate gains and archived the diagnostic. This
trial adds `nya_half_aligned` only to Q4_K/Q6_K row-dot scale reads and the
fused Q4_K gate/up reads. It loads one unsigned short, then uses the same
`cvt.f32.f16` instruction as the original byte-assembly helper. CUDA tensor
allocations and block/row offsets retain two-byte alignment, including Q6_K's
210-byte blocks. The generic helper and other callers remain unchanged.
Reference-math builds call the existing reference helper. No quantization,
accumulation sequence, graph policy, buffer allocation or transfer is changed.

## Validation

`check_kernel.py` compares both load paths for all 65,536 FP16 bit patterns,
including signed zeros, subnormals, infinities and NaNs. Output float bits must
match exactly. The input starts two bytes into its device allocation, ensuring
that four-byte alignment is not accidentally required. It also compares fused
Q4 output with two production dots plus SiLU on 35 edge shapes. `check_q6.py`
compares production Q6 dots with the archived original row-dot implementation
on the same 35 shapes. Each includes output guards, odd row counts and columns
256/768/2048/2304/8192. All outputs match byte-for-byte.

`validate_checks.py` runs the independent real-model CPU logit oracle with its
unchanged tolerance, exact fused/unfused graph integration, and all four GPU
sanitizer tools on both direct-driver harnesses. Sanitizer instrumentation has
no kernel-name filter, covering the new load and its consumers. The stricter
pre-existing CPU/GPU KV discrepancy remains unresolved; exact GPU comparison
does not replace that oracle. Full C17/C23 suite validation is separate.

## Performance protocol

The saved 137a7f0 executable is the before control. `compare.py` measures both
binaries with identical TinyLlama Q4_K_M, F32 KV, six CPU threads, two warmups,
five repetitions, 128 generated tokens and prefixes 0/512/1024. Two reversed
process pairs retain ten samples per cell. Each activity has an independent
GPU monitor and three consecutive <=30% utilization / <=15 W idle readings
before and after. No builds or validation run concurrently with timing.
Prefix preparation is untimed; this is decode optimization only. Results are
monitored interactive-desktop measurements, not headless performance claims.


## Paired results

All twelve activities passed process monitoring and both idle gates. The
baseline binary hash matches the previous accepted reference executable.
Means pool all ten samples per cell; +/- is sample standard deviation. Short
and prefix-512 controls were noisier; no sample was removed. Both versions
retain the same kernel counts, 640 graph replays, zero timed uploads and 640
host-logit downloads.

| Prefix | Before tok/s | After tok/s | Throughput gain |
| --- | ---: | ---: | ---: |
| 0 | 314.091 +/- 3.923 | 324.170 +/- 0.540 | 3.21% |
| 512 | 297.225 +/- 2.954 | 305.185 +/- 0.620 | 2.68% |
| 1,024 | 279.407 +/- 0.460 | 284.998 +/- 0.724 | 2.00% |

Both real-model checks exited zero. All four sanitizer tools passed on both
harnesses (eight instrumented runs), reporting zero errors/hazards. Validation
exits and raw logs are preserved. `validate_suites.py` completes configured
C17/C23 regression coverage after timing, without overlap.

C17 and C23 each passed 25/25 configured tests, including resident training
graph smoke coverage for the shared CUDA compilation unit. Both builds and
suites exited zero; see `suite-exits.json`.


## Accepted matched reference refresh

All eight final activities passed independent monitoring and both idle gates.
Means pool ten timed samples per cell; +/- denotes sample standard deviation.
One short Fyodor run was noisier (6.331 tok/s sample deviation), and its samples
remain included.

| Decode | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefix 0, 128 tokens | 323.217 +/- 5.065 | 384.279 +/- 1.289 | **84.1%** |
| Prefix 1,024, 128 tokens | 284.870 +/- 0.883 | 269.960 +/- 0.462 | **105.5%** |

Retain the aligned scale-load helper: correctness and safety gates pass and
paired full-runtime measurements show improvement. Long decode beats this
pinned reference on the measured workload; short decode remains below parity.
No general win across models, hardware or llama.cpp configurations is claimed.

Reference: llama.cpp b10809, commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e,
TinyLlama Q4_K_M, F32 KV, full GPU offload, reference flash attention off,
six CPU threads, maximum batch 512, decode-only `-p 0`, two warmups and five
repetitions per process, two reversed-order process pairs per context. Both
engines return host logits each token. Short capacity limits effective batch
to 128. Prefix preparation is untimed. RTX 5060 Ti 16 GB, driver 617.42,
Ryzen 5 9600X, Windows. No competing workload required termination during this
acceptance run. Prefill and training scores remain historical and unchanged.

`reference-20261015/metadata.json` records model/source/executable/library
hashes, environment and commands. `summary.json` validates and pools the
complete evidence; all raw outputs and monitors are retained.
