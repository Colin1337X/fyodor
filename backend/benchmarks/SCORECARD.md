# Performance scorecard - decode updated October 16, 2026

Percentages are **Fyodor throughput / reference throughput Ã— 100**. 100% is
parity on that workload, not a percentage of engineering completion.

| Workload | Fyodor | Reference | Relative throughput |
| --- | ---: | ---: | ---: |
| Inference prefill, 512 tokens | 4,595.77 Â± 33.02 tok/s | 15,021.92 Â± 591.61 tok/s | **30.6%** |
| Inference decode, 128 tokens, prefix 0 | 358.02 ± 0.70 tok/s | 384.22 ± 0.75 tok/s | **93.2%** |
| Inference decode, 128 tokens, prefix 1,024 | 308.64 ± 0.29 tok/s | 270.01 ± 1.25 tok/s | **114.3%** |
| Small full-rank CUDA training update | 7.699 Â± 0.029 ms | 5.466 Â± 0.450 ms | **71.0%** |

Decode was rerun October 16; prefill and training remain historical October 12
measurements. Work is focused exclusively on decode until both short and long decode reach at least 125% of llama.cpp throughput.
The current refresh combines query/key RoPE into one launch per eligible decode
layer. Exact logits and KV are preserved across 795 graph decodes; C17/C23
regression passes and all four bounded GPU sanitizers pass. The synthetic
harness needed explicit upload ordering; failed runs are retained. All twelve
monitored baseline activities and eight reference activities passed idle and
contamination checks, with no competitor termination needed.

Paired baseline gains pool to 2.72% short, 1.51% prefix 512 and 0.58% long.
Short baseline variance and small long gains limit precision. Both orders are
positive, but the reverse long gain is only 0.01%; larger-model gains are not
established. Both workloads remain below the 125% target.

[Decode methodology and validation](decode-rope-pair-20261016/README.md),
[decode scores](decode-rope-pair-20261016/summary.json), and
[decode provenance](decode-rope-pair-20261016/reference-20261016/metadata.json).
The [previous refresh](decode-residual-20261016/README.md) retains 92.4% / 113.5%.
Differences across refreshes are not isolated code effects.

Inference uses TinyLlama Q4_K_M against pinned llama.cpp b10809, F32 KV,
reference flash attention off and six CPU threads. Decode uses `-p 0`,
128 generated tokens, and prefix 0 or 1,024; maximum batch is 512
(short-workload capacity limits its effective batch to 128). Each value pools
ten timed samples across two process runs in reversed engine order, after two
warmups per process. Â± is sample standard deviation. This does not represent
the latest or fastest available llama.cpp configuration.

Training uses PyTorch 2.11.0+cu128 eager execution, width 64, FF 176, two layers,
four query heads, two KV heads and context 32. Four paired runs alternate order;
each has 30 updates, excludes ten warmups and the final diagnostic update, and
checks losses, parameters, gradients and AdamW moments at the unchanged 1e-5
tolerance. All passed. Training Â± is standard deviation across four run means.
Throughput ratio is reference mean step time / Fyodor mean step time.
This replaces the October 11 **74.8%** small-workload score. PyTorch paired
ratios ranged from 65.2% to 78.4%; the pooled score uses all four pairs. This
is a fresh measurement, not an isolated estimate of a code speedup or regression.
It does not establish parity for larger models, LoRA, or compiled PyTorch.
The larger training comparison remains rejected by its numerical gate.

Hardware: RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X, Windows. GPU
competitors and NVIDIA overlays were absent at preflight; no process needed
termination or suspension during the accepted reference refreshes. An earlier
fusion diagnostic contaminated by Roblox was excluded and that process stopped. Windows and development UI
remained active. These are **monitored interactive-desktop measurements**, not
headless results. Each activity passed an explicit desktop idle gate and
monitoring found no unapproved GPU process. The earlier October 11 results and
excluded attempts remain available in the historical evidence directory.

[Methodology and reproduction](score-refresh-20261012/README.md),
[machine-readable scorecard](score-refresh-20261012/scorecard.json),
[inference evidence](score-refresh-20261012/inference/metadata.json), and
[training evidence](score-refresh-20261012/training/results.json).

Separate internal LoRA comparison: the [October 12 small-batch scheduling
change](training-narrow-grid-20261012/README.md) improved rank-8 TinyLlama CPT
throughput by 19.6% at context 32 versus the previous Fyodor runtime, with exact
checkpoint/export matches. Rank-2 gains were 1.7% at context 32 and 0.6% at
context 128; context 8 was effectively unchanged. This uses a separately
documented windowed desktop protocol and does not update or increase the
inference/PyTorch percentages above. Training and prefill percentages come from the October 12 refresh; decode
percentages come from the October 15 refresh including aligned decode scale loads.


## October 26, 2026: selective FDPP decode

A fresh matched TinyLlama Q4_K_M comparison measures optional selective FDPP
at **94.70% of llama.cpp on short decode and 114.23% on long decode**.
Native-only in that same run measures 92.74% / 113.97%. The earlier dated
93.18% / 114.31% native result remains historical evidence, not a current rerun.

Selective mode uses four-partition FDPP through position255 and native
attention afterward. Set `NYA_CUDA_FLASH_DECODING_PP=short` in a build with
optional attention enabled. Default inference is unchanged. Short throughput
improves 2.12% versus native; the long-path 0.23% difference is timing variation.

F32 KV, full GPU offload, llama FA off, batch512, six CPU threads, prefixes0/1024,
128 decoded tokens, ten warmups and twenty repetitions in each order. All
12 monitored activities pass; all forty observations per cell are retained.
C17/C23 transition tests, real-model CPU checks and physical-removal build/smoke
pass. The 125% target remains unmet on both contexts.

[Raw results and method](fdpp-selective-20261026/README.md)


## October 27: optional fixed-width decode gate

With selective FDPP plus NYA_ENABLE_FIXED_GATE=ON and NYA_CUDA_FIXED_GATE=1,
matched TinyLlama F32-KV decode is **96.33% short / 115.36% long** of llama.cpp.
Forty observations per cell, ten warmups per process, both orders, all twelve
activities pass monitoring. Generic Fyodor with selective FDPP scores 95.13% /
114.35% in this same trial; fixed-width gains are 1.26% / 0.88%. Default dispatch
is unchanged; only 2048-column Q4 gate/up with matching strides opts in. C17/C23
lifecycle tests, real-model CPU oracles and physical-removal build/smoke pass.
The 125% target on both remains unmet. Full results, variance and limitations:
[fixed-width runtime](decode-fixed-runtime-20261027/README.md).

## Integer scale-product decode runtime (2026-10-27)

Accepted opt-in width-2048 Q4 gate/up optimization in root optcpp. Fresh matched
TinyLlama Q4_K_M/F32-KV results after stopping verified Epic GPU competitors:
**98.97% short / 117.99% long relative to llama.cpp**. Target >=125% both remains
unmet. Fixed baseline 366.000/309.815 tok/s; integer candidate 374.013/316.095;
llama.cpp 377.890/267.903 at prefixes 0/1024, 128 decode tokens. Candidate gains
2.19%/2.03% against the fixed baseline. Forty observations/cell, reverse orders,
10 warmups/process, full host logits per token, full GPU, llama FA off.

All 12 accepted environment gates pass. Tool interruption split the accepted
protocol into nine plus three observations with matching binary/model/dependency
hashes. Rejected competitor runs and the incomplete paired trial are retained
and excluded. C17/C23 six selected tests each, real-model CPU oracles at capacities
129/273, and fresh physical-removal build/inference smoke pass. No tolerances
relaxed. Enable NYA_ENABLE_INTEGER_GATE=ON and NYA_CUDA_INTEGER_GATE=1; remains
off by default, native/fixed fallbacks retained. Training/prefill unchanged.
See decode-integer-runtime-20261027/README.md and reference-summary.json.

## Q4 integer-product row-dot runtime (2026-10-28)

Fresh matched TinyLlama Q4_K_M/F32-KV decode reaches **101.88% short / 120.66%
long relative to llama.cpp**. Both exceed parity; >=125% on both remains unmet.
The accepted integer-gate baseline gives376.147/315.798tok/s; adding Q4 integer
dot/add/QKV gives388.168/323.800; llama.cpp gives380.999/268.351, at prefixes0/1024
and128 decode tokens. Gains versus baseline are3.20%/2.53%. Forty observations
per cell, reverse orders,10 warmups/process, host logits every token, full GPU,
llama FA off. All12 contamination gates pass in one uninterrupted run.

C17/C23 six selected tests each, real-model CPU comparisons at capacities129/273,
and fresh physical-removal build/inference smoke pass without tolerance changes.
Enable NYA_ENABLE_INTEGER_DOT=ON and NYA_CUDA_INTEGER_DOT=1 along with accepted
integer gate/selective FDPP settings. Default off; native fallback preserved;
explicit reference and training contexts excluded; Q6 arithmetic unchanged.
See decode-integer-dot-runtime-20261027/README.md and reference-summary.json.
Training and prefill scores remain unchanged.

## Larger Q6 integer-product runtime (2026-10-28)

Fresh matched TinyLlama Q4_K_M/F32-KV decode reaches **102.59% short /121.41%
long relative to llama.cpp**. >=125% on both remains unmet. Baseline with Q4
integer dot/gate and selective FDPP gives387.403/324.181tok/s; adding larger-Q6
dot/add gives390.452/326.137; llama.cpp gives380.598/268.616, prefixes0/1024 and
128 decode tokens. Gains against baseline are0.79%/0.60%, positive in both
execution orders. Forty observations/cell,10 warmups/process, full host logits,
full GPU, llama FA off. All12 contamination gates pass in one complete run.

C17/C23 six selected tests each, real-model CPU oracles at capacities129/273,
and fresh physical-removal build/inference smoke pass unchanged tolerances.
Enable NYA_ENABLE_INTEGER_Q6=ON and NYA_CUDA_INTEGER_Q6=1 alongside the accepted
Q4/FDPP settings. Optional dot/add wrappers use integer arithmetic only with
rows and columns >=2048; small shapes and QKV retain native Q6 arithmetic.
Default off; missing sources/symbols fall back; training/reference excluded.
See decode-q6-runtime-20261028/README.md and reference-summary.json. Training
and prefill scores remain unchanged. llama short-context variance is retained
in the evidence rather than substituted with a historical reference score.

## Aligned Q4 byte-metadata gate runtime (2026-10-28)

Fresh matched TinyLlama Q4_K_M/F32-KV decode reaches **106.29% short /124.95%
long relative to llama.cpp**. 124.95 remains below125; target on both unmet.
Accepted baseline390.547/325.883tok/s; packed candidate404.749/336.578; llama.cpp
380.815/269.371, prefixes0/1024 and128 decode tokens. Gains3.64%/3.28%. Forty
observations/cell, reverse orders,10 warmups/process, host logits every token,
full GPU, llama FA off. All12 contamination gates pass in a complete run.

Opt-in NYA_ENABLE_BYTE_GATE=ON plus NYA_CUDA_BYTE_GATE=1 adds aligned metadata
copies for width2048 Q4 gate/up pairs. Native weights remain for prefill/fallback:
extra VRAM317,194,240 bytes (302.5MiB) here, not merely11.1% overhead. C owns
packing/admission/cleanup. Insufficient budget or optional allocation failure
preserves resident native execution. Default off; no host C++ requirement.
C17/C23 six selected tests each, independent packing/guard/size tests, real-model
CPU oracles129/273, injected allocation failure,900MiB budget fallback, and fresh
physical-removal build/inference smoke pass. See decode-byte-runtime-20261028.
Training/prefill scores unchanged; preparation cost is excluded from decode timing.


## 2026-10-28: aligned byte-metadata Q4 projections

Matched TinyLlama Q4_K_M/F32-KV decode on RTX 5060 Ti, 40 observations per cell:
short **110.5081%** of llama.cpp (421.5243 vs 381.4420 tok/s), long
**129.4989%** (346.8666 vs 267.8529 tok/s). Same-run gains over accepted
byte-gate baseline: **4.0292% / 3.1136%**. Long reaches the 125% target;
short does not. C17/C23 selected tests, CPU parity, optional-source removal,
and allocation/budget fallback checks passed. Clean run passed all 12 gates
after a pre-timing Roblox competitor rejection and verified process stop.
Additional packed weights cost 505.625 MiB; defaults remain off.
Evidence: [byte projection runtime](decode-byte-dot-runtime-20261028/README.md).
PyTorch training scores unchanged and not remeasured.
