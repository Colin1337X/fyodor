# Performance scorecard - decode updated October 16, 2026

Percentages are **Fyodor throughput / reference throughput × 100**. 100% is
parity on that workload, not a percentage of engineering completion.

| Workload | Fyodor | Reference | Relative throughput |
| --- | ---: | ---: | ---: |
| Inference prefill, 512 tokens | 4,595.77 ± 33.02 tok/s | 15,021.92 ± 591.61 tok/s | **30.6%** |
| Inference decode, 128 tokens, prefix 0 | 354.82 ± 0.58 tok/s | 383.95 ± 1.15 tok/s | **92.4%** |
| Inference decode, 128 tokens, prefix 1,024 | 306.47 ± 0.37 tok/s | 269.97 ± 0.62 tok/s | **113.5%** |
| Small full-rank CUDA training update | 7.699 ± 0.029 ms | 5.466 ± 0.450 ms | **71.0%** |

Decode was rerun October 16; prefill and training remain historical October 12
measurements. Work is focused exclusively on decode until it exceeds parity.
The current refresh fuses eligible decode projections with residual addition,
removing another 44 launches per TinyLlama token without extra buffers or changed
arithmetic. Initial plus confirmation comparisons pool to +1.87% short throughput
(20 samples each); prefix 512 improves 1.13% and long 0.91% (ten each). Short
baseline variance limits precision, and every observation is retained. Both
orders improve in both short rounds. A wider isolated projection regressed
0.7%; gains on larger real models are not established. Short remains below
parity, so optimization stays focused on decode.

[Decode methodology and validation](decode-residual-20261016/README.md),
[decode scores](decode-residual-20261016/summary.json),
[pooled short baseline comparison](decode-residual-20261016/pooled-short.json), and
[decode provenance](decode-residual-20261016/reference-20261016/metadata.json).
The [previous refresh](decode-qkv-20261015/README.md) retains 91.2% / 112.4%.
Differences across refreshes are not isolated code effects.

Inference uses TinyLlama Q4_K_M against pinned llama.cpp b10809, F32 KV,
reference flash attention off and six CPU threads. Decode uses `-p 0`,
128 generated tokens, and prefix 0 or 1,024; maximum batch is 512
(short-workload capacity limits its effective batch to 128). Each value pools
ten timed samples across two process runs in reversed engine order, after two
warmups per process. ± is sample standard deviation. This does not represent
the latest or fastest available llama.cpp configuration.

Training uses PyTorch 2.11.0+cu128 eager execution, width 64, FF 176, two layers,
four query heads, two KV heads and context 32. Four paired runs alternate order;
each has 30 updates, excludes ten warmups and the final diagnostic update, and
checks losses, parameters, gradients and AdamW moments at the unchanged 1e-5
tolerance. All passed. Training ± is standard deviation across four run means.
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
