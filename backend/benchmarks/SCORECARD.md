# Performance scorecard — October 11, 2026

Percentages are **Fyodor throughput / reference throughput × 100**. 100% is
parity on that workload, not a percentage of engineering completion.

| Workload | Fyodor | Reference | Relative throughput |
| --- | ---: | ---: | ---: |
| Inference prefill, 512 tokens | 4,614.81 ± 25.31 tok/s | 15,091.22 ± 509.78 tok/s | **30.6%** |
| Inference decode, 128 tokens, prefix 0 | 258.58 ± 4.39 tok/s | 370.97 ± 1.69 tok/s | **69.7%** |
| Inference decode, 128 tokens, prefix 1,024 | 233.08 ± 4.52 tok/s | 257.75 ± 4.23 tok/s | **90.4%** |
| Small full-rank CUDA training update | 7.388 ± 0.197 ms | 5.525 ± 0.262 ms | **74.8%** |

Inference uses TinyLlama Q4_K_M against pinned llama.cpp b10809, F32 KV,
reference flash attention off, batch 512 and six CPU threads. Each value pools
ten timed samples across two process runs in reversed engine order, after two
warmups per process. ± is sample standard deviation. This does not represent
the latest or fastest available llama.cpp configuration.

Training uses PyTorch 2.11.0+cu128 eager execution, width 64, FF 176, two layers,
four query heads, two KV heads and context 32. Four paired runs alternate order;
each has 30 updates, excludes ten warmups and the final diagnostic update, and
checks losses, parameters, gradients and AdamW moments at the unchanged 1e-5
tolerance. All passed. Training ± is standard deviation across four process
means. Throughput ratio is reference mean step time / Fyodor mean step time.
This replaces the previous **65.4% provisional** small-workload score; it is a
fresh environment measurement, not evidence of a corresponding code speedup.
It does not establish parity for larger models, LoRA, or compiled PyTorch.
The larger training comparison remains rejected by its numerical gate.

Hardware: RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X, Windows. Nonessential
NVIDIA overlays were stopped and their user controller temporarily suspended,
then resumed after measurement. Essential Windows and development UI rendering
remained active. These are **monitored interactive-desktop measurements**, not
headless results. Each activity passed an explicit desktop idle gate and
monitoring found no unapproved GPU process. Earlier contaminated or biased
attempts are excluded and retained for inspection.

[Methodology and reproduction](score-refresh-20261011/README.md),
[machine-readable scorecard](score-refresh-20261011/scorecard.json),
[inference evidence](score-refresh-20261011/inference/metadata.json), and
[training evidence](score-refresh-20261011/training/results.json).
