# Decode address and packed-load diagnostics - October 15, 2026

Based on committed production source 744da34. The preceding profile still
places fused gate/up at 36.4% of matrix time. Original Fyodor experiments:
key 1 unchanged; key 2 uses a 32-bit base loop index; key 4 uses a 32-bit
row index; key 8 combines both; key 16 loads quantized nibbles through aligned
32-bit words with lane-dependent extraction instead of byte loads. All retain
the four-block compiler bound and unroll eight. No production source changed.

| Rows x columns | base32 | row32 | both32 | packed word |
| --- | ---: | ---: | ---: | ---: |
| 9 x 256 | +4.96% | -0.29% | +2.78% | -0.94% |
| 5632 x 2048 | -4.48% | -0.94% | -5.22% | -6.87% |
| 11008 x 4096 | -3.36% | -0.96% | -3.96% | -6.17% |
| 65 x 768 | -3.83% | -3.39% | -2.63% | -1.87% |
| 257 x 2304 | -13.54% | -1.35% | -13.60% | -0.59% |
| 28672 x 8192 | -2.06% | -0.49% | -2.74% | -5.54% |

Percentages are throughput change versus the same-run original kernel.
Reject all variants: each regresses the target 5632-by-2048 projection and
wider dense shapes. The tiny 9-row speedup does not justify specialization.
Thirty kernel/shape checks matched baseline outputs including guards; all
five variants launch at each shape in forward and reverse order. All samples
are preserved (ten per cell, 50 kernels per graph, ten replays per sample,
three warmup replays). Kernel resources and device limits are archived.

These are hot-weight diagnostics with process snapshots, not monitored
whole-model acceptance timings. Score remains **86.8% short / 107.5% long**
against pinned llama.cpp. A narrow-index production version would additionally
need explicit dimension/overflow guards; none is introduced because it loses.

Follow-up code inspection found residual-add/RMSNorm fusion already exists
for widths <=256. Wider states deliberately use separate kernels because
prior paired tests rejected wider fusion. Do not simply broaden that dispatch
again. A different normalization implementation would need to preserve the
existing reduction order and demonstrate benefit before revisiting it.
