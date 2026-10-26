# Q4 gate/up load experiments, October 26, 2026

A fresh selective-FDPP diagnostic profile reports matrix 988.470 ms,
normalization 128.328 ms, RoPE 42.889 ms, attention 87.617 ms and gather 1.368 ms.
Matrix work is about 79% of that total. Fused Q4 gate/up accounts for 395.772 ms,
about 40% of matrix time. Profiling disables graphs and includes submission
gaps; these numbers locate bottlenecks, not expected end-to-end speedups.

Original optional prototypes retain the baseline accumulation order and SiLU
operation. Variant 1 is native; variant 2 loads each lane's packed 32-bit payload
then redistributes bytes through warp shuffles; variant 4 loads the four scalar
payload bytes earlier; variant 8 stages the shared input vector per block.
The latter synchronizes before partial-block warps return.

Hot-weight checks cover six shapes, including partial blocks and non-power-of-
two widths. Rotating tests use eight distinct gate/up pairs totaling about
99 MiB or 387 MiB, checking every bank against the original kernel. Every
candidate matches outputs and guard regions exactly. Forward/reverse graph
timing captures 50 launches, performs three warmups and five samples of ten
replays per order. Raw observations and GPU process snapshots are preserved.
These diagnostics have no continuous contamination acceptance gate and do not
replace controlled runtime benchmarks.

At 5632x2048, the initial hot experiment measures 42.70 us native, 46.06 us packed,
and 42.24 us with earlier scalar loads. Rotating weights measure 45.30 us native,
47.76 us packed and 44.68 us with earlier scalar loads. A separate shared-input
comparison measures 45.52 us native versus 49.93 us shared with rotating banks.
Larger shapes show the same general pattern; small shapes do not benefit.
All variants use 64 registers and report zero local spill bytes on SM120.

Packed loads and shared staging are rejected for runtime selection. Earlier
scalar loading improves the principal isolated shapes by roughly 1-1.4%, but
has not been integrated or accepted as a runtime gain. No production kernel,
build configuration, arithmetic precision or tolerance changed. The prototypes
remain under optcpp for reproducibility. They are not full FDPP pipeline support.

Current matched selective-decode ratios remain 94.70% short / 114.23% long versus
llama.cpp. Both remain below 125%; no performance score is increased here.
