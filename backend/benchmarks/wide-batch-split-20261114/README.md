# Batched split attention prototype

The resident verification path currently uses unsplit wide-value attention
because its partial-result allocation holds only one query. This original
Fyodor experiment gives each query independent four-part storage, then uses
the existing native combiner with a flattened query/head grid. It preserves
the arithmetic of independently launched single-query split attention.

The source is isolated in optcpp/attention/wide_batch_split16.cu and is not
part of any runtime build or dispatch. Raw correctness and diagnostic timings are recorded below; no runtime integration yet.
The prepared harness checks batch two/four, width 256/512, several contexts,
linear/ring storage, exact serial-split parity, output/partial/score guards,
agreement with unsplit attention and an independent float64 oracle. The
existing 0.0005 bound is retained. The GPU checks ran after the paired generation benchmark completed.

All 32 raw cases pass exact serial-split comparison and guard checks. Maximum
scaled difference from unsplit attention is 3.28455e-7; maximum sampled float64
error is 4.42934e-8, both below the unchanged 0.0005 bound.

Kernel diagnostics use 16 query heads, actual Gemma KV-head/width combinations,
three warmup graphs, five observations in each variant order, 200 attention
invocations per observation. Split timings include the combine kernel. Inputs
and KV banks are synthetic and reused; these are not model throughput scores.

| Width | Batch | Position | Window | Unsplit ms | Split ms | Speed % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 256 | 0 | 0.034920 | 0.025261 | 138.239 |
| 512 | 2 | 1024 | 0 | 0.126700 | 0.085079 | 148.921 |
| 512 | 2 | 8192 | 0 | 1.558055 | 0.656696 | 237.257 |
| 512 | 4 | 256 | 0 | 0.041347 | 0.046206 | 89.484 |
| 512 | 4 | 1024 | 0 | 0.159210 | 0.168872 | 94.278 |
| 512 | 4 | 8192 | 0 | 2.146866 | 1.338833 | 160.354 |
| 256 | 2 | 256 | 1024 | 0.022775 | 0.014921 | 152.640 |
| 256 | 2 | 1024 | 1024 | 0.079076 | 0.046360 | 170.568 |
| 256 | 2 | 8192 | 1024 | 0.078994 | 0.046282 | 170.679 |
| 256 | 4 | 256 | 1024 | 0.027075 | 0.028579 | 94.737 |
| 256 | 4 | 1024 | 1024 | 0.095383 | 0.091609 | 104.119 |
| 256 | 4 | 8192 | 1024 | 0.095416 | 0.092814 | 102.802 |

Intermediate-length measurements (same diagnostic contract):

| Width | Batch | Position | Window | Unsplit ms | Split ms | Speed % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 2048 | 0 | 0.249339 | 0.166516 | 149.739 |
| 512 | 2 | 4096 | 0 | 0.495521 | 0.331429 | 149.511 |
| 512 | 4 | 2048 | 0 | 0.317933 | 0.334154 | 95.146 |
| 512 | 4 | 4096 | 0 | 0.654294 | 0.666773 | 98.128 |
| 256 | 2 | 2048 | 1024 | 0.079117 | 0.046396 | 170.527 |
| 256 | 2 | 4096 | 1024 | 0.079011 | 0.046332 | 170.533 |
| 256 | 4 | 2048 | 1024 | 0.095239 | 0.092531 | 102.926 |
| 256 | 4 | 4096 | 1024 | 0.095374 | 0.092121 | 103.531 |

Batch two benefits across this sweep. Batch four full attention still regresses
at 2K and 4K; its 8K improvement does not justify enabling it for all lengths.
Batch-four sliding gains are small. A later integration should retain unsplit
fallback for these marginal/regressing cells and validate whole-model effects.
No cache/occupancy explanation is claimed without hardware-counter evidence.

CUDA Compute Sanitizer 13.1.118 memcheck repeats all 32 cases with zero errors
and exit code zero. Command: compute-sanitizer --tool memcheck
--error-exitcode 99 --target-processes all <python> check_memcheck.py.
Separate memcheck-oracle.json retains the source hash and case results.
Real-model integration, C17/C23 runtime tests and integrated memory checking
remain pending; this source is still excluded from runtime builds.
