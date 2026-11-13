# Selective batched split attention

The previous resident verification path used unsplit wide-value attention
because its partial-result allocation held only one query. This original
Fyodor experiment gives each query independent four-part storage, then uses
the existing native combiner with a flattened query/head grid. It preserves
the arithmetic of independently launched single-query split attention.

The source is isolated in optcpp/attention/wide_batch_split16.cu. It is now
integrated behind default-off NYA_ENABLE_WIDE_BATCH_SPLIT and
NYA_CUDA_WIDE_BATCH_SPLIT switches, with native C planning and dispatch.
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
Batch-four sliding gains are small. The experimental integration retains unsplit
fallback for these marginal/regressing cells; whole-model effects are unproven.
No cache/occupancy explanation is claimed without hardware-counter evidence.

CUDA Compute Sanitizer 13.1.118 memcheck repeats all 32 cases with zero errors
and exit code zero. Command: compute-sanitizer --tool memcheck
--error-exitcode 99 --target-processes all <python> check_memcheck.py.
Separate memcheck-oracle.json retains the source hash and case results.
C17/C23 runtime validation each passes 31 targeted cases, including batch
limits 1/2/3/4/8, ring and linear KV, and 8K batch-four verification. Scratch
accounting is checked against a disabled reference. Physical removal passes
the 82-test suite (four expected hardware skips) and CPU train/export/infer.
Real-model and integrated memory validation are in progress.

## Integration validation status

The four prefix-257 Gemma cases pass. The first prefix-8192 case (linear KV,
two verification tokens) matches the serial-after-prefill logits exactly,
but fails the unchanged 0.0005 bound when comparing replacement-token replay
against a fully serial 8K prefix. validation.json and the complete failure log
are retained. The remaining 8K cases have not run, and full-generation timing
is gated on all eight cases passing. A feature-disabled control with quantified
replay error ran through diagnose_replay.py and reproduced the failure with
maximum scaled error 0.00261755031 (index 157813: actual 0.0598487109 versus
fully serial 0.0626301989). The new split path was disabled in both sessions.
The unsplit-attention control also fails at 0.00235611498 (same index), with
wide split and short FDPP disabled while retaining the matrix path. Its
initial verification matches exactly. Thus changing those attention paths
does not remove the replay mismatch. A native-matmul control is prepared but
has not run. This integration is not
validated for acceptance; neither tolerance nor reference behavior was relaxed.
The prefix_kv.c diagnostic (compiled cleanly under C17/C23) compares a batched
prefix with fully serial ingestion inside one resident session, snapshots
keys/values at positions 0, 4096 and 8191 for every layer, and records final
prefix logits. It uses the same synthetic token sequence as the replay test.
The control completes with final-prefix logit max scaled error 0.00420031703
(max absolute 0.00556683540), before any rewind or replacement. Layer-zero
maximum sampled value error is 1.90541e-6; the largest sampled value difference
is 0.00361178 at layer 44, position 4096. All 144 KV rows and binary/source
hashes are retained in prefix-kv-control.json; the summary retains all layer
maxima. This localizes a batched-versus-serial prefix difference without proving
which operation is responsible. summarize_prefix.py reports measured KV
differences without defining or relaxing an acceptance tolerance.
The native-matmul prefix probe completed with cuBLAS disabled; its serial
path and attention settings otherwise match the control. Final-prefix logit
max scaled error remains 0.00334618502 (max absolute 0.00426483154), above
the unchanged 0.0005 bound. The largest sampled value difference rises to
0.0172257 at layer 44, position 4096 (absolute 0.0727386). Disabling vendor
matmul therefore does not resolve the discrepancy; the smaller final logit
error alone is not evidence that native matmul is more accurate. All 144
rows, the full log, hashes and layer summary are retained separately.

The unsplit-attention prefix probe is in progress. Five PDH samples during
this diagnostic attribute all positive target-adapter engine counters to
its PID 16572; no competing app was killed. This short observation is not
full-run contamination clearance or benchmark acceptance. An additional
unfused-projections mode is prepared to disable Gemma QKV and gate/up fusion
while preserving the control's matrix and attention settings; it has not run.
