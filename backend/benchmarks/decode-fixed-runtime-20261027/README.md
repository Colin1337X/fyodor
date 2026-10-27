# Optional fixed-width Q4 gate/up runtime, October 27, 2026

Original Fyodor kernel specializes only the 2048-column Q4 gate/up pair in
runtime dispatch. C host code selects it only with both 1152-byte row strides,
while the existing single-token Q4 fusion eligibility checks remain in force.
All other shapes use the generic kernel. No arithmetic, accumulation order,
precision, weight allocation, KV format or graph topology changes.

Build with NYA_ENABLE_OPTCPP=ON and NYA_ENABLE_FIXED_GATE=ON (default OFF),
then set NYA_CUDA_FIXED_GATE=1 before creating the CUDA context. Absent source
or symbol retains generic dispatch. Device code stays in root optcpp/matvec;
NVRTC embeds/compiles it without a host C++ compiler. The source also retains
unselected diagnostic widths, which the runtime never dispatches. Native
inference remains the default. No upstream code was copied.

C17 and C23 each pass the six selected lifecycle tests, including selective
FDPP boundaries, fallback, reset/prefix replay and disabled graphs. These small
fixtures predominantly exercise generic-width fallback. Real TinyLlama CPU
oracles at capacities129 and273 exercise the specialized path and pass existing
tolerances. Earlier isolated kernels match native output/guard bytes on six
shapes. No tolerance was changed. A fresh source copy physically lacking optcpp,
CUTLASS and CuTe builds and runs real-model CUDA inference while the option and
environment request are enabled; no C++ compiler is configured. This is removal
build/smoke coverage, not a rerun of every training/packaging removal test.

Initial paired trial: TinyLlama Q4_K_M, F32 KV, six CPU threads, batch512, 128
decoded tokens, prefixes0/1024, selective FDPP in both variants. Full host logits
each token; four warmups/eight repetitions per process, reverse orders, 16 pooled
observations per cell. All eight activities pass idle/competitor monitoring.

| Prefix | Generic tok/s | Fixed tok/s | Gain |
| --- | ---: | ---: | ---: |
| 0 | 363.928 +/- 1.156 | 367.028 +/- 1.971 | +0.85% |
| 1024 | 308.186 +/- 0.858 | 311.144 +/- 0.953 | +0.96% |

Gains occur in both orders. Register/weight/memory-layout changes are absent;
constant row width allows compiler specialization. This does not imply the
isolated +2.93% rotating-kernel gain applies to the whole model. Wider fixed
kernels lost and remain unselected. This is a modest optional improvement;
it does not achieve the required 125% of llama.cpp on both decode workloads.

The reference.py/summarize_reference.py follow-up uses ten warmups and twenty
repetitions in forward/reverse baseline/fixed/llama orders. The pinned llama
revision is 5266f24da75dc449bd56cbed7addb9c8e4a6a73e, full GPU/F32 KV, FA off.
Exact commands, source/binary/model/library hashes, all process-level aggregate measurements, CPU/GPU
activity, clocks, temperatures and power are retained. No builds or tests run
during timing. RTX5060Ti, driver617.42. See reference-summary.json when complete
for the new matched scores; never combine old reference and new candidate runs.

## Matched result

All twelve reference activities pass the unchanged idle/competitor gates. Forty
observations per cell are pooled from two process aggregates without exclusions.
The larger short-baseline variance is retained; no fastest-run selection.

| Prefix | Generic Fyodor tok/s | Fixed tok/s | llama.cpp tok/s | Fixed / llama |
| --- | ---: | ---: | ---: | ---: |
| 0 | 363.035 +/- 5.014 | 367.615 +/- 1.404 | 381.622 +/- 2.494 | 96.33% |
| 1024 | 307.004 +/- 1.035 | 309.708 +/- 1.334 | 268.481 +/- 0.776 | 115.36% |

The matched gains are +1.26% short and +0.88% long relative to generic Fyodor
with selective FDPP. Accept the guarded specialization as opt-in only. Both
orders and the earlier paired trial favor it, with no changed numerical contract.
The scores are 96.33% short / 115.36% long of llama.cpp. The 125% target remains
unmet. These dated results supersede neither historical native-only measurements
nor the earlier selective-only comparison; they describe this optional combination.
