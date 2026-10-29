# Gemma wide-head split attention: long-context follow-up

Production change: 6851425, exact binaries/dependencies/model hashes in the
metadata. This runs the existing opt-in wide split implementation, not TurboQuant
or the 512-wide portable attention changes being developed alongside this test.
RTX 5060 Ti 16GB, Gemma4 12B Q4_K_M, F32 KV, full GPU, batch 512, six CPU threads,
32 decode tokens following 4096/8192 deterministic prefix tokens. One warmup and
three measurements per process, both execution orders (six observations/cell).
llama.cpp b10809 C API, FA off. Byte packing is budget-skipped at these contexts.

| Prefix | Native baseline tok/s | Wide split tok/s | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|---:|
| 4096 | 18.97081 ± 0.01741 | 27.89756 ± 0.02347 | 31.16875 ± 0.01708 | 89.505% |
| 8192 | 14.57291 ± 0.00199 | 23.98581 ± 0.08605 | 22.44376 ± 1.20447 | 106.871% |

Pooled sample standard deviations include within-process and between-order
variation. Individual timing samples were not exported. The 8K llama.cpp run
has appreciably higher variation; the ratio is a descriptive mean, not a
confidence-bound guarantee. Neither context reaches the 125% target.

The first ten activities passed monitoring. The unfinished 8K candidate reverse
activity was rejected when Thorium appeared as a competing C+G process. That
monitor trace is retained; its timing was not returned by the monitor wrapper
and no rejected numeric result is included. A relaunched Roblox process and its
crash helper were also verified and stopped (paths/PIDs retained). The two
missing activities ran in reference-resume with identical binary/model/library
and monitor hashes. Both passed. The high-variance llama measurement passed its
monitor and remains included, not cherry-picked away.

The preceding numerical checks for wide split (including 4097/8193 positions)
are in ../gemma4-profile-20261029. No precision-mode changes are mixed into these
results. Current short/1K Gemma and TinyLlama scores are unchanged by this run.
