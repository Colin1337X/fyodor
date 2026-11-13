# Rejected Gemma gate loop experiment — 2026-11-13

Decision: **do not integrate**. Disabling both outer and inner loop unrolling
reduced JIT register use from 64 to 48 per thread in the preceding resource query,
but the isolated probe showed a large slowdown. Register count alone is not a
performance acceptance criterion. The existing runtime kernel is unchanged.

The candidate is reproduced by a byte-level transformation of the original
`optcpp/matvec/gemma_gate.cu` in the two Python harnesses: rename the entry point,
change outer unroll 8 to 1, and disable inner unrolling. No duplicate dead runtime
kernel or new build option is retained. The transformation was checked byte-for-byte
against the source file used by the oracle, sanitizers and timing probe.
`source-manifest.json` records hashes. Host harnesses are not runtime dependencies.

Correctness: `check_kernel.py` compares the candidate against two packed Q4
projection calls plus native GELU, with deterministic synthetic weights and inputs.
All row counts 1, 7, 8, 9, 63, 64, 65 and 15360 (3840 columns) match bit-for-bit,
including guard canaries, and all outputs are finite. CUDA memcheck, racecheck,
synccheck and initcheck pass with error-exitcode 99; commands and logs are retained.
No real-model oracle was run because kernel timing gave no reason to integrate it.

Diagnostic probe: 15360 rows, 3840 columns, two packed weight matrices (73,728,000
bytes total), F32 input/output. Each captured graph contains 20 operations
(three kernels per separate operation, one per fused operation); three warmup
graph replays, then five samples of ten replays each, in forward and reverse order.
CUDA events measure elapsed device time, divided by 200 operations per sample.
Ten raw samples per variant are retained, including both orders.

| Variant | Mean ms per paired projection + activation | Sample SD ms |
| --- | ---: | ---: |
| Separate packed Q4 projections + GELU | 0.202512 | 0.005951 |
| Existing fused gate | 0.193430 | 0.002469 |
| Candidate with both loops unrolled once | 0.533487 | 0.004743 |

**Not accepted performance evidence:** desktop GPU competitors remained present,
and the probe repeatedly reuses the same synthetic weight banks. GPU/process
snapshots before and after each variant are retained, but this is not a clean
continuous environment gate or end-to-end model workload. No llama.cpp/PyTorch
score is updated and the existing fused kernel is not promoted from these numbers.
The large observed slowdown is sufficient reason to abandon this candidate pending
contrary clean evidence. Its cause has not been established by hardware counters;
loop/control/indexing overhead is a hypothesis, not a measured attribution.
