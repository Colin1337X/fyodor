# Optional integer scale-product runtime trial

Based on 419bf09. Original Fyodor algorithm; kernel diagnostics and exhaustive
finite-half coefficient evidence are in decode-integer-scale-20261027.

NYA_ENABLE_INTEGER_GATE defaults OFF. With root optcpp enabled and the source
present, CMake embeds the optional device source without a host C++ compiler.
NYA_CUDA_INTEGER_GATE=1 requests width-2048 dispatch on context creation. This
reuses the existing gate/up eligibility and requires both Q4 strides to equal
1152 bytes. It overrides NYA_CUDA_FIXED_GATE=1; missing source/symbol retains
fixed (if requested) or generic dispatch. Other widths remain unchanged.

Accepted as an opt-in optimization after the checks below. validate.py builds
C17/C23 and runs six selected lifecycle cases per configuration. oracle.py
checks real TinyLlama Q4_K_M against the CPU at capacities 129 and 273 with
unchanged tolerances. verify_removal.py builds a fresh source copy with root
optcpp, backend optcpp, and optional CUDA compartments physically absent, then
runs a real-model inference smoke; this is not the full removal test matrix.

paired.py compares fixed baseline (key baseline) against the integer candidate
(key fixed, retained for compatibility with the existing evidence parser).
Both use selective FDPP. Run it serially after all builds and correctness
checks, then summarize.py. Timing uses eight repetitions and four warmups in
each order at prefixes 0/1024 and 128 decode tokens. The paired trial was interrupted before its eighth observation and is excluded
from acceptance. The larger matched reference protocol below supplies acceptance.



## Accepted evidence

C17 and C23 each pass all six selected lifecycle tests (suite-exits.json).
The real-model CPU oracles at capacities 129 and 273 pass unchanged tolerances
(oracle-exits.json and logs). The fresh physical-removal build and real-model
smoke pass (removal-proof.json). These do not claim a full removal matrix,
AMD/Vulkan validation, or complete long-horizon numerical coverage.

Reference workload: TinyLlama Q4_K_M, F32 KV, full GPU, deterministic tokens,
128 decode tokens, prefixes 0/1024, host logits every token, CPU threads 6,
batch 512, llama FA off. RTX 5060 Ti SM120, driver 617.42; llama b10809 commit
5266f24da75dc449bd56cbed7addb9c8e4a6a73e. Metadata pins model, dependencies,
monitor, and executable hashes. Both Fyodor variants use selective FDPP and
fixed gate dispatch; only the candidate additionally requests integer gate.
Forty observations per cell: 20 repetitions and 10 warmups in each of two
opposite engine orders. Standard deviations pool the full aggregate counts;
the harness retains per-process mean/stddev, not individual repetition values.

| Prefix | Fixed baseline tok/s | Integer candidate tok/s | llama.cpp tok/s | Candidate / llama | Candidate gain |
|---|---:|---:|---:|---:|---:|
| 0 | 366.000 +/- 2.761 | 374.013 +/- 3.453 | 377.890 +/- 4.763 | 98.97% | +2.19% |
| 1024 | 309.815 +/- 2.388 | 316.095 +/- 2.333 | 267.903 +/- 1.053 | 117.99% | +2.03% |

The reference run was interrupted after nine complete observations; authoritative
process inspection found no surviving benchmark. reference-continuation-clean
completed the remaining three, first asserting identical binary, model,
dependency and monitor hashes. assemble_reference.py combines only completed,
passed observations in protocol order. Every one of the 12 accepted activities
passes pre/post idle gates and continuous competitor monitoring. The incomplete
activity remains in the original record but contributes nothing to acceptance.
This segmented desktop run is not a claim of uninterrupted laboratory timing.

Epic launcher and EOS GPU processes repeatedly returned. The earlier rejected
attempts and process identity records are retained; verified Epic processes were
stopped under user authorization. rejected-competitors,
rejected-reference-competitors, reference-continuation, and the interrupted paired
set are excluded. No contaminated results are substituted into the accepted set.

Decision: retain width-2048 integer scale-product dispatch as opt-in. Default
behavior is unchanged. The 125% short-and-long target remains unmet. These
results do not update training or prefill scores. Next: profile the accepted
path and investigate applying exact integer scale products to native Q4 row
dots, with independent correctness and timing evidence before integration.
