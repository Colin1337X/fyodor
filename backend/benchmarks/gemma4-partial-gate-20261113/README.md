# Partial Gemma gate packing — 2026-11-13

Original native C host-side experiment in `CUDA/resident.inc`. When the full
packed weight copy exceeds the existing memory budget, opt-in partial packing
selects complete, eligible Gemma gate/up pairs in layer order. It uses the
minimum of the configured remaining cache budget and driver-reported free memory
minus the existing reserve (10%, at least 256 MiB). It never packs only half of
a selected pair. Shared tensors are counted once. Native weights remain resident
for prefill and for every unselected layer. Allocation/upload failure discards the
optional copy and retains native execution. Nothing allocates during decode.

Enable the existing `NYA_ENABLE_OPTCPP=ON`, `NYA_ENABLE_GEMMA_GATE=ON` build
options, then set `NYA_CUDA_GEMMA_GATE=1` and the new
`NYA_CUDA_GEMMA_GATE_PARTIAL=1` before session creation. Both runtime options are
experimental and off by default. Full packing, when it fits, is unchanged. With
partial packing disabled, the existing all-or-nothing fallback is unchanged.
No new C++ dependency, public ABI, or kernel source is introduced.

The earlier gate profile attributes about 41% of Gemma short-context matmul time
to gate/up projections. Earlier 8K validation showed zero fused layers because
the full optional copy did not fit. This experiment enables some fusion within
the same budget; it does not establish an end-to-end throughput improvement.

| Validation | Packed layers | Extra bytes | Saved launches | Logits |
| --- | ---: | ---: | ---: | --- |
| C17, 3 tokens, 9000 MiB, byte-dot off | 28 | 2,064,384,000 | 168 | bit-exact |
| C17, 3 tokens, 7050 MiB, byte-dot off | 0 | 0 | 0 | bit-exact |
| C17, 8194 tokens, automatic budget, byte-dot off | 11 | 811,008,000 | 44 | bit-exact |
| C23, 3 tokens, 9000 MiB, byte-dot on | 28 | 2,064,384,000 | 168 | bit-exact |
| C23, same budget, injected optional allocation failure | 0 | 0 | 0 | bit-exact |

Three-token runs execute three individual decode calls; the 8194-token run
prefills 8192 then decodes twice. Each fused layer saves two launches per decode.
The oracle requires whole-pair memory increments, the corresponding launch
reduction, finite nonsuppressed logits, exact declared suppression masks, and
max scaled error <=0.001. Observed max absolute/scaled error and RMS are all zero.
For failure injection only the second session sets
`NYA_CUDA_FAIL_ALLOCATION_AFTER=1`: mandatory storage allocates, existing weights
are reused, and the optional packed storage allocation fails. Resident inference
continues with zero extra packed weight bytes and unchanged kernel counts.

GCC C17 and C23 CUDA builds pass. Commands, environments, exit codes, raw output,
source/binary/model hashes and the candidate patch are retained. `oracle.c` and
`allocation_oracle.c` are focused model harnesses derived from the preceding gate
validation. AMD and other architectures are not covered by these measurements.
Physical-removal integration was tested by the preceding gate change; this patch
only changes native CUDA host planning and adds no optional-source requirements.

Desktop GPU competitors remain active; these are correctness checks, not clean
speed measurements. No llama.cpp/PyTorch score is updated or claimed. Matched
short/long throughput, model quality on other inputs and broader GPU validation
remain pending. The option must remain experimental until speed benefit is measured.
