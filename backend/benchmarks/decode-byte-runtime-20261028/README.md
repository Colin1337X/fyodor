# Aligned byte-metadata Q4 gate runtime (2026-10-28)

Accepted opt-in trial, base f65b076. Original Fyodor C packing and CUDA kernel;
no upstream code copied. Kernel/layout diagnostics are in
 decode-byte-scales-aligned-20261028 (and rejected148-byte predecessor).

NYA_ENABLE_BYTE_GATE defaults OFF. With root optcpp present and
NYA_CUDA_BYTE_GATE=1 before context creation, the optional width2048 symbol is
loaded. Missing source/symbol retains native/integer gate dispatch. The C plan
marks eligible Q4 gate/up pairs (two-dimensional, columns2048, exact1152-byte
native row stride), deduplicates by tensor index, and prepares one packed device
allocation before graph capture. Non-Gemma, fused gate/up only. Other widths,
training/reference contexts and noneligible tensors remain unchanged.

CUDA/q4_byte_pack.h losslessly expands144-byte blocks to160: original half
coefficients, byte scales/minima, zero padding, unchanged128-byte payload.
Checked size arithmetic precedes allocation. C owns the table, packing buffer,
and plan-owned device copy; cleanup frees it after graph/stream work. Admission
checks remaining configured budget, actual free VRAM and reserve. Host failure,
allocation failure or upload failure leaves native execution available. No
packing/allocation occurs in decode. Upload failures are handled in code but
were not fault-injected in this validation.

Native weights are retained for prefill and fallback. Thus runtime memory cost
is an additional full packed copy, NOT merely11.1% extra on gate weights. For
TinyLlama this adds317,194,240 device bytes (302.5MiB), moving reported weights
from667,078,656 to984,272,896 bytes. Host staging is one largest packed tensor
(7,208,960 bytes here), plus the pointer table, freed before decoding. Stats
include the extra device weight bytes. Packing is repeated per plan; no shared
packed cache or preparation-time improvement is claimed.

## Validation

C17 and C23 each pass six selected lifecycle tests. The independent C pack test
passes32,768 scale/minimum encodings, original half/payload preservation,
padding/guards, multi-block stride, invalid sizes and capacity/overflow rejection
under both standards. validate_extra.py proves real-model packing activation,
then resident native fallback on the existing allocation fault injection and a
900MiB configured budget. Real-model CPU oracles at capacities129/273 pass
unchanged tolerances with packing activation logged. The optional-compartment
physical-removal copy builds and runs an inference smoke without a host C++
compiler. These do not claim the entire removal matrix, AMD/Vulkan execution,
malformed nonfinite model validation or a leak-sanitizer run. The initial build
signedness warning and its fix are retained in initial-build-error.log.

## Matched timing

RTX5060Ti SM120, driver617.42, TinyLlama Q4_K_M, F32 KV, full GPU,128 decode tokens,
prefixes0/1024, CPU threads6, batch512, deterministic tokens, full host logits
every token; llama FA off. llama b10809 commit5266f24da75dc449bd56cbed7addb9c8e4a6a73e.
Reference metadata pins executable, dependency, model and monitor hashes.
Baseline uses integer Q4 gate/dot, larger-Q6 and selective FDPP; candidate adds
the aligned packed gate (legacy JSON label fixed). Packing precedes timing.

Forty observations/cell:20 repetitions and10 warmups per process in opposite
engine orders. All12 environment activities pass pre/post idle and continuous
competitor gates. This protocol completed without interruption or competitor
kills; no other builds/tests overlapped timing. Statistics pool per-process
counts/means/deviations; harness does not emit individual repetition values.

| Prefix | Baseline tok/s | Packed candidate tok/s | llama.cpp tok/s | Candidate / llama | Gain vs baseline |
|---|---:|---:|---:|---:|---:|
| 0 | 390.547 +/- 1.563 | 404.749 +/- 3.655 | 380.815 +/- 2.503 | 106.29% | +3.64% |
| 1024 | 325.883 +/- 1.269 | 336.578 +/- 1.332 | 269.371 +/- 1.016 | 124.95% | +3.28% |

Retain as opt-in because of its memory tradeoff. Current throughput106.29% short /
124.95% long versus llama.cpp;124.95 is below125, and the target on both remains
unmet. Training and prefill scores are unchanged. Next reprofile this baseline;
do not infer a broad layout win for every width or backend.

Reproduce serially with validate.py, validate_extra.py, oracle.py,
verify_removal.py, then reference.py <fresh directory> and summarize_reference.py
(expects reference/). Preserve prior outputs. Original comparison hashes are in
manifest.json; no performance-relevant source changed after measurement.
