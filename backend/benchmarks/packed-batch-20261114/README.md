# Rejected packed-scale verification-batch experiment

Original Fyodor prototype `optcpp/matvec/small_batch_packed.cu` reuses expanded
Q4 byte scales from the existing 160-byte packed format across 2/4 inputs.
It preserves native small-batch multiplication/reduction order. No runtime
loading, build integration or default dispatch was added.

All 36 shape/batch cases pass byte-exact comparison with the native kernel and
output guards: rows 1/7/9, 3840x65, 3840x15360, 15360x3840; batches 1/2/3/4/5/8;
both candidate input tiles 2/4. Small 256-column cases also pass the existing
independent float64 bound (2e-6 sum of absolute products plus 1e-6). Synthetic
weights use random scale bytes/payloads and fixed exactly representable block
scales. Broader model-logit checks and memory sanitization were not performed.

CUDA graph microbenchmarks run forward and reverse variant order, three warmup
graphs, then five measurements of 100 launches per order. These reuse the same
synthetic weight bank and retain resource snapshots; they are not accepted
whole-model throughput. Packed throughput relative to native using the matching
input tile is 91.931%/93.417% for Q4 gate/up-shaped batches 2/4, and
92.404%/94.271% for Q4 down-shaped batches 2/4. Tile 4 on batch 2 and tile 2 on
batch 4 are also measured and slower. No result is discarded.

Reject for runtime integration. The packed representation reads 160 bytes per
block instead of 144; attributing the slowdown specifically to bandwidth would
require hardware counters not collected here. Arithmetic correctness alone is
not sufficient benefit. No llama.cpp score changes.
