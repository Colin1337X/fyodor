# Unsplit wide-value attention for narrow verification batches

The post-cleanup MTP profile found 537 ms / 392 ms of target attention for
window-2/window-4 verification at 8K, versus 182 ms for ordinary generation.
This opt-in reuses the existing original Fyodor wide-value kernel for batches
exactly 2/4 at positions >=256 and head widths 256/512. Set
`NYA_CUDA_WIDE_BATCH=1` before plan creation, alongside the existing compiled
and runtime wide-value options. It defaults off and needs no new optional source.

Only unsplit dispatch is used for batched queries: split partial/output storage
is single-query. Query/output/score offsets already support independent rows;
the existing fewer-than-eight-query allocation covers this restricted dispatch.
No extra allocations or changes to precision, scaling, causal masking or ring
retention are introduced. Missing optional kernels preserve native fallback.
`NYA_MTP_TRACE=1` reports batched-wide calls at plan destruction for dispatch proof.

check_kernel.py imports the existing kernel compiler, tests the actual checked-in
wide_value16.cu against native attention, and independently computes sampled
float64 dot/softmax/value results. All 32 cases pass exact full-output comparison
and guards: widths 256/512, batches 2/4, positions 255/256/1023/8192, linear and
wrapped ring KV. The established .0005 float64 bound is unchanged.

model.c derives from tests/test_verify.c with only the proposal count set to 2/4.
validate.py checks real Gemma scalar versus batched full-vocabulary logits,
invalid requests, reset, rewind, rejected suffix replay and allocation behavior
for linear/ring KV at prefixes 257/1025. It requires actual resident execution
and a batched-wide dispatch trace. All eight real-model cases pass, with 96
confirmed batched-wide calls each; maximum scaled logit error is below 1.7e-5.
Each of C17/C23 passes all 17 selected verification tests, including eight new
fixtures covering widths 256/512, batches 2/4, linear/wrapped ring KV, plus
CPU/Vulkan verification, shared KV, failure handling and disabled graphs.

evaluate.py requires the completed real-model checks before integration tests
and then builds a same-binary off/on generation probe. Weight reuse is enabled
and borrowed-wide attention disabled in both variants. The complete-generation
benchmark is running; no throughput improvement or accepted llama.cpp score is
claimed yet. CUDA memcheck of the new runtime dispatch remains pending.
