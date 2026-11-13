# Borrowed MTP attention wide-value experiment, 2026-11-13

Independent opt-in NYA_CUDA_MTP_WIDE_VALUE=1 selects the existing unsplit
wide-value kernel for borrowed prefixes >256 and widths 256/512, provided the
target plan already has NYA_CUDA_WIDE_VALUE=1 and compiled optional kernels.
Native fallback, explicit frozen prefix, scale 1, requested window, retained KV,
output shape, allocation and failure handling remain unchanged. No new C++ code.
C17/C23 builds and six selected borrow/MTP tests each pass.

The Windows C harness loads real Gemma weights, prefills once, then
compares off/on against the SAME plan and frozen prefix in both execution orders.
It checks exact baseline output, independent float64 attention at the unchanged
.0005 scaled bound, transfer/allocation counts, hidden preservation and rewind.
Head widths 256/512 and native/narrow windows are covered. Both linear and ring KV pass at prefix 2048; all outputs are byte-identical
to the basic path and the maximum scaled float64 error is 2.262e-7. NYA_MTP_TRACE=1 must report 108 wide calls per plan, proving dispatch;
the counter is printed at destruction outside the timed calls. Every timed call
includes query upload, attention, synchronization and output download.

The probe is diagnostic: repeated KV/query and current desktop activity do not
establish end-to-end MTP or llama.cpp performance. The real-model ring test also passes CUDA memcheck at prefix 257 with zero
errors and 108 confirmed wide calls. Microtimings are noisy at these submillisecond
shapes, so do not infer an end-to-end MTP gain. The feature remains independent
and opt-in; full-generation evaluation is still pending.

## Separate target-verification question

Normal target dispatch also requires count==1 for wide-value attention. Its
unsplit kernel already indexes independent query blocks, but the split output
and partial allocation are decode-sized; simply changing the condition would
be unsafe for split batches. Profile long-context verification by matrix and
attention layer before choosing a batched lowering. The current borrowed
attention change remains single-query, unsplit, and separate from this question.
