# Register-cached wide attention query experiment

Original Fyodor experiment derived from optcpp/attention/wide_batch_split16.cu.
The query vector is loaded once into lane-local registers before the key loop.
The dot-product accumulation sequence, stable softmax, value reduction and
four-partition combine are unchanged. Supported experimental widths are 256
and 512 with 256-thread blocks. No production dispatch or allocation changes.

The current validated long profile attributes 188-192 ms to target attention
per joint decode workload; this is a measured bottleneck, not a speedup claim.
Caching may remove repeated query loads at the cost of register pressure.

check_kernel.py passes 24 cases: widths 256/512, positions 256/8192,
batches 1/2/4, linear/ring storage, four query heads, and grouped KV heads.
All outputs match the original serial split calls byte-for-byte, output/partial/
score guards hold, and sampled float64 checks meet the unchanged 0.0005 bound.
Compute Sanitizer memcheck repeats all 24 cases with exit zero and one zero-error
summary. Input hashes and complete raw evidence are retained. These fixtures
are deterministic; they do not establish adversarial or whole-model correctness.

check_gemma.py adds 16 query heads, KV heads 8/1 and window 1024. Its outcome is
recorded separately. Timing, race/synchronization checks, graph lifecycle and
full-model validation remain pending. No accepted performance score changes.

The Gemma-geometry sweep also passed all 24 cases with byte-identical serial
split outputs and intact guards. gemma-manifest.json verifies source identities
and records the maximum sampled float64 error. It has not yet run under sanitizers.
