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

All three Gemma-geometry sanitizers completed with zero reported errors.
The initial wrapper incorrectly rejected racecheck because it expected memcheck's
summary format. Its failed run record remains unchanged. validate_sanitizers.py
checks the correct tool-specific summaries, exact case inventory, source and
evidence hashes; sanitizers-validated.json records the corrected interpretation.
The parser regression test covers valid, missing, duplicate and nonzero summaries.

The initial nonblocking-stream timing stopped on a byte mismatch at width 512,
position 256, batch four; it did not save partial sample distributions. Its
observed terminal failure is retained in timing-unfenced-failure.json. A separate
direct-launch diagnostic matched serial outputs on all four shapes. probe_fenced.py
fences input uploads and adds independent serial-reference output checks outside
timing. This fixes a harness ordering omission; it does not prove that omission
was the sole cause of the earlier mismatch.

The fenced sweep completed all twelve shapes with output parity and 28 verified
input identities. Each variant has 40 samples in both execution orders, 200
operations per sample and 1000 warmup operations per order. Cached-query latency
is 0.234-7.323% lower across this synthetic sweep. At position 8192, batch four,
width 512 is 1.3412741 ms baseline versus 1.2927074 ms cached (3.621% lower);
width 256/window 1024 is 0.09343785 versus 0.08659529 ms (7.323% lower).
All samples, standard deviations and environment snapshots remain in
timing-fenced.json. This uses repeated synthetic banks and has no continuous
coverage acceptance. Whole-model validation, lifecycle checks and integration
remain pending; no accepted llama.cpp score changes.
