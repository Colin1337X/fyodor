# Four-part portable online attention prototype

Original Fyodor implementation in `optcpp/attention/portable_f32.cu`. Four
independent KV ranges keep online max/denominator/numerator state and emit
partials to the existing stable native combine kernel. Empty partitions emit
negative infinity, zero denominator and zero numerator. The split entry is a
single-query decode kernel; batched queries use the original unsplit entry.
No runtime dispatch selects the new entry yet. No default changes.

`check_kernel.py OUT [split|unsplit]` compares against independent Python double
attention with unchanged 1.5e-4 scaled-error tolerance. Cases cover empty splits,
odd width 257, width 512, causal batched queries (unsplit only), grouped queries,
sliding windows, 8K context and large logits. Output canaries detect overruns.
The original split oracle's maximum scaled error is 9.10e-8; the unsplit
regression oracle's maximum is also 9.10e-8. CUDA sanitizer logs and their
commands are retained separately for each mode/tool.

## Diagnostic timings, not model performance

`probe.py` captures 20 calls per graph, three warmup graphs, then five samples
of ten graph replays per path, in both orders. Each sample is divided by 200.
This reuses a warm KV bank and omits model matmuls, so it cannot establish a
model throughput gain. The environment monitor accepted the timing activity.

| Head width / prefix / window | Portable ms | Portable four-part ms | Native four-part ms |
|---|---:|---:|---:|
| 256 / 1024 / 1024 | 0.193223 | 0.056251 | 0.046528 |
| 512 / 1024 / full | 0.359606 | 0.108733 | 0.086919 |
| 512 / 4096 / full | 1.426356 | 0.412005 | 0.329171 |
| 512 / 8192 / full | 2.930861 | 0.823488 | 0.884501 |

All cases use 16 query heads; the sliding case has eight KV heads, the others
one. Ten per-call timing samples are retained in `timings/results.json`.
Partitioning is 3.3-3.6 times faster than the unpartitioned portable path, but
still slower than native split attention except at the tested 8K shape (6.90%
lower kernel latency). Full-model validation and matched timing are required
before integrating a dispatch policy. No AMD or Vulkan split execution claim.

## Combined runtime regression found and fixed

A subsequent real-model runtime check found that the new helper argument had
not been propagated to `flash_decoding_pp.cu`'s stable retry path. The isolated
oracle compiled only portable attention and therefore missed this caller;
the assembled runtime failed NVRTC compilation and selected CPU fallback.
The retry now explicitly requests one partition. The direct oracle now compiles
all four optional attention source files together, matching their runtime
ordering. Earlier isolated sanitizer results remain valid for the tested
kernels, but did not prove runtime integration. Real-model resident execution
was rechecked after the fix; combined FDPP regression results are in
`../gemma4-qkv-20261029`.
