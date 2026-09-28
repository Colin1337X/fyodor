# Rejected CUDA dot-product experiment — September 26, 2026

The starting point is `b15e04d68455f8d1e6cebc140fadb04a0508bf3c`, including
the accepted [partitioned attention](STAGE9.md). Its post-change profile still
attributes most GPU time to matrix work. This experiment tested two independent
F32 accumulation chains in the existing one-warp-per-row Q4_K/Q6_K decode
kernels. Even and odd 32-element groups accumulate separately and combine
before the unchanged warp reduction. Storage, precision, dispatch geometry,
graph topology, and transfers are unchanged.

**Rejected and removed:** the candidate slows decode by 4.3–4.5% across all
three tested lengths. The production source is restored to `b15e04d`; there
is no new runtime flag, installed artifact, or precision contract.

## Source study and hypothesis

Read-only study of the pinned MIT-licensed ggml revision
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` traced
[`mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/mmvq.cu)
and [`vecdotq.cuh`](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/vecdotq.cuh).
Their SHA-256 values are respectively
`49a8f1d60c06a3c4b7b95fcc6aa0406338a4afc94fdf7fc2d30d94f3670aa525` and
`6607e651085e9ff6e653a6ad23666467d1e4aee23904fd0e3e5f1d90d17dd526`.
Upstream partitions quantized block dot products and uses packed integer
dot operations with Q8_1 activations. That arithmetic is different from
Fyodor's current F32 activation contract. No upstream code was copied and no
activation quantization was introduced. The smaller original experiment only
tested whether shorter accumulation dependencies improve Fyodor's F32 path.

## Correctness and measurements

The quantized-kernel oracle, resident tiny-model check, and Gemma architecture
check pass, along with their training/export fixture. Real TinyLlama forced
decode at 273/272/271 tokens passes against the validated optimized native CPU
path: maximum scaled error **6.93259e-6**, below the unchanged **0.001** bound.
This probe did not receive the full acceptance matrix or GPU sanitizer pass,
because performance rejects it.

Same Windows/GCC 16.2 Release C17, RTX 5060 Ti 16 GiB driver 610.88, Ryzen 5
9600X, TinyLlama Q4_K_M file and F32 KV as Stage 9. Native CUDA decode, cuBLAS
F32 prefill, CUTLASS disabled, batch 512, six CPU threads. The preserved
committed executable is compared with the candidate in both orders, serially,
with two warmups and five repetitions per process (ten observations per cell).
Loading, tokenization, prefix preparation and sampling are untimed. Every
decoded token returns host logits. Resource samples include startup/loading.

All twelve process runs finish with the 7–12% interactive-desktop idle gates
and GPU process identity guards satisfied. No builds, tests, or other agent GPU
work overlap measurements. Existing desktop graphics clients remain open;
this is not a headless measurement. Every observation is retained.

Mean tokens/s ± pooled sample standard deviation:

| Prefix | Workload | Committed | Two chains | Change |
|---:|---|---:|---:|---:|
| 0 | pp512 | 3557.68 ± 58.20 | 3604.21 ± 32.37 | +1.31% |
| 0 | tg128 | 206.48 ± 0.86 | 197.59 ± 1.07 | -4.31% |
| 512 | tg128 | 195.61 ± 1.03 | 186.77 ± 1.37 | -4.52% |
| 1024 | tg128 | 183.91 ± 1.10 | 175.68 ± 1.00 | -4.47% |

Prefill does not use these changed dot kernels; its fluctuation is not a
claimed improvement. No new upstream comparison was run for this rejected
candidate. [Commands, hashes and raw results](inference-20260926/two-chains/metadata.json),
[pooled values](inference-20260926/probe-summary.json), and
[resource summary](inference-20260926/resource-summary.json) are retained.

After timing completed, a separate NVRTC/driver resource inspection used the
same four compiler options and compute_120 target. Q4_K uses 39→40 registers
per thread; Q6_K stays at 40. Both variants have zero local and shared bytes.
This does not show spilling and does not establish the cause of the slowdown.
Instruction scheduling and load behavior need finer profiling before another
arithmetic rewrite. [JIT resource counts](inference-20260926/registers.json).

## Reproduction and restoration

The exact rejected edit is [two-chains.patch](inference-20260926/two-chains.patch).
Apply it to the starting revision, preserve the unmodified benchmark as
`.tools/fyodor-bench-before-dot-ilp.exe`, then build `fyodor-bench`,
`nya-test-inference`, and `nya-test-quant-kernels` in the configured CUDA build.
Run `probe.py <new-output-directory>` after correctness checks have finished.
Run `tables.py <output-directory>` to pool results; `registers.py` compares
the pinned baseline to the current candidate source and must be run while
the patch is applied, after benchmarks have exited.

The rejected executable remains ignored under
`.tools/fyodor-bench-rejected-dot-ilp.exe`. The committed source was restored,
the three CUDA targets rebuilt, and the quantized-kernel check passed again.
Only this diagnostic evidence and documentation are retained in Git.
