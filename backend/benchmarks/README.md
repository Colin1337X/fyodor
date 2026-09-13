# Runtime benchmark evidence

**Latest pass:** [September 10 optimizations, measurements and validation](STAGE2.md).
The tables below retain the September 8–9 milestone as historical evidence.

Measured September 8–9, 2026 on Windows, AMD Ryzen 5 9600X (6 physical/logical processors reported), RTX 5060 Ti 16 GiB, NVIDIA driver 610.88. Fyodor: GCC 16.2, Release C17, AVX-512 CPU dispatch with six threads including the caller. No llama.cpp/ggml code or library is linked into Fyodor.

Model: [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF), Q4_K_M: 668,788,096-byte file; 667,078,656-byte tensor payload; 22 layers, width 2048; 135 Q4_K, 21 Q6_K and 45 F32 tensors. SHA-256: `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`. The downloaded model is local test data, not bundled source.

## Before and after

Each row uses pp512/tg128, three measured repetitions and one complete warmup. Figures are arithmetic mean token rates ± sample standard deviation. Raw JSON is retained unchanged, including slower intermediate measurements. The original executable was preserved as `.tools/fyodor-bench-baseline.exe` before optimization.

| Stage / raw evidence | pp512 tokens/s | tg128 tokens/s |
|---|---:|---:|
| [Original CPU, scalar](baseline-cpu.json) | 2.04 ± 0.16 | 1.99 ± 0.07 |
| [CPU SIMD + threads](optimized-cpu.json) | 32.75 ± 0.75 | 35.99 ± 2.58 |
| [CPU batched row reuse](final-cpu/fyodor.json) | 78.45 ± 1.32 | 39.89 ± 2.58 |
| [Original CUDA, per-matvec transfers](baseline-cuda.json) | 26.70 ± 0.69 | 29.59 ± 1.17 |
| [CUDA resident, reference kernels](resident-reference-cuda.json) | 42.66 ± 0.14 | 44.39 ± 0.39 |
| [CUDA batched + graphs](resident-graphs-cuda.json) | 418.57 ± 7.79 | 64.31 ± 0.45 |
| [CUDA quant-block specialization](optimized-cuda.json) | 303.91 ± 1.69 | 126.44 ± 0.94 |
| [CUDA 512-token / 32×32 tiles](final-cuda/fyodor.json) | 736.36 ± 0.54 | 172.72 ± 0.34 |
| [CUDA cooperative attention (current)](warp-attention-cuda/fyodor.json) | 792.86 ± 0.33 | 188.91 ± 0.51 |

The final CPU and CUDA comparisons ran serially without concurrent builds. This remains a shared interactive desktop: background applications, clock/power state and thermal conditions vary. Initial CUDA baseline free memory was about 3.1 GiB; later about 15 GiB. TinyLlama fit in both. Some builds/tests overlapped the long original scalar CPU baseline. Thus ratios describe these recorded runs, not isolated kernel speedups or guaranteed hardware performance. The earlier 32×32-tile run (`tile32-batch512-cuda.json`) measured 534.97/127.14; its difference from the later 736.36/172.72 run without a kernel change illustrates environmental sensitivity.

A separate [Vulkan harness smoke measurement](vulkan-synthetic.json) used the synthetic two-layer, width-64, F32 model (436608 bytes): pp512 877.58 ± 10.22, tg128 839.04 ± 8.25 tokens/s. It exercised real Vulkan matvec dispatch, not a resident graph. This tiny model is not comparable to any TinyLlama row above; the record verifies benchmark/backend operation rather than competitive model throughput.

The current run is 29.70× faster in CUDA prefill and 6.38× in CUDA decode than the original recorded CUDA path. The more controlled consecutive attention change improved pp512 736.36→792.86, short tg128 172.72→188.91, and tg128 after an untimed 1024-token prefix 68.86→135.61 tokens/s.

## Same-hardware llama.cpp comparison

Official Windows [llama.cpp](https://github.com/ggml-org/llama.cpp) release b10809, commit `5266f24da`; independent CPU and CUDA binaries. Both use the same GGUF and six CPU threads. CPU batch/ubatch is 32, CUDA batch/ubatch is 512, and CUDA flash attention is disabled in llama-bench. These comparisons align batch sizes but **do not have exact configuration parity**: Fyodor KV is F32; this llama-bench build rejects F32 KV and uses F16. Synthetic token sequences and warmup policies also differ. Fyodor pp computes final-position logits; inspect reference benchmark semantics before extending the comparison.

| Engine / configuration | pp512 tokens/s | tg128 tokens/s | tg128 after 1024-token prefix |
|---|---:|---:|---:|
| Fyodor CPU, batch 32 | 78.45 ± 1.32 | 39.89 ± 2.58 | not measured |
| llama-bench CPU, batch 32 | 468.96 ± 37.91 | 61.03 ± 1.06 | not measured |
| Fyodor CUDA, batch 512 | 792.86 ± 0.33 | 188.91 ± 0.51 | 135.61 ± 0.68 |
| llama-bench CUDA, batch 512 | 11963.96 ± 603.28 | 326.73 ± 0.45 | 319.44 ± 3.14 |

Raw outputs, command lines, hashes and resource samples: [CPU](final-cpu/metadata.json), [CUDA](warp-attention-cuda/metadata.json), [long-context CUDA](warp-attention-context1024/metadata.json). Earlier llama runs at default batch/ubatch 2048/512 are retained as `llama-b10809-*.json`; they are not substituted into the aligned table.

### Latency, memory and transfers

Current Fyodor pp512 mean latency: CPU 6527.79 ms; CUDA 645.76 ms. Decode mean per-token latency: CPU 25.14 ms; CUDA 5.294 ms at short context, 7.374 ms after the 1024-token prefix. JSON also records summed measured elapsed time and per-workload latency.

| Measured resource | CPU | CUDA short | CUDA 1024-prefix |
|---|---:|---:|---:|
| Sampled process peak working set, MiB | 639.87 | 773.48 | 773.53 |
| Resident compressed weights, bytes | n/a | 667078656 | 667078656 |
| Resident KV, bytes | n/a | 23068672 | 51904512 |
| Resident activations/scratch/RoPE/logits, bytes | n/a | 73531136 | 115474176 |
| Sampled whole-GPU memory peak, MiB | 2318 | 3298 | 3364 |

Process samples include startup, mapped weights and CPU recovery buffers. Whole-GPU memory/utilization includes other applications and driver overhead; it is not Fyodor-owned allocation accounting. CUDA sampling ranged 4–99% utilization in the short run, including startup. `*-resources.json` retains individual timestamps, working-set/private bytes, and best-effort whole-device telemetry. Original baselines predate this sampler; baseline RSS/utilization cannot be reconstructed and is unavailable.

Measured resident transfers across three repetitions: pp512 has 3 final-logit downloads and 3 fences; tg128 has 384 downloads and 384 fences. No activation-buffer uploads occur between transformer layers. Token IDs are kernel parameters, which are not counted as buffer transfers. Weight upload, RoPE initialization and graph warmup are excluded. Kernel-node counts are 2532 for three pp runs and 127872 for three tg runs; CUDA graph nodes are counted even when one graph submission launches many nodes. CPU/Vulkan zero counters mean unavailable/nonresident instrumentation.

## Reproduce

Build `fyodor-bench` with normal CMake. It times synthetic token-ID inference, excluding model load, tokenizer, session creation, sampling, text rendering and HTTP. Use warmup >=1 to exclude lazy row-scratch allocation/graph instantiation. pp requests logits only at its final position; tg requests logits for every token and ignores EOS. Each repetition resets the valid prefix and reuses allocations. Backend changes during measurement fail the run rather than disguising fallback as acceleration.

```sh
fyodor-bench -m model.gguf -b cpu -p 512 -n 128 -r 3 --warmup 1 --json
fyodor-bench -m model.gguf -b cuda -p 0 -n 128 --context 1024 -r 3 --warmup 1 --json
python backend/benchmarks/compare.py --fyodor build-cuda/fyodor-bench.exe --llama /path/to/llama-bench.exe --model /path/to/model.gguf --backend cuda --batch 512 --output /new/results/directory
```

For the CPU comparison add `--backend cpu --batch 32` and use the CPU llama binary. The helper creates a new directory, runs both engines serially, sets six threads and identical configured batch limits, preserves commands/hash/environment/outputs, and samples resources. Python is development tooling only; the C engine and benchmark do not require it. Resource sampling adds small supervisory overhead. Select a reference executable containing the requested backend and inspect its reported backend. See the official [llama-bench documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench).

Useful controls: `NYA_CPU_ISA=scalar|avx2|avx512`, `NYA_CPU_THREADS=1..64`, `NYA_CPU_BATCH=1..512` (default 32), `NYA_CUDA_BATCH=1..512` (default 512, reduced on budget pressure), `NYA_CUDA_GRAPHS=0|1`, `NYA_CUDA_REFERENCE=1`. Keep thread count fixed when isolating ISA effects. Reference CUDA uses generic decode matvec, original attention reductions and disabled FMA; fast mode permits FMA, with no unsafe fast-math.

## Profile and remaining bottlenecks

`NYA_CUDA_PROFILE=1` records event timings per operation class and disables executable graphs. Its rates include diagnostic overhead and must not be quoted as normal throughput. [Current short-workload profile](profile-warp-attention.log), pp32 + tg16 without warmup: matrix kernels 136.993 ms, norm 6.046, RoPE 4.117, vector/gating 6.477, attention 3.074, gather 0.257. Matrix operations consume about 87.3% of recorded GPU kernel time. Profiles sum all plan work, including untimed prefix preparation when `--context` is used; they do not isolate decode automatically.

Ranked next optimizations by expected benefit (qualitative, not promised speedups):

1. **CUDA prefill tensor-core GEMM / optional dynamically loaded tuned math library.** The custom 32×32 tile still uses F32 scalar CUDA arithmetic. Preserve custom-kernel fallback and verify any reduced precision. The remaining ~15× pp gap to llama-bench is the largest measured opportunity.
2. **Blocked CPU GEMM microkernels and packed activation tiles.** Batched prefill decodes each row once but performs vector dots across token rows; it lacks a tuned multi-row/multi-token register tile. CPU attention/norm also remain scalar. Current CPU pp is about 6× below the aligned reference.
3. **More efficient quantized decode projections and QKV/norm fusion.** Compressed dot kernels dominate measured GPU time. Reduce redundant weight-scale decoding and combine skinny launches; tune occupancy per matrix shape.
4. **Tiled online-softmax/flash attention and optional F16 KV.** Cooperative attention nearly doubled long-context decode, but still materializes scores and reads F32 KV. Long-context tg remains about 2.36× below the reference.
5. **Stable decode graph metadata and device sampling.** Graphs are recaptured/updated per token; moving token/position metadata and sampling on-device can remove host graph construction and full-logit download overhead.

Residency is implemented for dense LLaMA and dense Gemma including shared KV. PLE, routed experts, MTP borrowed target KV and multimodal overlays retain their established reference paths. Vulkan remains synchronous F32 matvec, with no resident transformer lowering. A general tensor-buffer/op scheduler, deliberate per-layer hybrid offload, continuous request batching, dynamic removal of the 2048-tensor CUDA cache-entry bound, ARM NEON, and GPU training remain incomplete. Oversized resident plans deliberately fall back to whole CPU execution; they do not silently mix CPU layers with per-matrix GPU transfers. This work does not claim completion of every requested backend capability or llama.cpp performance parity.

## Correctness and portability

Current executed configurations: C17 CPU **17/17**; C17 CUDA+Vulkan **30/30**; LLVM-MinGW ASan+UBSan CPU **17/17**; standalone C23 + ONNX Runtime **18/18**; fresh source physically without `/CUDA` and CUDA requested: CPU **17/17**, independent Vulkan **20/20**. All passed. Live server/process protocol checks **36/36** passed. Frontend production build and Windows desktop/installer builds were also exercised; see the frontend README for launch behavior.

Coverage includes seven-format scalar/AVX2/AVX-512/CUDA fast/reference quantized projection parity, batched CPU and tiled CUDA projections for all seven formats with partial row/token tiles, full CPU/CUDA/Vulkan logits, prefill versus token steps, reset/prefix reuse and bounds, forced allocation-budget fallback, partial-allocation rollback, injected CUDA failure with complete CPU prefix replay, graph-disable mode, shared-KV Gemma, speculative/MTP behavior, and existing finite-difference/autograd/checkpoint/export workflows. Hardware-specific numerical tests return skip code 77 when the requested GPU is unavailable, rather than counting CPU fallback as GPU coverage.

On TinyLlama prefixes 8/7/6, current CUDA maximum absolute error was 0.000844, worst RMS 0.000206; CPU optimized maximum absolute error was 0.001307, worst RMS 0.000312. The optional real-model test enforces scaled error <=0.001 and prints absolute/RMS values. Independent Gemma fixtures retain 3e-5 bounds. The synthetic two-layer/64-wide model at 1024/1023/1022 tokens had maximum CUDA absolute error 1.79e-7. The supplied dense Gemma 12B Q4_K_M passed prefixes 3/2/1: maximum absolute error 0.000272, RMS <=0.000057 ([log](gemma12b-logit-parity.log)). These are numerical checks, not model-quality evaluations. The separate early Gemma pp16/tg8 smoke JSON is not comparable to the TinyLlama benchmark table.

The UB sweep retained byte-safe packed weight decoding, checked allocation/stride/cache arithmetic and disjoint worker outputs; corrected physical-core record walking; and exercised parser, generation and training paths under ASan/UBSan with no findings. This is not proof of absence of all UB. Windows GCC and LLVM were tested; Linux/macOS and MSVC inference execution were not.
