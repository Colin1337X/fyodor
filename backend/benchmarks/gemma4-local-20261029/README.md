# Local Gemma 4 12B evaluation — 2026-10-29

User-selected files in E:/models were read without modification. `model-inventory.json` records metadata, tensor shapes, quantization types and file sizes for the main Q4_K_M model, BF16 unified projector, Q4_0 MTP assistant, and Q8_0 Nomic MoE embedding model. Main model identity is also SHA256-pinned in reference metadata.

## Decode comparison

RTX 5060 Ti 16 GiB, SM120, driver 617.42; 6 CPU threads; full GPU offload; F32 KV; batch 512; 32 decode tokens after the listed untimed prefix. Each process has one complete workload warmup and three repetitions; forward/reverse engine orders give six observations per cell. Values after ± are pooled sample standard deviations, not confidence intervals. Individual repetition samples are unavailable; original process summaries/logs remain. `fixed` is the historical harness label for Fyodor. No prefill-performance claim is made.

llama.cpp b10809, commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e, FA off, host logits every token, same deterministic token IDs. Fyodor enables the previously accepted optional integer/byte projection flags and selective FDPP. All 16 clean activities passed monitoring. The initial attempt was rejected before timing due to Epic Launcher/EOS overlay processes; verified processes were stopped under user authorization and recorded. No build or test overlapped accepted timing. The smoke context sweep is diagnostic, not accepted benchmark evidence.

| Prefix | Fyodor tok/s | llama.cpp tok/s | Fyodor / llama.cpp |
| --- | --- | --- | --- |
| 0 | 41.0511 ± 0.0893 | 48.6580 ± 0.0406 | 84.3666% |
| 1024 | 25.8442 ± 0.0127 | 42.0848 ± 0.0307 | 61.4099% |
| 4096 | 18.9536 ± 0.0147 | 31.1867 ± 0.0189 | 60.7745% |
| 8192 | 14.5695 ± 0.0032 | 23.1048 ± 0.0105 | 63.0582% |

All measured Gemma workloads remain below parity and the 125% target. TinyLlama's separate 110.5081%/129.4989% result must not be generalized to this model. Training/PyTorch results are unchanged.

The main model is 7,381,381,760 file bytes, 48 layers, embedding width 3840, sliding head width 256/full head width 512. It advertises 262144 context tokens, not a tested hardware capacity. Resident native device weights are 7,365,558,464 bytes. Optional packed copies add 5,829,427,200 bytes when budget permits. At 4K and 8K, the C budget check skips packing and retains native resident weights. The 8K/32-token run uses 5,659,164,672 KV bytes and 454,144,520 scratch bytes. All tested contexts remain resident. This is not proof that the maximum advertised context fits.

## Correctness and companions

The native CPU/CUDA oracle at capacity 9 passes unchanged tolerances, covering prefixes 7/8/9. Largest observed scaled error is 0.000627708038. This is a small-prefix CPU parity check, not cross-runtime logit parity or long-context numerical certification.

The mmproj loads and produces finite 25x3840 image embeddings from a 240x240 zero-image smoke input and 1x3840 audio embeddings from one zero 640-feature row. These are execution/finite-output checks, not perceptual quality or end-to-end multimodal answer tests. Its actual patch weights consume 48x48 RGB; the metadata's 16-pixel teacher patch size does not define this unified projector input. A first 224x224 request was correctly rejected and its log is retained.

The MTP assistant loads with draft_supported=1 and is correctly rejected as standalone generation. The initial eight-token MTP validation used the wrong older Gemma chat syntax and timed out after 180 seconds; no success or speedup is claimed for it. `mtp-template.c` uses the simple no-tools/no-thinking branch of the embedded Gemma 4 template and disables optional byte packing for a bounded four-token comparison. Ordinary generation returns a sensible greeting. See its exact outcome/log for speculative parity.

Nomic's nomic-bert-moe architecture is inspectable but has neither native named-tensor execution nor generation support in this build. Its advertised context is 2048. The generic ONNX-disabled diagnostic is misleading for this GGUF model: enabling ONNX alone would not supply the missing native MoE embedding implementation. No embedding throughput or output-quality result is claimed.

Next work: profile Gemma's resident attention and projections; establish longer-context/cross-runtime numerical parity; audit MTP performance; implement the missing embedding path. No tolerances were relaxed and no production code was changed by these probes.

The corrected bounded MTP run passed exact greedy text/token-count parity: both paths returned four tokens, `Hello! I hope`; two draft tokens were proposed, one accepted, and 21 target steps reported. Byte packing was disabled for this check. No matched MTP latency benchmark was performed. This does not certify larger proposal windows or long-context speculative decoding.

Probe build: `.tools/w64devkit/bin/gcc.exe -std=c17 -O3 -Wall -Wextra -Werror -I backend/include <probe.c> build-cuda/libnya-engine.a -lbcrypt C:/Windows/System32/vulkan-1.dll -o <probe.exe>`. The compiler directory must be on PATH. Harnesses are diagnostic only; inspect recorded return codes and output assertions rather than treating registry load success as execution support.
