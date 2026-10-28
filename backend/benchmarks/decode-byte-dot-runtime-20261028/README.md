# Aligned byte-metadata Q4 decode integration

Experimental default-off `NYA_ENABLE_BYTE_DOT` embeds the root `optcpp/matvec/byte_scale_dot.cu` source only when present and OPTCPP is enabled. `NYA_CUDA_BYTE_DOT=1` enables the complete dot/add/QKV symbol group for non-reference inference. It does not affect training or nonresident dispatch.

The C plan packs eligible layer projections and output matrices into the existing plan-owned allocation. Matrices must have complete two-dimensional Q4_K data, positive rows and columns, and columns divisible by 256; overflow is checked. One original 144-byte block becomes 160 bytes. The original allocation remains resident for batched prefill and fallback, so the extra memory is the entire packed copy, not merely the 11.11% format expansion. The existing gate and new projection paths share copies.

Decode dispatch checks tensor shape and packed-pointer availability. QKV changes Q/K strides and Q4 V stride; Q6 V retains its original pointer, stride and arithmetic. Fused gate dispatch additionally requires its own loaded kernel. Missing symbols, absent optional source, host/device allocation failure, memory-budget rejection or upload failure retain native dispatch. Packing is complete before graph capture; cleanup follows the existing plan-owned allocation lifetime.

Validation and matched benchmark results are recorded alongside this file as completed. Kernel diagnostic evidence is in `../decode-byte-dot-20261028`; those kernel-only gains are not end-to-end performance claims.

## Validation and matched result

C17 and C23 builds and six selected tests each passed. TinyLlama CPU comparisons at capacities 129/273 passed unchanged tolerances with the packed allocation active. An independent projection-only run with BYTE_GATE=0 also passed CPU parity. C17/C23 packing tests, allocation-failure and 900 MiB budget fallbacks passed. A fresh source copy physically missing root/backend optcpp and optional cutlass/cute built and ran real-model inference with no C++ compiler configured. This removal smoke is not the full project test matrix. Upload-failure handling was inspected, not fault-injected; no leak sanitizer was run.

The initial reference attempt was rejected before timing because Roblox PID 2188 was using the GPU. Its executable identity was verified and the process stopped under user authorization; `stopped-competitor.json` and rejected `reference/environment.json` retain the evidence. The uninterrupted `reference-clean` run passed all 12 activity gates. No build or test overlapped accepted timing.

RTX 5060 Ti 16GB, SM120, driver 617.42; Ryzen 5 9600X; TinyLlama Q4_K_M; F32 KV; full GPU offload; 6 CPU threads; batch 512; 128 decode tokens with 0/1024 prefix tokens; llama.cpp b10809 commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e, FA off; host logits downloaded each token. Each cell pools 40 observations from two reverse-order processes, each with 10 warmups and 20 repetitions. Raw per-process means/stddevs and logs are retained; individual repetition samples are not available. `fixed` is the legacy label for the byte-dot candidate. The baseline includes all previously accepted options, including byte gate packing.

| Prefix | Baseline tok/s | Candidate tok/s | llama.cpp tok/s | Candidate / llama.cpp | Gain over baseline |
| --- | --- | --- | --- | --- | --- |
| 0 | 405.1980 ± 2.1818 | 421.5243 ± 2.4815 | 381.4420 ± 2.4133 | 110.5081% | 4.0292% |
| 1024 | 336.3926 ± 1.2814 | 346.8666 ± 1.2317 | 267.8529 ± 2.8261 | 129.4989% | 3.1136% |

Values after ± are pooled sample standard deviations, not confidence intervals. This passes the 125% long-decode target in this workload but not the short target. The baseline itself measured 125.5886% on long decode in this run; compare against the same-run baseline rather than attributing all change since prior sessions to this patch.

Native device weights are 667,078,656 bytes. Packed projections add 530,186,240 bytes (505.625 MiB), total 1,197,264,896 bytes, versus 984,272,896 with gate-only packing. Copies are per plan and prepared before timed decode. The experiment remains opt-in; no default or training-performance claim changes. The latest historical small-training PyTorch ratio remains 71.0%, not remeasured here.
