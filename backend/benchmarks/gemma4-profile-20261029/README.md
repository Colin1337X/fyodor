# Gemma wide-head split attention — 2026-10-29

`NYA_CUDA_WIDE_SPLIT=1` opts 256/512-wide heads into the existing four-way CUDA split-attention path after position 255. Default behavior is unchanged. The existing split-attention enable/reference gates still apply, and windows of 256 or fewer keys are excluded. C plan admission budgets partial storage for the new widths; combine launch uses enough threads for every output element (512 for a 512-wide head). The device kernel, F32 KV format, precision and softmax-rescaling algorithm are unchanged. This is a C routing/launch change, not a new FA4 backend or complete Flash-Decoding++ pipeline.

For the user's Gemma 4 12B model, ordinary attention exposes only sixteen head blocks; four context partitions expose sixty-four blocks. GEMV remains important, but this change addresses a measured context-dependent cost. Profiling disables graphs and includes prefix work: separate 1024-token prefix-only and prefix-plus-eight-decode measurements attribute approximately 119 ms to the eight decode attention steps, versus 4.709 ms near empty cache. Subtraction is diagnostic, not an exact hardware-counter breakdown or end-to-end acceptance.

## Matched 1K decode

Same Gemma Q4_K_M model and llama.cpp b10809 reference as `../gemma4-local-20261029`. RTX 5060 Ti 16GiB, driver 617.42, F32 KV, full GPU offload, 6 CPU threads, batch512, host logits each token, llama FA off, 1024 prefix tokens and 32 decode tokens. Forward/reverse process orders, one complete warmup and three repetitions each: six observations per cell. Both Fyodor variants enable the same accepted options, including packed Q4 copies; only WIDE_SPLIT changes. All six process-monitor gates passed without an interruption or competitor stop. No build/test overlapped timing.

| Path | tok/s (pooled sample stddev) | % llama.cpp |
| --- | --- | --- |
| Existing | 25.836252 ± 0.020412 | 61.4029% |
| Wide split | 33.858034 ± 0.014664 | 80.4676% |
| llama.cpp | 42.076618 ± 0.035256 | 100% |

Gain over same-run baseline: **31.0486%**. This remains below parity and the 125% target. This run updates the 1K score only; 0/4K/8K performance must be remeasured with the option before replacing their earlier scores. Per-process aggregates/logs are retained; individual repetition timings are unavailable.

## Validation

C17/C23 builds and six selected regression tests each pass. `wide_oracle.c` compares the new path to existing resident attention with native weights, at total token counts257/258/1025/1026, using two individual final decode steps to cover capture/replay and the position256 graph boundary. It uses the existing explicit real-model 0.001 scaled-error bound. Observed maximum is 0.0000204729312. Equal logits, including intentional negative-infinity masks, are skipped exactly as in the existing inference oracle. The first harness incorrectly rejected those identical masks; its diagnostic log is retained, and no kernel fix or relaxed tolerance was used to address it.

`wide_long_oracle.c` extends the comparison to4097/8193 total tokens. Logs and exit files record outcomes. These compare CUDA paths; the earlier CPU/CUDA small-prefix oracle remains the independent CPU evidence. They are not an independent long-context llama.cpp logit oracle. No claim of whole-project matrix completion, AMD execution or Vulkan improvement is made.

Only C host dispatch changes; existing kernel loading and storage cleanup remain in force, and deleting optional C++ directories introduces no new dependency from this change. Defaults stay off pending broader performance/shape validation.

The long-prefix checks passed: max scaled error 0.0000157977536 at4097 and 0.0000214904868 at8193. C23-linked real-model checks at257/258/1025/1026 also passed with the same observed bounds.

Compile each oracle with GCC `-std=c17` (or c23), `-O3 -Wall -Wextra -Werror -I backend/include -I backend/core`, its C source, the corresponding build's `libnya-engine.a`, `-lbcrypt`, and `C:/Windows/System32/vulkan-1.dll`. Run with `NYA_COMPUTE=cuda`, `NYA_CPU_THREADS=6` and the local cuBLAS library path. The harness toggles WIDE_SPLIT internally; optional packed copies are disabled for these numerical checks.
