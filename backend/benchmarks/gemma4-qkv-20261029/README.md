# Gemma Q/K/V projection fusion experiment

`NYA_CUDA_GEMMA_QKV=1` allows the existing Q4_K Q/K + Q4_K or Q6_K V
fused projection kernel on eligible Gemma decode layers. The general QKV toggle,
reference-mode exclusion, local KV ownership, value-tensor requirement and
matching Q/K row-stride checks still apply. Default is off. No kernel arithmetic,
normalization, RoPE, KV precision, or model representation changes. Gemma's
query/key/value normalization remains after projection. Missing V and shared-KV
layers retain their existing paths. The user's 12B model has 40 eligible sliding
layers; fusion can eliminate 80 launches per decode token.

This targets the measured decode matmul path (see the preceding Gemma profile),
without adding another packed weight copy or a new kernel dependency. The
existing plan-owned packed copies are reused when admitted by memory budgeting;
native layouts remain available when those copies are absent.

## Correctness and integration

The oracle compares all vocabulary logits after two individual decode steps,
including graph capture/replay and the position-256 boundary, for total lengths
3, 4, 257 and 1026. Native and packed weights both produce exactly equal logits.
The existing 0.001 scaled-error limit is unchanged. Equal intentional infinity
masks are handled as in the existing inference oracle. This is fused versus
separate CUDA validation, not an independent CPU/llama.cpp long-context oracle.

`native-oracle.log` and its exit record retain the first failed attempt, which
exposed the separate FDPP retry helper integration regression. That regression
was fixed in 593bf20. `native-success-observation.json` transcribes the successful
session's stdout; `packed-oracle.log` is directly captured output. The three
FDPP tests pass in both C17 and C23 after the fix, with no hardware skips.

## Performance

All 12 serial activities passed environmental checks. F32 KV, full GPU,
32 decode tokens, one warmup and three repeats per forward/reverse order (six
observations per cell), with llama.cpp b10809 FA off. Epic Games Launcher and its
overlay were stopped before timing; process identities are preserved. The run
varies only GEMMA_QKV between Fyodor variants. No build or test overlaps timing.

| Prefix | Separate tok/s | Fused tok/s | llama.cpp tok/s | Fused / llama |
|---|---:|---:|---:|---:|
| 0 | 41.02984 +/- 0.06998 | 41.62019 +/- 0.08406 | 48.61589 +/- 0.03010 | 85.6103% |
| 1024 | 33.84030 +/- 0.02397 | 34.21665 +/- 0.02281 | 42.06414 +/- 0.05827 | 81.3440% |

The gains over the matched native baseline are 1.4388% and 1.1121%. Standard
deviations pool all six observations, including differences between order means.
The option remains off by default pending broader shapes/contexts. It removes
80 launches per token on this model, confirmed by the benchmark counters, with
identical weight memory. The existing 4K/8K scores do not include this option;
no improvement there is inferred. The 125% target is still unmet.

`summary.json` is reproduced by `summarize.py`; exact binary/model/dependency
hashes and commands are in `reference/metadata.json`. The candidate patch and
source manifest preserve the uncommitted code used by that benchmark. Results
were collected October 30 and finalized November 13, 2026.

The updated direct attention oracle also passes all eight split cases after
compiling the complete optional attention chain, including the FDPP retry caller.
Its evidence is in `combined-attention-oracle`.
