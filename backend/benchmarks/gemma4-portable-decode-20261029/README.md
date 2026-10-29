# Gemma portable online attention: rejected decode promotion

All 12 serial activities passed the retained environment gates. RTX 5060 Ti
SM120; Gemma4 12B Q4_K_M; F32 KV; full GPU; 32 decode tokens; one warmup and
three measured repetitions per order, forward and reverse. Exact model,
binaries, dependencies, commands and flags are in `reference/metadata.json`.
llama.cpp b10809 / 5266f24 has FlashAttention off. Both native variants enable
wide native split attention, byte-dot and short-context FDPP. FDPP takes
precedence over portable attention on eligible short 256-wide heads.

| Prefix | Native split tok/s | Portable tok/s | llama.cpp tok/s | Portable / llama |
|---|---:|---:|---:|---:|
| 0 | 40.9338 +/- 0.2820 | 41.0884 +/- 0.0689 | 48.5288 +/- 0.0651 | 84.6682% |
| 1024 | 33.7309 +/- 0.2527 | 21.5824 +/- 0.0924 | 41.9556 +/- 0.1863 | 51.4410% |

Standard deviations pool all six observations using within-order sample
variance and between-order mean differences. `summarize.py` reproduces these
numbers from the preserved JSON. No significance claim is made for the 0.38%
short-context difference. The 36.02% long-context regression rejects promotion.
Accepted production performance is unchanged. The prior native result is
reproduced here at 84.35% short and 80.40% at 1K of matched llama.cpp.

The wrapper preserves raw outputs before environmental acceptance, so a rejected
activity does not lose its timing evidence. These runs predate the four-part
prototype; see `../portable-split-20261029` for its separate kernel experiment.
