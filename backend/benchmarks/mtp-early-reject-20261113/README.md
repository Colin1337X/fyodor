# Greedy first-proposal early rejection (isolated experiment)

Current generation creates a complete proposal window and verifies it even if
the first proposal differs from the already-available greedy target choice.
The recorded short workload executes 38/44 post-prefill target inputs for
window 2/4, versus 31 ordinary inputs. Not all excess work can be removed;
later proposal correctness is unavailable until verification runs.

build_probe.py prepares an isolated, default-off experiment controlled by
NYA_SPECULATIVE_EARLY_REJECT=1. It compares only the first greedy proposal with
the existing target logits. On a mismatch it stops drafting and bypasses batched
verification; the existing correction path advances the target and, for a
standalone draft, advances the draft with the corrected token. MTP state is
replaced from the target at the next round as usual. Sampled decoding is untouched.
The greedy sampler validates finite logits, preserves tie breaking, does not
mutate logits and consumes no RNG. Proposal/target counters reflect actual work.

The standalone C17 and C23 builds each pass eight off/on CPU/CUDA generation/
architecture test runs. They retain the existing greedy output checks, independent sampled
distribution checks and sampled MTP host/bridge comparisons. The deliberately
wrong standalone draft additionally must perform exactly one proposal per emitted
token and only prompt+generated-1 target steps when enabled, proving that known-
wrong work was removed. The generated test-source hashes are in probe-build.json.

This is not integrated into production. run_short.py first requires all eight
checks, then runs matched off/on short-context generation with the same binary,
weight reuse and batched wide attention enabled. Its counters and exact output
comparison must demonstrate real-model work savings as well as runtime benefit.
Only retain the optimization if savings justify the extra argmax scan.
Additional stop/context boundary validation remains a prerequisite for integration.

## Short-context diagnostic result

All 24 complete generation calls pass exact text/token/stop parity. First rounds
are excluded; the following means use two warmed observations per cell, with
sample deviations and all raw observations in short-summary.json.

| Mode | Off mean ms | On mean ms | Speed ratio |
| --- | ---: | ---: | ---: |
| Ordinary | 757.249 | 757.305 | 0.9999x |
| Serial MTP window 2 | 851.901 | 843.094 | 1.0104x |
| Batched MTP window 2 | 904.290 | 819.561 | 1.1034x |
| Batched MTP window 4 | 898.779 | 827.620 | 1.0860x |

Window 2 draft proposals fall 22->20 and total target steps 63->59; window 4
proposals fall 32->29 and target steps 69->65. Accepted proposals and output are
unchanged. Both modes still trail ordinary decoding. The environment gate rejects
these runs, so no accepted llama.cpp score is changed. Matched 8K timing is now
running via run_long.py. Production source remains unchanged by this experiment.
