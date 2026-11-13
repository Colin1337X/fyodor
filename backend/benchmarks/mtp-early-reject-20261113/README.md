# Greedy first-proposal early rejection

Previously generation created a complete proposal window and verified it even if
the first proposal differs from the already-available greedy target choice.
The recorded short workload executes 38/44 post-prefill target inputs for
window 2/4, versus 31 ordinary inputs. Not all excess work can be removed;
later proposal correctness is unavailable until verification runs.

build_probe.py reproduces an isolated, default-off experiment from the core and
test sources at d77bb85, controlled by
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

run_short.py first requires all eight
checks, then runs matched off/on short-context generation with the same binary,
weight reuse and batched wide attention enabled. Its counters and exact output
comparison must demonstrate real-model work savings as well as runtime benefit.
The results below justified native-C integration after stop/context boundary
validation. Production enables early rejection by default for greedy decoding;
`NYA_SPECULATIVE_EARLY_REJECT=0` restores the prior path for comparisons. The
setting is read once per generation. Sampled decoding keeps its existing path.

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
these runs, so no accepted llama.cpp score is changed.

## Long-context diagnostic result

All 24 8K generation calls pass exact output parity, with the same four target
steps saved in each batched mode. long-summary.json retains the two warmed
observations and sample deviations for each cell.

| Mode | Off mean ms | On mean ms | Speed ratio |
| --- | ---: | ---: | ---: |
| Ordinary | 935.921 | 933.215 | 1.0029x |
| Serial MTP window 2 | 1114.355 | 1097.123 | 1.0157x |
| Batched MTP window 2 | 1310.731 | 1179.191 | 1.1116x |
| Batched MTP window 4 | 1258.473 | 1155.286 | 1.0893x |

## Integration checks

Eight additional off/on C17/C23 CPU/CUDA runs each pass 96 boundary cases plus
two output-size failures. Matching and deliberately wrong drafts cover target
EOS, draft-only EOS, partial windows and context exhaustion against ordinary
decoding. CUDA runs also exercise sampled batch/serial parity and injected
verification failure. boundaries.json records source/binary hashes and commands.

The production integration passes nine selected generation/MTP tests in each
of C17/C23, including Vulkan, CUDA ring/linear KV and fallback. CPU ASan/UBSan
generation and architecture tests pass. Twelve real-Gemma calls through the
production library, with no early-rejection environment override, pass exact
output parity and confirm the reduced draft/target work. Production prompt
dispatch is unchanged; the standalone timing probe's batched ordinary prefill
is not integrated. Accepted llama.cpp ratios remain 85.61% short / 106.87% at 8K.
