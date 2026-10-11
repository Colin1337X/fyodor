# Post-Q4_K profile and rejected forward tile

At b742ebe, the synchronized resident-operation profile reports these steady
diagnostic ranges for TinyLlama Q4_K_M rank-2 LoRA, context 32:

| Operation group | Per-update diagnostic time |
| --- | ---: |
| Quantized forward | 55.96–57.56 ms |
| Quantized backward | 50.72–51.78 ms |
| Dense forward | 41.91–42.65 ms |
| Dense backward | 42.65–42.79 ms |

The previous post-narrow profile recorded approximately 65 ms for quantized
backward. This new diagnostic agrees with the direction of the separately
validated Q4_K backward optimization; it is not a controlled speedup estimate.
The random-init width-256, FF-704, two-layer, context-128 profile still places
dense forward first at 5.07–5.17 ms. Profiling on/off checkpoint and export hashes
match exactly for that random-init fixture. `resident-profile.json` records
commands, hashes and all three updates, including the first update omitted
from these ranges. The profiler synchronizes around each operation and includes
finite checks, so these values are not isolated kernel times.

The largest remaining LoRA group motivated a Q4_K forward reduction-tile
experiment: enlarge shared weight/input tiles from 32 to 64 columns, retaining
32 for other formats and the ordered F32 arithmetic. Shared memory doubles
from 8,448 to 16,640 bytes per block. This is original scheduling code, with no
external source adaptation. The full resident device suite passed, including
ordered-forward comparisons, backward checks, guards and scratch reuse.

The real-model matrix benchmark ran baseline-forward, candidate-forward,
candidate-reverse and baseline-reverse. Each shape has two warmups and five
samples of 20 dispatches per order; the summary pools all ten timed samples.
The benchmark's checked modes validate numerical results independently.

| Q4_K tensor | Tokens | Throughput change |
| --- | ---: | ---: |
| Query | 8 | −1.6% |
| Query | 64 | +1.8% |
| FFN up | 8 | +1.1% |
| FFN up | 64 | −3.2% |

The unchanged Q6_K down kernel also varied by roughly +1%. Standard deviations
and all samples are retained in the logs and `summary.json`. These short
diagnostics had no competing-process monitor attached and must not be used as
clean throughput acceptance. They provide no convincing evidence of benefit
and include a regression on the larger FFN-up case. **The forward candidate is
rejected and restored**, not shipped. `rejected-forward64.patch` preserves the
experiment; metadata records exact binary/source hashes. The restored build and
full device suite passed. The accepted Q4_K backward specialization remains.

Reproduce the profile with `training-torch-20261011/resident_profile.py --output
NEW_DIRECTORY --context 32 --rank 2`. Matrix logs come from
`nya-bench-training-device MODEL checked-forward` and `checked-reverse`.
Run `python backend/benchmarks/training-forward-tile-20261012/summarize.py` to
regenerate the pooled matrix summary.

The next investigation should distinguish matrix dimensions within the dense
and mapped groups before changing scheduling again. Increasing the forward
reduction tile alone did not reproduce the backward tile's benefit. Inference
and PyTorch reference scores are unchanged by this rejected experiment.
