# Post-narrow profile and rejected backward tile experiment

At revision a75b354, the same three-update TinyLlama rank-2, context-32
diagnostic profile now reports dense forward 41.86–42.34 ms, mapped forward
54.79–55.48 ms, dense backward 40.75–41.54 ms, and mapped backward
65.03–65.13 ms. Profiling on/off still yields exact checkpoint/export hashes.
`resident-profile.json` records source/binary/model hashes and full commands.
The profiling driver now accepts a fresh output directory, context and rank,
so this measurement is reproducible without editing hardcoded paths or
overwriting earlier evidence.

The dominant remaining operation group prompted a backward-input matrix
experiment: double the reduction tile from 32 to 64, including both shared
arrays, to halve block synchronization frequency. Products and additions remain
in their original ordered F32 sequence. Shared storage increases from 8448 to
16640 bytes per block. This is an original scheduling experiment, not adapted
third-party source.

The full resident matrix suite passed (all formats, exact ordered forward
comparisons, backward references, tails, guards, scratch and finite checks).
The real-model matrix benchmark also passed its independent checks in forward
and reverse job order. Each job has two warmups followed by five samples of
20 dispatches, using the existing `checked-forward`/`checked-reverse` modes.
Operation1 is dX; operation2 is dW and is not trained for frozen base weights.

Results were mixed: Q4_K query/up dX improved, whereas Q6_K down dX regressed.
`summarize.py` pools all ten samples per shape; `summary.json` includes means,
standard deviations and signed throughput changes. No concurrent GPU workload
monitor was attached to these short diagnostic microbenchmarks. They are not
accepted clean performance results or a replacement for end-to-end validation.

The universal 64-wide tile is **rejected and reverted**. Its exact source diff
is retained as `rejected-dx64.patch`. The restored device test also passed.
Future investigation may test a Q4_K-specific specialization, retaining the
32-wide Q6_K path, then validate whole training before accepting a change.
Do not select the best single sample or infer benefit for all formats.

The previously accepted narrow forward optimization remains intact. No
additional production kernel change is delivered by this experiment, and
none of the established numerical tolerances has changed.
