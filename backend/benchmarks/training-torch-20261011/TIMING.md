# Initial matched timing diagnostics

`timing.py` compares native whole-update CSV timing with synchronized PyTorch
whole-update wall time. Both execute the same random-initialized byte decoder,
corpus windows, optimizer settings, 30 updates, and six CPU workers. First ten
updates and the last diagnostic update are excluded. Four repetitions alternate
native/PyTorch order. Static RoPE and causal masks are cached in the reference.
PyTorch 2.11.0+cu128 uses eager F32 operations with TF32 disabled, double RMS,
RoPE and clipping, and stock fused CUDA AdamW. This is a transparent reference,
not a claim about optimized or compiled PyTorch training generally.

The width64/FF176/two-layer/four-head/two-KV-head/context32 CUDA case passed
all losses and final parameter/gradient/first-moment/second-moment checks at
the unchanged 1e-5 tolerance. Four throughput ratios were 67.12%, 66.52%,
61.78%, and 66.12% (mean 65.38%). These are **diagnostic, provisional timings**:
there was no concurrent GPU activity/resource monitor, and the reference and
native process lifetimes differ. They are not accepted clean competitive scores.
The first attempt stopped on a reference RoPE reshape bug; the corrected
four-run evidence is `timing-64-cuda-diagnostic.json`.

The width256/FF704/two-layer/eight-head/two-KV-head/context128 CUDA case
failed correctness on its first pair. Maximum absolute loss error was
1.0485551e-5 and scaled parameter error was 2.0243995e-5; raw gradients and
both moments passed. Its speed is rejected, recorded as null, and the run
stops immediately. Evidence is `timing-256-cuda-rejected.json`. The cause has
not yet been isolated; do not infer a native or reference defect from this alone.
Further work must localize the first divergence without loosening tolerances.

Raw checkpoints, native CSVs/logs, and exports remain in
`D:/fyodor-validation/training-timing-64-cuda-v2-20261011` and
`D:/fyodor-validation/training-timing-256-cuda-20261011`. Each JSON records the
native binary and harness hashes, configuration, per-step reference timings,
losses, command, and correctness results. No production code changed here.

## First-update localization

Diagnostic mode now permits a single update and one repetition, reports the
worst element of every state array, and never produces a throughput percentage.
`--optimizer scalar` selects stock scalar AdamW instead of fused/foreach AdamW.
Every result explicitly records that its timing environment is unverified.

The large case fails its parameter comparison immediately, while the loss,
raw gradient and both moments pass. CUDA parameter error is 1.9876350e-5;
CPU parameter error is 6.3729021e-5. Native CPU and CUDA parameters and raw
gradients are exactly equal for this update; their moments differ by at most
9.0948432e-13. `compare_checkpoints.py` checks every element and records input
hashes; its result is `divergence-native-cpu-cuda.json`.

Holding all gradients fixed to the native checkpoint, independent stock
PyTorch AdamW reproduces the native parameters within 5.9575228e-8 on both
devices. Selecting scalar CUDA AdamW also leaves the original mismatch
unchanged. This localizes the dominant discrepancy to the independently
computed gradients, rather than a CUDA-only fault or fused optimizer fault.
At the worst CUDA parameter (first layer FFN down weight, flat index149315),
the raw gradients are -2.14842455e-8 in PyTorch and -1.85355020e-8 natively.
Their difference is small in absolute terms but significant relative to the
optimizer epsilon; AdamW magnifies it into a 1.98832e-5 parameter difference.
This explains the failed parameter check without proving either gradient is
more accurate. The failed comparison remains rejected at the original 1e-5.

Reproduce localization with the timing harness and:

```
--width 256 --ff 704 --heads 8 --kv-heads 2 --context 128
--steps 1 --repeats 1 --diagnostic --device cpu|cuda --output NEW_DIRECTORY
```

The three `divergence-step1-*.json` files retain per-array coordinates,
values, raw gradients, losses and the identical-gradient optimizer control.
Corresponding raw files remain under
`D:/fyodor-validation/training-divergence-step1-control-{cpu,cuda}` and
`D:/fyodor-validation/training-divergence-step1-scalar-cuda`.
Next, compare operation-level forward/backward intermediates with a higher
precision independent reference before changing any native numerical behavior.

## Higher-precision independent investigation

The harness now supports diagnostic-only `--reference-precision float64`.
Parameters start from the same native F32 checkpoint, then all reference
activations, gradients and optimizer states use double precision. This mode
cannot generate a throughput percentage. It is not a replacement performance
baseline. Diagnostic reference arrays are saved on disk for subsequent analysis.

`--finite-difference` additionally checks the three implicated coordinates of
the width256 model with central differences at perturbations 1e-3 and 1e-4.
At 1e-4, absolute discrepancies from double autograd are 3.077906e-12,
1.098170e-12, and 2.971265e-13. These checks support the independent reference
at those coordinates; they do not constitute exhaustive derivative validation.
Raw evidence is `divergence-f64-finite-difference.json`.

`precision_analysis.py` compares every native and PyTorch F32 gradient array
against the saved double reference and reports maximum and RMS errors, along
with parameter errors after one update. Input hashes and results are in
`precision-analysis.json`. Native maximum scaled parameter error against the
double reference is 1.625568e-5; PyTorch F32's is 1.551979e-5. Neither is
uniformly closer. In particular, the original FFN-down coordinate's gradients
are native -1.85355020e-8, PyTorch F32 -2.14842455e-8, and double
-1.92881574e-8. The native result is closer at this coordinate.

The original pairwise failure remains a failure. This evidence identifies a
numerically sensitive workload and does not justify relaxing its tolerance,
changing optimizer epsilon, reporting its speed as accepted, or replacing
the native backward implementation blindly with PyTorch's accumulation order.
Operation-level accumulation analysis is still needed before selecting a fix.

Refactoring the reference forward pass into `model_loss` preserved the small
decoder's 30-update result: all four state arrays and losses passed unchanged,
maximum parameter error 6.9272714e-6. The diagnostic result is retained as
`reference-refactor-regression.json`. Production C/CUDA code is unchanged.
Raw new evidence remains under `D:/fyodor-validation/training-divergence-f64-fd-cuda`,
`training-divergence-f32-cuda`, and `training-reference-refactor-regression`.
